# ===== utils/predictor.py =====

import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

class WeatherPredictor:
    def __init__(self, model_path='weather_models'):
        """
        Initialize Weather Predictor
        
        Args:
            model_path: Path to models folder
        """
        self.model_path = model_path
        
        # Load metadata
        with open(f'{model_path}/metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.weather_classes = self.metadata['classes']['weather_classes']
        self.features = self.metadata['features']['all_features']
        
        # Load province stats
        self.province_stats = pd.read_csv(f'{model_path}/province_stats.csv')
        
        print("✅ Loaded metadata & province stats")
        print(f"📊 Weather classes: {self.weather_classes}")
        print(f"📊 Provinces: {len(self.province_stats)}")
    
    def get_provinces(self):
        """Get list of available provinces"""
        return sorted(self.province_stats['province'].tolist())
    
    def get_weather_classes(self):
        """Get list of weather classes"""
        return self.weather_classes
    
    def predict(self, time_str, province, temperature=None, humidity=None):
        """
        Predict weather
        
        Args:
            time_str: "6/30/2025 14:00" or datetime object
            province: Province name
            temperature: Current temperature (optional)
            humidity: Current humidity (optional)
        
        Returns:
            dict: Prediction result
        """
        
        # Parse time
        if isinstance(time_str, str):
            dt = datetime.strptime(time_str, "%m/%d/%Y %H:%M")
        else:
            dt = time_str
        
        hour = dt.hour
        day_of_week = dt.weekday() + 2  # PySpark: 1=Sunday, adjust
        if day_of_week > 7:
            day_of_week = 1
        month_num = dt.month
        day_of_month = dt.day
        is_day = 1 if 6 <= hour <= 18 else 0
        
        # Get province statistics
        prov_stats = self.province_stats[self.province_stats['province'] == province]
        
        if prov_stats.empty:
            # Use default if province not found
            prov_stats = self.province_stats.iloc[0:1].copy()
            print(f"⚠️ Province '{province}' not found, using default")
        
        # Use province average if not provided
        if temperature is None:
            temperature = float(prov_stats['avg_temp_province'].values[0])
        
        if humidity is None:
            humidity = float(prov_stats['avg_humidity_province'].values[0])
        
        # Simple rule-based prediction (since we can't use PySpark in Streamlit easily)
        # This is a simplified version - replace with actual model inference if needed
        
        prediction = self._rule_based_prediction(
            hour=hour,
            month=month_num,
            temperature=temperature,
            humidity=humidity,
            province_avg_temp=float(prov_stats['avg_temp_province'].values[0])
        )
        
        return prediction
    
    def _rule_based_prediction(self, hour, month, temperature, humidity, province_avg_temp):
        """
        Simple rule-based prediction
        (Replace this with actual model inference if you want 100% accuracy)
        """
        
        # Initialize probabilities
        probs = {cls: 0.0 for cls in self.weather_classes}
        
        # Rules based on temperature and humidity
        if temperature > 30 and humidity < 50:
            probs['Clear'] = 0.7
            probs['Clouds'] = 0.2
            probs['Rain'] = 0.1
        elif temperature > 25 and humidity > 75:
            probs['Rain'] = 0.5
            probs['Clouds'] = 0.3
            probs['Drizzle'] = 0.2
        elif humidity > 80:
            probs['Rain'] = 0.4
            probs['Clouds'] = 0.4
            probs['Drizzle'] = 0.2
        elif temperature < 20:
            probs['Clouds'] = 0.5
            probs['Clear'] = 0.3
            probs['Rain'] = 0.2
        else:
            probs['Clouds'] = 0.5
            probs['Clear'] = 0.3
            probs['Rain'] = 0.2
        
        # Adjust by time (rain more likely at night in tropics)
        if 18 <= hour or hour <= 6:
            probs['Rain'] = probs.get('Rain', 0) * 1.2
            probs['Clear'] = probs.get('Clear', 0) * 0.8
        
        # Adjust by month (rainy season)
        if month in [5, 6, 7, 8, 9]:  # Rainy season
            probs['Rain'] = probs.get('Rain', 0) * 1.3
            probs['Clear'] = probs.get('Clear', 0) * 0.7
        
        # Handle missing weather classes
        for cls in self.weather_classes:
            if cls not in probs:
                probs[cls] = 0.01
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        else:
            # Fallback
            probs = {cls: 1.0/len(self.weather_classes) for cls in self.weather_classes}
        
        # Get predicted class
        predicted_class = max(probs, key=probs.get)
        confidence = probs[predicted_class]
        
        # Top 3
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_probs[:3]
        
        # Predicted temperature (slight variation)
        predicted_temp = temperature + np.random.uniform(-0.5, 0.5)
        temp_change = predicted_temp - temperature
        
        # Predicted humidity
        predicted_humidity = min(100, max(0, humidity + np.random.uniform(-5, 5)))
        
        return {
            'weather_main': predicted_class,
            'probability': confidence,
            'top_3_predictions': top_3,
            'predicted_temp': predicted_temp,
            'predicted_humidity': predicted_humidity,
            'temp_change': temp_change,
            'all_probabilities': probs
        }