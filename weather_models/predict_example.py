
# ===== WEATHER PREDICTION EXAMPLE =====
# File này hướng dẫn cách sử dụng models đã train

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.regression import GBTRegressionModel
from pyspark.ml.feature import StandardScalerModel, StringIndexerModel, VectorAssembler
import pandas as pd
import json

# 1. Khởi tạo Spark
spark = SparkSession.builder.appName("WeatherPrediction").getOrCreate()

# 2. Load metadata
with open('weather_models/metadata.json') as f:
    metadata = json.load(f)

weather_classes = metadata['classes']['weather_classes']
features = metadata['features']['all_features']

# 3. Load models
rf_model = RandomForestClassificationModel.load("weather_models/rf_classifier")
province_indexer = StringIndexerModel.load("weather_models/province_indexer")
city_indexer = StringIndexerModel.load("weather_models/city_indexer")
scaler = StandardScalerModel.load("weather_models/scaler")

# 4. Load province statistics
province_stats = pd.read_csv('weather_models/province_stats.csv')

# ===== FUNCTION DỰ ĐOÁN =====
def predict_weather(time_str, province_str, current_temp=None, current_humidity=None):
    """
    Dự đoán thời tiết
    
    Args:
        time_str: "6/30/2025 14:00"
        province_str: "An Giang-Chau Doc"
        current_temp: Nhiệt độ hiện tại (optional, nếu không có sẽ dùng TB tỉnh)
        current_humidity: Độ ẩm hiện tại (optional)
    
    Returns:
        dict: {
            'weather_main': 'Clouds',
            'probability': 0.85,
            'top_3_predictions': [...]
        }
    """
    
    # Parse time
    from datetime import datetime
    dt = datetime.strptime(time_str, "%m/%d/%Y %H:%M")
    
    hour = dt.hour
    day_of_week = dt.weekday() + 1  # PySpark: 1=Monday
    month_num = dt.month
    day_of_month = dt.day
    is_day = 1 if 6 <= hour <= 18 else 0
    
    # Lấy province statistics
    prov_stat = province_stats[province_stats['province'] == province_str]
    
    if prov_stat.empty:
        print(f"⚠️ Province '{province_str}' không có trong dữ liệu!")
        return None
    
    # Nếu không có current values, dùng TB province
    if current_temp is None:
        current_temp = prov_stat['avg_temp_province'].values[0]
    if current_humidity is None:
        current_humidity = prov_stat['avg_humidity_province'].values[0]
    
    # Tạo input DataFrame
    input_data = spark.createDataFrame([{
        'hour': hour,
        'day_of_week': day_of_week,
        'month_num': month_num,
        'day_of_month': day_of_month,
        'is_day': is_day,
        'province': province_str,
        'city': province_str.split('-')[1] if '-' in province_str else province_str,
        'temperature': current_temp,
        'humidity': current_humidity,
        'pressure': prov_stat['avg_pressure_province'].values[0],
        'wind_speed': prov_stat['avg_wind_province'].values[0],
        # ... các features khác (dùng giá trị mặc định hoặc TB)
        'temp_range': 2.0,
        'cloudcover': 50.0,
        'precipitation': 0.0,
        'visibility': 10000.0,
        'feels_like': current_temp,
        'avg_temp_province': prov_stat['avg_temp_province'].values[0],
        'std_temp_province': prov_stat['std_temp_province'].values[0],
        'avg_humidity_province': prov_stat['avg_humidity_province'].values[0],
        'avg_pressure_province': prov_stat['avg_pressure_province'].values[0],
        'avg_wind_province': prov_stat['avg_wind_province'].values[0],
        # Lag features (giả sử bằng current)
        'temp_lag_1h': current_temp,
        'temp_lag_3h': current_temp,
        'humidity_lag_1h': current_humidity,
        'pressure_lag_1h': prov_stat['avg_pressure_province'].values[0],
        'temp_ma_3h': current_temp,
        'temp_ma_6h': current_temp,
        'humidity_ma_3h': current_humidity,
        'temp_change_1h': 0.0,
        'temp_min': current_temp - 1,
        'temp_max': current_temp + 1,
        'wind_gust': prov_stat['avg_wind_province'].values[0] * 1.5
    }])
    
    # Encode
    input_encoded = province_indexer.transform(input_data)
    input_encoded = city_indexer.transform(input_encoded)
    
    # Assemble features
    assembler = VectorAssembler(inputCols=features, outputCol='features_raw')
    input_assembled = assembler.transform(input_encoded)
    
    # Scale
    input_scaled = scaler.transform(input_assembled)
    
    # Predict
    prediction = rf_model.transform(input_scaled)
    
    # Lấy kết quả
    result = prediction.select('prediction', 'probability').collect()[0]
    predicted_class = int(result['prediction'])
    probabilities = result['probability'].toArray()
    
    # Map về label
    weather_label = weather_classes[predicted_class]
    confidence = probabilities[predicted_class]
    
    # Top 3 predictions
    top_3_indices = probabilities.argsort()[-3:][::-1]
    top_3 = [(weather_classes[i], probabilities[i]) for i in top_3_indices]
    
    return {
        'weather_main': weather_label,
        'probability': float(confidence),
        'top_3_predictions': top_3,
        'input_info': {
            'time': time_str,
            'province': province_str,
            'temperature': current_temp,
            'humidity': current_humidity
        }
    }

# ===== SỬ DỤNG =====
if __name__ == "__main__":
    # Ví dụ 1: Chỉ có time + province
    result1 = predict_weather("6/30/2025 14:00", "An Giang-Chau Doc")
    print("Kết quả 1:", result1)
    
    # Ví dụ 2: Có thêm nhiệt độ và độ ẩm hiện tại
    result2 = predict_weather(
        "7/1/2025 08:00", 
        "Ha Noi-Hanoi",
        current_temp=28.5,
        current_humidity=75
    )
    print("Kết quả 2:", result2)
