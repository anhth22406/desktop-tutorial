
# WEATHER BIG DATA PREDICTION MODELS

## 📋 MÔ TẢ
Models dự đoán thời tiết sử dụng PySpark và Machine Learning

## 📦 CẤU TRÚC FOLDER
```
weather_models/
├── rf_classifier/          Random Forest Classification Model
├── gbt_regressor/          Gradient Boosted Trees Regression Model
├── kmeans_clustering/      KMeans Clustering Model
├── scaler/                 StandardScaler
├── province_indexer/       Province StringIndexer
├── city_indexer/          City StringIndexer
├── weather_indexer/       Weather StringIndexer
├── province_stats.csv     Province statistics (cần cho prediction)
├── metadata.json          Thông tin models & features
├── predict_example.py     Code mẫu sử dụng
└── README.md              File này
```

## 🎯 MODELS

### 1. Classification (weather_main)
- Model: Random Forest (100 trees, depth=10)
- Accuracy: 92.12%
- F1-Score: 0.9195
- Classes: ['Clouds', 'Rain', 'Clear', 'Mist', 'Thunderstorm', 'Squall', 'Drizzle', 'Smoke']

### 2. Regression (temperature)
- Model: Gradient Boosted Trees
- RMSE: 0.64°C
- R²: 0.9637

### 3. Clustering (provinces)
- Model: KMeans
- K: 4 clusters
- Silhouette Score: 0.6058

## 🔧 SỬ DỤNG

### Load Models:
```python
from pyspark.ml.classification import RandomForestClassificationModel

model = RandomForestClassificationModel.load("weather_models/rf_classifier")
```

### Dự đoán:
Xem file `predict_example.py` để biết chi tiết

### Input cần thiết:
- time: Thời gian dự đoán
- province: Tỉnh/Thành phố
- (Optional) temperature, humidity hiện tại

### Output:
- weather_main: Dự đoán thời tiết
- probability: Độ tin cậy
- top_3_predictions: Top 3 khả năng

## 📊 FEATURES (32 features)

### Time Features:
- hour, day_of_week, month_num, day_of_month, is_day

### Location Features:
- province_encoded, city_encoded

### Weather Features:
- temperature, humidity, pressure, wind_speed, cloudcover, precipitation, visibility

### Derived Features:
- temp_range, temp_lag_1h, temp_ma_3h, temp_change_1h

## 📞 HỖ TRỢ

Nếu có vấn đề, xem file metadata.json để biết chi tiết cấu hình

---
Created: 2025-01-31
PySpark Version: 3.5.0
