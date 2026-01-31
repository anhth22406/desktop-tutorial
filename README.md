# 🌤️ Weather Prediction Streamlit App

## 📋 Mô tả

Web app dự đoán thời tiết sử dụng models đã train bằng PySpark

## 🚀 Cài đặt & Chạy

### Bước 1: Chuẩn bị

1. Download `weather_models.zip` từ Google Colab
2. Giải nén vào thư mục gốc:
```bash
unzip weather_models.zip
```

### Bước 2: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 3: Chạy app
```bash
streamlit run app.py
```

App sẽ mở tại: http://localhost:8501

## 📁 Cấu trúc
````
weather_streamlit_app/
├── app.py                 # Main app
├── requirements.txt
├── weather_models/        # Models (giải nén từ .zip)
│   ├── metadata.json
│   ├── province_stats.csv
│   └── ...
└── utils/
    ├── __init__.py
    └── predictor.py