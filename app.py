# ===== app.py =====

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from utils.predictor import WeatherPredictor
import json

# Page config
st.set_page_config(
    page_title="Weather Prediction",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize predictor
@st.cache_resource
def load_predictor():
    return WeatherPredictor()

try:
    predictor = load_predictor()
    provinces = predictor.get_provinces()
    weather_classes = predictor.get_weather_classes()
except Exception as e:
    st.error(f"❌ Lỗi load models: {e}")
    st.info("📥 Hãy đảm bảo file weather_models.zip đã được giải nén vào thư mục gốc!")
    st.stop()

# Header
st.markdown('<div class="main-header">🌤️ Weather Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dự đoán thời tiết sử dụng Big Data & Machine Learning</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1163/1163661.png", width=100)
    st.title("⚙️ Cấu hình")
    
    st.markdown("---")
    
    # Input mode
    input_mode = st.radio(
        "Chế độ nhập liệu:",
        ["Đơn giản (Time + Province)", "Chi tiết (Thêm nhiệt độ, độ ẩm)"],
        index=0
    )
    
    st.markdown("---")
    
    # Model info
    with st.expander("📊 Thông tin Models"):
        st.write("**Classification:**")
        st.write("- Model: Random Forest")
        st.write("- Accuracy: 92.12%")
        st.write("- F1-Score: 0.9195")
        
        st.write("\n**Regression:**")
        st.write("- Model: GBT")
        st.write("- RMSE: 0.64°C")
        st.write("- R²: 0.9637")
    
    st.markdown("---")
    st.markdown("**💡 Hướng dẫn:**")
    st.markdown("""
    1. Chọn thời gian
    2. Chọn tỉnh/thành phố
    3. (Tùy chọn) Nhập nhiệt độ/độ ẩm
    4. Nhấn **Dự đoán**
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Nhập thông tin dự đoán")
    
    # Date & Time
    col_date, col_time = st.columns(2)
    
    with col_date:
        selected_date = st.date_input(
            "Ngày:",
            value=datetime.now(),
            min_value=datetime.now() - timedelta(days=365),
            max_value=datetime.now() + timedelta(days=365)
        )
    
    with col_time:
        selected_time = st.time_input(
            "Giờ:",
            value=datetime.now().time()
        )
    
    # Combine datetime
    selected_datetime = datetime.combine(selected_date, selected_time)
    
    # Province
    selected_province = st.selectbox(
        "Tỉnh/Thành phố:",
        options=provinces,
        index=0
    )
    
    # Optional inputs
    if "Chi tiết" in input_mode:
        st.markdown("---")
        st.markdown("**🌡️ Thông tin chi tiết (Tùy chọn):**")
        
        col_temp, col_hum = st.columns(2)
        
        with col_temp:
            input_temp = st.number_input(
                "Nhiệt độ hiện tại (°C):",
                min_value=-10.0,
                max_value=50.0,
                value=28.0,
                step=0.5
            )
        
        with col_hum:
            input_humidity = st.number_input(
                "Độ ẩm hiện tại (%):",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )
    else:
        input_temp = None
        input_humidity = None
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Dự đoán thời tiết", type="primary", use_container_width=True):
        with st.spinner("Đang dự đoán..."):
            try:
                # Format datetime
                time_str = selected_datetime.strftime("%m/%d/%Y %H:%M")
                
                # Predict
                result = predictor.predict(
                    time_str=time_str,
                    province=selected_province,
                    temperature=input_temp,
                    humidity=input_humidity
                )
                
                # Store result in session state
                st.session_state['prediction_result'] = result
                st.session_state['input_info'] = {
                    'time': selected_datetime,
                    'province': selected_province,
                    'temperature': input_temp,
                    'humidity': input_humidity
                }
                
                st.success("✅ Dự đoán thành công!")
                
            except Exception as e:
                st.error(f"❌ Lỗi khi dự đoán: {e}")
                st.exception(e)

with col2:
    st.subheader("📋 Thông tin nhập")
    
    st.markdown(f"""
    <div class="metric-card">
        <strong>📅 Thời gian:</strong><br/>
        {selected_datetime.strftime('%d/%m/%Y %H:%M')}<br/><br/>
        <strong>📍 Địa điểm:</strong><br/>
        {selected_province}<br/><br/>
        <strong>🌡️ Nhiệt độ:</strong><br/>
        {f"{input_temp}°C" if input_temp else "Tự động"}<br/><br/>
        <strong>💧 Độ ẩm:</strong><br/>
        {f"{input_humidity}%" if input_humidity else "Tự động"}
    </div>
    """, unsafe_allow_html=True)

# Display prediction result
if 'prediction_result' in st.session_state:
    st.markdown("---")
    st.subheader("🎯 Kết quả dự đoán")
    
    result = st.session_state['prediction_result']
    input_info = st.session_state['input_info']
    
    # Main prediction
    col_pred1, col_pred2, col_pred3 = st.columns([2, 1, 1])
    
    with col_pred1:
        # Weather icon mapping
        weather_icons = {
            'Clear': '☀️',
            'Clouds': '☁️',
            'Rain': '🌧️',
            'Drizzle': '🌦️',
            'Thunderstorm': '⛈️',
            'Snow': '❄️',
            'Mist': '🌫️'
        }
        
        weather_main = result['weather_main']
        icon = weather_icons.get(weather_main, '🌤️')
        
        st.markdown(f"""
        <div class="prediction-box">
            <div style="font-size: 5rem;">{icon}</div>
            <h1 style="margin: 1rem 0;">{weather_main}</h1>
            <h3>Độ tin cậy: {result['probability']*100:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col_pred2:
        st.metric(
            label="🌡️ Nhiệt độ dự đoán",
            value=f"{result.get('predicted_temp', input_temp or 28):.1f}°C",
            delta=f"{result.get('temp_change', 0):.1f}°C"
        )
    
    with col_pred3:
        st.metric(
            label="💧 Độ ẩm dự đoán",
            value=f"{result.get('predicted_humidity', input_humidity or 75):.0f}%",
            delta=None
        )
    
    # Top 3 predictions
    st.markdown("---")
    st.subheader("📊 Top 3 Dự đoán có khả năng cao nhất")
    
    top_3 = result['top_3_predictions']
    
    cols = st.columns(3)
    for i, (weather, prob) in enumerate(top_3):
        with cols[i]:
            icon = weather_icons.get(weather, '🌤️')
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: #f5f5f5; border-radius: 10px;">
                <div style="font-size: 3rem;">{icon}</div>
                <h3>{weather}</h3>
                <p style="font-size: 1.5rem; color: #1E88E5; font-weight: bold;">{prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Probability chart
    st.markdown("---")
    st.subheader("📈 Phân phối xác suất")
    
    # Create chart data
    chart_data = pd.DataFrame([
        {'Weather': w, 'Probability': p*100} 
        for w, p in top_3
    ])
    
    fig = px.bar(
        chart_data,
        x='Weather',
        y='Probability',
        color='Probability',
        color_continuous_scale='Blues',
        title='Xác suất dự đoán (%)'
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Loại thời tiết",
        yaxis_title="Xác suất (%)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Details expander
    with st.expander("🔍 Xem chi tiết dự đoán"):
        st.json(result)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🎓 <strong>Đồ án Big Data & Ứng dụng</strong></p>
    <p>Phát triển bởi PySpark & Streamlit | 2025</p>
</div>
""", unsafe_allow_html=True)