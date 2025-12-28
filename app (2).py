import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
from datetime import timedelta

# --- 1. ການຕັ້ງຄ່າໜ້າຈໍ (UI Config) ---
st.set_page_config(layout="wide", page_title="ລະບົບ AI ຮ້ານກາເຟລາວ")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao:wght@400;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Noto Sans Lao', sans-serif; }
    .main { background-color: #f5f5f5; }
    .stMetric { background-color: white; border-radius: 10px; padding: 15px; border-left: 5px solid #D4AF37; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ຟັງຊັນໂຫລດຂໍ້ມູນ ແລະ Model ---
@st.cache_resource
def load_assets():
    # ໂຫລດ Model ທີ່ເຮົາ Save ຈາກ Colab
    model = joblib.load('best_coffee_model.pkl')
    return model

@st.cache_data
def load_data():
    file_path = 'Coffee Shop Sales.xlsx'
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        # ເຮັດ Data Cleaning ຄືກັບໃນ Colab
        df.columns = [c.lower().strip() for c in df.columns]
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        # ແປງເປັນເງິນກີບ (1$ = 23,000 ກີບ)
        df['total_sales_lak'] = df['transaction_qty'] * df['unit_price'] * 23000
        return df
    return None

# --- 3. ສ່ວນປະກອບຂອງໜ້າເວັບ (Sidebar & Navigation) ---
st.sidebar.title("☕ Cafe AI Automation")
st.sidebar.info("ລະບົບຕິດຕາມ ແລະ ພະຍາກອນຍອດຂາຍ")
menu = st.sidebar.radio("ເມນູຫຼັກ", ["📊 Dashboard ຕິດຕາມຍອດຂາຍ", "🔮 AI ພະຍາກອນຍອດຂາຍ"])

# ໂຫລດຂໍ້ມູນ
df = load_data()
model = load_assets()

if df is not None:
    # ຈັດກຸ່ມຂໍ້ມູນລາຍວັນ
    daily_df = df.groupby('transaction_date')['total_sales_lak'].sum().reset_index()
    daily_df.columns = ['ວັນທີ', 'ຍອດຂາຍ_ກີບ']
    
    if menu == "📊 Dashboard ຕິດຕາມຍອດຂາຍ":
        st.title("📊 Dashboard ຕິດຕາມຍອດຂາຍ (LAK)")
        
        # ສ່ວນສະແດງຕົວເລກ (Metrics)
        col1, col2, col3 = st.columns(3)
        total_all = daily_df['ຍອດຂາຍ_ກີບ'].sum()
        avg_daily = daily_df['ຍອດຂາຍ_ກີບ'].mean()
        
        col1.metric("ຍອດຂາຍລວມທັງໝົດ", f"₭ {total_all:,.0f}")
        col2.metric("ຍອດຂາຍສະເລ່ຍ/ວັນ", f"₭ {avg_daily:,.0f}")
        col3.metric("ຈຳນວນມື້ທີ່ບັນທຶກ", f"{len(daily_df)} ມື້")
        
        # ກຣາຟແນວໂນ້ມ
        st.subheader("📈 ແນວໂນ້ມຍອດຂາຍລາຍວັນ")
        fig = px.line(daily_df, x='ວັນທີ', y='ຍອດຂາຍ_ກີບ', 
                      markers=True, color_discrete_sequence=['#D4AF37'])
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "🔮 AI ພະຍາກອນຍອດຂາຍ":
        st.title("🔮 AI Forecasting (7 Days)")
        st.write("ລະບົບໃຊ້ Model XGBoost ໃນການຄາດການຍອດຂາຍລ່ວງໜ້າ")
        
        # ກຽມວັນທີໃນອະນາຄົດ
        last_date = daily_df['ວັນທີ'].max()
        future_dates = pd.date_range(last_date + timedelta(days=1), periods=7)
        future_X = pd.DataFrame({
            'ມື້ໃນອາທິດ': future_dates.dayofweek,
            'ເດືອນ': future_dates.month
        })
        
        # ທຳນາຍຜົນ
        preds = model.predict(future_X)
        
        res_df = pd.DataFrame({
            'ວັນທີ': future_dates.strftime('%d/%m/%Y'),
            'ຍອດພະຍາກອນ (₭)': preds
        })
        
        # ສະແດງຜົນ
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("📋 ຕາຕະລາງຜົນ")
            st.table(res_df.style.format({'ຍອດພະຍາກອນ (₭)': '{:,.0f}'}))
        
        with c2:
            st.subheader("📈 ກຣາຟພະຍາກອນ 7 ວັນ")
            fig_pred = px.bar(res_df, x='ວັນທີ', y='ຍອດພະຍາກອນ (₭)', 
                              text_auto='.2s', color_discrete_sequence=['#8B4513'])
            st.plotly_chart(fig_pred, use_container_width=True)

        st.success(f"💡 **AI Recommendation:** ຍອດຂາຍສະເລ່ຍ 7 ວັນຂ້າງໜ້າແມ່ນ ₭ {preds.mean():,.0f}. ກະລຸນາກຽມວັດຖຸດິບໃຫ້ພຽງພໍ!")

else:
    st.error("❌ ບໍ່ພົບໄຟລ໌ຂໍ້ມູນ 'Coffee Shop Sales.xlsx' ກະລຸນາກວດສອບໃນ GitHub")
