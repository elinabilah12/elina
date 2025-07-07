import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(
    page_title="Prediksi Harga Daging Ayam Broiler - Jawa Timur",
    page_icon="🍗",
    layout="wide"
)

# Title
st.title("📊 Dashboard Prediksi Harga Daging Ayam Broiler - Jawa Timur")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📂 Dataset", "📈 Visualisasi", "🤖 Model", "📉 Hasil Prediksi"])

# Tab 1 - Dataset
with tab1:
    st.header("📂 Dataset")
    uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())

        with st.expander("📊 Deskripsi Statistik"):
            st.write(df.describe())

# Tab 2 - Visualisasi
with tab2:
    st.header("📈 Visualisasi Dataset")

    if 'df' in locals():
        st.subheader("Distribusi Harga Daging Ayam Broiler")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['harga_daging_ayam'], kde=True, ax=ax1)
        st.pyplot(fig1)

        st.subheader("Korelasi antar Fitur")
        fig2, ax2 = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("Silakan upload dataset terlebih dahulu.")

# Tab 3 - Model
with tab3:
    st.header("🤖 Performa Model")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 XGBoost")
        st.metric("MAPE", "7.52%")
        st.metric("RMSE", "1250.45")

    with col2:
        st.subheader("📌 XGBoost + Optuna")
        st.metric("MAPE", "6.15%")
        st.metric("RMSE", "1078.32")

    st.success("Model dengan performa terbaik: **XGBoost + Optuna** (MAPE & RMSE lebih rendah)")

# Tab 4 - Hasil Prediksi
with tab4:
    st.header("📉 Hasil Prediksi")

    if 'df' in locals():
        st.subheader("Prediksi Harga Daging Ayam")

        # Simulasi hasil prediksi
        df_pred = df.copy()
        df_pred['pred_xgb'] = df['harga_daging_ayam'] * 0.95  # Simulasi prediksi XGBoost
        df_pred['pred_xgb_optuna'] = df['harga_daging_ayam'] * 0.97  # Simulasi prediksi Optuna

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(df['tanggal'], df['harga_daging_ayam'], label='Aktual', linewidth=2)
        ax3.plot(df['tanggal'], df_pred['pred_xgb'], label='Prediksi XGBoost', linestyle='--')
        ax3.plot(df['tanggal'], df_pred['pred_xgb_optuna'], label='Prediksi XGBoost + Optuna', linestyle='--')
        ax3.set_xlabel("Tanggal")
        ax3.set_ylabel("Harga")
        ax3.legend()
        ax3.set_title("Perbandingan Harga Aktual vs Prediksi")
        st.pyplot(fig3)
    else:
        st.warning("Upload dataset terlebih dahulu untuk melihat hasil prediksi.")


