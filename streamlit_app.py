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
# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Dataset", 
    "⚙️ Preprocessing", 
    "📈 Visualisasi", 
    "🤖 Model", 
    "📉 Hasil Prediksi"
])

# Tab 1 - Dataset
with tab1:
    st.header("📂 Dataset")
    
    required_columns = [
        'Date',
        'Harga Pakan Ternak Broiler',
        'Harga DOC Broiler',
        'Harga Jagung TK Peternak',
        'Harga Daging Ayam Broiler'
    ]

    uploaded_file = st.file_uploader("Upload Dataset Excel (.xlsx)", type=["xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            missing_cols = [col for col in required_columns if col not in df.columns]

            if missing_cols:
                st.error(f"❌ Kolom berikut tidak ditemukan di file Excel: {', '.join(missing_cols)}")
            else:
                st.session_state['df'] = df  # Simpan df ke session
                st.success("✅ Dataset valid!")
                st.write("Data Preview:")
                st.dataframe(df.head())

                with st.expander("📊 Deskripsi Statistik"):
                    st.write(df.describe())

        except Exception as e:
            st.error(f"❌ Gagal membaca file Excel. Pastikan formatnya benar. Error: {e}")
    else:
        st.info("Silakan upload file Excel (.xlsx) yang berisi semua variabel yang dibutuhkan.")

# Tab 2 - Preprocessing
with tab2:
    st.header("⚙️ Preprocessing Data")

    if 'df' in st.session_state:
        df = st.session_state['df'].copy()

        st.subheader("🔠 Normalisasi Nama Kolom")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        st.write("Kolom setelah dinormalisasi:")
        st.write(df.columns.tolist())

        st.subheader("🧼 Cek Missing Values")
        st.write(df.isnull().sum())

        st.subheader("🧮 Statistik Ringkas")
        st.write(df.describe())

        st.session_state['df_clean'] = df  # Simpan versi yang sudah diproses
    else:
        st.warning("Upload dataset terlebih dahulu di tab 📂 Dataset.")

# Tab 3 - Visualisasi
with tab3:
    st.header("📈 Visualisasi Dataset")

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']

        st.subheader("Distribusi Harga Daging Ayam Broiler")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['harga_daging_ayam_broiler'], kde=True, ax=ax1)
        st.pyplot(fig1)

        st.subheader("Korelasi antar Fitur")
        fig2, ax2 = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")

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


