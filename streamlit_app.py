import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# ================== CONFIG ==================
st.set_page_config(
    page_title="Prediksi Harga Daging Ayam Broiler - Jawa Timur",
    page_icon="🍗",
    layout="wide"
)

st.title("📊 Prediksi Harga Daging Ayam Broiler - Jawa Timur")

# CSS untuk tampilan nuansa kuning (kompatibel Streamlit)
yellow_css = """
<style>
/* Background utama (konten) */
[data-testid="stAppViewContainer"] {
    background-color: #fffde7;
}

/* Header teks (judul) */
h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #f9a825;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #fff8c6;
}

/* Tombol Streamlit */
div.stButton > button {
    background-color: #fbc02d;
    color: black;
    font-weight: bold;
    border: none;
    border-radius: 5px;
}

/* Teks biasa */
.stTextInput > label, .stNumberInput > label, .stSelectbox > label, p, div, span {
    color: #6d4c00;
}
</style>
"""
st.markdown(yellow_css, unsafe_allow_html=True)

# ================ SIDEBAR NAVIGATION =================
menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Beranda", "📂 Dataset", "⚙ Preprocessing", "📈 Visualisasi", "🤖 Model", "📉 Hasil Prediksi"]
)

# ================ MENU: BERANDA ======================
if menu == "🏠 Beranda":
    st.header("🏠 Hai Selamat Datang")
    st.markdown("""
    Selamat datang di **Dashboard Prediksi Harga Daging Ayam Broiler di Jawa Timur**.  
    Dashboard ini memanfaatkan model **XGBoost** dan **XGBoost dengan Optimasi Optuna** untuk memprediksi harga daging ayam broiler berdasarkan harga-harga komoditas pendukung seperti:
    - Harga Pakan Ternak Broiler
    - Harga DOC Broiler
    - Harga Jagung

    🔍 Anda dapat menavigasi melalui sidebar untuk melihat dataset, preprocessing, visualisasi data, pemodelan, dan hasil prediksi.
    """)


# ================ MENU: DATASET ======================
elif menu == "📂 Dataset":
    st.header("📂 Dataset")

    required_columns = [
        'Date', 'Harga Pakan Ternak Broiler', 'Harga DOC Broiler',
        'Harga Jagung TK Peternak', 'Harga Daging Ayam Broiler'
    ]

    uploaded_file = st.file_uploader("Upload Dataset Excel (.xlsx)", type=["xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)

            # Cek kolom yang dibutuhkan
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"Kolom tidak ditemukan: {', '.join(missing_cols)}")
            else:
                st.session_state['df'] = df.copy()
                st.success("✅ Dataset valid! Lanjut ke preprocessing.")
                st.dataframe(df.head())

                # Konversi kolom tanggal
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

                # Bersihkan dan konversi kolom harga (format lokal)
                harga_cols = [
                    'Harga Pakan Ternak Broiler', 'Harga DOC Broiler',
                    'Harga Jagung TK Peternak', 'Harga Daging Ayam Broiler'
                ]
                for col in harga_cols:
                    df[col] = df[col].astype(str).str.replace('.', '', regex=False)  # hapus titik ribuan
                    df[col] = df[col].str.replace(',', '.', regex=False)  # koma ke titik
                    df[col] = df[col].str.extract(r'(\d+\.?\d*)')[0]  # ambil angka valid
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Statistik numerik
                st.subheader("📊 Deskripsi Statistik")
                numeric_stats = df[harga_cols].describe().T

                # Statistik kolom tanggal
                valid_dates = df['Date'].dropna()
                if not valid_dates.empty:
                    date_stats = pd.DataFrame({
                        'count': [valid_dates.count()],
                        'mean': [valid_dates.mean()],
                        'min': [valid_dates.min()],
                        '25%': [valid_dates.quantile(0.25)],
                        '50%': [valid_dates.median()],
                        '75%': [valid_dates.quantile(0.75)],
                        'max': [valid_dates.max()]
                    }, index=['Date'])

                    # Gabungkan statistik
                    combined_stats = pd.concat([numeric_stats, date_stats])
                    st.dataframe(combined_stats)
                else:
                    st.dataframe(numeric_stats)

        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    else:
        if 'df' not in st.session_state:
            st.info("Silakan upload dataset terlebih dahulu.")


# ================ MENU: PREPROCESSING =================
elif menu == "⚙ Preprocessing":
    st.header("⚙ Preprocessing Data")

    if 'df' in st.session_state:
        df = st.session_state['df'].copy()

        df.rename(columns={
            'Harga Pakan Ternak Broiler': 'pakan',
            'Harga DOC Broiler': 'doc',
            'Harga Jagung TK Peternak': 'jagung',
            'Harga Daging Ayam Broiler': 'daging',
            'Date': 'tanggal'
        }, inplace=True)

        kolom_target = ['pakan', 'doc', 'jagung', 'daging']

        for col in kolom_target:
            df[col] = df[col].astype(str).str.replace(",", "").str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')

        st.subheader("2️⃣ Penanganan Missing Value")
        missing_before = df[kolom_target].isna().sum()
        df[kolom_target] = df[kolom_target].interpolate(method='linear')
        for col in kolom_target:
            df[col] = df[col].fillna(method='ffill')
            df[col] = df[col].fillna(method='bfill')
        missing_after = df[kolom_target].isna().sum()
        st.dataframe(pd.DataFrame({"Sebelum": missing_before, "Sesudah": missing_after}))

        st.subheader("3️⃣ Deteksi Outlier (IQR)")
        Q1 = df[kolom_target].quantile(0.25)
        Q3 = df[kolom_target].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[kolom_target] < (Q1 - 1.5 * IQR)) | (df[kolom_target] > (Q3 + 1.5 * IQR)))
        st.dataframe(outliers.sum())

        fig_outlier, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df[kolom_target], orient='h', palette='Set2', ax=ax)
        st.pyplot(fig_outlier)

        st.subheader("4️⃣ Transformasi Log")
        for col in kolom_target:
            df[f"{col}_log"] = np.log(df[col])
        st.dataframe(df[[f"{col}_log" for col in kolom_target]].head())

        fig_log, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()
        for i, col in enumerate([f"{c}_log" for c in kolom_target]):
            sns.histplot(df[col], kde=True, ax=axs[i], color='skyblue')
            axs[i].set_title(f'Distribusi Log: {col}')
        plt.tight_layout()
        st.pyplot(fig_log)

        st.session_state['df_clean'] = df
        st.success("✅ Preprocessing selesai.")
    else:
        st.warning("⚠ Silakan upload dataset terlebih dahulu.")

# ================ MENU: VISUALISASI ===================
elif menu == "📈 Visualisasi":
    st.header("📈 Visualisasi Dataset")

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']

        # Histogram harga daging
        fig, ax = plt.subplots()
        sns.histplot(df['daging'], kde=True, ax=ax)
        st.pyplot(fig)

        # Garis waktu pergerakan harga
        fig3, ax3 = plt.subplots(figsize=(10,5))
        ax3.plot(df['tanggal'], df['pakan'], label='Pakan')
        ax3.plot(df['tanggal'], df['doc'], label='DOC')
        ax3.plot(df['tanggal'], df['jagung'], label='Jagung')
        ax3.plot(df['tanggal'], df['daging'], label='Daging')
        ax3.set_title("Pergerakan Harga Komoditas")
        ax3.set_xlabel("Tanggal")
        ax3.set_ylabel("Harga")
        ax3.legend()
        st.pyplot(fig3)
    else:
        st.warning("Lakukan preprocessing terlebih dahulu.")

# ================ MENU: MODEL =========================
elif menu == "🤖 Model":
    st.header("🤖 Model")

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean'].copy()

        # Fitur Engineering
        df['rasio_pakan_daging'] = df['pakan'] / df['daging']
        df['rasio_doc_daging'] = df['doc'] / df['daging']
        df['rasio_jagung_pakan'] = df['jagung'] / df['pakan']
        df['ma7_daging'] = df['daging'].rolling(window=7).mean()
        df['ma7_pakan'] = df['pakan'].rolling(window=7).mean()
        df['ma7_doc'] = df['doc'].rolling(window=7).mean()
        df['ma7_jagung'] = df['jagung'].rolling(window=7).mean()
        df['lag1_daging'] = df['daging'].shift(1)
        df['lag2_daging'] = df['daging'].shift(2)
        df['pct_change_daging'] = df['daging'].pct_change()

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        fitur = [
            'rasio_pakan_daging', 'rasio_doc_daging', 'rasio_jagung_pakan',
            'ma7_daging', 'ma7_pakan', 'ma7_doc', 'ma7_jagung',
            'lag1_daging', 'lag2_daging', 'pct_change_daging'
        ]
        target = 'daging'

        X = df[fitur]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        def evaluate_model(y_true, y_pred):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            return rmse, mape

        # ========================
        # Model Default
        # ========================
        model_default = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=1,
            colsample_bytree=1,
            objective='reg:squarederror',
            random_state=42
        )
        model_default.fit(X_train_scaled, y_train)
        y_pred_default = model_default.predict(X_test_scaled)
        rmse_default, mape_default = evaluate_model(y_test, y_pred_default)

        # ========================
        # Model Tuned (Optuna Result)
        # ========================
        fixed_params = {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0.5,
            'reg_lambda': 1,
            'min_child_weight': 1,
            'objective': 'reg:squarederror'
        }

        model_optuna = XGBRegressor(**fixed_params, random_state=42)
        model_optuna.fit(X_train_scaled, y_train)
        y_pred_optuna = model_optuna.predict(X_test_scaled)
        rmse_best, mape_best = evaluate_model(y_test, y_pred_optuna)

        # Simpan ke session state
        st.session_state['model_default'] = model_default
        st.session_state['model_optuna'] = model_optuna
        st.session_state['X_test'] = X_test_scaled
        st.session_state['y_test'] = y_test
        st.session_state['df_clean'] = df

        st.success("✅ Model berhasil ditraining.")

        # ========================
        # Tampilkan Hasil Evaluasi
        # ========================
        st.markdown("### 📈 Perbandingan Performa Model")
        st.markdown(f"""
        | Model                  | RMSE     | MAPE     |
        |------------------------|----------|----------|
        | XGBoost Default        | {rmse_default:.2f} | {mape_default:.2f}% |
        | XGBoost + Optuna       | {rmse_best:.2f} | {mape_best:.2f}% |
        """)

        # ========================
        # Visualisasi Hasil Prediksi
        # ========================
        st.subheader("📉 Grafik Prediksi vs Aktual")

        tanggal_data = df['tanggal'].iloc[-len(y_test):].values if 'tanggal' in df.columns else pd.date_range(start='2020-01-01', periods=len(y_test))
        hasil_df = pd.DataFrame({
            'Tanggal': tanggal_data,
            'Aktual': y_test.values,
            'Prediksi Default': y_pred_default,
            'Prediksi Tuned': y_pred_optuna
        })

        hasil_df['Tanggal'] = pd.to_datetime(hasil_df['Tanggal'])

        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(hasil_df['Tanggal'], hasil_df['Aktual'], label='Aktual', linewidth=2)
        ax1.plot(hasil_df['Tanggal'], hasil_df['Prediksi Default'], label='Prediksi Default', linestyle='--')
        ax1.plot(hasil_df['Tanggal'], hasil_df['Prediksi Tuned'], label='Prediksi Tuned', linestyle='--')
        ax1.set_title("Perbandingan Harga Aktual vs Prediksi")
        ax1.set_xlabel("Tanggal")
        ax1.set_ylabel("Harga Daging Ayam")
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

    else:
        st.warning("Data belum tersedia. Silakan lakukan preprocessing terlebih dahulu.")


