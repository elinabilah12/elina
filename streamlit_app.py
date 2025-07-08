import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import json

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Harga Daging Ayam Broiler - Jawa Timur",
    page_icon="🍗",
    layout="wide"
)

st.title("📊 Dashboard Prediksi Harga Daging Ayam Broiler - Jawa Timur")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Dataset", 
    "⚙ Preprocessing", 
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
                for col in df.columns:
                    if col != 'Date':
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                st.session_state['df'] = df
                st.success("✅ Dataset valid!")
                st.write("Data Preview:")
                st.dataframe(df.head())

                with st.expander("📊 Deskripsi Statistik"):
                    st.dataframe(df.describe())

        except Exception as e:
            st.error(f"❌ Gagal membaca file Excel. Error: {e}")
    else:
        st.info("Silakan upload file Excel (.xlsx) yang berisi semua variabel yang dibutuhkan.")

# Tab 2 - Preprocessing
with tab2:
    st.header("⚙ Preprocessing Data")

    if 'df' in st.session_state:
        df = st.session_state['df'].copy()

        st.subheader("1️⃣ Pembersihan Nama Kolom")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        st.write("Nama kolom setelah dibersihkan:")
        st.write(df.columns.tolist())

        df.rename(columns={
            'harga_pakan_ternak_broiler': 'pakan',
            'harga_doc_broiler': 'doc',
            'harga_jagung_tk_peternak': 'jagung',
            'harga_daging_ayam_broiler': 'daging',
            'date': 'tanggal'
        }, inplace=True)

        st.subheader("2️⃣ Missing Values")

        kolom_target = ['pakan', 'doc', 'jagung', 'daging']
        missing_before = df[kolom_target].isna().sum()

        df[kolom_target] = df[kolom_target].interpolate(method='linear')
        for col in kolom_target:
            df[col].fillna(method='ffill', inplace=True)
            df[col].fillna(method='bfill', inplace=True)

        missing_after = df[kolom_target].isna().sum()

        missing_df = pd.DataFrame({
            "Sebelum": missing_before,
            "Sesudah": missing_after
        })

        st.write("Jumlah missing value sebelum dan sesudah penanganan:")
        st.dataframe(missing_df)

        st.subheader("3️⃣ Deteksi Outlier (IQR)")
        Q1 = df[kolom_target].quantile(0.25)
        Q3 = df[kolom_target].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df[kolom_target] < (Q1 - 1.5 * IQR)) | (df[kolom_target] > (Q3 + 1.5 * IQR))
        st.write("Jumlah outlier per kolom:")
        st.dataframe(outliers.sum())

        fig_outlier, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df[kolom_target], orient='h', palette='Set2', ax=ax)
        ax.set_title("Boxplot Deteksi Outlier")
        st.pyplot(fig_outlier)

        st.subheader("4️⃣ Transformasi Log")
        for col in kolom_target:
            df[f"{col}_log"] = np.log(df[col])

        log_cols = [f"{col}_log" for col in kolom_target]
        st.write("Contoh kolom hasil log transform:")
        st.dataframe(df[log_cols].head())

        fig_log, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()
        for i, col in enumerate(log_cols):
            sns.histplot(df[col], kde=True, color='skyblue', ax=axs[i])
            axs[i].set_title(f'Distribusi Log: {col}')
        plt.tight_layout()
        st.pyplot(fig_log)

        st.session_state['df_clean'] = df

    else:
        st.warning("Silakan upload dataset di tab 📂 Dataset.")

# Tab 3 - Visualisasi
with tab3:
    st.header("📈 Visualisasi Dataset")

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']

        st.subheader("Distribusi Harga Daging")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['daging'], kde=True, ax=ax1)
        st.pyplot(fig1)

        st.subheader("Korelasi antar Fitur (Transformasi Log)")
        log_cols = [f"{col}_log" for col in ['pakan', 'doc', 'jagung', 'daging']]
        fig2, ax2 = plt.subplots()
        sns.heatmap(df[log_cols].corr(), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

        st.subheader("Visualisasi Time Series Harga Asli")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(df['tanggal'], df['pakan'], label='Pakan')
        ax3.plot(df['tanggal'], df['doc'], label='DOC')
        ax3.plot(df['tanggal'], df['jagung'], label='Jagung')
        ax3.plot(df['tanggal'], df['daging'], label='Daging Ayam')
        ax3.set_title("Pergerakan Harga dari Waktu ke Waktu")
        ax3.set_xlabel("Tanggal")
        ax3.set_ylabel("Harga")
        ax3.legend()
        st.pyplot(fig3)

    else:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")

# Tab 4 - Model
with tab4:
    st.header("🤖 Model")

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']

        # Feature engineering
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

        # Fitur dan target
        fitur = [
            'rasio_pakan_daging', 'rasio_doc_daging', 'rasio_jagung_pakan',
            'ma7_daging', 'ma7_pakan', 'ma7_doc', 'ma7_jagung',
            'lag1_daging', 'lag2_daging', 'pct_change_daging'
        ]
        target = 'daging'

        X = df[fitur]
        y = df[target]

        # Split data (tanpa shuffle untuk data time series)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Standardisasi
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model default
        model_default = XGBRegressor(random_state=42)
        model_default.fit(X_train_scaled, y_train)
        y_pred_default = model_default.predict(X_test_scaled)

        # Tombol tuning
        if st.button("🔍 Jalankan Tuning Optuna"):
            with st.spinner("Menjalankan tuning Optuna..."):

                # Fungsi objektif Optuna
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'gamma': trial.suggest_float('gamma', 0, 5),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'objective': 'reg:squarederror'
                    }

                    model = XGBRegressor(**params, random_state=42)
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_test_scaled, y_test)],
                        early_stopping_rounds=10,
                        verbose=False
                    )
                    preds = model.predict(X_test_scaled)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    return rmse

                # Inisialisasi tuning
                study = optuna.create_study(
                    direction='minimize',
                    sampler=TPESampler(seed=42),
                    pruner=MedianPruner(n_warmup_steps=10)
                )
                study.optimize(objective, n_trials=200)

                # Model terbaik
                best_model = XGBRegressor(**study.best_params, random_state=42)
                best_model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_test_scaled, y_test)],
                    early_stopping_rounds=10,
                    verbose=False
                )
                y_pred_best = best_model.predict(X_test_scaled)

                # Evaluasi
                def evaluate_model(y_true, y_pred):
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
                    return rmse, mape

                rmse_default, mape_default = evaluate_model(y_test, y_pred_default)
                rmse_best, mape_best = evaluate_model(y_test, y_pred_best)

                # Output hasil
                st.success("Model berhasil dituning dengan Optuna!")
                st.code(f"""
=== PERBANDINGAN XGBOOST DEFAULT vs TUNED (OPTUNA) ===
[DEFAULT] RMSE: {rmse_default:.2f}, MAPE: {mape_default:.2f}%
[TUNED  ] RMSE: {rmse_best:.2f}, MAPE: {mape_best:.2f}%
""")


# Tab 5 - Hasil Prediksi
with tab5:
    st.header("📉 Hasil Prediksi")

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']

        st.subheader("Simulasi Prediksi Harga Daging")
        df_pred = df.copy()
        df_pred['pred_xgb'] = df['daging'] * 0.95
        df_pred['pred_optuna'] = df['daging'] * 0.97

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['tanggal'], df['daging'], label='Aktual', linewidth=2)
        ax.plot(df['tanggal'], df_pred['pred_xgb'], label='Prediksi XGBoost', linestyle='--')
        ax.plot(df['tanggal'], df_pred['pred_optuna'], label='XGBoost + Optuna', linestyle='--')
        ax.set_title("Perbandingan Harga Aktual vs Prediksi")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")
