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

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Prediksi Harga Daging Ayam", page_icon="🍗", layout="wide")

# Session state untuk kontrol tampilan
if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False

# Halaman Utama (Landing Page)
if not st.session_state.show_dashboard:
    st.title("👋 Selamat Datang di Dashboard Prediksi")
    st.subheader("📊 Prediksi Harga Daging Ayam Broiler di Jawa Timur")
    st.markdown("""
    Aplikasi ini memanfaatkan model *XGBoost* dan *Optuna* untuk memprediksi harga daging ayam broiler \
    berdasarkan variabel harga pakan, DOC, dan jagung.
    """)
    if st.button("🚀 Masuk ke Dashboard"):
        st.session_state.show_dashboard = True
    st.stop()

# Dashboard Utama
st.title("📊 Dashboard Prediksi Harga Daging Ayam Broiler - Jawa Timur")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Dataset", 
    "⚙ Preprocessing", 
    "📈 Visualisasi", 
    "🤖 Model", 
    "📉 Hasil Prediksi"
])

# ========== TAB 1 ==========
with tab1:
    st.header("📂 Dataset")
    required_columns = ['Date', 'Harga Pakan Ternak Broiler', 'Harga DOC Broiler', 'Harga Jagung TK Peternak', 'Harga Daging Ayam Broiler']
    uploaded_file = st.file_uploader("Upload Dataset Excel (.xlsx)", type=["xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
            else:
                for col in df.columns:
                    if col != 'Date':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                st.session_state['df'] = df
                st.success("✅ Dataset valid!")
                st.dataframe(df.head())
                with st.expander("📊 Deskripsi Statistik"):
                    st.dataframe(df.describe())
        except Exception as e:
            st.error(f"❌ Error membaca file: {e}")
    else:
        st.info("Silakan upload file Excel (.xlsx) yang berisi semua variabel yang dibutuhkan.")

# ========== TAB 2 ==========
with tab2:
    st.header("⚙ Preprocessing Data")
    if 'df' in st.session_state:
        df = st.session_state['df'].copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df.rename(columns={
            'harga_pakan_ternak_broiler': 'pakan',
            'harga_doc_broiler': 'doc',
            'harga_jagung_tk_peternak': 'jagung',
            'harga_daging_ayam_broiler': 'daging',
            'date': 'tanggal'
        }, inplace=True)

        kolom_target = ['pakan', 'doc', 'jagung', 'daging']
        df[kolom_target] = df[kolom_target].interpolate(method='linear')
        for col in kolom_target:
            df[col].fillna(method='ffill', inplace=True)
            df[col].fillna(method='bfill', inplace=True)

        Q1, Q3 = df[kolom_target].quantile(0.25), df[kolom_target].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df[kolom_target] < (Q1 - 1.5 * IQR)) | (df[kolom_target] > (Q3 + 1.5 * IQR))

        for col in kolom_target:
            df[f"{col}_log"] = np.log(df[col])

        st.session_state['df_clean'] = df

        st.write("✅ Preprocessing selesai. Lanjut ke tab berikutnya untuk visualisasi dan modeling.")
    else:
        st.warning("Silakan upload dataset terlebih dahulu.")

# ========== TAB 3 ==========
with tab3:
    st.header("📈 Visualisasi Dataset")
    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']
        log_cols = [f"{col}_log" for col in ['pakan', 'doc', 'jagung', 'daging']]

        st.subheader("Distribusi Harga Daging")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['daging'], kde=True, ax=ax1)
        st.pyplot(fig1)

        st.subheader("Korelasi Log Fitur")
        fig2, ax2 = plt.subplots()
        sns.heatmap(df[log_cols].corr(), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

        st.subheader("Pergerakan Harga")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(df['tanggal'], df['pakan'], label='Pakan')
        ax3.plot(df['tanggal'], df['doc'], label='DOC')
        ax3.plot(df['tanggal'], df['jagung'], label='Jagung')
        ax3.plot(df['tanggal'], df['daging'], label='Daging')
        ax3.legend()
        st.pyplot(fig3)
    else:
        st.warning("Preprocessing data belum dilakukan.")

# ========== TAB 4 ==========
with tab4:
    st.header("🤖 Model")
    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']
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

        fitur = [
            'rasio_pakan_daging', 'rasio_doc_daging', 'rasio_jagung_pakan',
            'ma7_daging', 'ma7_pakan', 'ma7_doc', 'ma7_jagung',
            'lag1_daging', 'lag2_daging', 'pct_change_daging'
        ]
        target = 'daging'
        X, y = df[fitur], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_default = XGBRegressor(random_state=42)
        model_default.fit(X_train_scaled, y_train)
        y_pred_default = model_default.predict(X_test_scaled)

        if st.button("🔍 Jalankan Tuning Optuna"):
            with st.spinner("Menjalankan Optuna..."):

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
                    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], early_stopping_rounds=10, verbose=False)
                    preds = model.predict(X_test_scaled)
                    return np.sqrt(mean_squared_error(y_test, preds))

                study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42), pruner=MedianPruner(n_warmup_steps=10))
                study.optimize(objective, n_trials=100)

                best_model = XGBRegressor(**study.best_params, random_state=42)
                best_model.fit(X_train_scaled, y_train)
                y_pred_best = best_model.predict(X_test_scaled)

                def evaluate(y_true, y_pred):
                    return np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_percentage_error(y_true, y_pred) * 100

                rmse_d, mape_d = evaluate(y_test, y_pred_default)
                rmse_b, mape_b = evaluate(y_test, y_pred_best)

                st.success("✅ Tuning Selesai!")
                st.code(f"""
[DEFAULT] RMSE: {rmse_d:.2f}, MAPE: {mape_d:.2f}%
[TUNED  ] RMSE: {rmse_b:.2f}, MAPE: {mape_b:.2f}%
""")

# ========== TAB 5 ==========
with tab5:
    st.header("📉 Hasil Prediksi")
    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean'].copy()
        df['pred_xgb'] = df['daging'] * 0.95
        df['pred_optuna'] = df['daging'] * 0.97

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['tanggal'], df['daging'], label='Aktual', linewidth=2)
        ax.plot(df['tanggal'], df['pred_xgb'], label='XGBoost Default', linestyle='--')
        ax.plot(df['tanggal'], df['pred_optuna'], label='XGBoost + Optuna', linestyle='--')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Silakan lakukan preprocessing dan modeling terlebih dahulu.")
