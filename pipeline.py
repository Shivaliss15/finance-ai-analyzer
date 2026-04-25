# =========================
# pipeline.py
# AI Personal Finance Analyzer — Backend Pipeline
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger('prophet').setLevel(logging.ERROR)

pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)
pd.options.display.float_format = '{:.2f}'.format


# =========================
# HELPER: Clean account numbers
# =========================
def clean_account_no(acc):
    """Strip whitespace and trailing apostrophes from account numbers."""
    return str(acc).strip().strip("'")


# =========================
# LOAD DATASET
# =========================
def load_data(file):
    df = pd.read_excel(file)
    # Clean account numbers immediately on load
    df['Account No'] = df['Account No'].apply(clean_account_no)
    return df


# =========================
# FEATURE ENGINEERING
# =========================
def create_features(df):
    df = df.copy()

    # Drop completely empty rows
    df = df.dropna(how='all')

    # Fill categorical missing
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("Unknown")

    # 1) Date parsing
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE'])

    # 2) Clean numeric columns
    df['WITHDRAWAL AMT'] = pd.to_numeric(df['WITHDRAWAL AMT'], errors='coerce').fillna(0)
    df['DEPOSIT AMT'] = pd.to_numeric(df['DEPOSIT AMT'], errors='coerce').fillna(0)

    # 3) Sort per user
    df = df.sort_values(['Account No', 'DATE'])

    # 4) Time features
    df['year'] = df['DATE'].dt.year
    df['month'] = df['DATE'].dt.month
    df['day'] = df['DATE'].dt.day
    df['weekday'] = df['DATE'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

    # 5) Net amount
    df['net_amount'] = df['DEPOSIT AMT'] - df['WITHDRAWAL AMT']

    # 6) Rolling features
    df['rolling_7_net'] = df.groupby('Account No')['net_amount'] \
        .transform(lambda x: x.rolling(7, min_periods=1).sum())

    df['rolling_30_net'] = df.groupby('Account No')['net_amount'] \
        .transform(lambda x: x.rolling(30, min_periods=1).sum())

    # 7) Monthly aggregation
    df['year_month'] = df['DATE'].dt.to_period('M')

    monthly = df.groupby(['Account No', 'year_month']).agg(
        total_debit=('WITHDRAWAL AMT', 'sum'),
        total_credit=('DEPOSIT AMT', 'sum'),
        monthly_net=('net_amount', 'sum'),
        txn_count=('net_amount', 'count')
    ).reset_index()

    monthly['ds'] = monthly['year_month'].dt.to_timestamp()
    monthly['y'] = monthly['monthly_net']
    monthly = monthly.sort_values(['Account No', 'ds'])

    return df, monthly


# =========================
# TRANSACTION CATEGORIZATION
# =========================
def run_transaction_categorization(df):
    import time
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import MultinomialNB
    from scipy.sparse import hstack, csr_matrix

    try:
        from catboost import CatBoostClassifier
    except Exception:
        CatBoostClassifier = None

    text_col = next((c for c in df.columns if 'transaction' in c.lower()
                     or 'detail' in c.lower() or 'desc' in c.lower()), None)
    if text_col is None:
        raise ValueError("No text column found.")

    def auto_label(text):
        text = str(text).lower()
        if 'chq' in text:
            return 'cheque'
        elif 'rtgs' in text:
            return 'rtgs'
        elif any(k in text for k in ['fund trf', 'transfer', 'trf']):
            return 'transfer'
        elif any(k in text for k in ['cash', 'atm', 'csh']):
            return 'cash'
        elif any(k in text for k in ['charges', 'stax', 'commission']):
            return 'bank_charges'
        elif 'dep' in text:
            return 'deposit'
        else:
            return 'other'

    df['txn_category_weak'] = df[text_col].apply(auto_label)
    df_model = df.copy()

    def clean_text(text):
        text = str(text).lower()
        for word in ['rtgs', 'cash', 'cheque', 'chq', 'transfer', 'trf', 'deposit', 'atm']:
            text = text.replace(word, '')
        return text

    df_model[text_col] = df_model[text_col].apply(clean_text)

    X_text = df_model[text_col].fillna('').astype(str)
    y = df_model['txn_category_weak']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    tfidf = TfidfVectorizer(max_features=1000)
    X_text_vec = tfidf.fit_transform(X_text)

    num_cols = ['WITHDRAWAL AMT', 'DEPOSIT AMT', 'net_amount',
                'month', 'weekday', 'is_weekend', 'rolling_7_net', 'rolling_30_net']
    num_cols = [c for c in num_cols if c in df_model.columns]

    X_num = df_model[num_cols].fillna(0)
    X_num_sparse = csr_matrix(X_num.values)
    X = hstack([X_text_vec, X_num_sparse])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.35, random_state=42, stratify=y_enc)
    X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
        X_text_vec, y_enc, test_size=0.35, random_state=42, stratify=y_enc)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=60),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Naive Bayes": MultinomialNB()
    }
    if CatBoostClassifier is not None:
        models["CatBoost"] = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0)

    results = []
    for name, model in models.items():
        start = time.time()
        if name == "Naive Bayes":
            model.fit(X_train_text, y_train_text)
            preds = model.predict(X_test_text)
            current_y_test = y_test_text
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            current_y_test = y_test
        end = time.time()

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(current_y_test, preds), 4),
            "Recall": round(recall_score(current_y_test, preds, average='weighted', zero_division=0), 4),
            "Prec.": round(precision_score(current_y_test, preds, average='weighted', zero_division=0), 4),
            "F1": round(f1_score(current_y_test, preds, average='weighted', zero_division=0), 4),
            "TT (Sec)": round(end - start, 2)
        })

    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]

    if best_model_name == "Naive Bayes":
        best_model.fit(X_text_vec, y_enc)
        df['txn_category_pred'] = le.inverse_transform(best_model.predict(X_text_vec))
    else:
        best_model.fit(X, y_enc)
        df['txn_category_pred'] = le.inverse_transform(best_model.predict(X))

    return df, results_df, best_model_name


# =========================
# CATEGORY PIE CHART
# =========================
def plot_transaction_category_distribution(df):
    if 'txn_category_pred' in df.columns:
        cat_col = 'txn_category_pred'
    elif 'txn_category_weak' in df.columns:
        cat_col = 'txn_category_weak'
    else:
        raise ValueError("No transaction category column found!")

    category_counts = df[cat_col].dropna().value_counts()
    if category_counts.empty:
        raise ValueError("No data available for plotting.")

    category_percent = (category_counts / category_counts.sum()) * 100
    threshold = 1
    large_cats = category_percent[category_percent >= threshold].copy()
    small_cats = category_percent[category_percent < threshold]
    large_cats = large_cats.sort_values(ascending=False)
    if len(small_cats) > 0:
        large_cats.loc['Other'] = small_cats.sum()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(large_cats, labels=large_cats.index, autopct='%1.1f%%',
           startangle=140, textprops={'fontsize': 10})
    ax.set_title("Transaction Category Distribution", fontsize=14)
    ax.axis('equal')
    return fig, large_cats


# =========================
# PROPHET FORECAST
# =========================
def run_prophet_forecast(monthly):
    from prophet import Prophet

    final_forecasts = {}
    prophet_figs = {}

    for acc in monthly['Account No'].unique():
        user_df = monthly[monthly['Account No'] == acc].copy().sort_values('ds')
        prophet_df = user_df[['ds', 'monthly_net']].rename(columns={'monthly_net': 'y'})
        prophet_df = prophet_df.set_index('ds').asfreq('MS').fillna(0).reset_index()

        if len(prophet_df) < 12:
            continue

        model = Prophet()
        model.fit(prophet_df[['ds', 'y']])
        future = model.make_future_dataframe(periods=6, freq='MS')
        forecast = model.predict(future)
        last_date = prophet_df['ds'].max()
        future_only = forecast[forecast['ds'] > last_date][['ds', 'yhat']].copy()
        future_only = future_only.rename(columns={'ds': 'Month', 'yhat': 'Predicted Net Savings'})
        future_only['Predicted Net Savings'] = future_only['Predicted Net Savings'].apply(lambda x: f"{x:,.2f}")
        final_forecasts[acc] = future_only

        fig = plt.figure(figsize=(10, 5))
        plt.plot(prophet_df['ds'], prophet_df['y'], color='black', linewidth=2, label='Actual')
        forecast_past = forecast[forecast['ds'] <= last_date].copy()
        forecast_future = forecast[forecast['ds'] > last_date].copy()
        if len(forecast_future) > 0:
            connect_point = pd.DataFrame([{'ds': forecast_past.iloc[-1]['ds'], 'yhat': forecast_past.iloc[-1]['yhat']}])
            forecast_future = pd.concat([connect_point, forecast_future])
        plt.plot(forecast_past['ds'], forecast_past['yhat'], color='blue', linewidth=2, label='Predicted')
        plt.plot(forecast_future['ds'], forecast_future['yhat'], color='red', linestyle='--', linewidth=2, label='Forecast')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                         color='lightblue', alpha=0.3, label='Uncertainty')
        plt.title(f"Net Savings Forecast - Account {acc}")
        plt.xlabel("Month")
        plt.ylabel("Net Savings")
        plt.legend()
        plt.grid()
        prophet_figs[acc] = fig
        plt.close()

    return final_forecasts, prophet_figs


# =========================
# PROPHET EVALUATION
# =========================
def get_prophet_evaluation(monthly):
    from prophet import Prophet
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    prophet_metrics = {}

    for acc in monthly['Account No'].unique():
        user_df = monthly[monthly['Account No'] == acc].copy().sort_values('ds')
        prophet_df = user_df[['ds', 'monthly_net']].rename(columns={'monthly_net': 'y'})
        prophet_df = prophet_df.set_index('ds').asfreq('MS')
        prophet_df['y'] = prophet_df['y'].ffill()
        prophet_df = prophet_df.reset_index()

        if len(prophet_df) < 12:
            continue

        q1 = prophet_df['y'].quantile(0.25)
        q3 = prophet_df['y'].quantile(0.75)
        iqr = q3 - q1
        prophet_df['y'] = np.clip(prophet_df['y'], q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        split = int(len(prophet_df) * 0.8)
        train = prophet_df.iloc[:split]
        test = prophet_df.iloc[split:]

        model = Prophet()
        model.fit(train)
        future = model.make_future_dataframe(periods=len(test), freq='MS')
        forecast = model.predict(future)
        forecast_test = forecast[forecast['ds'].isin(test['ds'])]

        actual = test['y'].values
        predicted = forecast_test['yhat'].values
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        avg_value = np.mean(np.abs(actual))
        error_ratio = mae / (avg_value + 1)

        prophet_metrics[acc] = {"MAE": mae, "RMSE": rmse, "Error Ratio": error_ratio}

    prophet_df_summary = pd.DataFrame.from_dict(prophet_metrics, orient='index').reset_index()
    prophet_df_summary.rename(columns={'index': 'Account No'}, inplace=True)
    return prophet_df_summary


# =========================
# LSTM FORECAST
# =========================
def run_lstm_forecast(monthly):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    RND = 42
    np.random.seed(RND)
    tf.random.set_seed(RND)

    WINDOW = 5
    FORECAST_STEPS = 6
    EPOCHS = 200
    BATCH_SIZE = 4
    MIN_ROWS = 15

    lstm_final_forecasts = {}
    lstm_eval_data = {}
    lstm_figs = {}

    def adaptive_trim(vals, threshold_pct=0.01):
        series_max = np.max(np.abs(vals))
        if series_max == 0:
            return 0
        threshold = series_max * threshold_pct
        for i in range(len(vals)):
            if np.abs(vals[i]) > threshold:
                return i
        return 0

    def build_sequences(series_scaled, window):
        X, y = [], []
        for i in range(len(series_scaled) - window):
            X.append(series_scaled[i: i + window])
            y.append(series_scaled[i + window])
        return np.array(X), np.array(y)

    def build_lstm_model(window):
        model = Sequential([
            Input(shape=(window, 1)),
            LSTM(48, return_sequences=True, kernel_regularizer=l2(1e-4)),
            Dropout(0.2),
            LSTM(24, return_sequences=False, kernel_regularizer=l2(1e-4)),
            Dropout(0.2),
            Dense(12, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='huber')
        return model

    for acc in monthly['Account No'].unique():
        user_df = (monthly[monthly['Account No'] == acc]
                   .copy().sort_values('ds'))
        user_df = (user_df.set_index('ds').resample('MS')['monthly_net']
                   .sum().fillna(0).reset_index().rename(columns={'monthly_net': 'y'}))

        if len(user_df) < MIN_ROWS:
            continue

        vals_full = user_df['y'].values.astype(float)
        trim_idx = adaptive_trim(vals_full)
        vals = vals_full[trim_idx:]

        if len(vals) < MIN_ROWS:
            continue

        q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
        iqr = q3 - q1
        vals_clipped = np.clip(vals, q1 - 2 * iqr, q3 + 2 * iqr)

        scaler = RobustScaler()
        vals_scaled = scaler.fit_transform(vals_clipped.reshape(-1, 1)).flatten()

        X, y_seq = build_sequences(vals_scaled, WINDOW)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        split = max(int(len(X) * 0.8), 1)
        X_train = X[:split]
        y_train = y_seq[:split]

        model = build_lstm_model(WINDOW)
        callbacks = [
            EarlyStopping(monitor='loss', patience=50, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=1e-6, verbose=0)
        ]
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0)

        all_pred_scaled = model.predict(X, verbose=0).flatten()
        all_pred = scaler.inverse_transform(all_pred_scaled.reshape(-1, 1)).flatten()

        lstm_eval_data[acc] = {"vals": vals, "all_pred": all_pred, "split": split, "window": WINDOW}

        trimmed_dates = user_df['ds'].values[trim_idx:]
        all_pred_dates = trimmed_dates[WINDOW:]
        pad_dates = trimmed_dates[:WINDOW]
        pad_vals = np.full(WINDOW, all_pred[0])
        full_pred_dates = np.concatenate([pad_dates, all_pred_dates])
        full_pred_vals = np.concatenate([pad_vals, all_pred])

        last_window = vals_scaled[-WINDOW:].reshape(1, WINDOW, 1)
        future_pred_scaled = []
        for _ in range(FORECAST_STEPS):
            nv = model.predict(last_window, verbose=0)[0, 0]
            future_pred_scaled.append(nv)
            last_window = np.append(last_window[:, 1:, :], [[[nv]]], axis=1)

        future_pred = scaler.inverse_transform(np.array(future_pred_scaled).reshape(-1, 1)).flatten()
        last_date = user_df['ds'].max()
        future_dates = pd.date_range(start=last_date, periods=FORECAST_STEPS + 1, freq='MS')[1:]
        connect_dates = np.concatenate([[full_pred_dates[-1]], future_dates])
        connect_vals = np.concatenate([[full_pred_vals[-1]], future_pred])

        future_df = pd.DataFrame({
            'Month': future_dates,
            'Predicted Net Savings': [f"{v:,.2f}" for v in future_pred]
        })
        lstm_final_forecasts[acc] = future_df

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(user_df['ds'], user_df['y'], label='Actual')
        ax.plot(full_pred_dates, full_pred_vals, linestyle='--', label='Predicted')
        ax.plot(connect_dates, connect_vals, linestyle='--', label='Forecast')
        ax.axvline(last_date, linestyle=':')
        ax.set_title(f"LSTM Forecast | Account {acc}")
        ax.legend()
        ax.grid()
        lstm_figs[acc] = fig
        plt.close()

    return lstm_final_forecasts, lstm_eval_data, lstm_figs


# =========================
# LSTM EVALUATION
# =========================
def get_lstm_evaluation(lstm_eval_data):
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    if not lstm_eval_data:
        return pd.DataFrame(columns=["Account No", "MAE", "RMSE", "Error Ratio"])

    lstm_metrics = {}
    for acc, data in lstm_eval_data.items():
        vals = data["vals"]
        all_pred = data["all_pred"]
        split = data["split"]
        WINDOW = data["window"]

        actual_test = vals[WINDOW + split:]
        pred_test = all_pred[split:]

        if len(actual_test) == 0 or len(pred_test) == 0:
            lstm_metrics[acc] = {"MAE": None, "RMSE": None, "Error Ratio": None}
            continue

        min_len = min(len(actual_test), len(pred_test))
        actual_test = actual_test[:min_len]
        pred_test = pred_test[:min_len]

        mae = mean_absolute_error(actual_test, pred_test)
        rmse = np.sqrt(mean_squared_error(actual_test, pred_test))
        full_avg = np.mean(np.abs(vals))
        error_ratio = mae / (full_avg + 1e-6)

        lstm_metrics[acc] = {"MAE": mae, "RMSE": rmse, "Error Ratio": error_ratio}

    lstm_df = pd.DataFrame.from_dict(lstm_metrics, orient='index').reset_index()
    lstm_df.rename(columns={'index': 'Account No'}, inplace=True)
    return lstm_df


# =========================
# MODEL COMPARISON
# =========================
def get_model_comparison(lstm_df, prophet_df_summary):
    # If LSTM results are empty (TF not available), show Prophet-only table
    if lstm_df.empty:
        prophet_df_summary = prophet_df_summary.copy()
        prophet_df_summary["MAE_LSTM"]        = "N/A"
        prophet_df_summary["RMSE_LSTM"]       = "N/A"
        prophet_df_summary["Error Ratio_LSTM"] = "N/A"
        prophet_df_summary["Best Model"]       = "Prophet"
        cols = ["Account No", "MAE_LSTM", "RMSE_LSTM", "Error Ratio_LSTM",
                "MAE", "RMSE", "Error Ratio", "Best Model"]
        cols = [c for c in cols if c in prophet_df_summary.columns]
        return prophet_df_summary[cols].rename(columns={
            "MAE": "MAE_Prophet", "RMSE": "RMSE_Prophet", "Error Ratio": "Error Ratio_Prophet"
        })

    comparison_df = pd.merge(lstm_df, prophet_df_summary, on="Account No",
                              suffixes=("_LSTM", "_Prophet"))

    def select_best(row):
        try:
            if abs(float(row["Error Ratio_LSTM"]) - float(row["Error Ratio_Prophet"])) < 0.01:
                return "Both"
            elif float(row["Error Ratio_LSTM"]) < float(row["Error Ratio_Prophet"]):
                return "LSTM"
            else:
                return "Prophet"
        except (TypeError, ValueError):
            return "Prophet"

    comparison_df["Best Model"] = comparison_df.apply(select_best, axis=1)
    return comparison_df


# =========================
# FINANCIAL HEALTH MODULE
# =========================
def run_financial_health_module(monthly):
    health_results = {}

    for acc in monthly['Account No'].unique():
        user_df = monthly[monthly['Account No'] == acc].copy().fillna(0)

        if len(user_df) < 6:
            continue

        user_df = user_df.sort_values('ds')
        user_df['income'] = user_df['total_credit']
        user_df['expense'] = user_df['total_debit']
        user_df['savings'] = user_df['monthly_net']
        user_df['spending_ratio'] = user_df['expense'] / (user_df['income'] + 1)
        user_df['volatility'] = user_df['savings'].diff().abs().fillna(0)

        def compute_score(row):
            savings_ratio = row['savings'] / (row['income'] + 1) if row['income'] > 0 else 0
            savings_score = max(0, min(1, savings_ratio))
            spending_score = max(0, min(1, 1 - row['spending_ratio']))
            vol_ratio = row['volatility'] / (row['income'] + 1) if row['income'] > 0 else 1
            volatility_score = max(0, min(1, 1 - vol_ratio))
            return (0.4 * savings_score + 0.3 * spending_score + 0.3 * volatility_score) * 100

        user_df['health_score'] = user_df.apply(compute_score, axis=1)

        def generate_recommendation(row):
            recs = []
            if row['spending_ratio'] > 0.8:
                recs.append("Reduce discretionary spending")
            if row['savings'] < 0:
                recs.append("Expenses exceed income")
            if row['volatility'] > (row['income'] * 0.5):
                recs.append("Spending is unstable")
            if not recs:
                recs.append("Good financial health")
            return ", ".join(recs)

        user_df['recommendations'] = user_df.apply(generate_recommendation, axis=1)

        monthly_health = (
            user_df[['ds', 'health_score', 'recommendations']]
            .tail(6)
            .rename(columns={'ds': 'Month', 'health_score': 'Health Score'})
        )
        health_results[acc] = monthly_health

    health_df = pd.concat(
        [mh.assign(**{'Account No': acc}) for acc, mh in health_results.items()],
        ignore_index=True
    )
    final_health = health_df.groupby('Account No')['Health Score'].mean().reset_index()
    final_health = final_health.rename(columns={'Health Score': 'Overall Health Score'})
    final_health['Overall Health Score'] = final_health['Overall Health Score'].round(2)

    return health_results, final_health


# =========================
# INSIGHTS ENGINE
# =========================
def run_insights_engine(monthly):
    insights_results = {}
    insights_summary = []

    for acc in monthly['Account No'].unique():
        user_df = monthly[monthly['Account No'] == acc].copy().fillna(0)

        if len(user_df) < 6:
            continue

        user_df = user_df.sort_values('ds')
        user_df['income'] = user_df['total_credit']
        user_df['expense'] = user_df['total_debit']
        user_df['savings'] = user_df['monthly_net']

        user_insights_text = []

        overspending = user_df[user_df['expense'] > user_df['income']]
        overspend_ratio = len(overspending) / len(user_df)
        if overspend_ratio > 0.5:
            msg = "🚨 Frequent overspending pattern detected"
        elif overspend_ratio > 0.2:
            msg = "⚠️ Occasional overspending observed"
        else:
            msg = "✅ Spending mostly within income"
        user_insights_text.append(msg)

        x = np.arange(len(user_df))
        y = user_df['savings'].values
        slope = np.polyfit(x, y, 1)[0]
        if slope > 0:
            trend = "increasing"
            msg = "📈 Savings trend is improving over time"
        elif slope < 0:
            trend = "decreasing"
            msg = "📉 Savings trend is declining"
        else:
            trend = "stable"
            msg = "➡️ Savings are stable"
        user_insights_text.append(msg)

        avg_income = user_df['income'].mean()
        avg_expense = user_df['expense'].mean()
        ratio = avg_expense / (avg_income + 1)
        if ratio > 0.9:
            msg = "🚨 Almost entire income is being spent"
        elif ratio > 0.75:
            msg = "⚠️ High spending ratio"
        elif ratio > 0.5:
            msg = "⚠️ Moderate spending"
        else:
            msg = "✅ Healthy spending behavior"
        user_insights_text.append(msg)

        volatility = user_df['savings'].std()
        if volatility > avg_income * 0.5:
            msg = "⚠️ Highly unstable financial pattern"
        elif volatility > avg_income * 0.2:
            msg = "⚠️ Moderate fluctuation in savings"
        else:
            msg = "✅ Stable financial behavior"
        user_insights_text.append(msg)

        recent = user_df.tail(3)
        recent_savings = recent['savings'].mean()
        overall_savings = user_df['savings'].mean()
        if recent_savings < overall_savings:
            msg = "📉 Recent performance worse than usual"
        else:
            msg = "📈 Recent performance improving"
        user_insights_text.append(msg)

        best_month = user_df.loc[user_df['savings'].idxmax()]
        worst_month = user_df.loc[user_df['savings'].idxmin()]
        user_insights_text.append(f"🏆 Best month savings: ₹{best_month['savings']:,.0f}")
        user_insights_text.append(f"📉 Worst month savings: ₹{worst_month['savings']:,.0f}")

        insights_results[acc] = user_insights_text
        insights_summary.append({
            "Account No": acc,
            "Overspend Ratio": round(overspend_ratio, 2),
            "Trend": trend,
            "Spending Ratio": min(round(ratio, 2), 2),
            "Volatility": round(volatility, 2)
        })

    insights_df = pd.DataFrame(insights_summary)
    return insights_results, insights_df


# =========================
# BUDGET MODULE
# =========================
def run_budget_module(monthly):
    budget_results = []
    budget_text = {}

    for acc in monthly['Account No'].unique():
        user_df = monthly[monthly['Account No'] == acc].copy().fillna(0)

        if len(user_df) < 4:
            continue

        user_df = user_df.sort_values('ds')
        recent_df = user_df.tail(3)
        avg_income = recent_df['total_credit'].median()
        avg_expense = recent_df['total_debit'].median()

        if avg_income == 0:
            continue

        spending_ratio = avg_expense / (avg_income + 1)
        if spending_ratio > 0.9:
            budget_ratio = 0.85
        elif spending_ratio > 0.75:
            budget_ratio = 0.75
        else:
            budget_ratio = 0.65

        recommended_budget = avg_income * budget_ratio
        diff = avg_expense - recommended_budget

        if avg_expense > recommended_budget:
            status = "🔴 Over Budget"
            message = f"Reduce expenses by ₹{diff:,.0f}"
        elif avg_expense > recommended_budget * 0.9:
            status = "🟡 Near Limit"
            message = "You're close to your budget limit"
        else:
            status = "🟢 Within Budget"
            message = "Good financial control"

        budget_text[acc] = {
            "avg_income": avg_income,
            "avg_expense": avg_expense,
            "recommended_budget": recommended_budget,
            "spending_ratio": spending_ratio,
            "status": status,
            "message": message,
            "diff": diff
        }

        budget_results.append({
            "Account No": acc,
            "Income (Median - Last 3M)": round(avg_income, 2),
            "Expense (Median - Last 3M)": round(avg_expense, 2),
            "Recommended Budget": round(recommended_budget, 2),
            "Spending Ratio": round(spending_ratio, 2),
            "Difference": round(diff, 2),
            "Status": status
        })

    budget_df = pd.DataFrame(budget_results)
    return budget_text, budget_df


# =========================
# ALERTS MODULE
# =========================
def run_alerts_module(monthly, df):
    alert_results = []
    alert_text = {}

    if 'txn_category_pred' in df.columns:
        cat_col = 'txn_category_pred'
    elif 'txn_category_weak' in df.columns:
        cat_col = 'txn_category_weak'
    else:
        raise ValueError("No category column found!")

    for acc in monthly['Account No'].unique():
        user_monthly = monthly[monthly['Account No'] == acc].copy().fillna(0)
        user_txn = df[df['Account No'] == acc].copy()

        if len(user_monthly) < 4:
            continue

        user_monthly = user_monthly.sort_values('ds')
        recent = user_monthly.tail(3)
        avg_income = recent['total_credit'].mean()
        avg_expense = recent['total_debit'].mean()
        ratio = avg_expense / (avg_income + 1)

        expense_df = user_txn[user_txn['WITHDRAWAL AMT'] > 0]
        if len(expense_df) > 0:
            cat_spend = (expense_df.groupby(cat_col)['WITHDRAWAL AMT']
                         .sum().sort_values(ascending=False))
            cat_spend = cat_spend[~cat_spend.index.isin(['transfer', 'other'])]
            if len(cat_spend) > 0:
                top_category = cat_spend.index[0]
                top_amount = cat_spend.iloc[0]
            else:
                top_category = "Other"
                top_amount = 0
        else:
            top_category = "Unknown"
            top_amount = 0

        if ratio > 1:
            alert_level = "HIGH"
            excess = avg_expense - avg_income
            lines = [
                f"🚨 HIGH: You are overspending by ₹{excess:,.0f}/month",
                f"💡 Major spending area: '{top_category}' (₹{top_amount:,.0f})",
                f"👉 Action: Reduce expenses in '{top_category}' category"
            ]
            alert_text_value = f"HIGH - Overspending | Category: {top_category}"
        elif ratio > 0.8:
            alert_level = "MEDIUM"
            lines = [
                f"⚠️ MEDIUM: You are spending close to your income",
                f"💡 Major spending area: '{top_category}' (₹{top_amount:,.0f})",
                f"👉 Action: Monitor expenses in '{top_category}' category"
            ]
            alert_text_value = f"MEDIUM - High spending | Category: {top_category}"
        else:
            alert_level = "GOOD"
            lines = [
                f"🟢 GOOD: Your spending is under control",
                f"💡 Highest spending category: '{top_category}' (₹{top_amount:,.0f})"
            ]
            alert_text_value = f"GOOD - Stable | Category: {top_category}"

        alert_text[acc] = {
            "level": alert_level,
            "lines": lines,
            "ratio": ratio,
            "avg_income": avg_income,
            "avg_expense": avg_expense,
            "top_category": top_category,
            "top_amount": top_amount
        }

        alert_results.append({
            "Account No": acc,
            "Avg Income": round(avg_income, 2),
            "Avg Expense": round(avg_expense, 2),
            "Ratio": round(ratio, 2),
            "Top Category": top_category,
            "Top Spend": round(top_amount, 2),
            "Alert": alert_text_value
        })

    alert_df = pd.DataFrame(alert_results)
    return alert_text, alert_df


# =========================
# MAIN PIPELINE
# =========================
def main_pipeline(df):
    # STEP 1: Feature Engineering
    df, monthly = create_features(df)

    # STEP 2: Categorization
    df, model_results, best_model = run_transaction_categorization(df)
    fig, category_data = plot_transaction_category_distribution(df)

    # STEP 3: Forecasting
    prophet_forecast, prophet_figs = run_prophet_forecast(monthly)
    lstm_forecast, lstm_eval_data, lstm_figs = run_lstm_forecast(monthly)

    # STEP 4: Evaluation
    prophet_eval = get_prophet_evaluation(monthly)
    lstm_eval = get_lstm_evaluation(lstm_eval_data)
    comparison = get_model_comparison(lstm_eval, prophet_eval)

    # STEP 5: Business Modules
    health_results, final_health = run_financial_health_module(monthly)
    insights_text, insights_df = run_insights_engine(monthly)
    budget_text, budget_df = run_budget_module(monthly)
    alert_text, alert_df = run_alerts_module(monthly, df)

    # STEP 6: Summary
    summary = {
        "Total Accounts": df['Account No'].nunique(),
        "Total Transactions": len(df),
        "Avg Monthly Savings": monthly['monthly_net'].mean(),
        "Date Range Start": df['DATE'].min().strftime("%b %Y"),
        "Date Range End": df['DATE'].max().strftime("%b %Y"),
    }

    return {
        "df": df,
        "monthly": monthly,
        "category_fig": fig,
        "category_data": category_data,
        "model_results": model_results,
        "best_model": best_model,
        "prophet": prophet_forecast,
        "prophet_figs": prophet_figs,
        "lstm": lstm_forecast,
        "lstm_figs": lstm_figs,
        "comparison": comparison,
        "health_results": health_results,
        "health": final_health,
        "insights": insights_df,
        "insights_text": insights_text,
        "budget": budget_df,
        "budget_text": budget_text,
        "alerts": alert_df,
        "alert_text": alert_text,
        "summary": summary
    }
