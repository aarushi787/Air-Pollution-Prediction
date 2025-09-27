import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="AQI Prediction Dashboard", layout="wide")

st.title("🌍 Air Quality Index (AQI) Prediction Dashboard")
st.markdown("Upload a `city_day.csv` or `city_hour.csv` (or similar air-quality CSV). The app will clean data, compute AQI labels, let you explore, train models and make predictions.")

# --------------------- Helpers ---------------------

def numeric(x):
    # remove common junk characters and convert to float; keep NaN on failure
    try:
        if pd.isna(x):
            return np.nan
        s = str(x)
        s = s.replace('x','').replace('#','').replace('*','').strip()
        # empty -> nan
        if s == '':
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def compute_aqi_label(df, pollutant_cols):
    # Simplified AQI: take max pollutant value per row among pollutant_cols
    df = df.copy()
    df['AQI_raw'] = df[pollutant_cols].max(axis=1)
    # bins similar to notebook
    bins = [0,50,100,150,200,500000]
    labels = [0,1,2,3,4]
    df['label'] = pd.cut(df['AQI_raw'], bins=bins, labels=labels, include_lowest=True).astype('float')
    return df


def download_link_df(df, filename="data.csv"):
    towrite = BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    return towrite

# --------------------- Sidebar ---------------------
st.sidebar.header("1) Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
missing_strategy = st.sidebar.selectbox("Missing value strategy", ['Drop Rows', 'Fill Mean', 'Fill Median', 'Forward Fill', 'Back Fill'])
max_rows = st.sidebar.number_input("Max rows to use (0 = all)", min_value=0, value=5000, step=500)
model_choice = st.sidebar.multiselect("Models to train", ['RandomForest', 'LogisticRegression', 'SVM'], default=['RandomForest'])
random_seed = st.sidebar.number_input("Random seed", value=42)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed for educational/demo use. AQI calculation here is simplified (max pollutant); use official formulas for production.")

# --------------------- Main ---------------------
if uploaded_file is not None:
    # load
    raw = pd.read_csv(uploaded_file)
    st.subheader("1. Raw Data Preview")
    st.write(f"Rows: {raw.shape[0]}, Columns: {raw.shape[1]}")
    st.dataframe(raw.head())

    # show columns
    st.write("**Columns in file:**")
    st.write(raw.columns.tolist())

    # Clean numeric columns we expect or detect
    expected_cols = ['O3','PM2.5','PM10','CO','NO','NO2','NOx','SO2','NMHC','CH4','TEMP','RH']
    present = [c for c in expected_cols if c in raw.columns]
    pollutant_candidates = [c for c in ['PM2.5','PM10','NO2','O3','CO','SO2'] if c in raw.columns]

    st.write(f"Detected expected columns: {present}")
    st.write(f"Pollutant columns used for AQI (detected): {pollutant_candidates}")

    # Apply numeric cleaning to present cols
    cleaned = raw.copy()
    for c in present:
        cleaned[c] = cleaned[c].apply(numeric)

    # missing handling
    if missing_strategy == 'Drop Rows':
        cleaned = cleaned.dropna()
    elif missing_strategy == 'Fill Mean':
        cleaned = cleaned.fillna(cleaned.mean(numeric_only=True))
    elif missing_strategy == 'Fill Median':
        cleaned = cleaned.fillna(cleaned.median(numeric_only=True))
    elif missing_strategy == 'Forward Fill':
        cleaned = cleaned.fillna(method='ffill')
    elif missing_strategy == 'Back Fill':
        cleaned = cleaned.fillna(method='bfill')

    # limit rows
    if max_rows > 0 and cleaned.shape[0] > max_rows:
        cleaned = cleaned.iloc[:max_rows]

    st.subheader("2. Cleaned Data Preview")
    st.dataframe(cleaned.head())
    st.write("Missing values after cleaning:")
    st.write(cleaned.isna().sum())

    # If no pollutant columns detected, warn
    if len(pollutant_candidates) == 0:
        st.error("No pollutant columns detected (PM2.5, PM10, NO2, O3, CO, SO2). Cannot compute AQI or train models.")
    else:
        # compute AQI label
        data_labeled = compute_aqi_label(cleaned, pollutant_candidates)
        st.subheader("3. AQI Labels")
        st.write(data_labeled['label'].value_counts(dropna=False))

        st.markdown("**AQI category mapping:** 0=Good, 1=Satisfactory, 2=Moderately Polluted, 3=Poor, 4=Very Poor/Severe")

        # show distribution plot
        fig1, ax1 = plt.subplots()
        data_labeled['label'].value_counts().sort_index().plot(kind='bar', ax=ax1)
        ax1.set_xlabel('AQI Category')
        ax1.set_ylabel('Count')
        st.pyplot(fig1)

        # correlation heatmap for numeric columns
        numeric_cols = data_labeled.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            fig2, ax2 = plt.subplots(figsize=(8,6))
            sns.heatmap(data_labeled[numeric_cols].corr(), annot=True, fmt='.2f', ax=ax2)
            st.subheader('Correlation heatmap (numeric features)')
            st.pyplot(fig2)

        # prepare data for ML: drop non-numeric cols
        ml_df = data_labeled.copy()
        ml_df = ml_df.select_dtypes(include=[np.number])
        # drop columns that would leak label like AQI_raw
        if 'AQI_raw' in ml_df.columns:
            ml_df = ml_df.drop(columns=['AQI_raw'])

        if 'label' not in ml_df.columns:
            st.error('Label column missing after labeling step — aborting ML.')
        else:
            # features and target
            X = ml_df.drop(columns=['label'])
            y = ml_df['label']

            # small safety: drop rows with any nan
            X = X.dropna()
            y = y.loc[X.index]

            # train/test split
            if X.shape[0] < 10:
                st.warning('Not enough numeric rows to train models. Need at least 10 samples.')
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(random_seed))

                trained_models = {}
                results = []

                if 'RandomForest' in model_choice:
                    rfc = RandomForestClassifier(n_estimators=200, random_state=int(random_seed))
                    rfc.fit(X_train, y_train)
                    preds = rfc.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    trained_models['RandomForest'] = rfc
                    results.append(('RandomForest', acc))

                if 'LogisticRegression' in model_choice:
                    try:
                        lr = LogisticRegression(max_iter=1000)
                        lr.fit(X_train, y_train)
                        preds = lr.predict(X_test)
                        acc = accuracy_score(y_test, preds)
                        trained_models['LogisticRegression'] = lr
                        results.append(('LogisticRegression', acc))
                    except Exception as e:
                        st.warning(f'Logistic Regression failed: {e}')

                if 'SVM' in model_choice:
                    try:
                        svc = SVC()
                        svc.fit(X_train, y_train)
                        preds = svc.predict(X_test)
                        acc = accuracy_score(y_test, preds)
                        trained_models['SVM'] = svc
                        results.append(('SVM', acc))
                    except Exception as e:
                        st.warning(f'SVM failed: {e}')

                st.subheader('4. Model Performance Summary')
                if len(results) > 0:
                    res_df = pd.DataFrame(results, columns=['Model','Accuracy']).sort_values('Accuracy', ascending=False)
                    st.dataframe(res_df)
                else:
                    st.write('No models were successfully trained.')

                # Show classification report and confusion matrix for the best model
                if len(trained_models) > 0:
                    best_name = res_df.iloc[0]['Model'] if len(results)>0 else list(trained_models.keys())[0]
                    best_model = trained_models[best_name]
                    preds = best_model.predict(X_test)

                    st.write(f'### Detailed report for {best_name}')
                    st.text(classification_report(y_test, preds, zero_division=0))
                    cm = confusion_matrix(y_test, preds)
                    fig3, ax3 = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax3)
                    ax3.set_xlabel('Predicted')
                    ax3.set_ylabel('Actual')
                    st.pyplot(fig3)

                    # Feature importance for RandomForest
                    if best_name == 'RandomForest':
                        try:
                            fi = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                            st.write('Feature importances (RandomForest)')
                            st.dataframe(fi.reset_index().rename(columns={'index':'feature',0:'importance'}).head(20))
                        except Exception:
                            pass

                    # ---------------- Prediction widget ----------------
                    st.subheader('5. Real-time Prediction')
                    with st.form(key='predict_form'):
                        input_vals = {}
                        cols = X.columns.tolist()
                        for c in cols:
                            # number_input needs a default
                            try:
                                val = float(X[c].mean())
                            except Exception:
                                val = 0.0
                            input_vals[c] = st.number_input(f'Input {c}', value=val)

                        submit = st.form_submit_button('Predict')
                        if submit:
                            input_df = pd.DataFrame([input_vals])
                            pred = best_model.predict(input_df)[0]
                            st.success(f'Predicted AQI category: {int(pred)}')
                            # health advice
                            adv_map = {
                                0: 'Good — Air quality is satisfactory.',
                                1: 'Satisfactory — Acceptable air quality.',
                                2: 'Moderately Polluted — People with respiratory issues should reduce activity.',
                                3: 'Poor — Reduce outdoor physical activity.',
                                4: 'Very Poor/Severe — Avoid outdoor exposure; sensitive groups stay indoors.'
                            }
                            st.info(adv_map.get(int(pred),'No advice available'))

                # ---------------- Export options ----------------
                st.subheader('6. Export / Download')
                csv_bytes = download_link_df(data_labeled)
                st.download_button(label='Download cleaned & labeled CSV', data=csv_bytes, file_name='labeled_data.csv', mime='text/csv')

                # Also allow downloading training/testing splits if available
                if 'X_train' in locals():
                    buf = BytesIO()
                    X_train.to_csv(buf, index=False)
                    buf.seek(0)
                    st.download_button('Download X_train.csv', data=buf, file_name='X_train.csv', mime='text/csv')

else:
    st.info('Upload a CSV file to begin. Try the Kaggle "Air Quality Data in India" dataset or any CSV with pollutant columns like PM2.5, PM10, NO2, O3, CO, SO2.')

# --------------------- Footer ---------------------
st.markdown("---")
st.markdown("**Notes:** This app uses a simplified AQI labeling method (max pollutant). For regulatory or health-critical systems use official AQI computation methods. Visuals and model choices are meant for exploratory and educational purposes.")
st.caption('Built with ❤️ — modify freely for your project')
