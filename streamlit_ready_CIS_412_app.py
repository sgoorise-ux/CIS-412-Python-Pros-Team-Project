
"""
Streamlit-ready wrapper for CIS_412 Team Project
Auto-generated from the user's notebook/script.

Usage:
  - Run locally: `streamlit run streamlit_ready_CIS_412_app.py`
  - Or push this file and your dataset (test.csv) to GitHub and deploy on Streamlit Cloud.

This script:
  - Tries to load the CSV from several sensible locations (repo-local, common Colab path)
  - Allows the user to provide a raw CSV URL
  - Falls back to a Streamlit file uploader
  - Runs the original analysis and replaces plt.show() calls with Streamlit-friendly displays
"""

import streamlit as st
st.set_page_config(page_title="CIS 412 - Airline Satisfaction", layout="wide")

import pandas as pd
import numpy as np
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
pip install matplotlib

st.title("CIS 412 — Airline Passenger Satisfaction (Streamlit)")
st.markdown("Auto-converted Streamlit app. If the dataset isn't found automatically, upload it below or provide a raw URL.")

# --- CSV Loading logic ---
def try_read_paths(paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                df = pd.read_csv(p)
                st.success(f"Loaded CSV from local path: {p}")
                return df
            except Exception as e:
                st.warning(f"Found file at {p} but failed to parse: {e}")
    return None

# sensible default paths to try (includes Colab path)
default_paths = [
    "test.csv",
    "/content/test.csv",
    "/mnt/data/test.csv",
    "/mnt/data/CIS_412_test.csv",
]

df = try_read_paths(default_paths)

# Allow user to input a raw CSV URL (e.g., raw.githubusercontent.com link)
csv_url = st.text_input("Optional: raw CSV URL (leave empty to use repo/upload)", value="")
if not df and csv_url:
    try:
        df = pd.read_csv(csv_url)
        st.success("Loaded CSV from provided URL.")
    except Exception as e:
        st.error(f"Failed to read CSV from URL: {e}")

# File uploader fallback
if df is None:
    uploaded = st.file_uploader("Upload CSV (fallback)", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success("Uploaded CSV read successfully.")
        except Exception as e:
            st.error(f"Uploaded file could not be parsed: {e}")
            st.stop()
    else:
        st.warning("No dataset found yet. Please upload a CSV or add 'test.csv' to the app repo.")
        st.stop()

# show basic info
st.subheader("Data preview")
st.write("Shape:", df.shape)
st.dataframe(df.head())

# --- Below: original analysis adapted for Streamlit ---
# Note: small defensive checks added where appropriate.

# Drop columns if present
cols_to_drop = [c for c in ['Unnamed: 0', 'id'] if c in df.columns]
if cols_to_drop:
    df = df.drop(cols_to_drop, axis=1)
    st.info(f"Dropped columns: {cols_to_drop}")

# Fill missing Arrival Delay in Minutes if present
if 'Arrival Delay in Minutes' in df.columns:
    if df['Arrival Delay in Minutes'].isnull().any():
        df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median())
        st.info("Filled missing 'Arrival Delay in Minutes' with median.")

# Convert target column 'satisfaction' to binary
if 'satisfaction' in df.columns:
    if df['satisfaction'].dtype == object:
        mapping = {'satisfied':1, 'neutral or dissatisfied':0}
        if set(df['satisfaction'].unique()) & set(mapping.keys()):
            df['satisfaction'] = df['satisfaction'].map(mapping)
            st.info("Mapped 'satisfaction' to binary (1 = satisfied, 0 = neutral/dissatisfied).")
else:
    st.error("Column 'satisfaction' not found in dataset. Make sure the CSV includes it.")
    st.stop()

# Encode categorical columns if present
from sklearn.preprocessing import LabelEncoder
label_cols = [c for c in ['Gender', 'Customer Type', 'Type of Travel', 'Class'] if c in df.columns]
if label_cols:
    le = LabelEncoder()
    for col in label_cols:
        try:
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception as e:
            st.warning(f"Could not encode {col}: {e}")

st.subheader("Column types & missing values")
st.write(df.info())
st.write(df.isnull().sum())

# Basic plots replaced with Streamlit pyplot
st.subheader("Satisfaction distribution")
if 'satisfaction' in df.columns:
    plt.figure(figsize=(6,3))
    sns.countplot(data=df, x='satisfaction')
    plt.title('Satisfaction Count')
    st.pyplot(plt.gcf())
    plt.clf()

# Correlation heatmap
st.subheader("Correlation heatmap (numeric columns)")
try:
    correlation = df.corr()
    plt.figure(figsize=(10,6))
    sns.heatmap(correlation, annot=True, fmt='.2f', annot_kws={'size':7}, linewidths=0.5, cmap='coolwarm')
    st.pyplot(plt.gcf())
    plt.clf()
except Exception as e:
    st.warning(f"Could not generate heatmap: {e}")

# Drop Departure Delay if highly correlated with Arrival Delay
if 'Departure Delay in Minutes' in df.columns and 'Arrival Delay in Minutes' in df.columns:
    try:
        corrval = df['Departure Delay in Minutes'].corr(df['Arrival Delay in Minutes'])
        st.write(f"Departure vs Arrival delay correlation: {corrval:.2f}")
        if abs(corrval) > 0.9:
            df = df.drop('Departure Delay in Minutes', axis=1)
            st.info("Dropped 'Departure Delay in Minutes' due to high correlation with arrival delay.")
    except Exception as e:
        st.warning(f"Could not compute correlation: {e}")

# Train/test split and model training (logistic regression + decision tree)
st.subheader("Model training: Logistic Regression & Decision Tree")
from sklearn.model_selection import train_test_split

if 'satisfaction' not in df.columns:
    st.error("'satisfaction' column missing; cannot train models.")
    st.stop()

X = df.drop('satisfaction', axis=1)
y = df['satisfaction']

# simple check: ensure X is numeric; if not, attempt to convert
non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
if non_numeric:
    st.warning(f"Converting non-numeric columns to numeric where possible: {non_numeric}")
    for c in non_numeric:
        try:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
        except Exception:
            X[c] = X[c].astype('category').cat.codes

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

st.write("Train/test shapes:", X_train.shape, X_test.shape)

# Logistic Regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

scaler = StandardScaler()
X_train_lr_scaled = scaler.fit_transform(X_train)
X_test_lr_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=2000, random_state=42)
lr.fit(X_train_lr_scaled, y_train)

y_pred = lr.predict(X_test_lr_scaled)
y_proba = lr.predict_proba(X_test_lr_scaled)[:,1]

st.write("Logistic Regression accuracy:", accuracy_score(y_test, y_pred))
st.text("Classification report (Logistic Regression):")
st.text(classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix (LR)')
st.pyplot(plt.gcf()); plt.clf()

# ROC curve
try:
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.3f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve (LR)')
    plt.legend()
    st.pyplot(plt.gcf()); plt.clf()
except Exception as e:
    st.warning(f"Could not compute ROC for logistic regression: {e}")

# Feature coefficients
try:
    feature_names = X.columns.tolist()
    coefs = lr.coef_.ravel()
    coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    st.subheader("Top coefficients (Logistic Regression)")
    st.dataframe(coef_df.head(12)[['feature','coef']])
except Exception as e:
    st.warning(f"Could not display coefficients: {e}")

# Decision Tree
from sklearn.tree import DecisionTreeClassifier, plot_tree

dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

st.write("Decision Tree accuracy:", accuracy_score(y_test, y_pred_dt))
st.text("Classification report (Decision Tree):")
st.text(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(4,3))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix (DT)')
st.pyplot(plt.gcf()); plt.clf()

# Feature importances
try:
    importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': dt_model.feature_importances_}).sort_values(by='Importance', ascending=False)
    st.subheader("Top features by importance (Decision Tree)")
    st.dataframe(importances.head(10))
except Exception as e:
    st.warning(f"Could not compute feature importances: {e}")

# Tree plot (may be large)
try:
    plt.figure(figsize=(18,10))
    plot_tree(dt_model, filled=True, feature_names=X_train.columns, class_names=['Dissatisfied','Satisfied'], rounded=True, fontsize=8)
    st.pyplot(plt.gcf()); plt.clf()
except Exception as e:
    st.warning(f"Could not render decision tree visualization: {e}")

st.success("Analysis complete. You can download this script and deploy with Streamlit Cloud or run locally.")
