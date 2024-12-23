from operator import index
import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Streamlit Sidebar
with st.sidebar:
    st.image("ml.png")
    st.title("AutoStreamMl")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This application allows you to build an automated ML pipeline with Streamlit, Pandas Profiling, and scikit-learn. Developed by: Iyed Zarrougui")

# Load dataset if it exists
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

# Upload dataset
if choice == "Upload":
    st.title("Upload your Dataset for Modelling!")
    file = st.file_uploader("Upload your dataset here")
    
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

# Profiling
if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

# Machine Learning with scikit-learn
if choice == "ML":
    st.title("Automated Machine Learning with scikit-learn")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    
    if st.button('Run Modelling'):
        # Encode categorical variables
        le = LabelEncoder()
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[col] = le.fit_transform(df_encoded[col])
        
        # Handle missing values
        num_cols = df_encoded.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns

        imputer_num = SimpleImputer(strategy='mean')
        df_encoded[num_cols] = imputer_num.fit_transform(df_encoded[num_cols])

        if not cat_cols.empty:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df_encoded[cat_cols] = imputer_cat.fit_transform(df_encoded[cat_cols])
        
        # Split the data
        X = df_encoded.drop(columns=[chosen_target])
        y = df_encoded[chosen_target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and fit the model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Evaluate the model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        st.write("Model Accuracy: ", accuracy)

        # Save the model
        joblib.dump(model, 'best_model.pkl')
        st.success("Model trained and saved successfully!")

# Download model
if choice == "Download":
    st.title("Download your model now!!!")
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
    else:
        st.warning("No model found. Please run the ML pipeline first.")
