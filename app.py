from operator import index
import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
#from pycaret.classification import setup, compare_models, pull, save_model 


with st.sidebar:
    st.image("ml.png")
    st.title("AutoStreamMl")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This applications allows you to build an automated ML pipeline with streamlit, Pandas Profiling and Pycart. Developed by : Iyed Zarrougui")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload your Dataset for Modelling!")
    file = st.file_uploader("Upload your dataset here")
    
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)


'''if choice == "ML": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')'''

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")