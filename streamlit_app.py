import streamlit as st
st.sidebar.title("Student Success Predictor!!")
st.sidebar.write("Are you going to survive this semester?")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns

st.markdown("<h1 style='color: cornflowerblue;'>Student Success Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: lavender;'>Are you going to survive the semester?.</p>", unsafe_allow_html=True)

df = pd.read_csv("StudentPerformanceFactors.csv")
## Automated Report 
#from ydata_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report

## Model Part 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

## page configurations
st.set_page_config(page_title="Student Success Factor Dashboard 📚",layout="wide",page_icon="📚")

## Full password logic
def check_password():
    st.sidebar.header("🔐 Login")
    password = st.sidebar.text_input("Enter Password",type="password")
    # good password
    if password == "california123":
        st.sidebar.success("Access Granted ✅")
        return True
    # wrong password
    elif password:
        st.sidebar.error("incorrect password ❌")
        return False
    # no password
    else:
        st.sidebar.info("Please enter the password to continue")
        return False
    
if not check_password():
    st.stop()



## PAGES
page = st.sidebar.selectbox("Select Page",["Introduction","Data Viz","Prediction"])
##Introduction Page
if page == "Introduction":
   # st.header("Introduction")
    #st.write("Welcome! We looked at many metrics to determine what brings a student academic success in whatever whatever whatever")
    st.subheader("01 Introduction")

    st.markdown("##### Data Preview")

    rows = st.slider("Select a number of rows",5,20,5)
    
    st.dataframe(df.head(rows))

    st.markdown("##### Missing Values")

    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum()==0:
        st.success("No missing values found")
    else:
        st.warning("You have some missing values")

    st.markdown("#### Statistical Summary")

    if st.button("Generate Statistical Summary"):
        st.dataframe(df.describe())
