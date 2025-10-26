import streamlit as st
st.sidebar.title("🧑‍🎓 Student Success Predictor")
st.sidebar.write("Are you going to survive this semester?")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
import plotly.express as px

st.markdown("<h1 style='color: cornflowerblue;'>Student Success Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: lavender;'>Are you going to survive the semester? Our data will tell!</p>", unsafe_allow_html=True)
st.image("joshua-hoehne-iggWDxHTAUQ-unsplash.jpg", width=700)

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
    st.header("Introduction")
    st.subheader("🎯 Objective")
    st.image("albert-vincent-wu-m7yOJx0ALME-unsplash.jpg", width=389)
    st.write("As university students ourselves, we constantly juggle sleep, classes, and extracurriculars while striving to stay motivated and balanced. Managing these competing demands can be overwhelming, so our app leverages data visualization and predictive insights to help students better understand their habits, anticipate challenges, and make smarter decisions about how to manage their time and well-being.")
    st.write("Our aim is to arm you with information so you can grow as a student. Take a breath, take a look, you've got this semester!")
    
    st.subheader("📊 Data Set")
    st.write("Take a look here to see what we use to inform our student success visualizations and predictions!")

    st.markdown("##### Data Preview")

    rows = st.slider("Select a number of rows",5,20,5)
    
    st.dataframe(df.head(rows))

    st.markdown("##### Missing Values")

    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum()==0:
        st.success("No missing values found")
    else:
        st.warning("This data has some missing values")

    st.markdown("#### Statistical Summary")

    if st.button("Click Here to Generate Statistical Summary!"):
        st.dataframe(df.describe())

## Data Viz Page
if page == "Data Viz":
    ## Data Preview
    st.header("Data Visualization Preview")
    st.dataframe(df.head())
    st.dataframe(df.tail())
    ## Basic Info
    st.sidebar.header("Pick Your Values")
    st.sidebar.subheader("Chart Settings")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    all_cols = df.columns.tolist()
    ## Sidebar Filters
    x_axis = st.sidebar.selectbox("X Value",all_cols)
    y_axis = st.sidebar.selectbox("Y Value",numeric_cols)
    chart_type = st.sidebar.selectbox("Chart Type",["Scatter", "Line", "Bar", "Box"])
    ## Visualization
    st.subheader("📈 Chart Type")
    if chart_type == "Scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis, color=df.columns[0])
    elif chart_type == "Line":
        fig = px.line(df, x=x_axis, y=y_axis, color=df.columns[0])
    elif chart_type == "Bar":
        fig = px.bar(df, x=x_axis, y=y_axis, color=df.columns[0])
    else:
        fig = px.box(df, x=x_axis, y=y_axis, color=df.columns[0])
    st.plotly_chart(fig, use_container_width=True)

    ##Summery Stats
    st.subheader("📊 Summary Statistics")
    st.write(df.describe())

