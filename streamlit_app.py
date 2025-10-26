import streamlit as st
st.sidebar.title("🧑‍🎓 Student Success Predictor")
st.sidebar.write("Are you going to survive this semester?")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error


st.image("header.jpg", use_container_width=True)

st.markdown("<h1 style='color: cornflowerblue; text-align: center;'>Student Success Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: lavender; text-align: center;'>Are you going to survive the semester? Our data will tell!</p>", unsafe_allow_html=True)

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
page = st.sidebar.selectbox("Select Page",["Introduction","Data Viz","Prediction", "Crystal Ball",])
##Introduction Page
if page == "Introduction":
    st.header("Introduction")
    st.image("boitumelo-o_tcYADlSt8-unsplash.jpg", use_container_width=True)
    st.subheader("🎯 Objective")
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
    st.image("conny-schneider-pREq0ns_p_E-unsplash.jpg", use_container_width=True)
    st.header("Data Visualization Preview")
    st.write("Observe and manipulate the data as it applies to your situation to see what factors are impacting your academic performance!")
    st.dataframe(df.head())
    st.dataframe(df.tail())
    ## Basic Info
    st.subheader("🧐 Pick Your Values")
    st.markdown("Chart Settings")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    all_cols = df.columns.tolist()
    ## Sidebar Filters
    x_axis = st.selectbox("X Value",all_cols)
    y_axis = st.selectbox("Y Value",numeric_cols)
    chart_type = st.selectbox("Chart Type",["Scatter", "Line", "Bar", "Box"])
    ## Visualization
    st.subheader("📈 Make a Chart!")
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


## Prediction Page
if page == "Prediction":
    st.info("Using sample data (Student Performance)")
    df2 = pd.DataFrame({
           "study_hours": [2, 3, 4, 5, 6, 8, 10],
           "attendance": [60, 65, 70, 75, 80, 85, 90],
           "score": [50, 55, 60, 65, 70, 80, 90]
       })
    st.subheader("1️⃣ Data Preview")
    st.dataframe(df2)
    st.subheader("2️⃣ Select Target Variable (to predict)")
    target_col = st.selectbox("Target variable", df2.columns)
    if not target_col:
       st.warning("Please choose a target column.")
       st.stop()
    X = df2.drop(columns=[target_col])
    y = df2[target_col]
    problem_type = "regression" if pd.api.types.is_numeric_dtype(y) else "classification"
    st.write(f"Detected problem type: **{problem_type}**")
    test_size = st.slider("Test size (%)", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test).reindex(columns=X_train.columns, fill_value=0)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    st.subheader("3️⃣ Choose Model")
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    if problem_type == "classification":
       model = RandomForestClassifier(random_state=42)
    else:
       model = RandomForestRegressor(random_state=42)
       model.fit(X_train_scaled, y_train)
       preds = model.predict(X_test_scaled)
    st.subheader("📈 Model Performance")
    if problem_type == "classification":
       acc = accuracy_score(y_test, preds)
       st.metric("Accuracy", f"{acc:.3f}")
    else:
       mae = mean_absolute_error(y_test, preds) 
       st.metric("MAE", f"{mae:.3f}")
    st.write("Predicted accuracy of our model in terms of percentage points")
    st.subheader("🧪 Try your own input")
    input_data = {}
    for col in X.columns:
        val = st.number_input(f"Enter {col}", value=float(df2[col].mean()))
        input_data[col] = val
        
    if st.button("Predict"):
       new_df = pd.DataFrame([input_data])
       new_df = pd.get_dummies(new_df).reindex(columns=X_train.columns, fill_value=0)
       new_df_scaled = scaler.transform(new_df)
       prediction = model.predict(new_df_scaled)[0]
       st.success(f"Predicted **{target_col}**: {prediction}")
    


#st.image()
    

if page == "Crystal Ball":
    st.image("nicole-avagliano-TeLjs2pL5fA-unsplash.jpg", use_container_width=True)
    st.header("Crystal Ball")
    st.write("Sometimes all a student needs is a sign that they will do well! Use this page to build up belief in yourself, everything you need is inside of you.")
    import random
    st.subheader("🔮 Will I succeed?")
    if st.button("Click here to reveal your fate!"):
        st.header(random.choice(["Absolutely, keep going!", "Obviously, you got this!", "Definitely, you're almost there!", "Of course, just focus and shine!"]))
    
    st.subheader("🍀 Find A Lucky Charm!")
    if st.button("Click here to be gifted a lucky charm to bring you good fortune!"):
        st.write("Your lucky charm is:")
        st.header(random.choice(["Lucky Rabbit 🐰", "Lucky Duck 🦆", "Red Envelope 🧧", "Fortune Cookie 🥠", "Four Leaf Clover 🍀", "Evil Eye Protection 🧿", "Lucky Ladybug 🐞"]))
    
