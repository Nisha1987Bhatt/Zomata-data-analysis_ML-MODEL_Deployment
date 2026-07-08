import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

try:
    scaler = joblib.load("scaler.pkl")
except Exception:
    scaler = StandardScaler()

if not hasattr(scaler, "scale_"):
    df_encoded = pd.read_csv("Zomato Encoded Dataset.csv")
    scaler.fit(df_encoded[["Average Cost for two", "Has Table booking", "Has Online delivery", "Price range"]].values)

st.title("Restaurant Rating Prediction App")
st.set_page_config(layout = "wide")
st.caption("This app helps you to predict a restaurants review class")
st.divider()

averagecost = st.number_input("Please enter the estimated average cost for two", min_value=50,max_value=999999,value=1000,step=200)
tablebooking = st.selectbox("Restaurant has table booking?", ["Yes","No"])
onlinedelivery = st.selectbox("Restaurants have online delivery",["Yes","No"])
pricerange = st.selectbox("What is the price range(1 Cheapest,4 Most Expensive)",[1,2,3,4])

predictbutton = st.button("Predict the review")
st.divider()
model = joblib.load("mlmodel.pkl")

bookingstatus = 1 if tablebooking == "Yes" else 0
deliverystatus = 1 if onlinedelivery == "Yes" else 0

values = [[averagecost,bookingstatus,deliverystatus,pricerange]]
X_values = np.array(values)
X = scaler.transform(X_values)

if predictbutton:
    st.snow()
    
    prediction = model.predict(X)
    if len(prediction) > 0:
        prediction_value = float(np.ravel(prediction)[0])
    else:
        prediction_value = None

    if prediction_value is None:
        st.error("Prediction failed.")
    elif prediction_value < 2.5:
        st.write("Poor")
    elif prediction_value < 3.5:
        st.write("Average")
    elif prediction_value < 4.0:
        st.write("Good")
    elif prediction_value < 4.5:
        st.write("Very Good")
    else:
        st.write("Excellent")
        
    
    
