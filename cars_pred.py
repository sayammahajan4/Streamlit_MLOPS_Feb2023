import datetime
import pickle

import pandas as pd
import sklearn
import streamlit as st


st.header('Cars-24-Price-Prediction', divider='rainbow')
# df=pd.read_csv("/Users/LENOVO/PycharmProjects/StreamLitTutorial/cars24-car-price-cleaned.csv")
# st.dataframe(df)


fuel_type = st.selectbox(
    "Select Fuel Type",
    ("Diesel", "Electric", "LPG", "Petrol"))

engine = st.slider("Set Engine power ", 500, 5000, step=100)

col1, col2 = st.columns(2)

with col1:
    transmission_type = st.selectbox("Select Transmission Type",
                                     ("Automatic", "Manual"))
with col2:
    d_start = st.date_input("Year of Manufacture", datetime.date(2014,1,1))
    year= d_start.year


encode_dict= {"fuel_type": {"Diesel": 1, "Petrol": 2,  "Electric": 3, "LPG": 4} ,
              "transmission_type" : {"Manual" : 1 ,"Automatic" : 2}
}


def model_pred(fuel_type, engine, transmission_type, year ):
    with open("car_pred","rb") as handle :
        model=pickle.load(handle)
        return model.predict([[year,1,120000,fuel_type,transmission_type,19.7,engine,46.3,5.0]])


if st.button("Predict"):
    fuel_type = encode_dict["fuel_type"][fuel_type]
    transmission_type=encode_dict["transmission_type"][transmission_type]
    price= model_pred(fuel_type, engine,transmission_type, year )
    st.write("Estimated Price(in Lakhs)",price)