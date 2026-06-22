import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("Iris.csv")

X = df[[
"SepalLengthCm",
"SepalWidthCm",
"PetalLengthCm",
"PetalWidthCm"
]]

y = df["Species"]

model = DecisionTreeClassifier()
model.fit(X, y)

st.title("Iris Flower Classification")

sl = st.number_input("Sepal Length", 5.1)
sw = st.number_input("Sepal Width", 3.5)
pl = st.number_input("Petal Length", 1.4)
pw = st.number_input("Petal Width", 0.2)

if st.button("Predict"):
    result = model.predict([[sl, sw, pl, pw]])
    st.success(result[0])