import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.title("🌸 Iris Flower Species Predictor")

iris = load_iris()
x=iris.data
y=iris.target
model =RandomForestClassifier()
model.fit(x,y)

sepal_length = st.slider("Sepal Length (cm)",4.0,8.0,5.1)
sepal_width=st.slider("Sepal Width (cm)",2.0,4.5,3.5)
petal_length = st.slider("Petal Length (cm)",1.0,7.0,1.4)
petal_width=st.slider("Petal Width (cm)",0.1,2.5,0.2)

input_data=[[sepal_length,sepal_width,petal_length,petal_width]]
prediction = model.predict(input_data)
predicted_species=iris.target_names[prediction[0]]

st.subheader("prediction")
st.success(f"The predicted spiecse is **{predicted_species}**")