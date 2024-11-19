import streamlit as st
import pandas as pd

from pickle import load

st.header("✍️ Model Test Page")

with open("model.pkl", "rb") as f:
    clf = load(f)

df_categories = pd.read_csv("categories.csv")

input_text = st.text_input("Input text: ", "Badischer Weinwanderweg")
pred = clf.predict_proba([input_text])

pred_categories = df_categories.copy()
pred_categories["proba"] = [x.tolist()[0][1] for x in pred]
st.dataframe(pred_categories.sort_values("proba", ascending=False).head())