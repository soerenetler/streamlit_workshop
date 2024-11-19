import streamlit as st
import pandas as pd

from pickle import load

st.header("✨ Insights")

with open("model.pkl", "rb") as f:
    clf = load(f)

df_categories = pd.read_csv("categories.csv")

for category_index, category_name in enumerate(df_categories["name"]):
    st.subheader(category_name)
    st.write(sorted(zip(clf["preprocess"].get_feature_names_out(), clf["clf"].estimators_[category_index].coef_[0]), key=lambda x:x[1], reverse=True)[:5])