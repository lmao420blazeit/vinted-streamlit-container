import pandas as pd
import streamlit as st
import pickle
import xgboost as xgb
import numpy as np


def main():
    st.set_page_config(layout="wide")
    st.title("Prediction Service")
    labels = pd.read_csv("model_development/model_artifacts/params_xgboost.csv")
    with open("model_development/model_artifacts/xgboost_regression.pkl", "rb") as f:
        # Load the pickled object from the file
        model = pickle.load(f)

    catalog = st.selectbox(
        'Select the product catalog.',
        options = labels[labels['Decoded'].str.startswith("catalog_id", na = False)]["Decoded"]
    )

    brand = st.selectbox(
        'Select the brand.',
        options = labels[labels['Decoded'].str.startswith("brand_title", na = False)]["Decoded"] #.unique()
    )

    size = st.selectbox(
        'Select the size.',
        options = labels[labels['Decoded'].str.startswith("size_title", na = False)]["Decoded"] #.unique()
    )

    status = st.selectbox(
        'Select the status.',
        options = labels[labels['Decoded'].str.startswith("status", na = False)]["Decoded"] #.unique()
    )
    # Define a function to map size titles to 1 if it matches the unique size title, else 0
    def map_size_title(title):
        return 1 if title in [size, brand, catalog] else 0

    # Add a new column 'is_size_80' indicating whether each row corresponds to the unique size title
    labels['input_vector'] = labels['Decoded'].apply(lambda x: map_size_title(x))
    print(labels)
    ins = pd.DataFrame(labels['input_vector']).T
    #print(labels[["Labels", 'input_vector']])
    #matrix = xgb.DMatrix(labels[["input_vector", 'Labels']].to_numpy())
    print(ins.shape)
    prediction = model.predict(ins.values)
    print(prediction)

main()