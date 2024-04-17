import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import json
import pickle

def load_credentials(path = "aws_rds_credentials.json"):
     with open(path, 'r') as file:
          config = json.load(file)

     # set up credentials
     for key in config.keys():
          os.environ[key] = config[key]

     return


load_credentials()

aws_rds_url = f"postgresql://{os.environ['user']}:{os.environ['password']}@{os.environ['host']}:{os.environ['port']}/{os.environ['database']}?sslmode=require"

from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objs as go

def plotly_wordcloud(text):
    wc = WordCloud(stopwords = set(STOPWORDS),
                   max_words = 200,
                   max_font_size = 100)
    wc.generate(text)
    
    word_list=[]
    freq_list=[]
    fontsize_list=[]
    position_list=[]
    orientation_list=[]
    color_list=[]

    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)
        
    # get the positions
    x=[]
    y=[]
    for i in position_list:
        x.append(i[0])
        y.append(i[1])
            
    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(int(i*100))
    new_freq_list
    
    trace = go.Scatter(x=x, 
                       y=y, 
                       hoverinfo='text',
                       textfont = dict(size=new_freq_list,
                                       color=color_list),
                       hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                       mode='text',  
                       text=word_list
                      )
    
    layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})
    
    fig = go.Figure(data=[trace], layout=layout)
    
    return fig

# Load a sample dataset
def load_data(brand, catalog):
    engine = create_engine(aws_rds_url)
    sql_query = f"""SELECT * 
                    FROM public.products_catalog 
                    WHERE brand_title = '{brand}' AND catalog_id = '{catalog}'
                    ORDER BY date DESC
                    """
    df = pd.read_sql(sql_query, engine)
    return (df)

# Main function
def main():
    st.set_page_config(layout="wide", page_title="Price")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # loading fonts
    st.markdown("""
        <link href="https://fonts.googleapis.com/css?family=Nunito:200,300,700" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css?family=Bungee" rel="stylesheet">
        """,
        unsafe_allow_html= True)

    st.write("<h2 style='font-family: Bungee; color: orange'>Prices</h2>", 
             unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns([0.5, 0.5, 0.5, 0.5, 0.5])
    with col1:
        catalog = st.selectbox(
            'Select the product catalog.',
            options = pd.read_csv("prediction_service/catalog_id.csv")["0"]
        )

    with col2:
        brand = st.selectbox(
            'Select the brand.',
            options = pd.read_csv("prediction_service/brand_title.csv")["0"]
        )

    with col3:
        color = st.selectbox(
            'Select the color.',
            options = pd.read_csv("prediction_service/color1_id.csv")["0"]
        )

    with col4:
        status = st.selectbox(
            'Select the status of the item.',
            options = pd.read_csv("prediction_service/status.csv")["0"]
        )

    col1, col2, col3 = st.columns([0.5, 0.5, 0.5])
    with col1:
        country = st.selectbox(
            'Select the country of origin.',
            options = pd.read_csv("prediction_service/country.csv")["0"]
        )

    with col2:
        package_size_id = st.selectbox(
            'Select the package size.',
            options = pd.read_csv("prediction_service/package_size_id.csv")["0"]
        )

    with col3:
        size_title = st.selectbox(
            'Select the size.',
            options = pd.read_csv("prediction_service/size_title.csv")["0"]
        )
    
main()
        