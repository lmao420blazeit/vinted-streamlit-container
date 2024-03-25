import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import json

def load_credentials(path = "aws_rds_credentials.json"):
     with open(path, 'r') as file:
          config = json.load(file)

     # set up credentials
     for key in config.keys():
          os.environ[key] = config[key]

     return


load_credentials()

aws_rds_url = f"postgresql://{os.environ['user']}:{os.environ['password']}@{os.environ['host']}:{os.environ['port']}/{os.environ['database']}?sslmode=require"

import subprocess
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

def load_labels():
    engine = create_engine(aws_rds_url)
    sql_query = f"SELECT DISTINCT catalog_id, brand_title FROM public.products_catalog GROUP BY brand_title, catalog_id HAVING COUNT(product_id) > 300"
    df = pd.read_sql(sql_query, engine)
    return (df)   

# Load a sample dataset
def load_data(brand, catalog):
    engine = create_engine(aws_rds_url)
    sql_query = f"SELECT * FROM public.products_catalog WHERE brand_title = '{brand}' and catalog_id = '{catalog}' ORDER BY date DESC"
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

    st.session_state.labels = load_labels()

    st.write("<h2 style='font-family: Bungee;'>Prices</h2>", 
             unsafe_allow_html=True)

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        catalog = st.selectbox(
            'Select the product catalog.',
            options = st.session_state.labels.catalog_id.unique()
        )

    with col2:
        brand = st.selectbox(
            'Select the brand.',
            options = st.session_state.labels[st.session_state.labels["catalog_id"] == catalog]["brand_title"] #.unique()
        )
    # Load data
    global products_catalog
    if 'products_catalog' not in st.session_state:
        st.session_state.products_catalog = load_data(brand = brand, catalog = catalog)
        # remove outliers for viz purposes
        q_high = st.session_state.products_catalog["price"].quantile(0.95)
        q_low = st.session_state.products_catalog["price"].quantile(0.05)
        st.session_state.products_catalog = st.session_state.products_catalog[(st.session_state.products_catalog["price"] < q_high) & 
                                                                              (st.session_state.products_catalog["price"] > q_low)]
        
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(label="**Sizes**", 
                    value=st.session_state.products_catalog['size_title'].nunique(),
                    help="Number of unique products in the sample")
        with col2:
            st.metric(label="**Products**", 
                    value=st.session_state.products_catalog['product_id'].nunique(),
                    help="Number of unique products in the sample")
        with col3:
            st.metric(label="**Active Users**", 
                    value=st.session_state.products_catalog['user_id'].nunique(),
                    help="Number of unique users in the sample")
        with col4:
            st.metric(label="**Median Price**", 
                    value=st.session_state.products_catalog['price'].median(),
                    help="Median price (€) in the sample")
        with col5:
            st.metric(label="**Total Volume**", 
                    value=st.session_state.products_catalog['price'].sum(),
                    help="Total volume (€) in the sample")

    cols = st.columns([0.4, 0.4])
    with cols[1]:
        plt.subplots(figsize = (8,8))
        wordcloud = WordCloud (
            background_color = 'white',
                ) \
            .generate(' '.join(st.session_state.products_catalog["title"]))
        plt.imshow(wordcloud, interpolation='bilinear')
        st.pyplot(use_container_width= True)        

    with cols[0]: 
        fig = px.pie(st.session_state.products_catalog[["status", "product_id"]], 
                    values="product_id", 
                    names="status", 
                    title='Products by status')
        st.plotly_chart(fig)

    fig = px.histogram(st.session_state.products_catalog, 
                       x="price", 
                       marginal="box", 
                       barmode= "overlay", 
                       facet_col="status",
                       category_orders={"status":["Satisfatório", "Bom", "Muito bom", "Novo sem etiquetas", "Novo com etiquetas"]})

    st.write("<h6 style='font-family: Bungee;'>Price per status</h6>", 
             unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)  

    import numpy as np
    from scipy.stats import bootstrap

    def calculate_quantiles(data):
        return np.percentile(data, [25, 50, 75])
    
    cols = st.columns(5)
    for i, index in enumerate(["Satisfatório", "Bom", "Muito bom", "Novo sem etiquetas", "Novo com etiquetas"]):
        __ = st.session_state.products_catalog[st.session_state.products_catalog["status"] == index]
        bootstrap_results = bootstrap((__["price"], ), 
                                    statistic=calculate_quantiles,
                                    method = "basic")
        
        df_confidence_interval = pd.DataFrame({
            'low': bootstrap_results.confidence_interval.low,
            'high': bootstrap_results.confidence_interval.high
        },
        index =  ["Q25%", "Q50%", "Q75%"])
        with cols[i]:
            st.write(index, __["price"].count())
            st.table(df_confidence_interval)


main()