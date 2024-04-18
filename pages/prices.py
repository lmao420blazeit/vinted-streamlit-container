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

def load_labels():
    engine = create_engine(aws_rds_url)
    sql_query = f"""SELECT DISTINCT catalog_id, brand_title 
                    FROM public.products_catalog 
                    WHERE date >= CURRENT_DATE - INTERVAL '90 days'
                    GROUP BY brand_title, catalog_id 
                    HAVING COUNT(*) > 100
                    """
    df = pd.read_sql(sql_query, engine)
    return (df)   

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

    st.session_state.labels = load_labels()

    st.write("<h2 style='font-family: Bungee; color: orange'>Prices</h2>", 
             unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Overview", "Price Prediction"])
    with tab1:
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
        st.session_state.products_catalog = load_data(brand = brand, 
                                                        catalog = catalog)
        # remove outliers for viz purposes
        q_high = st.session_state.products_catalog["price"].quantile(0.95)
        q_low = st.session_state.products_catalog["price"].quantile(0.05)
        st.session_state.products_catalog = st.session_state.products_catalog[(st.session_state.products_catalog["price"] < q_high) & 
                                                                                (st.session_state.products_catalog["price"] > q_low)]
            
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns(6)
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
                        value="{:.2f} €".format(st.session_state.products_catalog['price'].median()),
                        help="Median price (€) in the sample")
            with col5:
                st.metric(label="**Standard Dev.**", 
                        value="{:.2f} €".format(st.session_state.products_catalog['price'].std()),
                        help="Std. Dev price (€) in the sample")
            with col6:
                st.metric(label="**Total Volume**", 
                        value="{:.2f} €".format(st.session_state.products_catalog['price'].sum()),
                        help="Total volume (€) in the sample")

        #######################
        # CSS styling
        st.markdown("""
        <style>

        [data-testid="block-container"] {
            padding-left: 2rem;
            padding-right: 2rem;
            padding-top: 1rem;
            padding-bottom: 0rem;
            margin-bottom: -7rem;
        }

        [data-testid="stVerticalBlock"] {
            padding-left: 0rem;
            padding-right: 0rem;
        }

        [data-testid="stMetric"] {
            background-color: #393939;
            text-align: center;
            padding: 15px 0;
            border-radius: 10px;
        }

        [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        align-items: center;
        }

        [data-testid="stMetricDeltaIcon-Up"] {
            position: relative;
            left: 38%;
            -webkit-transform: translateX(-50%);
            -ms-transform: translateX(-50%);
            transform: translateX(-50%);
        }

        [data-testid="stMetricDeltaIcon-Down"] {
            position: relative;
            left: 38%;
            -webkit-transform: translateX(-50%);
            -ms-transform: translateX(-50%);
            transform: translateX(-50%);
        }
        .st-emotion-cache-q8sbsg {
            font-family: Nunito, sans-serif;
            font-size: 30px;
        }
        </style>
        """, unsafe_allow_html=True)

        cols = st.columns([0.25, 0.25, 0.25, 0.25])
        data = st.session_state.products_catalog.groupby(["date"])["price"].agg(['median', 'count']).reset_index()
        with cols[3]:
            fig = px.bar(data, 
                        x='date', 
                        y='count')
            fig.update_layout(title = "Number of articles per day")
            st.plotly_chart(fig, use_container_width= True)  
        with cols[2]:
            #plt.subplots(figsize = (8,8))
            #wordcloud = WordCloud (
            #    background_color = 'black',
            #        ) \
            #    .generate(' '.join(st.session_state.products_catalog["title"]))
            #plt.imshow(wordcloud, interpolation='bilinear')
            #st.pyplot(use_container_width= True)   
            fig = px.bar(data, 
                        x='date', 
                        y='median')
            fig.update_layout(title = "Median price of articles per day")
            st.plotly_chart(fig, use_container_width= True)  

        with cols[1]: 
            fig = px.pie(st.session_state.products_catalog[["status", "product_id"]], 
                        values="product_id", 
                        names="status", 
                        title='Products by status')
            st.plotly_chart(fig, use_container_width= True)

        with cols[0]: 
            fig = px.pie(st.session_state.products_catalog[["size_title", "product_id"]], 
                        values="product_id", 
                        names="size_title", 
                        title='Products by size')
            st.plotly_chart(fig, use_container_width= True)

        fig = px.histogram(st.session_state.products_catalog, 
                        x="price", 
                        marginal="box", 
                        barmode= "overlay", 
                        facet_col="status",
                        category_orders={"status":["Satisfatório", "Bom", "Muito bom", "Novo sem etiquetas", "Novo com etiquetas"]})

        st.write("<h6 style='font-family: Bungee; color: orange'>Price per status</h6>", 
                unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)  

        import numpy as np
        from scipy.stats import bootstrap

        def calculate_quantiles(data):
            return np.percentile(data, [25, 50, 75])
        
        cols = st.columns(5)
        for i, index in enumerate(["Satisfatório", "Bom", "Muito bom", "Novo sem etiquetas", "Novo com etiquetas"]):
            __ = st.session_state.products_catalog[st.session_state.products_catalog["status"] == index]
            try:
                bootstrap_results = bootstrap((__["price"], ), 
                                            statistic=calculate_quantiles,
                                            method = "basic")
                low = bootstrap_results.confidence_interval.low
                high = bootstrap_results.confidence_interval.high
            except:
                low = 0
                high = 0

            df_confidence_interval = pd.DataFrame({
                'low': low,
                'high': high
            },
            index =  ["Q1", "Q2", "Q3"])

            with cols[i]:
                st.write(index, __["price"].count())
                st.dataframe(
                    df_confidence_interval,
                    column_config={
                        "low": st.column_config.NumberColumn(
                            "Lower CI",
                            help="Lower CI",
                            format="%.2f €",
                        ),
                        "high": st.column_config.NumberColumn(
                            "Upper CI",
                            help="Upper CI",
                            format="%.2f €",
                        )
                    },
                    hide_index=False,
                    use_container_width= True
                )


        fig = px.histogram(st.session_state.products_catalog[~st.session_state.products_catalog['size_title'].isin(["XXXL", "4XL"])], 
                        x="price", 
                        marginal="box", 
                        barmode= "overlay", 
                        facet_col="size_title",
                        category_orders={"size_title":["XS", "S", "M", "L", "XL", "XXL"]})

        st.write("<h6 style='font-family: Bungee; color: orange'>Price per size</h6>", 
                unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)  

    with tab2:
        with st.expander("LightGBM"):
            st.write("""
                # Regression modelling using LightGBM
                    
                The model implemented for the task of mostly in sample prediction is LightGBM. LightGBM is a lightweight tree based gradient boosting machine useful to solve problems with categorical data (column oriented inputs).
                The algorithm provides decent results in this dataset, but there is still a lot of room to improve here, including: 
                - increasing the size of the dataset, which partially tackles the out-sample prediction (extrapolation) which it isn't good at
                - refining the original dataset, such as uniformization of country names, segmentation of catalogs and sizes
                - hyperparameter tunning

            """)
        col1, col2, col3, col4, col5, col6, col7 = st.columns([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
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
                options = pd.read_csv("prediction_service/color1.csv")["0"]
            )

        with col4:
            status = st.selectbox(
                'Select the status of the item.',
                options = pd.read_csv("prediction_service/status.csv")["0"]
            )

        with col5:
            country = st.selectbox(
                'Select the country of origin.',
                options = pd.read_csv("prediction_service/country.csv")["0"]
            )

        with col6:
            package_size_id = st.selectbox(
                'Select the package size.',
                options = pd.read_csv("prediction_service/package_size_id.csv")["0"]
            )

        with col7:
            size_title = st.selectbox(
                'Select the size.',
                options = pd.read_csv("prediction_service/size_title.csv")["0"]
            )

        with open('prediction_service/lgb.pkl', 'rb') as f:
            clf = pickle.load(f)
        
        test_df = pd.read_pickle("prediction_service/test_sample.pkl")
        labels = {"size_title": size_title, 
                "color1": color,
                "brand_title": brand,
                "status": status,
                "catalog_id": catalog,
                "package_size_id": package_size_id,
                "country": country}

        input_df = pd.DataFrame([labels]) 
        
        for col in input_df.columns:
            input_df[col] = input_df[col].astype("category")

        y_pred = clf.predict(input_df)
        st.write("Expected selling price is {:.2f}€".format(float(y_pred)))


main()