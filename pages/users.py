import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.graph_objects as go
import os
import json
import plotly.express as px
import numpy as np
import altair as alt

def load_credentials(path = "aws_rds_credentials.json"):
     with open(path, 'r') as file:
          config = json.load(file)

     # set up credentials
     for key in config.keys():
          os.environ[key] = config[key]

     return


load_credentials()

aws_rds_url = f"postgresql://{os.environ['user']}:{os.environ['password']}@{os.environ['host']}:{os.environ['port']}/{os.environ['database']}?sslmode=require"

# Load a sample dataset
def load_data():
    engine = create_engine(aws_rds_url)
    sql_query = f"""SELECT * 
                    FROM public.users_staging 
                    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                        AND (user_id, date) IN (
                            SELECT user_id, MAX(date)
                            FROM public.users_staging
                            WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                            GROUP BY user_id
                        )"""
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
    
    st.write("<h2 style='font-family: Bungee; color: orange'>Users</h2>", 
             unsafe_allow_html=True)

    st.session_state.users = load_data()

    latest_date = st.session_state.users["date"].max()
    st.write(f"Latest updated on {latest_date} \n (last 30 days)", 
             unsafe_allow_html=True)

    # wrapping up with containers
    with st.container():
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col6:
            st.metric(label="**Active users**", 
                    value="{}".format(st.session_state.users['user_id'].nunique()),
                    help="Average number of unique users in the sample")            
        with col1:
            st.metric(label="**Item count**", 
                    value="{:,.0f}".format(st.session_state.users['item_count'].mean()),
                    help="Average number of unique brands in the sample")
        with col2:
            st.metric(label="**Positive feedback**", 
                    value="{:,.0f} 	👍".format(st.session_state.users['positive_feedback_count'].mean()),
                    help="Average number of positive feedback per user")
        with col3:
            st.metric(label="**Negative feedback**", 
                    value="{:,.0f} 👎".format(st.session_state.users['negative_feedback_count'].mean()),
                    help="Average number of negative feedback per user")
        with col4:
            st.metric(label="**Total feedback**", 
                    value="{:,.0f}".format(st.session_state.users['feedback_count'].mean()),
                    help="Average of total feedback per user")
        with col5:
            st.metric(label="**Feedback reputation**", 
                    value="{:,.2f} ⭐".format(st.session_state.users['feedback_reputation'].median()*5),
                    help="Feedback reputation")

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

    data = st.session_state.users.groupby("country_title")["item_count"].sum().reset_index()

    # Mapping ISO format codes to country names
    # Move this into a lib dependency
    iso_mapping = {
        'BEL': ['Belgien', 'Belgio', 'Belgique', 'Belgium', 'België', 'Bélgica'],
        'ESP': ['Espagne', 'Espanha', 'España'],
        'FRA': ['France', 'Francia', 'Francúzsko', 'Frankreich', 'Frankrijk', 'França'],
        'ITA': ['Italia', 'Italie', 'Italien', 'Italija', 'Italië', 'Italy', 'Itália'],
        'NLD': ['Nederland', 'Netherlands', 'Niederlande', 'Nyderlandai', 'Paesi Bassi', 'Pays-Bas', 'Países Baixos', 'Países Bajos'],
        'PRT': ['Portogallo', 'Portugal'],
        'ESP': ['Spagna', 'Spain', 'Spanien', 'Spanje']
    }

    # Reverse mapping from country names to ISO format codes
    reverse_mapping = {value: key for key, values in iso_mapping.items() for value in values}
    data['iso_code'] = data['country_title'].map(reverse_mapping)
    data = data.groupby('iso_code')['item_count'].sum().reset_index()

    fig = go.Figure(
            data=go.Choropleth(
                locations=data["iso_code"],  # Country names
                z=data["item_count"],  # Values for each country
                locationmode='ISO-3',  # Use country names to identify locations
                colorbar_title='Item count',  # Title for the color bar
    ))

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=350
    )

    # Update layout
    fig.update_layout(
        title_text='Total items listed per country',
        geo=dict(
            showcoastlines=True,  # Show coastlines on the map
        )
    )
    fig.update_geos(
        lonaxis_range=[-15, 30],
        lataxis_range=[35, 60]
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig, use_container_width= True)

    # Mapping ISO format codes to country names
    iso_mapping = {
        'BEL': ['Belgien', 'Belgio', 'Belgique', 'Belgium', 'België', 'Bélgica'],
        'ESP': ['Espagne', 'Espanha', 'España'],
        'FRA': ['France', 'Francia', 'Francúzsko', 'Frankreich', 'Frankrijk', 'França'],
        'ITA': ['Italia', 'Italie', 'Italien', 'Italija', 'Italië', 'Italy', 'Itália'],
        'NLD': ['Nederland', 'Netherlands', 'Niederlande', 'Nyderlandai', 'Paesi Bassi', 'Pays-Bas', 'Países Baixos', 'Países Bajos'],
        'PRT': ['Portogallo', 'Portugal'],
        'ESP': ['Spagna', 'Spain', 'Spanien', 'Spanje']
    }

    # Reverse mapping from country names to ISO format codes
    data = st.session_state.users[["country_title", "feedback_count", "feedback_reputation", "given_item_count", "taken_item_count", "item_count", "user_id"]]
    reverse_mapping = {value: key for key, values in iso_mapping.items() for value in values}
    data['iso_code'] = data['country_title'].map(reverse_mapping)
    data["feedback_reputation"] = data["feedback_reputation"]*5
    data = data.groupby("iso_code").agg({
        "feedback_count": "mean",
        "feedback_reputation": "mean",
        "given_item_count": "mean",
        "taken_item_count": "mean",
        "item_count": "mean",
        "user_id": "nunique"
    })
    with col2:
        st.dataframe(
            data,
            column_config={
                "iso_code": "ISO-3"
                ,
                "feedback_count": st.column_config.NumberColumn(
                    "Reviews",
                    help="Average number of reviews",
                    format="%.0f"),
                "feedback_reputation": st.column_config.NumberColumn(
                    "Stars",
                    help="Average number of user stars",
                    format="%.2f ⭐"),
                "given_item_count": st.column_config.ProgressColumn(
                    "Items sold",
                    help="Average number of items sold to date",
                    format="%.0f",
                    max_value=max(data.given_item_count)),
                "taken_item_count": st.column_config.ProgressColumn(
                    "Items bought",
                    format="%.0f",
                    min_value=0,
                    max_value=max(data.taken_item_count),
                    ),
                "item_count": st.column_config.ProgressColumn(
                    "Items listed",
                    format="%.0f",
                    min_value=0,
                    max_value=max(data.item_count),
                    help="Average number of items listed at present date"
                    ),
                "user_id": "Users"
            },
            hide_index=True,
            use_container_width= True
        )


    st.write("<h6 style='font-family: Bungee; color: orange'>Distribution</h6>", 
             unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    fig1 = px.histogram(st.session_state.users, 
                        x="given_item_count", 
                        marginal="box", 
                        barmode="overlay",
                        nbins=75
                    )
    
    fig1.update_layout(
        xaxis_title="# Sold items",
        yaxis_title="Count",
        title= "Sold items"
    )
    percentiles = np.percentile(st.session_state.users["given_item_count"], [25, 50, 75, 95])
    for percentile in percentiles:
        fig1.add_shape(
            type="line",
            x0=percentile,
            x1=percentile,
            y0=0,
            y1=fig1.data[0].x.max(),  # Set y1 to the maximum count on the y-axis
            line=dict(
                color="red",
                width=2,
                dash="dashdot"
            )
        )

    # Create the second histogram
    fig2 = px.histogram(st.session_state.users, 
                        x="taken_item_count", 
                        marginal="box", 
                        barmode="overlay",
                        nbins=75,
                    )
    fig2.update_layout(
        xaxis_title="# Bought items",
        yaxis_title="Count",
        title= "Bought items"
    )
    percentiles = np.percentile(st.session_state.users["taken_item_count"], [25, 50, 75, 95])
    for percentile in percentiles:
        fig2.add_shape(
            type="line",
            x0=percentile,
            x1=percentile,
            y0=0,
            y1=fig2.data[0].x.max(),  # Set y1 to the maximum count on the y-axis
            line=dict(
                color="red",
                width=1,
                dash="dash"
            )
        )

    # Share y-axis between the plots
    fig1.update_yaxes(matches='y')
    fig2.update_yaxes(matches='y')
    fig1.update_yaxes(matches='x')
    fig2.update_yaxes(matches='x')

    # Display the plots in Streamlit
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1)

    with col2:
        st.plotly_chart(fig2)

    st.write("<h6 style='font-family: Bungee; color: orange'>Profiles</h6>", 
             unsafe_allow_html=True)

    dataframe_data = st.session_state.users[["user_id", "city", "country_title", "profile_url", "feedback_count", "feedback_reputation", "given_item_count", "taken_item_count", "item_count"]]
    dataframe_data["feedback_reputation"] *= 5 
    st.dataframe(
        dataframe_data,
        column_config={
            "user_id": st.column_config.NumberColumn(
                "User ID",
                help="User ID",
                format="%i",
            ),
            "city": "City",
            "country_title": "Country",
            "profile_url": st.column_config.LinkColumn("URL"),
            "feedback_count": "Reviews",
            "feedback_reputation": st.column_config.NumberColumn(
                "Stars",
                help="Number of user stars",
                format="%.2f ⭐"),
            "given_item_count": "Items sold",
            "taken_item_count": "Items bought",
            "item_count": "Items listed"
        },
        hide_index=True,
        use_container_width= True
    )

main()