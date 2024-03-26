import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.graph_objects as go
import os
import json
import plotly.express as px

def load_credentials(path = "aws_rds_credentials.json"):
     with open(path, 'r') as file:
          config = json.load(file)

     # set up credentials
     for key in config.keys():
          os.environ[key] = config[key]

     return


load_credentials()

aws_rds_url = "postgresql://postgres:9121759591mM!@vinted.cl2cus64cwps.eu-north-1.rds.amazonaws.com:5432/postgres?sslmode=require"

# Load a sample dataset
def load_data():
    engine = create_engine(aws_rds_url)
    sql_query = f"SELECT * FROM public.users_staging LIMIT 20000"
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
    st.write(f"Latest updated on {latest_date}", 
             unsafe_allow_html=True)

    # wrapping up with containers
    with st.container():
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col6:
            st.metric(label="**Active users**", 
                    value="{}".format(st.session_state.users['user_id'].nunique()),
                    help="Number of unique users in the sample")            
        with col1:
            st.metric(label="**Item count**", 
                    value="{:,.2f}".format(st.session_state.users['item_count'].mean()),
                    help="Number of unique brands in the sample")
        with col2:
            st.metric(label="**Positive feedback**", 
                    value="{:,.2f} 	👍".format(st.session_state.users['positive_feedback_count'].mean()),
                    help="Average number of positive feedback per user")
        with col3:
            st.metric(label="**Negative feedback**", 
                    value="{:,.2f} 👎".format(st.session_state.users['negative_feedback_count'].mean()),
                    help="Average number of negative feedback per user")
        with col4:
            st.metric(label="**Total feedback**", 
                    value="{:,.2f}".format(st.session_state.users['feedback_count'].mean()),
                    help="Average of total feedback per user")
        with col5:
            st.metric(label="**Feedback reputation**", 
                    value="{:,.2f} ⭐".format(st.session_state.users['feedback_reputation'].median()*5),
                    help="Feedback reputation")

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

    # Update layout
    fig.update_layout(
        title_text='Total items listed per country',
        geo=dict(
            showcoastlines=True,  # Show coastlines on the map
            showlakes=True
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
                    "Reviews (mean)",
                    help="Number of reviews",
                    format="%.2f"),
                "feedback_reputation": st.column_config.NumberColumn(
                    "Stars (mean)",
                    help="Number of user stars",
                    format="%.2f ⭐"),
                "given_item_count": st.column_config.NumberColumn(
                    "Items sold (mean)",
                    help="Number of items sold to date",
                    format="%.2f"),
                "taken_item_count": st.column_config.NumberColumn(
                    "Items bought (mean)",
                    help="Number of items bought to date",
                    format="%.2f"),
                "item_count": st.column_config.NumberColumn(
                    "Items listed (mean)",
                    help="Number of items listed at present date",
                    format="%.2f"),
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
                        nbins=100
                    )

    # Create the second histogram
    fig2 = px.histogram(st.session_state.users, 
                        x="taken_item_count", 
                        marginal="box", 
                        barmode="overlay",
                        nbins=100,
                    )

    # Share y-axis between the plots
    fig1.update_yaxes(matches='y')
    fig2.update_yaxes(matches='y')

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