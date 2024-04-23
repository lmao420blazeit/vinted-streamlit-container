import pandas as pd
from sqlalchemy import create_engine
import os
import json
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.manifold import TSNE
import plotly.figure_factory as ff
import numpy as np
import plotly.express as px

def load_credentials(path = "aws_rds_credentials.json"):
     with open(path, 'r') as file:
          config = json.load(file)

     # set up credentials
     for key in config.keys():
          os.environ[key] = config[key]

     return

time_interval = 90 #days

load_credentials()

aws_rds_url = f"postgresql://{os.environ['user']}:{os.environ['password']}@{os.environ['host']}:{os.environ['port']}/{os.environ['database']}?sslmode=require"

engine = create_engine(aws_rds_url)
sql_query = f"""
            WITH catalogs_table AS (
                SELECT catalog_id
                FROM public.tracking_staging
                WHERE date >= CURRENT_DATE - INTERVAL '{time_interval} days'
                GROUP BY catalog_id
                HAVING COUNT(*) > 250
                ORDER BY catalog_id
            )
            SELECT price_numeric, catalog_id, size_title
            FROM public.tracking_staging
            WHERE catalog_id IN (SELECT catalog_id FROM catalogs_table)
            ORDER BY date DESC
            LIMIT 40000;
               """


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

    data = pd.read_sql(sql_query, engine)

    st.write("<h2 style='font-family: Bungee; color: orange'>Catalog Clustering</h2>", 
             unsafe_allow_html=True)
    
    with st.expander("Agglomerative Clustering + t-SNE"):
        st.write("""
            #### Agglomerative Clustering using Size Title Sizes and Ward Distance
                
            The problem we aim to solve here is to cluster product catalogs into higher levels of catalogs. By nature, Vinted already has an hierarchical sequence of natural catalogs. For instance, the primary labels are: Woman, Men, Children, House, Animals, Others

            - Woman -> Bras, dresses, leggings, pantyhoses, as well as common clothes (trousers, shirts, shoes, etc)
            - Men -> Shoes, shorts, t shirts, shirts, etc
            - Children -> same as above
            - House -> blankets, towels, pillows, decoration, etc
            - Animals -> Toys, grooming, training, etc
            - Entertainment -> ...
                 
            ##### Why Ward Distance
                 
            Wards Linkage is based on the idea of minimizing Total Sums of Squares (intra cluster variance) when joining two clusters. 
            By mathematical expression its much more likely to result in aggregate and homogeneous groups. 
            In addition, Wards Linkage is inherently hierarchical since it provides means to group cluster centroids into higher dimensional clusters.
                
        """)

    pivot_size = data.pivot_table(values='price_numeric', columns='size_title', index='catalog_id', aggfunc='count')

    pivot_combined = pivot_size.fillna(0)
    pivot_combined = pivot_combined.T
    for col in pivot_combined.columns:
        pivot_combined[col] = MinMaxScaler().fit_transform(X = pivot_combined[[col]]) #/pivot_combined[col].sum()

    linkage_matrix = linkage(pivot_combined.T, 
                             'ward')

    # flattening labels
    t = 2.7
    res = fcluster(linkage_matrix, 
                   criterion = "distance", 
                   t = t)
    df = pd.DataFrame(res, index = pivot_combined.columns).reset_index()

    tab1, tab2 = st.tabs(["t-SNE", "Dendrogram"])
    with tab1:
        # Compute t-SNE embeddings
        tsne = TSNE(n_components=2, random_state=42, perplexity=10)
        X_tsne = tsne.fit_transform(pivot_combined.T)

        pca_df = pd.DataFrame(data=X_tsne, 
                            columns=['PC1', 'PC2'], 
                            index = pivot_combined.T.index)

        concat_df = pd.concat([pca_df, 
                            df.set_index("catalog_id")], 
                            ignore_index= False, 
                            axis=1)
        
        concat_df = concat_df.rename(columns={0: 'Cluster'})
        concat_df["Cluster"] = concat_df["Cluster"].astype("category")

        st.write(f"Num clusters: {len(concat_df['Cluster'].unique())}")
        st.write(f"Num catalogs: {len(concat_df['Cluster'])}")
        fig = px.scatter(concat_df, x='PC1', y='PC2', color='Cluster', title='t-SNE Plot', hover_data={'Catalog': concat_df.index})
        fig = fig.update_layout(height = 600,
                                yaxis={'visible': False, 'showticklabels': False},
                                xaxis={'visible': False, 'showticklabels': False})
        st.plotly_chart(fig, use_container_width= True)

    with tab2:
        X = dendrogram(linkage_matrix)
        fig = ff.create_dendrogram(np.array(X["icoord"]),
                                   orientation= 'left')
        fig.update_layout(height=8000)
        st.plotly_chart(fig, use_container_width= True)
    
main()