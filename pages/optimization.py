import pandas as pd
from sqlalchemy import create_engine
import os
import json
#import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
#import seaborn as sns
import streamlit as st
from sklearn.impute import SimpleImputer
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns: 
        col_type = df[col].dtypes
        if col_type in numerics: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: st.write('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def load_credentials(path = "aws_rds_credentials.json"):
     with open(path, 'r') as file:
          config = json.load(file)

     # set up credentials
     for key in config.keys():
          os.environ[key] = config[key]

     return

def load_data(time_interval, n_samples, min_asset_price):
    engine = create_engine(aws_rds_url)
    sql_query = f"""
                WITH catalogs AS (
                    SELECT catalog_id
                    FROM public.tracking_staging
                    WHERE date >= CURRENT_DATE - INTERVAL '{time_interval} days'
                    GROUP BY catalog_id
                    HAVING COUNT(DISTINCT date) > {time_interval * 0.4}          
                )
                SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_numeric) as price, catalog_id, date -- need to replace for median
                FROM public.tracking_staging 
                WHERE date >= CURRENT_DATE - INTERVAL '{time_interval} days'
                        AND catalog_id IN (SELECT catalog_id FROM catalogs)
                GROUP BY date, catalog_id
                HAVING AVG(price_numeric) > {min_asset_price} AND COUNT(product_id) > {n_samples} ;
                """
    data = pd.read_sql(sql_query, engine)
    return data

def preprocess_data(data):
    data = data.pivot_table(index = "date", 
                            columns="catalog_id", 
                            values = "price")

    imputer = SimpleImputer(strategy='median')

    for col in data.columns:
        data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1))

    return (data)

# generator function, creates an iterator over the number of portfolios we want to generate
# its a better practice, specially if num_port -> inf
def generate_random_portfolios(data, num_port, var_quantile):
    num_assets = len(data.columns)
    var_matrix = data.cov()

    for _ in range(num_port):
        # each asset has either value 0 or 1
        weights = np.random.randint(0, 2, size=num_assets)
        # the returns of the portfolio is the matrix multiplication between (weights,)*(,expected_returns)
        returns = np.dot(weights, data.median())
        # portfolio variance is the double sum of covariance between assets as in the formula
        var = var_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        std = np.sqrt(var)
        var_proxy = np.dot(weights, (data.median()-data.quantile(var_quantile)))

        yield weights, returns, std, var_proxy

def compute_portfolio_stats(data, iterations, var_percentile):
    port_weights = []
    port_returns = []
    port_volatility = []
    port_var = []
    with st.spinner('Generating random portfolios...'):
        for weights, returns, volatility, var_proxy in generate_random_portfolios(data, iterations, var_percentile):
            port_weights.append(weights)
            port_returns.append(returns)
            port_volatility.append(volatility)
            port_var.append(var_proxy)

        new_data = {"Revenue": port_returns, 
                    "Volatility": port_volatility,
                    "VaR": port_var}

        for counter, symbol in enumerate(data.columns.tolist()):
            new_data[str(symbol)+'_weight'] = [w[counter] for w in port_weights]

    return(pd.DataFrame(new_data))

def plot_portfolio(portfolio, **kwargs):
    st.write("<h5 style='font-family: Bungee; color: orange'>Portfolios</h5>", 
             unsafe_allow_html=True)
    
    # Create heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=portfolio.drop(columns=["Revenue", "Volatility", "VaR"], axis = 1).head(50).values,
        x=portfolio.drop(columns=["Revenue", "Volatility", "VaR"], axis = 1).head(50).columns,
        y=portfolio.drop(columns=["Revenue", "Volatility", "VaR"], axis = 1).head(50).index,
        colorscale='YlGnBu',
        colorbar=dict(title='Number of products')
    ))

    fig.update_layout(
        title='Product distribution across portfolios',
        xaxis_title='Number of sample portfolios',
        yaxis_title='Catalog_id'
    )

    st.plotly_chart(fig, **kwargs)
    return

def plot_portfolio_scatter(portfolio, **kwargs):
    fig = px.scatter(
        data_frame=portfolio,
        x='Volatility',
        y='Revenue',
        color='Sharpe',
        title='Portfolios Mean-Variance',
        labels={'Volatility': 'Standard Dev. (€)', 'Revenue': 'Expected Returns (€)', 'Sharpe': 'Sharpe Ratio'},
        marginal_x='histogram',
        marginal_y='histogram', 
    )

    fig.update_layout(
        width=1200,  
        height=800,  
    )
    st.plotly_chart(fig, **kwargs)
    return
    

def plot_portfolio_hist(portfolio, x = "Sharpe", **kwargs):
    fig = px.histogram(
        portfolio,
        x=x,
        title=x,  # Set title of the plot
        labels={x: x, 'count': 'Frequency'},  # Set labels for axes
        opacity=0.7,  # Optional: set opacity of bars
        color_discrete_sequence=['skyblue']  # Optional: set color of bars
    )

    fig.update_layout(
        xaxis_title=x,  # Set label for x-axis
        yaxis_title='Frequency'  # Set label for y-axis
    )
    st.plotly_chart(fig, **kwargs)

def plot_portfolio_3d(portfolio, z = "Revenue", **kwargs):
    scatter3d_trace = go.Scatter3d(
        x=portfolio[z],
        y=portfolio["Volatility"],
        z=portfolio["Sharpe"],
        mode='markers',
        marker=dict(
            size=3,                    
            color=portfolio[z],                   
            colorscale='Viridis',      
            opacity=0.8,
            line=dict(width=0.5, color='black')
        ),
        text=[f'Return: {r}<br>Volatility: {v}<br>Sharpe: {s}' for r, v, s in zip(portfolio["Revenue"], portfolio["Volatility"], portfolio[z])]
    )

    layout = go.Layout(
        title='Sharpe curve',
        scene=dict(
            xaxis=dict(title='Volatility'),
            yaxis=dict(title='Revenue'),
            zaxis=dict(title=z)
        )
    )

    fig = go.Figure(data=[scatter3d_trace], layout=layout)

    fig.update_layout(autosize=True,
                        hovermode='closest',
                        scene = {"aspectratio": {"x": 1, "y": 2.2, "z": 1},
                                'camera': {'eye':{'x': 2, 'y':0.4, 'z': 0.8}},
                                'xaxis_title':'Revenue (€)',
                              'yaxis_title':'Standard Dev. (€)',
                              'zaxis_title':'Sharpe'
                                },
                        margin=dict(t=40),
                        annotations=[
                            dict(
                                text="Data Source: Vinted",
                                x=0,
                                y=-0.15,
                                xref="paper",
                                yref="paper",
                                showarrow=False
                            )
                        ]

                    )

    st.plotly_chart(fig, **kwargs)
    return

def plot_var_3d(portfolio, **kwargs):
    scatter3d_trace = go.Scatter3d(
        x=portfolio["Revenue"],
        y=portfolio["Sharpe"],
        z=portfolio["VaR"],
        mode='markers',
        marker=dict(
            size=3,                    
            color=portfolio["VaR"],                   
            colorscale='Viridis',      
            opacity=0.8,
            line=dict(width=0.5, color='black')
        ),
        text=[f'Revenue: {r}<br>Sharpe: {v}<br>VaR: {s}' for r, v, s in zip(portfolio["Revenue"], portfolio["Sharpe"], portfolio["VaR"])]
    )

    layout = go.Layout(
        title='Sharpe curve',
        scene=dict(
            xaxis=dict(title='Revenue'),
            yaxis=dict(title='Sharpe'),
            zaxis=dict(title='VaR')
        )
    )

    fig = go.Figure(data=[scatter3d_trace], layout=layout)

    fig.update_layout(autosize=True,
                        hovermode='closest',
                        scene = {"aspectratio": {"x": 1, "y": 2.2, "z": 1},
                                'camera': {'eye':{'x': 2, 'y':0.4, 'z': 0.8}},
                                'xaxis_title':'Revenue (€)',
                              'yaxis_title':'Sharpe',
                              'zaxis_title':'VaR'
                                },
                        margin=dict(t=40),
                        annotations=[
                            dict(
                                text="Data Source: Vinted",
                                x=0,
                                y=-0.15,
                                xref="paper",
                                yref="paper",
                                showarrow=False
                            )
                        ]

                    )

    st.plotly_chart(fig, **kwargs)
    return

def surface_3d(df, z = "Revenue", **kwargs):
    '''
    3d surface plot - History of Yield Curve on a monthly basis from 1m to 30Y rates
    '''


    x = np.array(df[z])
    y = np.array(df["Volatility"])
    z = np.array(df["Sharpe"])


    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)

    X,Y = np.meshgrid(xi,yi)

    Z = griddata((x,y),
                 z,
                 (X,Y), 
                 method='cubic')

    fig = go.Figure(data=[go.Surface(x=xi,
                                    y=yi,
                                    z=Z,
                                    opacity=0.95,
                                    connectgaps=True,
                                    colorscale='rdbu',
                                    showscale=True,
                                    reversescale=True,
                                    )
                        ]
                )

    fig.update_layout(autosize=True,
                        hovermode='closest',
                        scene = {"aspectratio": {"x": 1, "y": 2.2, "z": 1},
                                'camera': {'eye':{'x': 2, 'y':0.4, 'z': 0.8}},
                                'xaxis_title':'Revenue (€)',
                              'yaxis_title':'Standard Dev. (€)',
                              'zaxis_title':'Sharpe'
                                },
                        margin=dict(t=40),
                        annotations=[
                            dict(
                                text="Data Source: Vinted",
                                x=0,
                                y=-0.15,
                                xref="paper",
                                yref="paper",
                                showarrow=False
                            )
                        ]

                    )

    st.plotly_chart(fig, **kwargs)
    return

def surface_var_3d(df, **kwargs):
    '''
    3d surface plot - History of Yield Curve on a monthly basis from 1m to 30Y rates
    '''


    x = np.array(df["Revenue"])
    y = np.array(df["Sharpe"])
    z = np.array(df["VaR"])


    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)

    X,Y = np.meshgrid(xi,yi)

    Z = griddata((x,y),
                 z,
                 (X,Y), 
                 method='cubic')

    fig = go.Figure(data=[go.Surface(x=xi,
                                    y=yi,
                                    z=Z,
                                    opacity=0.95,
                                    connectgaps=True,
                                    colorscale='rdbu',
                                    showscale=True,
                                    reversescale=True,
                                    )
                        ]
                )

    fig.update_layout(autosize=True,
                        hovermode='closest',
                        scene = {"aspectratio": {"x": 1, "y": 2.2, "z": 1},
                                'camera': {'eye':{'x': 2, 'y':0.4, 'z': 0.8}},
                                'xaxis_title':'Revenue (€)',
                              'yaxis_title':'Sharpe',
                              'zaxis_title':'VaR'
                                },
                        margin=dict(t=40),
                        annotations=[
                            dict(
                                text="Data Source: Vinted",
                                x=0,
                                y=-0.15,
                                xref="paper",
                                yref="paper",
                                showarrow=False
                            )
                        ]

                    )

    st.plotly_chart(fig, **kwargs)
    return

def main_controlflow(days, num_port, samples, var_percentile, min_asset_price):
    # load melted dataframe
    data = load_data(days, samples, min_asset_price)
    unique = data["catalog_id"].nunique()
    st.write(f"Unique product catalogs: {unique}") 
    #data2 = data.pivot_table(index = "date", 
    #                        columns="catalog_id", 
    #                        values = "price")
    #st.write(data2.describe())
    # explode into tabular and fill missing values
    data = preprocess_data(data)

    # reduce memory usage
    data = reduce_mem_usage(data, False)

    portfolio = compute_portfolio_stats(data, num_port, var_percentile)

    tab1, tab2, tab3 = st.tabs(["Portfolio", "Data (line chart)", "Date (dataframe)"])
    with tab1:
        plot_portfolio(portfolio, use_container_width = True)

    with tab2:
        fig = px.line(data, 
                    x=data.index, 
                    y=data.columns)

        st.plotly_chart(fig, use_container_width= True)

    with tab3:
        st.write(data.describe())

    portfolio["Sharpe"] = portfolio["Revenue"]/portfolio["Volatility"] 
    #portfolio["VaR"] = portfolio["Revenue"]/portfolio["Volatility"] 

    # surface_3d(portfolio)

    tab1, tab2, tab3 = st.tabs(["Sharpe", "VaR", "Mean-variance"])
    with tab1:
        cols = st.columns([0.5, 0.5])
        with cols[0]:
            plot_portfolio_3d(portfolio, 
                              use_container_width = True)
        with cols[1]:
            surface_3d(portfolio, 
                       use_container_width = True)
    with tab2:
        cols = st.columns([0.5, 0.5])
        with cols[0]:
            plot_var_3d(portfolio, 
                              use_container_width = True)
        with cols[1]:
            surface_var_3d(portfolio, 
                       use_container_width = True)

    with tab3:
        plot_portfolio_scatter(portfolio, use_container_width = True)

    st.write("<h5 style='font-family: Bungee; color: orange'>Analysis</h5>", 
            unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Sharpe", "VaR", "Portfolios"])
    with tab1:
        with st.expander("Linear Regression Modelling"):
            st.write("""
                In order to find how labels (catalogs) correlate to Sharpe ratio, we implemented Linear Regression Modelling. 
                The purpose is to extract the relative importance of each label through sign and magnitude.
                    
                **These coefficients do not imply causation**.

            """)
        reg = LinearRegression().fit(portfolio.drop(["Sharpe", "Revenue", "Volatility", "VaR"], axis = 1), 
                                     portfolio["Sharpe"])
        coefficients = reg.coef_
        top_5_indices = coefficients.argsort()[-5:][::-1]

        top_5_labels = portfolio.columns[top_5_indices]
        df = pd.DataFrame({"Coefficient": np.abs(coefficients[top_5_indices]), 
                        "Label": top_5_labels})
        fig = px.bar(df, 
                    y="Label", 
                    x="Coefficient", 
                    title="Top 5 Catalogs by Sharpe",
                    labels={"Coefficient": "Coefficient Value", 
                            "Label": "Catalog"},
                    orientation='h')
        
        fig.update_traces(marker_color = 'green', marker_line_color = 'white',
                        marker_line_width = 1, opacity = 1)

        cols = st.columns([0.2, 0.2, 0.4])

        with cols[0]:
            st.plotly_chart(fig, use_container_width= True)

        top_5_indices = coefficients.argsort()[:5][::-1]

        top_5_labels = portfolio.columns[top_5_indices]
        df = pd.DataFrame({"Coefficient": coefficients[top_5_indices], 
                        "Label": top_5_labels})

        fig = px.bar(df, 
                    y="Label", 
                    x="Coefficient", 
                    title="Bottom 5 Catalogs by Sharpe",
                    labels={"Coefficient": "Coefficient Value", 
                            "Label": "Catalog"},
                    orientation='h')
        
        fig.update_traces(marker_color = 'red', 
                          marker_line_color = 'white',
                        marker_line_width = 1, 
                        opacity = 1)

        with cols[1]:
            st.plotly_chart(fig, use_container_width= True)

        with cols[2]:
            plot_portfolio_hist(portfolio, use_container_width = True, x = "Sharpe")
    
    with tab2:
        with st.expander("Linear Regression Modelling"):
            st.write("""
                In order to find how labels (catalogs) correlate to Sharpe ratio, we implemented Linear Regression Modelling. 
                The purpose is to extract the relative importance of each label through sign and magnitude.
                    
                **These coefficients do not imply causation**.

            """)
        reg = LinearRegression().fit(portfolio.drop(["Sharpe", "Revenue", "Volatility", "VaR"], axis = 1), portfolio["VaR"])
        coefficients = reg.coef_
        top_5_indices = coefficients.argsort()[-5:][::-1]

        top_5_labels = portfolio.columns[top_5_indices]
        df = pd.DataFrame({"Coefficient": np.abs(coefficients[top_5_indices]), 
                        "Label": top_5_labels})
        fig = px.bar(df, 
                    y="Label", 
                    x="Coefficient", 
                    title="Worst Catalogs by VaR",
                    labels={"Coefficient": "Coefficient Value", 
                            "Label": "Catalog"},
                    orientation='h')
        
        fig.update_traces(marker_color = 'red', marker_line_color = 'white',
                        marker_line_width = 1, opacity = 1)

        cols = st.columns([0.2, 0.2, 0.4])

        with cols[0]:
            st.plotly_chart(fig, use_container_width= True)

        top_5_indices = coefficients.argsort()[:5][::-1]

        top_5_labels = portfolio.columns[top_5_indices]
        df = pd.DataFrame({"Coefficient": coefficients[top_5_indices], 
                        "Label": top_5_labels})

        fig = px.bar(df, 
                    y="Label", 
                    x="Coefficient", 
                    title="Best Catalogs by VaR",
                    labels={"Coefficient": "Coefficient Value", 
                            "Label": "Catalog"},
                    orientation='h')
        
        fig.update_traces(marker_color = 'green', 
                          marker_line_color = 'white',
                        marker_line_width = 1, 
                        opacity = 1)

        with cols[1]:
            st.plotly_chart(fig, 
                            use_container_width= True)

        with cols[2]:
            plot_portfolio_hist(portfolio, 
                                use_container_width = True, 
                                x = "VaR")

    with tab3:
        top_5_port = portfolio.sort_values("Sharpe", ascending=False).reset_index(drop= True).head(5)
        top_5_port = pd.concat([top_5_port["Sharpe"], 
                                top_5_port.drop("Sharpe", axis=1)], 
                                axis=1)
        top_5_port.index = [f"Top {n}" for n in range(1, len(top_5_port.index) + 1)]
        st.write(top_5_port)

    cols = st.columns([0.5, 0.5, 0.5, 0.5])
    # computing baseline portfolio
    var_matrix = data.cov()
    num_assets = len(data.columns) # sharpe, returns, std
    weights = np.random.randint(1, 2, size=num_assets)
    # the returns of the portfolio is the matrix multiplication between (weights,)*(,expected_returns)
    returns = np.dot(weights, data.median())
    # portfolio variance is the double sum of covariance between assets as in the formula
    var = var_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
    std = np.sqrt(var)
    sharpe = returns/std
    rets_per_asset = returns/num_assets
    with cols[0]:
        st.write(f"""<div style="font-family: Nunito, sans-serif; color: #333333;">
                    <h6>Baseline portfolio performance:</h6>
                    <p>Revenue:{returns:.2f}€</p>
                    <p><b>Revenue per asset:{rets_per_asset:.2f}€</b></p>
                    <p>Standard Dev.:{std:.2f}€</p>
                    <p>Sharpe:{sharpe:.2f}</p>
                    <p>Nº items:{num_assets:.0f}</p>
                    </div>""", 
                    unsafe_allow_html=True)

    with cols[1]:
        __revenue = top_5_port["Revenue"].iloc[0]
        __sharpe = top_5_port["Sharpe"].iloc[0]
        __volatility = top_5_port["Volatility"].iloc[0]
        __num_assets = top_5_port.drop(["Revenue", "Volatility", "Sharpe", "VaR"], axis = 1).iloc[0].sum()
        __rev_per_asset = __revenue/__num_assets
        with st.container:
            st.write(f"""<div style="font-family: Nunito, sans-serif; color: #333333;">
                        <h6>Mean-Variance Optimized Portfolio:</h6>
                        <p><b>Revenue:{__revenue:.2f}€</b></p>
                        <p><b>Revenue per asset:{__rev_per_asset:.2f}€</b></p>
                        <p><b>Standard Dev.:{__volatility:.2f}€</b></p>
                        <p><b>Sharpe:{__sharpe:.2f}</b></p>
                        <p>Nº items:{__num_assets:.0f}</p>
                        </div>""", 
                        unsafe_allow_html=True)
    
    with cols[2]:
        top_5_port = portfolio.sort_values("Revenue", ascending=False).reset_index(drop= True).head(1)
        __revenue = top_5_port["Revenue"].iloc[0]
        __sharpe = top_5_port["Sharpe"].iloc[0]
        __volatility = top_5_port["Volatility"].iloc[0]
        __num_assets = top_5_port.drop(["Revenue", "Volatility", "Sharpe", "VaR"], axis = 1).iloc[0].sum()
        __rev_per_asset = __revenue/__num_assets
        with st.container:
            st.write(f"""<div style="font-family: Nunito, sans-serif; color: #333333;">
                        <h6>Revenue Optimized Portfolio:</h6>
                        <p><b>Revenue:{__revenue:.2f}€</b></p>
                        <p><b>Revenue per asset:{__rev_per_asset:.2f}€</b></p>
                        <p><b>Standard Dev.:{__volatility:.2f}€</b></p>
                        <p><b>Sharpe:{__sharpe:.2f}</b></p>
                        <p>Nº items:{__num_assets:.0f}</p>
                        </div>""", 
                        unsafe_allow_html=True)

    with cols[3]:
        top_5_port = portfolio.sort_values("Volatility", ascending=True).reset_index(drop= True).head(1)
        __revenue = top_5_port["Revenue"].iloc[0]
        __sharpe = top_5_port["Sharpe"].iloc[0]
        __volatility = top_5_port["Volatility"].iloc[0]
        __num_assets = top_5_port.drop(["Revenue", "Volatility", "Sharpe", "VaR"], axis = 1).iloc[0].sum()
        __rev_per_asset = __revenue/__num_assets
        with st.container:
            st.write(f"""<div style="font-family: Nunito, sans-serif; color: #333333;">
                        <h6>Volatility Optimized Portfolio:</h6>
                        <p><b>Revenue:{__revenue:.2f}€</b></p>
                        <p><b>Revenue per asset:{__rev_per_asset:.2f}€</b></p>
                        <p><b>Standard Dev.:{__volatility:.2f}€</b></p>
                        <p><b>Sharpe:{__sharpe:.2f}</b></p>
                        <p>Nº items:{__num_assets:.0f}</p>
                        </div>""", 
                        unsafe_allow_html=True)

    #st.write("Use other risk measures other than volatility (IQR, Range, etc)")
    #st.pyplot(sns.clustermap(_data.corr()))

aws_rds_url = f"postgresql://{os.environ['user']}:{os.environ['password']}@{os.environ['host']}:{os.environ['port']}/{os.environ['database']}?sslmode=require"

load_credentials()

def main():
    st.set_page_config(layout="wide", page_title="Optimization")
    #st.set_option('deprecation.showPyplotGlobalUse', False)

    # loading fonts
    st.markdown("""
        <link href="https://fonts.googleapis.com/css?family=Nunito:200,300,700" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css?family=Bungee" rel="stylesheet">
        """,
        unsafe_allow_html= True)

    st.write("<h2 style='font-family: Bungee; color: orange'>Mean-Variance Optimization</h2>", 
             unsafe_allow_html=True)
    
    with st.expander("Mean-Variance Portfolio Optimization (Markowitz Approach)"):
        st.write("""
            # A product basket optimization approach using Markowitz Portfolio Theory on Vinted dataset

            The efficient frontier represents the set of portfolios that offer the highest expected return for a given level of risk, or the lowest risk for a given level of expected return.

            ### Key Assumptions

            - **Defining expected return**: The expected return of a portfolio is the weighted average of the expected returns of its individual assets.

            $E(R_p) = \sum_{i=1}^{n} w_i \cdot E(R_i)$

            Where:
                - $E(R_p)$ is the expected return of the portfolio.
                - $w_i$ is the weight of asset \(i\) in the portfolio.
                - $E(R_i)$ is the expected return of asset $i$.

            - **Defining risk**: The proxy of risk in MPT is the variance of the portfolio.

            - **Diversification**: One of the key principles of MPT is diversification, which involves spreading investments across different asset classes with uncorrelated or negatively correlated returns.

            ### Mathematical Formulation

            The optimization problem in MPT can be formulated as a quadratic programming problem to find the optimal portfolio weights that maximize the expected return for a given level of risk or minimize the risk for a given level of expected return, subject to certain constraints such as budget constraints and minimum or maximum weight constraints.

            Maximize $\quad E(R_p) = \mathbf{w}^T \mathbf{R}$

            
            Subject to: 
            $\quad
            \begin{cases}
            \mathbf{w}^T \mathbf{1} = 1 & \text{(Budget constraint)} \\
            \mathbf{w}^T \mathbf{\Sigma} \mathbf{w} \leq \sigma^2 & \text{(Risk constraint)} \\
            w_i \geq 0 & \text{(Non-negativity constraint)}
            \end{cases}$
            

            Where:
            - $E(R_p)$ is the expected return of the portfolio.
            - $\mathbf{w}$ is the vector of portfolio weights.
            - $\mathbf{R}$ is the vector of expected returns of the assets.
            - $\mathbf{\Sigma}$ is the covariance matrix of asset returns.
            - $\sigma^2$ is the target risk level.
        """)

    cols = st.columns([0.1, 0.2, 0.2, 0.1, 0.2])
    with cols[0]:
        selected_interval = st.selectbox("Time interval:", 
                                         ["7 days", "30 days", "90 days"],
                                         help = "This is the time horizon for which data will be considered.")

    with cols[1]:
        num_port = st.selectbox("Number of portfolio iterations:", 
                                [1000, 2000, 3000],
                                help = "This will be the number of portfolios that are going to be simulated.")

    with cols[2]:
        samples = st.selectbox("Minimum sample size:", 
                               [30, 45, 60],
                               help = "Minimum number of products for each catalog and day to consider the catalog. This means a value of 30 will select only the catalogs that have 30 or more products worth of data for every day during the period.")

    with cols[3]:
        var_percentile = st.selectbox("VaR proxy:", 
                                      [0.25, 0.05],
                                      help = "The percentile of the revenues used as a proxy for Value at Risk.")

    with cols[4]:
        min_asset_price = st.selectbox("Minimum expected catalog price:", 
                                       [0, 5, 10, 15, 20],
                                       help= "Minimum expected price of a catalog for it to be considered. For value 10, only catalogs with an expected price > 10 will be considered.")

    # Convert selected interval to days
    if selected_interval == "7 days":
        days = 7
    elif selected_interval == "30 days":
        days = 30
    elif selected_interval == "90 days":
        days = 90

    main_controlflow(days, num_port, samples, var_percentile, min_asset_price)

main()