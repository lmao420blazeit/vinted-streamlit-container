import pandas as pd
from sqlalchemy import create_engine
import os
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
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

def load_data(time_interval, n_samples):
    engine = create_engine(aws_rds_url)
    sql_query = f"""
                WITH catalogs AS (
                    SELECT catalog_id
                    FROM public.tracking_staging
                    WHERE date >= CURRENT_DATE - INTERVAL '{time_interval} days'
                    GROUP BY catalog_id
                    HAVING COUNT(DISTINCT date) > {time_interval * 0.4}
                )
                SELECT PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY price_numeric) as price, catalog_id, date
                FROM public.tracking_staging 
                WHERE date >= CURRENT_DATE - INTERVAL '{time_interval} days'
                        AND catalog_id IN (SELECT catalog_id FROM catalogs)
                GROUP BY date, catalog_id
                HAVING COUNT(catalog_id) > {n_samples};
                """
    data = pd.read_sql(sql_query, engine)
    return data

def preprocess_data(data):
    data = data.pivot_table(index = "date", columns="catalog_id", values = "price")

    imputer = SimpleImputer(strategy='median')

    for col in data.columns:
        data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1))

    return (data)

# generator function, creates an iterator over the number of portfolios we want to generate
# its a better practice, specially if num_port -> inf
def generate_random_portfolios(data, num_port):
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

        yield weights, returns, std

def compute_portfolio_stats(data, iterations):
    port_weights = []
    port_returns = []
    port_volatility = []
    with st.spinner('Generating random portfolios...'):
        for weights, returns, volatility in generate_random_portfolios(data, iterations):
            port_weights.append(weights)
            port_returns.append(returns)
            port_volatility.append(volatility)

        new_data = {"Revenue": port_returns, 
                    "Volatility": port_volatility}

        for counter, symbol in enumerate(data.columns.tolist()):
            new_data[str(symbol)+'_weight'] = [w[counter] for w in port_weights]

    return(pd.DataFrame(new_data))

def plot_portfolio(portfolio, **kwargs):
    st.write("<h5 style='font-family: Bungee; color: orange'>Portfolios</h5>", 
             unsafe_allow_html=True)
    
    # Create heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=portfolio.drop(columns=["Revenue", "Volatility"], axis = 1).head(50).values,
        x=portfolio.drop(columns=["Revenue", "Volatility"], axis = 1).head(50).columns,
        y=portfolio.drop(columns=["Revenue", "Volatility"], axis = 1).head(50).index,
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

def plot_portfolio_hist(portfolio, **kwargs):
    fig = px.histogram(
        portfolio,
        x='Sharpe',
        title='Sharpe',  # Set title of the plot
        labels={'Sharpe': 'Sharpe', 'count': 'Frequency'},  # Set labels for axes
        opacity=0.7,  # Optional: set opacity of bars
        color_discrete_sequence=['skyblue']  # Optional: set color of bars
    )

    fig.update_layout(
        xaxis_title='Sharpe',  # Set label for x-axis
        yaxis_title='Frequency'  # Set label for y-axis
    )
    st.plotly_chart(fig, **kwargs)

def plot_portfolio_3d(portfolio, **kwargs):
    scatter3d_trace = go.Scatter3d(
        x=portfolio["Revenue"],
        y=portfolio["Volatility"],
        z=portfolio["Sharpe"],
        mode='markers',
        marker=dict(
            size=3,                    
            color=portfolio["Sharpe"],                   
            colorscale='Viridis',      
            opacity=0.8,
            line=dict(width=0.5, color='black')
        ),
        text=[f'Return: {r}<br>Volatility: {v}<br>Sharpe: {s}' for r, v, s in zip(portfolio["Revenue"], portfolio["Volatility"], portfolio["Sharpe"])]
    )

    layout = go.Layout(
        title='Sharpe curve',
        scene=dict(
            xaxis=dict(title='Volatility'),
            yaxis=dict(title='Revenue'),
            zaxis=dict(title='Sharpe')
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

def surface_3d(df, **kwargs):
    '''
    3d surface plot - History of Yield Curve on a monthly basis from 1m to 30Y rates
    '''


    x = np.array(df["Revenue"])
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

def main_controlflow(days, num_port, samples):
    # load melted dataframe
    data = load_data(days, samples)
    unique = data["catalog_id"].nunique()
    st.write(f"Unique product catalogs: {unique}") 
    # explode into tabular and fill missing values
    data = preprocess_data(data)

    # reduce memory usage
    data = reduce_mem_usage(data, False)

    portfolio = compute_portfolio_stats(data, num_port)

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

    # surface_3d(portfolio)

    tab1, tab2 = st.tabs(["3D Scatterplot", "Mean-variance"])
    with tab1:
        cols = st.columns([0.5, 0.5])
        with cols[0]:
            plot_portfolio_3d(portfolio, use_container_width = True)
        with cols[1]:
            surface_3d(portfolio, use_container_width = True)

    with tab2:
        plot_portfolio_scatter(portfolio, use_container_width = True)

    st.write("<h5 style='font-family: Bungee;; color: orange'>Analysis</h5>", 
            unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Portfolio", "Top Portfolios"])
    with tab1:
        with st.expander("Linear Regression Modelling"):
            st.write("""
                In order to find how labels (catalogs) correlate to Sharpe ratio, we implemented Linear Regression Modelling. 
                The purpose is to extract the relative importance of each label through sign and magnitude.
                    
                **These coefficients do not imply causation**.

            """)
        reg = LinearRegression().fit(portfolio.drop(["Sharpe", "Revenue", "Volatility"], axis = 1), portfolio["Sharpe"])
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
                    orientation='h',
                    color_continuous_scale=["green"])

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
                    labels={"Coefficient": "Coefficient Value", "Label": "Catalog"},
                    orientation='h',
                    color_continuous_scale=["red"])

        with cols[1]:
            st.plotly_chart(fig, use_container_width= True)

        with cols[2]:
            plot_portfolio_hist(portfolio, use_container_width = True)

    with tab2:
        top_5_port = portfolio.sort_values("Sharpe", ascending=False).reset_index(drop= True).head(5)
        top_5_port = pd.concat([top_5_port["Sharpe"], top_5_port.drop("Sharpe", axis=1)], axis=1)
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
        __num_assets = top_5_port.drop(["Revenue", "Volatility", "Sharpe"], axis = 1).iloc[0].sum()
        __rev_per_asset = __revenue/__num_assets
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
        __num_assets = top_5_port.drop(["Revenue", "Volatility", "Sharpe"], axis = 1).iloc[0].sum()
        __rev_per_asset = __revenue/__num_assets
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
        __num_assets = top_5_port.drop(["Revenue", "Volatility", "Sharpe"], axis = 1).iloc[0].sum()
        __rev_per_asset = __revenue/__num_assets
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

    cols = st.columns([0.2, 0.4, 0.4])
    with cols[0]:
        selected_interval = st.selectbox("Select a time interval:", ["7 days", "30 days", "90 days"])

    with cols[1]:
        num_port = st.selectbox("Select number of portfolio iterations:", [1000, 2000, 3000])

    with cols[2]:
        samples = st.selectbox("Minimum products per sample (day and catalog):", [30, 45, 60])

    # Convert selected interval to days
    if selected_interval == "7 days":
        days = 7
    elif selected_interval == "30 days":
        days = 30
    elif selected_interval == "90 days":
        days = 90

    main_controlflow(days, num_port, samples)

main()