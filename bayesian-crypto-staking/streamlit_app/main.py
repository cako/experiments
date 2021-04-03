import streamlit as st

import cryptocompare as cc
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import scipy as sp

import utils

# ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark,
# presentation, xgridoff, ygridoff, gridon, none
PLOTLY_TEMPLATE = "ggplot2"
PLOTLY_COLORS = plotly.colors.qualitative.D3
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "lasso2d",
        "select2d",
        "hoverClosestCartesian",
    ],
}

st.set_page_config(
    page_title="Locked Staking Analysis",
    page_icon=utils.ICON,
    layout="wide",
    initial_sidebar_state="auto",
)


# Data-loading functions
@st.cache
def get_exchanges():
    exchanges = cc.get_exchanges()
    return ["All"] + list(exchanges.keys())


@st.cache
def get_pairs(exchange=None):
    pairs = cc.get_pairs(exchange=exchange)

    coins = list(set(e["fsym"] for v in pairs.values() for e in v if e["fsym"]))
    currs = list(set(e["tsym"] for v in pairs.values() for e in v if e["tsym"]))

    return sorted(coins) + [""], sorted(currs) + [""]


@st.cache(allow_output_mutation=True)
def get_predictions_simple(logreturns, n_sims, n_days):
    model = utils.get_student_t_model(logreturns)
    spp = utils.sample_model(model, n_sims)
    return spp["logreturns"][:, -n_days:]


@st.cache(allow_output_mutation=True)
def get_predictions_stochastic(logreturns, n_sims, n_days):
    model = utils.get_stochastic_model(logreturns)
    spp = utils.sample_model(model, n_sims)
    return spp["logreturns"][:, -n_days:]


# Display functions
def write_intro(price, logreturns):
    _, col, _ = st.beta_columns([0.2, 1, 0.2])
    col.title("Cryptocurrency Price Ranges")
    col.subheader("by [Carlos Costa](https://github/cako)")
    col.markdown("")
    col.markdown("")
    col.markdown(
        """
    Cryptocurrencies have gone from obscure digital curiosities to multibillion dollar markets. 
    The vast amount of trading that takes place using these cryptos have allowed them to behave similarly to traditional stocks.
    This app explores a couple of models for pricing cryptoassets. One uses a naive "best-fit" predictor, another uses a simple Bayesian model, and the last one uses a time-varying stochastic model.
    Jump into the models by choosing an option on the Dashboard, or read more about the models below.
    """
    )
    col.markdown(
        "You can find the source code for this app from the top-right menu or from my [GitHub repo](https://github.com/cako/experiments)."
    )
    with col.beta_expander("Naive Predictor"):
        st.markdown(
            r"""
Average returns and volatility in finance are usually measured in terms of log variations in price.
We can calculate the log of the daily returns as defined by

$$\qquad r_t = \log(S_{t}/S_{t-1}) = \log(S_{t}) - \log(S_{t-1})$$

where $S_t$ and $r_t$ are the price and the log-return at time $t$, respectively.

We can hypothesize that the distribution of these returns over a period of time follows a certain distribution.
Without any prior information, one could be tempted to choose a normal distribution, but empirical studies have shown that the [Student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution), which is better at describing the returns.
Therefore, a simple way of modeling returns is to obtain parameters for a Student's t-distribution which best describes our data.
From these returns we can use the equation above to make many predictions of prices.
From these many predictions we can display some useful information like confidence intervals.
In the Naive Prediction tab you are able to view the 50% and 90% confidence intervals.
        """
        )
    with col.beta_expander("Bayesian Predictor"):
        st.markdown(
            r"""
In the Naive Predictor, parameters for the Student's t-distribution are chosen so as to best match the data. One of these parameters is "location", that is, which return is the most likely.
Suppose for some data, the most likely return is 0.2%.
If we predict according to this model, on average our stock would increase 0.2% a day.
Now, it is possible that our most likely return was actually 0.015%, but we had some crazy days in the past which spiked our returns and skewed them towards a higher average.
We can't simply remove these values: we would ideally like to know how likely it is that our average returns are 0.15% instead of 0.2%.

The exact probability of the values of "location" (and other parameters of our model) is exactly what a Bayesian analysis gives us.
And this adds a lot more information about our model.
Instead of obtaining one value for "location", we will obtain an entire probability distribution, informed by the data of course.
The distribution after we have informed it with data is called the posterior.
A very wide posterior can mean that the model is poor (does not match our data), or that the parameter cannot be inferred from the data.
How would be know this from a simple fitting? We can't!

The most common way of specifyind Bayesian models is by a model. In our case our simple Bayesian model looks like this:

$$\qquad \nu \sim \mathrm{Exponential}$$

$$\qquad \mu \sim \mathrm{Normal}$$

$$\qquad \sigma \sim \mathrm{Half Cauchy}$$

$$\qquad r_t \sim \mathrm{StudentT}(\nu, \mu, \sigma)$$

This model tells us that our returns are distributed according to Student's t, and the prior information we have about its parameters are Exponential, Normal and Half Cauchy (these are essentially arbitrary).
        """
        )
    with col.beta_expander("Stochastic Predictor"):
        st.markdown("Work in progress!")


def write_overview(price, logreturns):
    st.title(f"Overview")
    col1, col2 = st.beta_columns([1.5, 1])
    col1.markdown(f"### Price History")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price,
            mode="lines",
            name=price.name,
            line={"color": "black"},
        )
    )
    fig.update_layout(
        showlegend=True,
        xaxis_type="date",
        yaxis=dict(type="linear", title=f"Price [{price.name.split('-')[1]}]"),
        margin=dict(l=0, r=0, b=0, t=80, pad=0),
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h"),
        height=400,
        template=PLOTLY_TEMPLATE,
    )
    col1.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)

    col2.markdown(f"### Log-returns")
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=logreturns, histnorm="probability density", name=price.name
        )
    )
    fig.update_layout(
        bargap=0.2,
        showlegend=True,
        yaxis=dict(type="linear", title="Probability density"),
        margin=dict(l=0, r=0, b=0, t=80, pad=0),
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h"),
        height=400,
        template=PLOTLY_TEMPLATE,
        xaxis_range=2 * np.array([-1, 1]) * logreturns.std(),
    )
    col2.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)

    return


def write_naive_prediction(price, logreturns):
    # Process data
    n_sims, n_days = 500, 20
    nu, loc, scale = sp.stats.t.fit(logreturns)
    logrets_pred = sp.stats.t(nu, loc, scale).rvs(size=(n_sims, n_days))

    index_pred = pd.date_range(
        start=price.index[-1], periods=n_days + 1, freq="D"
    )
    percs_pred = utils.get_percentiles(
        logrets_pred, starting_price=price[-1], index=index_pred
    )

    x = np.linspace(logreturns.min(), logreturns.max(), 1001)
    pdf_pred = sp.stats.t.pdf(x, nu, loc, scale)

    st.title(f"Naive Prediction")
    col1, col2 = st.beta_columns([1.5, 1])
    col1.markdown(f"### Price Prediction")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price,
            mode="lines",
            name=price.name,
            line={"color": "black"},
            showlegend=True,
        )
    )
    cone_color, cone_group = PLOTLY_COLORS[0], "cone90"
    fig.add_trace(
        go.Scatter(
            x=index_pred,
            y=percs_pred[5],
            fill=None,
            mode="lines",
            legendgroup=cone_group,
            showlegend=False,
            line_color=cone_color,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=index_pred,
            y=percs_pred[95],
            fill="tonexty",  # fill area between trace0 and trace1
            mode="lines",
            name=f"90% Confidence",
            legendgroup=cone_group,
            line_color=cone_color,
        )
    )
    cone_color, cone_group = PLOTLY_COLORS[1], "cone50"
    fig.add_trace(
        go.Scatter(
            x=index_pred,
            y=percs_pred[25],
            fill=None,
            mode="lines",
            legendgroup=cone_group,
            showlegend=False,
            line_color=cone_color,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=index_pred,
            y=percs_pred[75],
            fill="tonexty",  # fill area between trace0 and trace1
            mode="lines",
            name=f"50% Confidence",
            legendgroup=cone_group,
            line_color=cone_color,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=index_pred,
            y=percs_pred[50],
            mode="lines",
            name=f"Predicted price",
            line={"color": "black", "dash": "dash"},
        )
    )

    fig.update_layout(
        xaxis_type="date",
        yaxis=dict(type="linear", title=f"Price [{price.name.split('-')[1]}]"),
        margin=dict(l=0, r=0, b=0, t=80, pad=0),
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h"),
        height=400,
        template=PLOTLY_TEMPLATE,
        xaxis_range=[
            price.index[-1] - pd.to_timedelta(3 * n_days, unit="d"),
            index_pred[-1],
        ],
    )
    col1.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)

    col2.markdown(f"### Log-returns")
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=logreturns, histnorm="probability density", name=price.name
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf_pred,
            mode="lines",
            name=f"Best-fit Student's t",
            line_color=PLOTLY_COLORS[4],
        )
    )
    fig.update_layout(
        bargap=0.2,
        showlegend=True,
        yaxis=dict(type="linear", title="Probability density"),
        margin=dict(l=0, r=0, b=0, t=80, pad=0),
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h"),
        height=400,
        template=PLOTLY_TEMPLATE,
        xaxis_range=2 * np.array([-1, 1]) * logreturns.std(),
    )
    col2.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)

    return


def write_simple_bayesian(price, logreturns):
    # Process data
    n_sims, n_days = 500, 20
    logrets_pred = get_predictions_simple(logreturns, n_sims, n_days)

    index_pred = pd.date_range(
        start=price.index[-1], periods=n_days + 1, freq="D"
    )
    percs_pred = utils.get_percentiles(
        logrets_pred, starting_price=price[-1], index=index_pred
    )

    # x = np.linspace(logreturns.min(), logreturns.max(), 1001)
    # pdf_pred = sp.stats.t.pdf(x, nu, loc, scale)

    st.title(f"Simple Bayesian Prediction")
    col1, col2 = st.beta_columns([1.5, 1])
    col1.markdown(f"### Price Prediction")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price,
            mode="lines",
            name=price.name,
            line={"color": "black"},
            showlegend=True,
        )
    )
    cone_color, cone_group = PLOTLY_COLORS[0], "cone90"
    fig.add_trace(
        go.Scatter(
            x=index_pred,
            y=percs_pred[5],
            fill=None,
            mode="lines",
            legendgroup=cone_group,
            showlegend=False,
            line_color=cone_color,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=index_pred,
            y=percs_pred[95],
            fill="tonexty",  # fill area between trace0 and trace1
            mode="lines",
            name=f"90% Confidence",
            legendgroup=cone_group,
            line_color=cone_color,
        )
    )
    cone_color, cone_group = PLOTLY_COLORS[1], "cone50"
    fig.add_trace(
        go.Scatter(
            x=index_pred,
            y=percs_pred[25],
            fill=None,
            mode="lines",
            legendgroup=cone_group,
            showlegend=False,
            line_color=cone_color,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=index_pred,
            y=percs_pred[75],
            fill="tonexty",  # fill area between trace0 and trace1
            mode="lines",
            name=f"50% Confidence",
            legendgroup=cone_group,
            line_color=cone_color,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=index_pred,
            y=percs_pred[50],
            mode="lines",
            name=f"Predicted price",
            line={"color": "black", "dash": "dash"},
        )
    )

    fig.update_layout(
        xaxis_type="date",
        yaxis=dict(type="linear", title=f"Price [{price.name.split('-')[1]}]"),
        margin=dict(l=0, r=0, b=0, t=80, pad=0),
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h"),
        height=400,
        template=PLOTLY_TEMPLATE,
        xaxis_range=[
            price.index[-1] - pd.to_timedelta(3 * n_days, unit="d"),
            index_pred[-1],
        ],
    )
    col1.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)

    col2.markdown(f"### Log-returns")
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=logreturns,
            histnorm="probability density",
            name=price.name,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=logrets_pred.flatten(),
            histnorm="probability density",
            name="Predicted",
            marker_color="black",
            opacity=0.5,
        )
    )
    fig.update_layout(
        bargap=0.2,
        barmode="overlay",
        showlegend=True,
        yaxis=dict(type="linear", title="Probability density"),
        margin=dict(l=0, r=0, b=0, t=80, pad=0),
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h"),
        height=400,
        template=PLOTLY_TEMPLATE,
        xaxis_range=2 * np.array([-1, 1]) * logreturns.std(),
    )
    # fig.update_traces(opacity=0.75)
    col2.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)

    return


def write_stochastic_bayesian(price, logreturns):
    # Process data
    n_sims, n_days = 500, 20
    logrets_pred = get_predictions_stochastic(logreturns, n_sims, n_days)

    index_pred = pd.date_range(
        start=price.index[-1], periods=n_days + 1, freq="D"
    )
    percs_pred = utils.get_percentiles(
        logrets_pred, starting_price=price[-1], index=index_pred
    )

    # x = np.linspace(logreturns.min(), logreturns.max(), 1001)
    # pdf_pred = sp.stats.t.pdf(x, nu, loc, scale)

    st.title(f"Stochastic Bayesian")
    col1, col2 = st.beta_columns([1.5, 1])
    col1.markdown(f"### Price Prediction")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price,
            mode="lines",
            name=price.name,
            line={"color": "black"},
            showlegend=True,
        )
    )
    cone_color, cone_group = PLOTLY_COLORS[0], "cone90"
    fig.add_trace(
        go.Scatter(
            x=index_pred,
            y=percs_pred[5],
            fill=None,
            mode="lines",
            legendgroup=cone_group,
            showlegend=False,
            line_color=cone_color,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=index_pred,
            y=percs_pred[95],
            fill="tonexty",  # fill area between trace0 and trace1
            mode="lines",
            name=f"90% Confidence",
            legendgroup=cone_group,
            line_color=cone_color,
        )
    )
    cone_color, cone_group = PLOTLY_COLORS[1], "cone50"
    fig.add_trace(
        go.Scatter(
            x=index_pred,
            y=percs_pred[25],
            fill=None,
            mode="lines",
            legendgroup=cone_group,
            showlegend=False,
            line_color=cone_color,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=index_pred,
            y=percs_pred[75],
            fill="tonexty",  # fill area between trace0 and trace1
            mode="lines",
            name=f"50% Confidence",
            legendgroup=cone_group,
            line_color=cone_color,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=index_pred,
            y=percs_pred[50],
            mode="lines",
            name=f"Predicted price",
            line={"color": "black", "dash": "dash"},
        )
    )

    fig.update_layout(
        xaxis_type="date",
        yaxis=dict(type="linear", title=f"Price [{price.name.split('-')[1]}]"),
        margin=dict(l=0, r=0, b=0, t=80, pad=0),
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h"),
        height=400,
        template=PLOTLY_TEMPLATE,
        xaxis_range=[
            price.index[-1] - pd.to_timedelta(3 * n_days, unit="d"),
            index_pred[-1],
        ],
    )
    col1.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)

    col2.markdown(f"### Log-returns")
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=logreturns,
            histnorm="probability density",
            name=price.name,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=logrets_pred.flatten(),
            histnorm="probability density",
            name="Predicted",
            marker_color="black",
            opacity=0.5,
        )
    )
    fig.update_layout(
        bargap=0.2,
        barmode="overlay",
        showlegend=True,
        yaxis=dict(type="linear", title="Probability density"),
        margin=dict(l=0, r=0, b=0, t=80, pad=0),
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h"),
        height=400,
        template=PLOTLY_TEMPLATE,
        xaxis_range=2 * np.array([-1, 1]) * logreturns.std(),
    )
    # fig.update_traces(opacity=0.75)
    col2.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)

    return


# Definitions
PAGES = {
    "Introduction": write_intro,
    "Overview": write_overview,
    "Naive Prediction": write_naive_prediction,
    "Simple Bayesian": write_simple_bayesian,
    "Stochastic Bayesian": write_stochastic_bayesian,
}

# Load data
exchanges = get_exchanges()
coins_all, currencies_all = get_pairs()  # Preload All


def main():
    st.sidebar.title("Crypto Price Ranges")
    st.sidebar.markdown("Bayesian analysis of cryptocurrencies")

    # DASHBOARD
    st.sidebar.markdown("## Dashboard")
    selection = st.sidebar.radio("", list(PAGES.keys()))

    # OPTIONS
    with st.sidebar.beta_expander("Exchange"):
        exchange = st.selectbox("", options=exchanges)
    if exchange == "All":
        coins, currencies = coins_all, currencies_all
    else:
        coins, currencies = get_pairs()
    try:
        idx_coin = coins.index("CHR")
    except ValueError:
        idx_coin = 0
    with st.sidebar.beta_expander("Coin", expanded=True):
        coin = st.selectbox("", options=coins, index=idx_coin)
    try:
        idx_curr = currencies.index("EUR")
    except ValueError:
        idx_curr = 0
    with st.sidebar.beta_expander("Currency", expanded=True):
        curr = st.selectbox("", options=currencies, index=idx_curr)

    ohlcv = utils.get_ohlcv(coin, curr)
    price = ohlcv[["high", "low", "open", "close"]].mean(axis=1)
    price.name = f"{coin}-{curr}"
    logreturns = utils.get_logreturns(price)

    with st.spinner(
        (
            f"Running prediction on {utils.CPUS} cores."
            "\nThese are heavy models, they may take a while!"
        )
    ):
        PAGES[selection](price, logreturns)


if __name__ == "__main__":
    main()