import streamlit as st
import pymc3 as pm
import cryptocompare as cc
import plotly
import plotly.graph_objects as go

import utils


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

with open("icon.b64") as f:
    icon = f.read()
    st.set_page_config(
        page_title="Locked Staking Analysis",
        page_icon=icon,
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


# Display functions
def write_overview(price):
    st.title(f"Overview of {price.name}")
    col1, col2 = st.beta_columns([1.5, 1])
    col1.markdown(f"### Price History")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price,
            mode="lines",
            name=price.name,
        )
    )
    fig.update_layout(
        showlegend=False,
        xaxis_type="date",
        yaxis=dict(type="linear", title=f"Price [{price.name.split('-')[1]}]"),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h"),
        height=300,
        template="ggplot2",
    )
    col1.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)

    col2.markdown(f"### Log-returns")
    logreturns = utils.get_logreturns(price)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=logreturns, histnorm="percent"))
    fig.update_layout(
        showlegend=False,
        yaxis=dict(type="linear", title="Frequency [%]"),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h"),
        height=300,
        template="ggplot2",
    )
    col2.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)

    return


def write_naive_prediction():
    st.title(pm.__version__)
    return


def write_simple_bayesian():
    return


def write_stochastic_bayesian():
    return


# Definitions
PAGES = {
    "Overview": write_overview,
    "Naive Prediction": write_naive_prediction,
    "Simple Bayesian": write_simple_bayesian,
    "Stochastic Bayesian": write_stochastic_bayesian,
}

# Load data
exchanges = get_exchanges()
coins_all, currencies_all = get_pairs()  # Preload All


def main():
    st.sidebar.title("Locked Staking Analysis")
    st.sidebar.markdown("Bayesian analysis of locked staking")

    # DASHBOARD
    st.sidebar.markdown("## Dashboard")
    selection = st.sidebar.radio("", list(PAGES.keys()))

    # OPTIONS
    st.sidebar.markdown("## Options")
    exchange = st.sidebar.selectbox("Exchange", options=exchanges)
    if exchange == "All":
        coins, currencies = coins_all, currencies_all
    else:
        coins, currencies = get_pairs()
    try:
        idx_coin = coins.index("CHR")
    except ValueError:
        idx_coin = 0
    coin = st.sidebar.selectbox("Coin", options=coins, index=idx_coin)
    try:
        idx_curr = currencies.index("EUR")
    except ValueError:
        idx_curr = 0
    curr = st.sidebar.selectbox("Currency", options=currencies, index=idx_curr)

    ohlcv = utils.get_ohlcv(coin, curr)
    price = ohlcv[["high", "low", "open", "close"]].mean(axis=1)
    price.name = f"{coin}-{curr}"

    with st.spinner(f"Loading {selection} ..."):
        PAGES[selection](price)


if __name__ == "__main__":
    main()