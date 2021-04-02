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

    st.title(f"Stochastic Prediction")
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