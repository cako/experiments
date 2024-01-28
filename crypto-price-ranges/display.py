import streamlit as st

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


def price_figure(price, percs_pred, beg_date=None, end_date=None):
    index_pred = percs_pred[50].index
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
    if beg_date:
        fig.add_vline(
            x=beg_date, line_width=3, line_dash="dash", line_color="gray"
        )
    if end_date:
        fig.add_vline(
            x=end_date, line_width=3, line_dash="dash", line_color="gray"
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
            price.index[-1]
            - pd.to_timedelta(3 * max(len(index_pred) - 1, 10), unit="d"),
            index_pred[-1],
        ],
    )
    return fig


def write_intro(**kwargs):
    _, col, _ = st.beta_columns([0.2, 1, 0.2])
    col.title("Cryptocurrency Price Ranges")
    col.subheader("by [Carlos Costa](https://github/cako)")
    col.markdown(
        """
    ### ⬅ Jump into the models by choosing on the Dashboard, or read more below.


    Cryptocurrencies have gone from obscure digital curiosities to multibillion dollar markets. 
    The vast amount of trading that takes place have allowed traditional models to be used for pricing cryptocurrencies.
    This app explores a couple of these models. The first uses a naive "best-fit" predictor. The second uses the same model but in a Bayesian format. The last one uses a time-varying stochastic model.
    """
    )
    col.markdown(
        "You can find the source code for this app from the top-right menu or from my [GitHub repo](https://github.com/cako/experiments). It contains an accompanying [notebook](https://github.com/cako/experiments/blob/main/bayesian-crypto-staking/BayesianCryptoStaking.ipynb) detailing the implementation."
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
        st.markdown(
            r"""
A common way of modeling stock prices (as used in the Black-Scholes-Merton model for pricing options) is to assume that the [log-returns follow a "random walk"](https://en.wikipedia.org/wiki/Geometric_Brownian_motion).
Mathematically, you may write this as:

$\qquad\mathrm{d} \ln(S_t) = \mu\,\mathrm{d}t + \sigma\,\mathrm{d}W_t$

It says that at any point in time, the variation of the log of price, $\mathrm{d} \ln(S_t)$, is given by a time-independent (but still random) return $\mu$ and a [Wiener process](https://en.wikipedia.org/wiki/Wiener_process), $W_t$, with some volatility $\sigma$.
Finally, we will suppose that the average returns $\mu$, will be summed with the Wiener process as a Student's t-distribution as we did above.

$$\qquad \nu \sim \mathrm{Exponential}$$

$$\qquad \mu \sim \mathrm{Normal}$$

$$\qquad \ln\sigma \sim \mathrm{Exponential}$$

$$\qquad \sigma_t \sim \mathrm{Wiener(\ln\sigma)}$$

$$\qquad r_t \sim \mathrm{StudentT}(\nu, \mu, \sigma_t)$$
        """
        )


def write_overview(**kwargs):
    price, logreturns = kwargs["price"], kwargs["logreturns"]
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
        yaxis=dict(type="linear", title="Probability density", side="right"),
        margin=dict(l=0, r=0, b=0, t=80, pad=0),
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h"),
        height=400,
        template=PLOTLY_TEMPLATE,
        xaxis_range=2 * np.array([-1, 1]) * logreturns.std(),
    )
    col2.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)

    return


def write_naive_prediction(**kwargs):
    price, logreturns = kwargs["price"], kwargs["logreturns"]

    st.title(f"Naive Prediction")

    # Selectors
    cols = st.beta_columns([0.45, 0.1, 0.45, 0.1, 0.6, 0.3])
    n_days = cols[0].slider("Number of days for prediction", 1, 90, 10)
    n_sims = cols[2].slider("Number of simulations", 100, 5000, 500, step=100)

    beg_date, end_date = cols[-2].select_slider(
        "Only use returns between these dates",
        options=list(price.index.date),
        value=(price.index.date[0], price.index.date[-1]),
    )

    # Process data
    nu, loc, scale = sp.stats.t.fit(logreturns.loc[beg_date:end_date])
    logrets_pred = sp.stats.t(nu, loc, scale).rvs(size=(n_sims, n_days))

    index_pred = pd.date_range(
        start=price.index[-1], periods=n_days + 1, freq="D"
    )
    percs_pred = utils.get_percentiles(
        logrets_pred, starting_price=price[-1], index=index_pred
    )

    x = np.linspace(logreturns.min(), logreturns.max(), 1001)
    pdf_pred = sp.stats.t.pdf(x, nu, loc, scale)

    col1, col2 = st.beta_columns([1.5, 1])
    col1.markdown(f"### Price Prediction")
    fig = price_figure(price, percs_pred, beg_date=beg_date, end_date=end_date)
    col1.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)

    col2.markdown(f"### Log-returns")
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=logreturns.loc[beg_date:end_date],
            histnorm="probability density",
            name=price.name,
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
        yaxis=dict(type="linear", title="Probability density", side="right"),
        margin=dict(l=0, r=0, b=0, t=80, pad=0),
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h"),
        height=400,
        template=PLOTLY_TEMPLATE,
        xaxis_range=2 * np.array([-1, 1]) * logreturns.std(),
    )
    col2.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)

    return


def write_bayesian(**kwargs):
    price, logreturns, kind = (
        kwargs["price"],
        kwargs["logreturns"],
        kwargs["kind"],
    )

    st.title(kind)

    # Selectors
    cols = st.beta_columns([0.45, 0.1, 0.45, 0.1, 0.6, 0.3])
    n_days = cols[0].slider("Number of days for prediction", 1, 90, 10)
    n_sims = cols[2].slider("Number of simulations", 100, 5000, 500, step=100)

    beg_date, end_date = cols[-2].select_slider(
        "Only use returns between these dates",
        options=list(price.index.date),
        value=(price.index[0], price.index[-1]),
    )

    # Process data
    if kind.startswith("Simple"):
        logrets_pred = utils.get_predictions_simple(
            logreturns.loc[beg_date:end_date], n_sims, n_days
        )
    else:
        logrets_pred = utils.get_predictions_stochastic(
            logreturns.loc[beg_date:end_date], n_sims, n_days
        )

    index_pred = pd.date_range(
        start=price.index[-1], periods=n_days + 1, freq="D"
    )
    percs_pred = utils.get_percentiles(
        logrets_pred, starting_price=price[-1], index=index_pred
    )

    col1, col2 = st.beta_columns([1.5, 1])
    col1.markdown(f"### Price Prediction")
    fig = price_figure(price, percs_pred, beg_date=beg_date, end_date=end_date)
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
        yaxis=dict(type="linear", title="Probability density", side="right"),
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
