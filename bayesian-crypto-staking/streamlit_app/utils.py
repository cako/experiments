import streamlit as st

import cryptocompare as cc
import multiprocessing
import numpy as np
import pandas as pd
import pymc3 as pm

CPUS = multiprocessing.cpu_count()


@st.cache(allow_output_mutation=True)
def get_predictions_simple(logreturns, n_sims, n_days):
    model = get_student_t_model(logreturns)
    spp = sample_model(model, n_sims)
    return spp["logreturns"][:, -n_days:]


@st.cache(allow_output_mutation=True)
def get_predictions_stochastic(logreturns, n_sims, n_days):
    model = get_stochastic_model(logreturns)
    spp = sample_model(model, n_sims)
    return spp["logreturns"][:, -n_days:]


class OHLCVNotFoundError(Exception):
    def __init__(self, message):
        self.message = message


def find_first_nonzero(response):
    """
    Parameters
    ----------
        response : iterable
            Every item is a dictionary which must contain at least keys:
            'open','close', 'high', 'low'.

    Returns
    -------
        int
            First index where the entry is not zero. If none found, return -1.
    """
    first_nonzero = -1
    for i, datapoint in enumerate(response):
        if (
            datapoint["open"] != 0
            or datapoint["close"] != 0
            or datapoint["high"] != 0
            or datapoint["low"] != 0
        ):
            first_nonzero = i
            break  # Find first nonzero point
    return first_nonzero


def get_ohlcv(coin, currency, limit=None):
    """
    Parameters
    ----------
        coin : str
        currency: str
            Specifies the currency you would like the `coin` priced at.
        limit : int, optional
            Limits the response to `limit` entries.

    Returns
    -------
        DataFrame
            Containns columns 'high', 'low', 'open', 'close', 'volume'.
            'Volume' is priced in the `coin`.

    Raises
    ------
        OHLCVNotFoundError
            If all entries are zero, raises OHLCVNotFoundError
    """
    if limit is None:
        response = cc.get_historical_price_day(coin, currency)
    else:
        response = cc.get_historical_price_day(coin, currency, limit=limit)
    # `response` may contain many identically zero entries. We need
    #  to find the first nonzero so as to drop all the useless info
    first_nonzero = find_first_nonzero(response)
    if first_nonzero == -1:
        message = f"No nonzero values found for pair {coin}/{currency}"
        raise OHLCVNotFoundError(message)

    df = pd.DataFrame(response[first_nonzero:])
    df = df.rename({"volumefrom": "volume"}, axis=1)
    df.index = pd.to_datetime(df.time, unit="s")
    df.index.name = "Date"
    return df[["high", "low", "open", "close", "volume"]].asfreq("D")


def get_logreturns(price: pd.DataFrame):
    return np.log(price).diff().dropna()


def predict_price(logreturns, starting_price=1.0):
    return np.maximum(
        np.cumprod(np.exp(logreturns), axis=-1) * starting_price, 0.0
    )


def get_percentiles(simulations, starting_price, index):
    prices_t_arr = predict_price(
        simulations, starting_price=starting_price
    )  # predict prices
    prices_t_arr = np.insert(
        prices_t_arr, 0, starting_price, axis=-1
    )  # insert starting price

    return {
        p: pd.Series(np.percentile(prices_t_arr, p, axis=0), index=index)
        for p in [5, 25, 50, 75, 95]
    }


# Simple Bayesian Model
def get_student_t_model(data):
    with pm.Model() as model:
        mean = pm.Normal("mean", mu=0, sd=0.05, testval=data.mean())
        sigma = pm.HalfCauchy("volatility", beta=1, testval=data.std())
        nu = pm.Exponential("nu", lam=1.0 / 10.0, testval=3.0)

        pm.StudentT("logreturns", nu=nu, mu=mean, sd=sigma, observed=data)
    return model


# Stochastic Model
def get_stochastic_model(data):
    with pm.Model() as model:
        # Average return
        mu = pm.Normal("mu", 0.01, sigma=5)

        # Average volatility
        sigma = pm.Exponential("sigma", 1.0 / 0.02, testval=0.1)

        # Time-dependent volatility
        log_vol = pm.GaussianRandomWalk("log_vol", sigma ** -2, shape=len(data))
        lam = pm.Deterministic(
            "lam", pm.math.exp(-2 * log_vol)
        )  # Simple way of parametrizing Student's t
        volatility = pm.Deterministic("volatility", pm.math.exp(log_vol))

        # Log-returns
        nu = pm.Exponential("nu", 1.0 / 10, testval=5.0)  # Degrees of freedom
        pm.StudentT("logreturns", nu=nu, lam=lam, mu=mu, observed=data)
    return model


def sample_model(model, n_sims):
    with model:
        trace = pm.sample(
            n_sims,
            tune=n_sims // 2,
            chains=2,
            cores=CPUS,
            return_inferencedata=False,
        )
        spp = pm.sample_posterior_predictive(trace, samples=n_sims)
    return spp


# Icon
ICON = "data:image/png;base64,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"