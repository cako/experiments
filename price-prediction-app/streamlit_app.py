import streamlit as st

import cryptocompare as cc

import utils
import display

st.set_page_config(
    page_title="Locked Staking Analysis",
    page_icon=utils.ICON,
    layout="wide",
    initial_sidebar_state="auto",
)

# External data loading
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


exchanges = get_exchanges()
coins_all, currencies_all = get_pairs()  # Preload All

# Navigation
PAGES = {
    "Introduction": display.write_intro,
    "Overview": display.write_overview,
    "Naive Prediction": display.write_naive_prediction,
    "Simple Bayesian": display.write_bayesian,
    "Stochastic Bayesian": display.write_bayesian,
}


def main():
    col1, col2 = st.sidebar.beta_columns([2, 10])
    col1.image(utils.ICON, width=50)
    col2.title("Crypto Price Ranges")
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

    msg = f"Running prediction on {utils.CPUS} cores.\nThese are heavy models, they may take a while!"
    with st.spinner(msg):
        PAGES[selection](price=price, logreturns=logreturns, kind=selection)


if __name__ == "__main__":
    main()