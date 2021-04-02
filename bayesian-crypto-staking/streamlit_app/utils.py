import numpy as np
import pandas as pd
import cryptocompare as cc


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