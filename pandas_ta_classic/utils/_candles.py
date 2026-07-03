from pandas import Series


def candle_color(open_: Series, close: Series) -> Series:
    color = close.copy().astype(int)
    color[close >= open_] = 1
    color[close < open_] = -1
    return color
