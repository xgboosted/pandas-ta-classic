#!/usr/bin/env python
# coding: utf-8

# # Technical Analysis with Pandas ([pandas_ta_classic](https://github.com/xgboosted/pandas-ta-classic))
# * Below contains examples of simple charts that can be made from pandas_ta_classic indicators
# * Examples below are for **educational purposes only**
# * **NOTE:** The **watchlist** module is independent of Pandas TA Classic. To easily use it, copy it from your local pandas_ta_classic installation directory into your project directory.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import datetime as dt
import random as rnd

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import mplfinance as mpf

# Optional import for AlphaVantage API
try:
    from alphaVantageAPI.alphavantage import AlphaVantage
except ImportError:
    print("[!] alphaVantageAPI not available. Install with: pip install alphaVantage-api")
    AlphaVantage = None
import pandas_ta_classic as ta

from watchlist import colors # Is this failing? If so, copy it locally. See above.

print(f"\nPandas TA Classic v{ta.version}\nTo install the Latest Version:\n$ pip install pandas-ta-classic\n")

get_ipython().run_line_magic('pylab', 'inline')


# ### List of Indicators (post an [issue](https://github.com/xgboosted/pandas-ta-classic/issues) if the indicator doc needs updating)

# In[2]:


e = pd.DataFrame()
e.ta.indicators()


# ### Individual Indicator help

# In[3]:


help(ta.ema)


# In[4]:


# Function to format Millions
def format_millions(x, pos):
    "The two args are the value and tick position"
    return "%1.1fM" % (x * 1e-6)


# In[5]:


def ctitle(indicator_name, ticker="SPY", length=100):
    return f"{ticker}: {indicator_name} from {recent_startdate} to {recent_enddate} ({length})"

# # All Data: 0, Last Four Years: 0.25, Last Two Years: 0.5, This Year: 1, Last Half Year: 2, Last Quarter: 3
# yearly_divisor = 1
# recent = int(ta.RATE["TRADING_DAYS_PER_YEAR"] / yearly_divisor) if yearly_divisor > 0 else df.shape[0]
# print(recent)
def recent_bars(df, tf: str = "1y"):
    # All Data: 0, Last Four Years: 0.25, Last Two Years: 0.5, This Year: 1, Last Half Year: 2, Last Quarter: 4
    yearly_divisor = {"all": 0, "10y": 0.1, "5y": 0.2, "4y": 0.25, "3y": 1./3, "2y": 0.5, "1y": 1, "6mo": 2, "3mo": 4}
    yd = yearly_divisor[tf] if tf in yearly_divisor.keys() else 0
    return int(ta.RATE["TRADING_DAYS_PER_YEAR"] / yd) if yd > 0 else df.shape[0]

def ta_ylim(series: pd.Series, percent: float = 0.1):
    smin, smax = series.min(), series.max()
    if isinstance(percent, float) and 0 <= float(percent) <= 1:
        y_min = (1 + percent) * smin if smin < 0 else (1 - percent) * smin
        y_max = (1 - percent) * smax if smax < 0 else (1 + percent) * smax
        return (y_min, y_max)
    return (smin, smax)

price_size = (16, 8)
ind_size = (16, 3.25)


# ### Load Daily Ticker Data using yfinance and clean it

# In[6]:


# help(e.ta.ticker)


# In[7]:


# Load sample data from local CSV file instead of external API
ticker = "SPY"
import os
# Load sample data (go up one directory to find data folder)
data_path = os.path.join('..', 'data', 'SPY_D.csv')
df = pd.read_csv(data_path, index_col='date', parse_dates=True)
df.name = ticker
# Ensure column names are lowercase for consistency
df.columns = df.columns.str.lower()
recent_startdate = df.tail(recent_bars(df)).index[0]
recent_enddate = df.tail(recent_bars(df)).index[-1]
print(f"{df.name}{df.tail(recent_bars(df)).shape} from {recent_startdate} to {recent_enddate}")
df.tail(recent_bars(df)).head()


# ### Aliases

# In[8]:


opendf = df["open"]
highdf = df["high"]
lowdf = df["low"]
closedf = df["close"]
volumedf = df["volume"]


# ## DataFrame **constants**: When you need some simple lines for charting

# In[9]:


# help(df.ta.constants) # for more info
chart_lines = np.append(np.arange(-5, 6, 1), np.arange(-100, 110, 10))
df.ta.constants(True, chart_lines) # Adding the constants for the charts
df.ta.constants(False, np.array([-60, -40, 40, 60])) # Removing some constants from the DataFrame
print(f"Columns: {', '.join(list(df.columns))}")


# ## Example Charting Class utilizing **mplfinance** panels

# In[10]:


class Chart(object):
    def __init__(self, df: pd.DataFrame = None, strategy: ta.Strategy = ta.CommonStrategy, *args, **kwargs):
        self.verbose = kwargs.pop("verbose", False)

        if isinstance(df, pd.DataFrame) and df.ta.datetime_ordered:
            self.df = df
            if self.df.name is not None and self.df.name != "":
                df_name = str(self.df.name)
            else:
                df_name = "DataFrame"
            if self.verbose: print(f"[i] Loaded {df_name}{self.df.shape}")
        else:
            print(f"[X] Oops! Missing 'ohlcv' data or index is not datetime ordered.\n")
            return None

        self._validate_mpf_kwargs(**kwargs)
        self._validate_chart_kwargs(**kwargs)
        self._validate_ta_strategy(strategy)

        # Build TA and Plot
        self.df.ta.strategy(self.strategy, verbose=self.verbose)
        self._plot(**kwargs)

    def _validate_ta_strategy(self, strategy):
        if strategy is not None or isinstance(strategy, ta.Strategy):
            self.strategy = strategy
        elif len(self.strategy_ta) > 0:
            print(f"[+] Strategy: {self.strategy_name}")
        else:
            self.strategy = ta.CommonStrategy        

    def _validate_chart_kwargs(self, **kwargs):
        """Chart Settings"""
        self.config = {}
        self.config["last"] = kwargs.pop("last", recent_bars(self.df))
        self.config["rpad"] = kwargs.pop("rpad", 10)
        self.config["title"] = kwargs.pop("title", "Asset")
        self.config["volume"] = kwargs.pop("volume", True)

    def _validate_mpf_kwargs(self, **kwargs):
        # mpf global chart settings
        default_chart = mpf.available_styles()[-1]
        default_mpf_width = {
            'candle_linewidth': 0.6,
            'candle_width': 0.525,
            'volume_width': 0.525
        }
        mpfchart = {}

        mpf_style = kwargs.pop("style", "")
        if mpf_style == "" or mpf_style.lower() == "random":
            mpf_styles = mpf.available_styles()
            mpfchart["style"] = mpf_styles[rnd.randrange(len(mpf_styles))]
        elif mpf_style.lower() in mpf.available_styles():
            mpfchart["style"] = mpf_style

        mpfchart["figsize"] = kwargs.pop("figsize", (12, 10))
        mpfchart["non_trading"] = kwargs.pop("nontrading", False)
        mpfchart["rc"] = kwargs.pop("rc", {'figure.facecolor': '#EDEDED'})
        mpfchart["plot_ratios"] = kwargs.pop("plot_ratios", (12, 1.7))
        mpfchart["scale_padding"] = kwargs.pop("scale_padding", {'left': 1, 'top': 4, 'right': 1, 'bottom': 1})
        mpfchart["tight_layout"] = kwargs.pop("tight_layout", True)
        mpfchart["type"] = kwargs.pop("type", "candle")
        mpfchart["width_config"] = kwargs.pop("width_config", default_mpf_width)
        mpfchart["xrotation"] = kwargs.pop("xrotation", 15)

        self.mpfchart = mpfchart

    def _attribution(self):
        print(f"\nPandas v: {pd.__version__} [pip install pandas] https://github.com/pandas-dev/pandas")
        print(f"Data from AlphaVantage v: 1.0.19 [pip install alphaVantage-api] http://www.alphavantage.co https://github.com/twopirllc/AlphaVantageAPI")
        print(f"Technical Analysis with Pandas TA Classic v: {ta.version} [pip install pandas-ta-classic] https://github.com/xgboosted/pandas-ta-classic")
        print(f"Charts by Matplotlib Finance v: {mpf.__version__} [pip install mplfinance] https://github.com/matplotlib/mplfinance\n")

    def _right_pad_df(self, rpad: int, delta_unit: str = "D", range_freq: str = "B"):
        if rpad > 0:
            dfpad = self.df[-rpad:].copy()
            dfpad.iloc[:,:] = np.NaN

            df_frequency = self.df.index.value_counts().mode()[0] # Most common frequency
            freq_delta = pd.Timedelta(df_frequency, unit=delta_unit)
            new_dr = pd.date_range(start=self.df.index[-1] + freq_delta, periods=rpad, freq=range_freq)
            dfpad.index = new_dr # Update the padded index with new dates
            self.df = self.df.append(dfpad)


    def _plot(self, **kwargs):
        if not isinstance(self.mpfchart["plot_ratios"], tuple):
            print(f"[X] plot_ratios must be a tuple")
            return

        # Override Chart Title Option
        chart_title = self.config["title"]
        if "title" in kwargs and isinstance(kwargs["title"], str):
            chart_title = kwargs.pop("title")

        # Override Right Bar Padding Option
        rpad = self.config["rpad"]
        if "rpad" in kwargs and kwargs["rpad"] > 0:
            rpad = int(kwargs["rpad"])

        def cpanel():
            return len(self.mpfchart["plot_ratios"])

        # Last Second Default TA Indicators
        linreg = kwargs.pop("linreg", False)
        linreg_name = self.df.ta.linreg(append=True).name if linreg else ""

        midpoint = kwargs.pop("midpoint", False)
        midpoint_name = self.df.ta.midpoint(append=True).name if midpoint else ""

        ohlc4 = kwargs.pop("ohlc4", False)
        ohlc4_name = self.df.ta.ohlc4(append=True).name if ohlc4 else ""

        clr = kwargs.pop("clr", False)
        clr_name = self.df.ta.log_return(cumulative=True, append=True).name if clr else ""

        rsi = kwargs.pop("rsi", False)
        rsi_length = kwargs.pop("rsi_length", None)
        if isinstance(rsi_length, int) and rsi_length > 1:
            rsi_name = self.df.ta.rsi(length=rsi_length, append=True).name
        elif rsi:
            rsi_name = self.df.ta.rsi(append=True).name
        else: rsi_name = ""

        zscore = kwargs.pop("zscore", False)
        zscore_length = kwargs.pop("zscore_length", None)
        if isinstance(zscore_length, int) and zscore_length > 1:
            zs_name = self.df.ta.zscore(length=zscore_length, append=True).name
        elif zscore:
            zs_name = self.df.ta.zscore(append=True).name
        else: zs_name = ""

        macd = kwargs.pop("macd", False)
        macd_name = ""
        if macd:
            macds = self.df.ta.macd(append=True)
            macd_name = macds.name

        squeeze = kwargs.pop("squeeze", False)
        lazybear = kwargs.pop("lazybear", False)
        squeeze_name = ""
        if squeeze:
            squeezes = self.df.ta.squeeze(lazybear=lazybear, detailed=True, append=True)
            squeeze_name = squeezes.name

        ama = kwargs.pop("archermas", False)
        ama_name = ""
        if ama:
            amas = self.df.ta.amat(append=True)
            ama_name = amas.name

        aobv = kwargs.pop("archerobv", False)
        aobv_name = ""
        if aobv:
            aobvs = self.df.ta.aobv(append=True)
            aobv_name = aobvs.name

        # Pad and trim Chart
        self._right_pad_df(rpad)
        mpfdf = self.df.tail(self.config["last"]).copy()
        mpfdf_columns = list(self.df.columns)

        tsig = kwargs.pop("tsignals", False)
        if tsig:
            # Long Trend requires Series Comparison (<=. <, = >, >=)
            # or Trade Logic that yields trends in binary.
            default_long = mpfdf["SMA_10"] > mpfdf["SMA_20"]
            long_trend = kwargs.pop("long_trend", default_long)
            if not isinstance(long_trend, pd.Series):
                raise(f"[X] Must be a Series that has boolean values or values of 0s and 1s")
            mpfdf.ta.percent_return(append=True)
            mpfdf.ta.tsignals(long_trend, append=True)
            buys = np.where(mpfdf.TS_Entries > 0, 1, np.nan)
            sells = np.where(mpfdf.TS_Exits > 0, 1, np.nan)
            mpfdf["ACTRET_1"] = mpfdf.TS_Trends * mpfdf.PCTRET_1

        # BEGIN: Custom TA Plots and Panels
        # Modify the area below 
        taplots = [] # Holds all the additional plots

        # Panel 0: Price Overlay
        if linreg_name in mpfdf_columns:
            taplots += [mpf.make_addplot(mpfdf[linreg_name], type=kwargs.pop("linreg_type", "line"), color=kwargs.pop("linreg_color", "black"), linestyle="-.", width=1.2, panel=0)]

        if midpoint_name in mpfdf_columns:
            taplots += [mpf.make_addplot(mpfdf[midpoint_name], type=kwargs.pop("midpoint_type", "scatter"), color=kwargs.pop("midpoint_color", "fuchsia"), width=0.4, panel=0)]

        if ohlc4_name in mpfdf_columns:
            taplots += [mpf.make_addplot(mpfdf[ohlc4_name], ylabel=ohlc4_name, type=kwargs.pop("ohlc4_type", "scatter"), color=kwargs.pop("ohlc4_color", "blue"), alpha=0.85, width=0.4, panel=0)]

        if self.strategy.name == ta.CommonStrategy.name:
            total_sma = 0 # Check if all the overlap indicators exists before adding plots
            for c in ["SMA_10", "SMA_20", "SMA_50", "SMA_200"]:
                if c in mpfdf_columns: total_sma += 1
                else: print(f"[X] Indicator: {c} missing!")
            if total_sma == 4:
                ta_smas = [
                    mpf.make_addplot(mpfdf["SMA_10"], color="green", width=1.5, panel=0),
                    mpf.make_addplot(mpfdf["SMA_20"], color="orange", width=2, panel=0),
                    mpf.make_addplot(mpfdf["SMA_50"], color="red", width=2, panel=0),
                    mpf.make_addplot(mpfdf["SMA_200"], color="maroon", width=3, panel=0),
                ]
                taplots += ta_smas

        if tsig:
            taplots += [
                mpf.make_addplot(0.985 * mpfdf.close * buys, type="scatter", marker="^", markersize=26, color="blue", panel=0),
                mpf.make_addplot(1.015 * mpfdf.close * sells, type="scatter", marker="v", markersize=26, color="fuchsia", panel=0),
            ]

        if len(ama_name):
            amat_sr_ = mpfdf[amas.columns[-1]][mpfdf[amas.columns[-1]] > 0]
            amat_sr = amat_sr_.index.to_list()
        else:
            amat_sr = None

        # Panel 1: If volume=True, the add the VOL MA. Since we know there is only one, we immediately pop it.
        if self.config["volume"]:
            volma = [x for x in list(self.df.columns) if x.startswith("VOL_")].pop()
            max_vol = mpfdf["volume"].max()
             # Volume axis
            ta_volume = [mpf.make_addplot(mpfdf[volma], color="black", width=1.2, panel=1, ylim=(-.2 * max_vol, 1.5 * max_vol))]
            taplots += ta_volume

        # Panels 2 - 9
        common_plot_ratio = (3,)

        if len(aobv_name):
            _p = kwargs.pop("aobv_percenty", 0.2)
            aobv_ylim = ta_ylim(mpfdf[aobvs.columns[0]], _p)
            taplots += [
                mpf.make_addplot(mpfdf[aobvs.columns[0]], ylabel=aobv_name, color="black", width=1.5, panel=cpanel(), ylim=aobv_ylim),
                mpf.make_addplot(mpfdf[aobvs.columns[2]], color="silver", width=1, panel=cpanel(), ylim=aobv_ylim),
                mpf.make_addplot(mpfdf[aobvs.columns[3]], color="green", width=1, panel=cpanel(), ylim=aobv_ylim),
                mpf.make_addplot(mpfdf[aobvs.columns[4]], color="red", width=1.2, panel=cpanel(), ylim=aobv_ylim),
            ]
            self.mpfchart["plot_ratios"] += common_plot_ratio # Required to add a new Panel

        if clr_name in mpfdf_columns:
            _p = kwargs.pop("clr_percenty", 0.1)
            clr_ylim = ta_ylim(mpfdf[clr_name], _p)

            taplots += [mpf.make_addplot(mpfdf[clr_name], ylabel=clr_name, color="black", width=1.5, panel=cpanel(), ylim=clr_ylim)]
            if (1 - _p) * mpfdf[clr_name].min() < 0 and (1 + _p) * mpfdf[clr_name].max() > 0:
                taplots += [mpf.make_addplot(mpfdf["0"], color="gray", width=1.2, panel=cpanel(), ylim=clr_ylim)]
            self.mpfchart["plot_ratios"] += common_plot_ratio # Required to add a new Panel

        if rsi_name in mpfdf_columns:
            rsi_ylim = (0, 100)
            taplots += [
                mpf.make_addplot(mpfdf[rsi_name], ylabel=rsi_name, color=kwargs.pop("rsi_color", "black"), width=1.5, panel=cpanel(), ylim=rsi_ylim),
                mpf.make_addplot(mpfdf["20"], color="green", width=1, panel=cpanel(), ylim=rsi_ylim),
                mpf.make_addplot(mpfdf["50"], color="gray", width=0.8, panel=cpanel(), ylim=rsi_ylim),
                mpf.make_addplot(mpfdf["80"], color="red", width=1, panel=cpanel(), ylim=rsi_ylim),
            ]
            self.mpfchart["plot_ratios"] += common_plot_ratio # Required to add a new Panel

        if macd_name in mpfdf_columns:
            _p = kwargs.pop("macd_percenty", 0.15)
            macd_ylim = ta_ylim(mpfdf[macd_name], _p)
            taplots += [
                mpf.make_addplot(mpfdf[macd_name], ylabel=macd_name, color="black", width=1.5, panel=cpanel()),#, ylim=macd_ylim),
                mpf.make_addplot(mpfdf[macds.columns[-1]], color="blue", width=1.1, panel=cpanel()),#, ylim=macd_ylim),
                mpf.make_addplot(mpfdf[macds.columns[1]], type="bar", alpha=0.8, color="dimgray", width=0.8, panel=cpanel()),#, ylim=macd_ylim),
                mpf.make_addplot(mpfdf["0"], color="black", width=1.2, panel=cpanel()),#, ylim=macd_ylim),
            ]
            self.mpfchart["plot_ratios"] += common_plot_ratio # Required to add a new Panel            

        if zs_name in mpfdf_columns:
            _p = kwargs.pop("zascore_percenty", 0.2)
            zs_ylim = ta_ylim(mpfdf[zs_name], _p)
            taplots += [
                mpf.make_addplot(mpfdf[zs_name], ylabel=zs_name, color="black", width=1.5, panel=cpanel(), ylim=zs_ylim),
                mpf.make_addplot(mpfdf["-3"], color="red", width=1.2, panel=cpanel(), ylim=zs_ylim),
                mpf.make_addplot(mpfdf["-2"], color="orange", width=1, panel=cpanel(), ylim=zs_ylim),
                mpf.make_addplot(mpfdf["-1"], color="silver", width=1, panel=cpanel(), ylim=zs_ylim),
                mpf.make_addplot(mpfdf["0"], color="black", width=1.2, panel=cpanel(), ylim=zs_ylim),
                mpf.make_addplot(mpfdf["1"], color="silver", width=1, panel=cpanel(), ylim=zs_ylim),
                mpf.make_addplot(mpfdf["2"], color="orange", width=1, panel=cpanel(), ylim=zs_ylim),
                mpf.make_addplot(mpfdf["3"], color="red", width=1.2, panel=cpanel(), ylim=zs_ylim)
            ]
            self.mpfchart["plot_ratios"] += common_plot_ratio # Required to add a new Panel

        if squeeze_name in mpfdf_columns:
            _p = kwargs.pop("squeeze_percenty", 0.6)
            sqz_ylim = ta_ylim(mpfdf[squeeze_name], _p)
            taplots += [
                mpf.make_addplot(mpfdf[squeezes.columns[-4]], type="bar", color="lime", alpha=0.65, width=0.8, panel=cpanel(), ylim=sqz_ylim),
                mpf.make_addplot(mpfdf[squeezes.columns[-3]], type="bar", color="green", alpha=0.65, width=0.8, panel=cpanel(), ylim=sqz_ylim),
                mpf.make_addplot(mpfdf[squeezes.columns[-2]], type="bar", color="maroon", alpha=0.65, width=0.8, panel=cpanel(), ylim=sqz_ylim),
                mpf.make_addplot(mpfdf[squeezes.columns[-1]], type="bar", color="red", alpha=0.65, width=0.8, panel=cpanel(), ylim=sqz_ylim),
                mpf.make_addplot(mpfdf["0"], color="black", width=1.2, panel=cpanel(), ylim=sqz_ylim),
                mpf.make_addplot(mpfdf[squeezes.columns[4]], ylabel=squeeze_name, color="green", width=2, panel=cpanel(), ylim=sqz_ylim),
                mpf.make_addplot(mpfdf[squeezes.columns[5]], color="red", width=1.8, panel=cpanel(), ylim=sqz_ylim),
            ]
            self.mpfchart["plot_ratios"] += common_plot_ratio # Required to add a new Panel

        if tsig:
            _p = kwargs.pop("tsig_percenty", 0.23)
            treturn_ylim = ta_ylim(mpfdf["ACTRET_1"], _p)
            taplots += [
                mpf.make_addplot(mpfdf["ACTRET_1"], ylabel="Active % Return", type="bar", color="green", alpha=0.45, width=0.8, panel=cpanel(), ylim=treturn_ylim),
                mpf.make_addplot(pd.Series(mpfdf["ACTRET_1"].mean(), index=mpfdf["ACTRET_1"].index), color="blue", width=1, panel=cpanel(), ylim=treturn_ylim),
                mpf.make_addplot(mpfdf["0"], color="black", width=1, panel=cpanel(), ylim=treturn_ylim),
            ]
            self.mpfchart["plot_ratios"] += common_plot_ratio # Required to add a new Panel

            _p = kwargs.pop("cstreturn_percenty", 0.58)
            mpfdf["CUMACTRET_1"] = mpfdf["ACTRET_1"].cumsum()
            cumactret_ylim = ta_ylim(mpfdf["CUMACTRET_1"], _p)
            taplots += [
                mpf.make_addplot(mpfdf["CUMACTRET_1"], ylabel="Cum Trend Return", type="bar", color="silver", alpha=0.45, width=1, panel=cpanel(), ylim=cumactret_ylim),
                mpf.make_addplot(0.9 * buys * mpfdf["CUMACTRET_1"], type="scatter", marker="^", markersize=14, color="green", panel=cpanel(), ylim=cumactret_ylim),
                mpf.make_addplot(1.1 * sells * mpfdf["CUMACTRET_1"], type="scatter", marker="v", markersize=14, color="red", panel=cpanel(), ylim=cumactret_ylim),
                mpf.make_addplot(mpfdf["0"], color="black", width=1, panel=cpanel(), ylim=cumactret_ylim),
            ]            
            self.mpfchart["plot_ratios"] += common_plot_ratio # Required to add a new Panel

        # END: Custom TA Plots and Panels

        if self.verbose:
            additional_ta = []
            chart_title  = f"{chart_title} [{self.strategy.name}] (last {self.config['last']} bars)"
            chart_title += f"\nSince {mpfdf.index[0]} till {mpfdf.index[-1]}"
            if len(linreg_name) > 0: additional_ta.append(linreg_name)
            if len(midpoint_name) > 0: additional_ta.append(midpoint_name)
            if len(ohlc4_name) > 0: additional_ta.append(ohlc4_name)
            if len(additional_ta) > 0:
                chart_title += f"\nIncluding: {', '.join(additional_ta)}"

        if amat_sr:
            vlines_ = dict(vlines=amat_sr, alpha=0.1, colors="red")
        else:
            # Hidden because vlines needs valid arguments even if None 
            vlines_ = dict(vlines=mpfdf.index[0], alpha=0, colors="white")

        # Create Final Plot
        mpf.plot(mpfdf,
            title=chart_title,
            type=self.mpfchart["type"],
            style=self.mpfchart["style"],
            datetime_format="%-m/%-d/%Y",
            volume=self.config["volume"],
            figsize=self.mpfchart["figsize"],
            tight_layout=self.mpfchart["tight_layout"],
            scale_padding=self.mpfchart["scale_padding"],
            panel_ratios=self.mpfchart["plot_ratios"], # This key needs to be update above if adding more panels
            xrotation=self.mpfchart["xrotation"],
            update_width_config=self.mpfchart["width_config"],
            show_nontrading=self.mpfchart["non_trading"],
            vlines=vlines_,
            addplot=taplots
        )

        self._attribution()


# ### Charting Example
# #### Play with the parameters to see different charts and results
# * This is an example chart so it's not perfect. Enough to get started with common and uncommon plots.
# * There is a maximum of 10 Panels. In this example, panels 0 and 1 are reserved for Price and Volume respectively.
#     * 

# In[11]:


# Used for example Trend Return Long Trend Below
macd_ = ta.macd(closedf)
macdh = macd_[macd_.columns[1]]

Chart(df,
    # style: which mplfinance chart style to use. Added "random" as an option.
    # rpad: how many bars to leave empty on the right of the chart
    style="yahoo", title=ticker, last=recent_bars(df), rpad=10,

    # Overlap Indicators
    linreg=True, midpoint=False, ohlc4=False, archermas=True,

    # Example Indicators with default parameters
    volume=True, rsi=True, clr=False, macd=True, zscore=False, squeeze=False, lazybear=False,

    # Archer OBV and OBV MAs (https://www.tradingview.com/script/Co1ksara-Trade-Archer-On-balance-Volume-Moving-Averages-v1/)
    archerobv=False,

    # Create trends and see their returns
    tsignals=True,
    # Example Trends or create your own. Trend must yield Booleans
    long_trend=ta.sma(closedf,10) > ta.sma(closedf,20), # trend: sma(close,10) > sma(close,20) [Default Example]
#     long_trend=closedf > ta.ema(closedf,5), # trend: close > ema(close,5)
#     long_trend=ta.sma(closedf,10) > ta.ema(closedf,50), # trend: sma(close,10) > ema(close,50)
#     long_trend=ta.increasing(ta.ema(closedf), 10), # trend: increasing(ema, 10)
#     long_trend=macdh > 0, # trend: macd hist > 0
#     long_trend=macd_[macd_.columns[0]] > macd_[macd_.columns[-1]], # trend: macd > macd signal
#     long_trend=ta.increasing(ta.sma(ta.rsi(closedf), 10), 5, asint=False), # trend: rising sma(rsi, 10) for the previous 5 periods
#     long_trend=ta.squeeze(highdf, lowdf, closedf, lazybear=True, detailed=True).SQZ_PINC > 0,
#     long_trend=ta.amat(closedf, 50, 200, mamode="sma").iloc[:,0], # trend: amat(50, 200) long signal using sma
    show_nontrading=False, # Intraday use if needed
    verbose=True, # More detail
)


# ## Indicator Examples
# * Examples of simple and complex indicators.  Most indicators return a Series, while a few return DataFrames.
# * All indicators can be called one of three ways. Either way, they return the result.
# 
# ### Three ways to use pandas_ta
# 1. Stand Alone like TA-Lib  ta.**indicator**(*kwargs*).
# 2. As a DataFrame Extension like df.ta.**indicator**(*kwargs*).  Where df is a DataFrame with columns named "open", "high", "low", "close, "volume" for simplicity.
# 3. Similar to #2, but by calling: df.ta(kind="**indicator**", *kwargs*).

# ### Cumulative Log Return

# In[12]:


clr_ma_length = 8
clrdf = df.ta.log_return(cumulative=True, append=True)
clrmadf = ta.ema(clrdf, length=clr_ma_length)
clrxdf = pd.DataFrame({f"{clrdf.name}": clrdf, f"{clrmadf.name}({clrdf.name})": clrmadf})
clrxdf.tail(recent_bars(df)).plot(figsize=ind_size, color=colors("BkBl"), linewidth=1, title=ctitle(clrdf.name, ticker=ticker, length=recent_bars(df)), grid=True)


# ### MACD

# In[13]:


macddf = df.ta.macd(fast=8, slow=21, signal=9, min_periods=None, append=True)
macddf[[macddf.columns[0], macddf.columns[2]]].tail(recent_bars(df)).plot(figsize=(16, 2), color=colors("BkBl"), linewidth=1.3)
macddf[macddf.columns[1]].tail(recent_bars(df)).plot.area(figsize=ind_size, stacked=False, color=["silver"], linewidth=1, title=ctitle(macddf.name, ticker=ticker, length=recent_bars(df)), grid=True).axhline(y=0, color="black", lw=1.1)


# ### ZScore

# In[14]:


zscoredf = df.ta.zscore(length=30, append=True)
zcolors = ["darkgreen", "green", "silver", "silver", "red", "maroon", "black"]
zcols = df[["-4", "-3", "-2", "2", "3", "4", zscoredf.name]].tail(recent_bars(df))
zcols.plot(figsize=ind_size, color=zcolors, linewidth=1.2, title=ctitle(zscoredf.name, ticker=ticker, length=recent_bars(df)), grid=True).axhline(y=0, color="black", lw=1.1)


# In[15]:


# Now Volume Z Score
zvscoredf = df.ta.zscore(close="volume", length=30, prefix="VOL", append=True)
zcolors = ["darkgreen", "green", "silver", "silver", "red", "maroon", "black"]
zvcols = df[["-4", "-3", "-2", "2", "3", "4", zvscoredf.name]].tail(recent_bars(df))
zvcols.plot(figsize=ind_size, color=zcolors, linewidth=1.2, title=ctitle(zvscoredf.name, ticker=ticker, length=recent_bars(df)), grid=True).axhline(y=0, color="black", lw=1.1)


# # New Features

# ## Squeeze Indicator (John Carter and Lazybear Versions)
# Squeeze Indicator (__squeeze__)

# In[16]:


# help(ta.squeeze)


# In[17]:


Chart(df, style="yahoo", title=ticker, verbose=False,
    last=recent_bars(df), rpad=10, clr=True, squeeze=True,
    show_nontrading=False, # Intraday use if needed
)


# ### Lazybear's TradingView Squeeze

# In[18]:


Chart(df, style="yahoo", title=ticker, verbose=False,
    last=recent_bars(df), rpad=10, clr=True, squeeze=True, lazybear=True,
    show_nontrading=False, # Intraday use if needed
)


# ### Archer Moving Averages Trends
# Archer Moving Average Trends (__amat__) returns the long and short run trends of fast and slow moving averages.
# * The _pink_ background, on the Price chart, is when Archer MAs are bearish. Conversely, a _white_ background is bullish

# In[19]:


Chart(df, style="yahoo", title=ticker, verbose=False,
    last=recent_bars(df), rpad=10,
    volume=True, midpoint=False, ohlc4=False,
    rsi=False, clr=True, macd=False, zscore=False, squeeze=False, lazybear=False,
    archermas=True, archerobv=False,
    show_nontrading=False, # Intraday use if needed
)


# ### Archer On Balance Volume
# Archer On Balance Volume (__aobv__) returns a DataFrame of OBV, OBV min and max, fast and slow MAs of OBV, and the long and short run trends of the two OBV MAs.
# * On the chart below, only **OBV**, **OBV min**, _fast_ and _slow_ **OBV MAs**.
# * Not on the chart are: **OBV LR** and **OBV SR** trends.

# In[20]:


Chart(df, style="yahoo", title=ticker, verbose=False,
    last=recent_bars(df), rpad=10,
    volume=True, midpoint=False, ohlc4=False,
    rsi=False, clr=True, macd=False, zscore=False, squeeze=False, lazybear=False,
    archermas=False, archerobv=True,
    show_nontrading=False, # Intraday use if needed
)


# # Disclaimer
# * All investments involve risk, and the past performance of a security, industry, sector, market, financial product, trading strategy, or individual’s trading does not guarantee future results or returns. Investors are fully responsible for any investment decisions they make. Such decisions should be based solely on an evaluation of their financial circumstances, investment objectives, risk tolerance, and liquidity needs.
# 
# * Any opinions, news, research, analyses, prices, or other information offered is provided as general market commentary, and does not constitute investment advice. I will not accept liability for any loss or damage, including without limitation any loss of profit, which may arise directly or indirectly from use of or reliance on such information.
