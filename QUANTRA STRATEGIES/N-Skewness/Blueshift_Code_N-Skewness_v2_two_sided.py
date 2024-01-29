"""
    Title: Markov Markowitz Strategy for N-Skewness

    Description: This strategy has won the Best In-Batch Project Award in Batch 52
    Does not include the money management module. This code is Intellectual Property
    of Markov Markowitz Company. Usage, sharing, partial or whole copying is prohibited.


    Style tags: Mean reversion
    Asset class: Equities
    Dataset: US Equities

    ############################# DISCLAIMER #############################
    This is a strategy template only and should not be
    used for live trading without appropriate backtesting and tweaking of
    the strategy parameters.
    ######################################################################
"""
# Import libraries
from scipy.stats import skew
import pandas as pd
import talib

SHORT_WINDOW = 5
LONG_WINDOW = 27
LOOKBACK =  300
KELLY = 1 #0.2886

BUY_ORDER_PRESENT = False
SELL_ORDER_PRESENT = False
buy_signal = False
sell_signal = False
sqr_off_signal = False


# Import blueshift libraries
from blueshift.api import(
                            symbol,
                            order_target_percent,
                            schedule_function,
                            date_rules,
                            time_rules,
                            get_datetime
                        )

def initialize(context):
    # Define symbol
    context.stocks = [
                        symbol("WBA")
                     ]
#                     IR       Signal       MDD    R_Skew    Delta Skew
#0   MRK      MSC  0.0558          BUY  0.056859 -0.283097    0.051206
#1   WMT      MSC  0.0753          BUY  0.046166 -0.436259    0.142821
#2   JNJ      MSC  0.4086  STRONG SELL  0.063830 -0.408801   -1.743138
#3   WBA      MSC  2.2084   STRONG BUY  0.083401 -0.437309    1.234136
#4   JPM      MSC  0.4931         SELL  0.083846 -0.040894   -0.023610
#5  NVDA      MSC -0.5220          BUY  0.160609 -0.197903    0.103678

    # Define lookback in days
    context.lookback = LOOKBACK


    # Rebalance every day
    schedule_function(
        rebalance,
        date_rules.every_day(),
        time_rules.market_close()
    )

def rebalance(context, data):
    global ORDER_PRESENT
    for ticker in context.stocks:

        # GENERATE TIMESERIES
        returns = data.history(ticker, 'close', context.lookback, '1d').pct_change()

        long_skewness = returns.rolling(LONG_WINDOW).skew()
        long_lag_skewness = long_skewness.shift(1)

        short_skewness = returns.rolling(SHORT_WINDOW).skew()
        short_lag_skewness = short_skewness.shift(1)

        # Generate trading signals

        buy_mask = (short_skewness > long_skewness) & (short_lag_skewness < long_lag_skewness)
        sell_mask = (short_skewness < long_skewness) & (short_lag_skewness > long_lag_skewness)
        sq_off_mask = (((long_skewness > 0) & (short_skewness < long_skewness) & (short_lag_skewness > long_lag_skewness)) | ((long_skewness < 0) & (short_skewness > long_skewness ) & (short_lag_skewness < long_lag_skewness)))

        buy_signal = buy_mask.iloc[-1]
        sell_signal = sell_mask.iloc[-1]
        sq_off_signal = sq_off_mask.iloc[-1]


        if BUY_ORDER_PRESENT == False and buy_signal == True and sell_signal == False:
            order_target_percent(ticker, KELLY)
            buy_signal = False
            BUY_ORDER_PRESENT = True

        elif BUY_ORDER_PRESENT == True and buy_signal == False and sell_signal == True:
            order_target_percent(ticker, -2*KELLY)
            sell_signal = False
            SELL_ORDER_PRESENT = True
            BUY_ORDER_PRESENT = False

        elif SELL_ORDER_PRESENT==False and sell_signal==True and buy_signal == False:
            order_target_percent(ticker, -1*KELLY)
            sell_signal = False
            SELL_ORDER_PRESENT = True

        elif SELL_ORDER_PRESENT==True and sell_signal==False and buy_signal == True:
            order_target_percent(ticker, 2*KELLY)
            buy_signal = False
            SELL_ORDER_PRESENT = False
            BUY_ORDER_PRESENT = True

        elif BUY_ORDER_PRESENT==True or SELL_ORDER_PRESENT==True and sq_off_signal==True:
            sq_off_signal = False
            order_target_percent(ticker, 0)
            BUY_ORDER_PRESENT = False
            SELL_ORDER_PRESENT = False
