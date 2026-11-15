"""The column names of every indicator, from one hardcoded table, for the function and for ``df.ta``.

A column name is public API: a strategy, a saved frame or a chart reads it, so a
rename breaks callers and is a **BREAKING** change. ``EXPECTED_COLUMNS`` writes
the names out instead of deriving them from the code, so a change shows up here
as a decision. Each row is ``(indicator, keyword arguments, name, columns)``:
``columns`` is ``None`` for a Series result, otherwise the DataFrame's columns in
order, and ``name`` is the result's ``.name``.

Every row is checked twice: the function called on SPY_D, and ``df.ta`` with
``append=True``, which must append exactly those columns. The inputs come from
``tests.contracts.inputs``; an optional ``open_`` is passed only where a row asks
for it with ``OPEN``, as ``df.ta`` fetches it only on ``open="open"``.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.contracts import INDICATORS, inputs, spy

# The open column, for the indicators that take an optional one (ad gives ADo).
OPEN = "open"

EXPECTED_COLUMNS: list[tuple[str, dict, str, list[str] | None]] = [
    # candles
    ('cdl_doji', {}, 'CDL_DOJI_10_0.1', None),
    ('cdl_inside', {}, 'CDL_INSIDE', None),
    ('cdl_pattern', {'name': '2crows'}, 'CDL_PATTERN', ['CDL_2CROWS']),
    ('cdl_pattern', {'name': '3blackcrows'}, 'CDL_PATTERN', ['CDL_3BLACKCROWS']),
    ('cdl_pattern', {'name': '3inside'}, 'CDL_PATTERN', ['CDL_3INSIDE']),
    ('cdl_pattern', {'name': '3linestrike'}, 'CDL_PATTERN', ['CDL_3LINESTRIKE']),
    ('cdl_pattern', {'name': '3outside'}, 'CDL_PATTERN', ['CDL_3OUTSIDE']),
    ('cdl_pattern', {'name': '3starsinsouth'}, 'CDL_PATTERN', ['CDL_3STARSINSOUTH']),
    ('cdl_pattern', {'name': '3whitesoldiers'}, 'CDL_PATTERN', ['CDL_3WHITESOLDIERS']),
    ('cdl_pattern', {'name': 'abandonedbaby'}, 'CDL_PATTERN', ['CDL_ABANDONEDBABY']),
    ('cdl_pattern', {'name': 'advanceblock'}, 'CDL_PATTERN', ['CDL_ADVANCEBLOCK']),
    ('cdl_pattern', {'name': 'belthold'}, 'CDL_PATTERN', ['CDL_BELTHOLD']),
    ('cdl_pattern', {'name': 'breakaway'}, 'CDL_PATTERN', ['CDL_BREAKAWAY']),
    ('cdl_pattern', {'name': 'closingmarubozu'}, 'CDL_PATTERN', ['CDL_CLOSINGMARUBOZU']),
    ('cdl_pattern', {'name': 'concealbabyswall'}, 'CDL_PATTERN', ['CDL_CONCEALBABYSWALL']),
    ('cdl_pattern', {'name': 'counterattack'}, 'CDL_PATTERN', ['CDL_COUNTERATTACK']),
    ('cdl_pattern', {'name': 'darkcloudcover'}, 'CDL_PATTERN', ['CDL_DARKCLOUDCOVER']),
    ('cdl_pattern', {'name': 'doji'}, 'CDL_PATTERN', ['CDL_DOJI_10_0.1']),
    ('cdl_pattern', {'name': 'dojistar'}, 'CDL_PATTERN', ['CDL_DOJISTAR']),
    ('cdl_pattern', {'name': 'dragonflydoji'}, 'CDL_PATTERN', ['CDL_DRAGONFLYDOJI']),
    ('cdl_pattern', {'name': 'engulfing'}, 'CDL_PATTERN', ['CDL_ENGULFING']),
    ('cdl_pattern', {'name': 'eveningdojistar'}, 'CDL_PATTERN', ['CDL_EVENINGDOJISTAR']),
    ('cdl_pattern', {'name': 'eveningstar'}, 'CDL_PATTERN', ['CDL_EVENINGSTAR']),
    ('cdl_pattern', {'name': 'gapsidesidewhite'}, 'CDL_PATTERN', ['CDL_GAPSIDESIDEWHITE']),
    ('cdl_pattern', {'name': 'gravestonedoji'}, 'CDL_PATTERN', ['CDL_GRAVESTONEDOJI']),
    ('cdl_pattern', {'name': 'hammer'}, 'CDL_PATTERN', ['CDL_HAMMER']),
    ('cdl_pattern', {'name': 'hangingman'}, 'CDL_PATTERN', ['CDL_HANGINGMAN']),
    ('cdl_pattern', {'name': 'harami'}, 'CDL_PATTERN', ['CDL_HARAMI']),
    ('cdl_pattern', {'name': 'haramicross'}, 'CDL_PATTERN', ['CDL_HARAMICROSS']),
    ('cdl_pattern', {'name': 'highwave'}, 'CDL_PATTERN', ['CDL_HIGHWAVE']),
    ('cdl_pattern', {'name': 'hikkake'}, 'CDL_PATTERN', ['CDL_HIKKAKE']),
    ('cdl_pattern', {'name': 'hikkakemod'}, 'CDL_PATTERN', ['CDL_HIKKAKEMOD']),
    ('cdl_pattern', {'name': 'homingpigeon'}, 'CDL_PATTERN', ['CDL_HOMINGPIGEON']),
    ('cdl_pattern', {'name': 'identical3crows'}, 'CDL_PATTERN', ['CDL_IDENTICAL3CROWS']),
    ('cdl_pattern', {'name': 'inneck'}, 'CDL_PATTERN', ['CDL_INNECK']),
    ('cdl_pattern', {'name': 'inside'}, 'CDL_PATTERN', ['CDL_INSIDE']),
    ('cdl_pattern', {'name': 'invertedhammer'}, 'CDL_PATTERN', ['CDL_INVERTEDHAMMER']),
    ('cdl_pattern', {'name': 'kicking'}, 'CDL_PATTERN', ['CDL_KICKING']),
    ('cdl_pattern', {'name': 'kickingbylength'}, 'CDL_PATTERN', ['CDL_KICKINGBYLENGTH']),
    ('cdl_pattern', {'name': 'ladderbottom'}, 'CDL_PATTERN', ['CDL_LADDERBOTTOM']),
    ('cdl_pattern', {'name': 'longleggeddoji'}, 'CDL_PATTERN', ['CDL_LONGLEGGEDDOJI']),
    ('cdl_pattern', {'name': 'longline'}, 'CDL_PATTERN', ['CDL_LONGLINE']),
    ('cdl_pattern', {'name': 'marubozu'}, 'CDL_PATTERN', ['CDL_MARUBOZU']),
    ('cdl_pattern', {'name': 'matchinglow'}, 'CDL_PATTERN', ['CDL_MATCHINGLOW']),
    ('cdl_pattern', {'name': 'mathold'}, 'CDL_PATTERN', ['CDL_MATHOLD']),
    ('cdl_pattern', {'name': 'morningdojistar'}, 'CDL_PATTERN', ['CDL_MORNINGDOJISTAR']),
    ('cdl_pattern', {'name': 'morningstar'}, 'CDL_PATTERN', ['CDL_MORNINGSTAR']),
    ('cdl_pattern', {'name': 'onneck'}, 'CDL_PATTERN', ['CDL_ONNECK']),
    ('cdl_pattern', {'name': 'piercing'}, 'CDL_PATTERN', ['CDL_PIERCING']),
    ('cdl_pattern', {'name': 'rickshawman'}, 'CDL_PATTERN', ['CDL_RICKSHAWMAN']),
    ('cdl_pattern', {'name': 'risefall3methods'}, 'CDL_PATTERN', ['CDL_RISEFALL3METHODS']),
    ('cdl_pattern', {'name': 'separatinglines'}, 'CDL_PATTERN', ['CDL_SEPARATINGLINES']),
    ('cdl_pattern', {'name': 'shootingstar'}, 'CDL_PATTERN', ['CDL_SHOOTINGSTAR']),
    ('cdl_pattern', {'name': 'shortline'}, 'CDL_PATTERN', ['CDL_SHORTLINE']),
    ('cdl_pattern', {'name': 'spinningtop'}, 'CDL_PATTERN', ['CDL_SPINNINGTOP']),
    ('cdl_pattern', {'name': 'stalledpattern'}, 'CDL_PATTERN', ['CDL_STALLEDPATTERN']),
    ('cdl_pattern', {'name': 'sticksandwich'}, 'CDL_PATTERN', ['CDL_STICKSANDWICH']),
    ('cdl_pattern', {'name': 'takuri'}, 'CDL_PATTERN', ['CDL_TAKURI']),
    ('cdl_pattern', {'name': 'tasukigap'}, 'CDL_PATTERN', ['CDL_TASUKIGAP']),
    ('cdl_pattern', {'name': 'thrusting'}, 'CDL_PATTERN', ['CDL_THRUSTING']),
    ('cdl_pattern', {'name': 'tristar'}, 'CDL_PATTERN', ['CDL_TRISTAR']),
    ('cdl_pattern', {'name': 'unique3river'}, 'CDL_PATTERN', ['CDL_UNIQUE3RIVER']),
    ('cdl_pattern', {'name': 'upsidegap2crows'}, 'CDL_PATTERN', ['CDL_UPSIDEGAP2CROWS']),
    ('cdl_pattern', {'name': 'xsidegap3methods'}, 'CDL_PATTERN', ['CDL_XSIDEGAP3METHODS']),
    ('cdl_z', {}, 'CDL_Z_30_1', ['open_Z_30_1', 'high_Z_30_1', 'low_Z_30_1', 'close_Z_30_1']),
    ('ha', {}, 'Heikin-Ashi', ['HA_open', 'HA_high', 'HA_low', 'HA_close']),
    # cycles
    ('dsp', {}, 'DSP_14', None),
    ('ebsw', {}, 'EBSW_40_10', None),
    ('ht_dcperiod', {}, 'HT_DCPERIOD', None),
    ('ht_dcphase', {}, 'HT_DCPHASE', None),
    ('ht_phasor', {}, 'HT_PHASOR', ['HT_PHASOR_INPHASE', 'HT_PHASOR_QUAD']),
    ('ht_sine', {}, 'HT_SINE', ['HT_SINE', 'HT_LEADSINE']),
    ('ht_trendmode', {}, 'HT_TRENDMODE', None),
    ('msw', {}, 'MSW_5', ['MSW_SINE_5', 'MSW_LEAD_5']),
    # math
    ('acos', {}, 'ACOS', None),
    ('add', {}, 'ADD', None),
    ('asin', {}, 'ASIN', None),
    ('atan', {}, 'ATAN', None),
    ('ceil', {}, 'CEIL', None),
    ('cos', {}, 'COS', None),
    ('cosh', {}, 'COSH', None),
    ('div', {}, 'DIV', None),
    ('exp', {}, 'EXP', None),
    ('floor', {}, 'FLOOR', None),
    ('ln', {}, 'LN', None),
    ('log10', {}, 'LOG10', None),
    ('maxindex', {}, 'MAXINDEX_30', None),
    ('minindex', {}, 'MININDEX_30', None),
    ('minmax', {}, 'MINMAX_30', ['MIN_30', 'MAX_30']),
    ('minmaxindex', {}, 'MINMAXINDEX_30', ['MINIDX_30', 'MAXIDX_30']),
    ('mult', {}, 'MULT', None),
    ('npabs', {}, 'ABS', None),
    ('npround', {}, 'ROUND', None),
    ('rolling_max', {}, 'MAX_30', None),
    ('rolling_min', {}, 'MIN_30', None),
    ('rolling_sum', {}, 'SUM_30', None),
    ('sin', {}, 'SIN', None),
    ('sinh', {}, 'SINH', None),
    ('sqrt', {}, 'SQRT', None),
    ('sub', {}, 'SUB', None),
    ('tan', {}, 'TAN', None),
    ('tanh', {}, 'TANH', None),
    ('todeg', {}, 'TODEG', None),
    ('torad', {}, 'TORAD', None),
    ('trunc', {}, 'TRUNC', None),
    # momentum
    ('ao', {}, 'AO_5_34', None),
    ('apo', {}, 'APO_12_26', None),
    ('bias', {}, 'BIAS_SMA_26', None),
    ('bop', {}, 'BOP', None),
    ('brar', {}, 'BRAR_26', ['AR_26', 'BR_26']),
    ('cci', {}, 'CCI_14_0.015', None),
    ('cfo', {}, 'CFO_9', None),
    ('cg', {}, 'CG_10', None),
    ('cmo', {}, 'CMO_14', None),
    ('coppock', {}, 'COPC_11_14_10', None),
    ('cti', {}, 'CTI_12', None),
    ('dm', {}, 'DM_14', ['PLUS_DM_14', 'MINUS_DM_14']),
    ('er', {}, 'ER_10', None),
    ('eri', {}, 'ERI_13', ['BULLP_13', 'BEARP_13']),
    ('fisher', {}, 'FISHERT_9_1', ['FISHERT_9_1', 'FISHERTs_9_1']),
    ('fosc', {}, 'FOSC_14', None),
    ('inertia', {'refined': True}, 'INERTIAr_20_14', None),
    ('inertia', {'thirds': True}, 'INERTIAt_20_14', None),
    ('inertia', {}, 'INERTIA_20_14', None),
    ('kdj', {}, 'KDJ_9_3', ['K_9_3', 'D_9_3', 'J_9_3']),
    ('kst', {}, 'KST_10_15_20_30_10_10_10_15_9', ['KST_10_15_20_30_10_10_10_15', 'KSTs_9']),
    ('lrsi', {}, 'LRSI_14', None),
    ('macd', {'asmode': True}, 'MACDAS_12_26_9', ['MACDAS_12_26_9', 'MACDASh_12_26_9', 'MACDASs_12_26_9']),
    ('macd', {}, 'MACD_12_26_9', ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']),
    ('macdext', {}, 'MACDEXT_12_26_9', ['MACDEXT_12_26_9', 'MACDEXTs_12_26_9', 'MACDEXTh_12_26_9']),
    ('macdfix', {}, 'MACDFIX_9', ['MACDFIX_9_9', 'MACDFIXh_9_9', 'MACDFIXs_9_9']),
    ('mom', {}, 'MOM_10', None),
    ('pgo', {}, 'PGO_14', None),
    ('po', {}, 'PO_14', None),
    ('ppo', {}, 'PPO_12_26_9', ['PPO_12_26_9', 'PPOh_12_26_9', 'PPOs_12_26_9']),
    ('psl', {}, 'PSL_12', None),
    ('pvo', {}, 'PVO_12_26_9', ['PVO_12_26_9', 'PVOh_12_26_9', 'PVOs_12_26_9']),
    ('qqe', {}, 'QQE_14_5_4.236', ['QQE_14_5_4.236', 'QQE_14_5_4.236_RSIMA', 'QQEl_14_5_4.236', 'QQEs_14_5_4.236', 'QQEb_l_14_5_4.236', 'QQEb_s_14_5_4.236', 'QQEd_14_5_4.236']),
    ('roc', {}, 'ROC_10', None),
    ('rocp', {}, 'ROCP_10', None),
    ('rocr', {}, 'ROCR_10', None),
    ('rocr100', {}, 'ROCR100_10', None),
    ('rsi', {}, 'RSI_14', None),
    ('rsx', {}, 'RSX_14', None),
    ('rvgi', {}, 'RVGI_14_4', ['RVGIh_14_4', 'RVGI_14_4', 'RVGIs_14_4']),
    ('slope', {'as_angle': True, 'to_degrees': True}, 'ANGLEd_1', None),
    ('slope', {'as_angle': True}, 'ANGLEr_1', None),
    ('slope', {}, 'SLOPE_1', None),
    ('smc_sweep', {}, 'SMC_SWEEP_15_1.5', None),
    ('smi', {'scalar': 10}, 'SMI_5_20_5_10', ['SMI_5_20_5_10', 'SMIs_5_20_5_10', 'SMIo_5_20_5_10']),
    ('smi', {}, 'SMI_5_20_5', ['SMI_5_20_5', 'SMIs_5_20_5', 'SMIo_5_20_5']),
    ('squeeze', {'tr': False}, 'SQZhlr_20_2.0_20_1.5', ['SQZhlr_20_2.0_20_1.5', 'SQZ_ON_usetr0', 'SQZ_OFF_usetr0', 'SQZ_NO_usetr0']),
    ('squeeze', {}, 'SQZ_20_2.0_20_1.5', ['SQZ_20_2.0_20_1.5', 'SQZ_ON', 'SQZ_OFF', 'SQZ_NO']),
    ('squeeze_pro', {'tr': False}, 'SQZPROhlr_20_2.0_20_2_1.5_1', ['SQZPROhlr_20_2.0_20_2_1.5_1', 'SQZPRO_ON_WIDE_usetr0', 'SQZPRO_ON_NORMAL_usetr0', 'SQZPRO_ON_NARROW_usetr0', 'SQZPRO_OFF_usetr0', 'SQZPRO_NO_usetr0']),
    ('squeeze_pro', {}, 'SQZPRO_20_2.0_20_2_1.5_1', ['SQZPRO_20_2.0_20_2_1.5_1', 'SQZPRO_ON_WIDE', 'SQZPRO_ON_NORMAL', 'SQZPRO_ON_NARROW', 'SQZPRO_OFF', 'SQZPRO_NO']),
    ('stc', {}, 'STC_10_12_26_0.5', ['STC_10_12_26_0.5', 'STCmacd_10_12_26_0.5', 'STCstoch_10_12_26_0.5']),
    ('stoch', {}, 'STOCH_14_3_3', ['STOCHk_14_3_3', 'STOCHd_14_3_3']),
    ('stochf', {}, 'STOCHF_5_3', ['STOCHFk_5_3', 'STOCHFd_5_3']),
    ('stochrsi', {}, 'STOCHRSI_14_14_3_3', ['STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3']),
    ('td_seq', {'show_all': False}, 'TD_SEQ', ['TD_SEQ_UP', 'TD_SEQ_DN']),
    ('td_seq', {'show_all': True}, 'TD_SEQ', ['TD_SEQ_UPa', 'TD_SEQ_DNa']),
    ('td_seq', {}, 'TD_SEQ', ['TD_SEQ_UPa', 'TD_SEQ_DNa']),
    ('trix', {}, 'TRIX_30_9', ['TRIX_30_9', 'TRIXs_30_9']),
    ('trixh', {}, 'TRIXH_18_9', ['TRIX_18_9', 'TRIXs_18_9', 'TRIXh_18_9']),
    ('tsi', {}, 'TSI_13_25_13', ['TSI_13_25_13', 'TSIs_13_25_13']),
    ('uo', {}, 'UO_7_14_28', None),
    ('vwmacd', {}, 'VWMACD_12_26_9', ['VWMACD_12_26_9', 'VWMACDh_12_26_9', 'VWMACDs_12_26_9']),
    ('willr', {}, 'WILLR_14', None),
    # overlap
    ('alma', {}, 'ALMA_10_6.0_0.85', None),
    ('avgprice', {}, 'AVGPRICE', None),
    ('dema', {}, 'DEMA_10', None),
    ('ema', {}, 'EMA_10', None),
    ('fwma', {}, 'FWMA_10', None),
    ('hilo', {}, 'HILO_13_21', ['HILO_13_21', 'HILOl_13_21', 'HILOs_13_21']),
    ('hl2', {}, 'HL2', None),
    ('hlc3', {}, 'HLC3', None),
    ('hma', {}, 'HMA_10', None),
    ('ht_trendline', {}, 'HT_TRENDLINE', None),
    ('hwma', {}, 'HWMA_0.2_0.1_0.1', None),
    ('ichimoku', {}, 'ICHIMOKU_9_26_52', ['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26']),
    ('jma', {}, 'JMA_7_0', None),
    ('kama', {}, 'KAMA_10_2_30', None),
    ('linreg', {'angle': True}, 'LRa_14', None),
    ('linreg', {'intercept': True}, 'LRb_14', None),
    ('linreg', {'r': True}, 'LRr_14', None),
    ('linreg', {'slope': True}, 'LRm_14', None),
    ('linreg', {}, 'LR_14', None),
    ('linregangle', {}, 'LRa_14', None),
    ('linregintercept', {}, 'LRb_14', None),
    ('linregslope', {}, 'LRm_14', None),
    ('mama', {}, 'MAMA_0.5_0.05', ['MAMA_0.5_0.05', 'FAMA_0.5_0.05']),
    ('mavp', {}, 'MAVP_2_30', None),
    ('mcgd', {}, 'MCGD_10', None),
    ('medprice', {}, 'MEDPRICE', None),
    ('midpoint', {}, 'MIDPOINT_2', None),
    ('midprice', {}, 'MIDPRICE_2', None),
    ('mmar', {}, 'MMAR_10_5_6', ['MMAR_10', 'MMAR_15', 'MMAR_20', 'MMAR_25', 'MMAR_30', 'MMAR_35']),
    ('ohlc4', {}, 'OHLC4', None),
    ('pwma', {}, 'PWMA_10', None),
    ('rainbow', {}, 'RAINBOW_2_10', ['RAINBOW_1', 'RAINBOW_2', 'RAINBOW_3', 'RAINBOW_4', 'RAINBOW_5', 'RAINBOW_6', 'RAINBOW_7', 'RAINBOW_8', 'RAINBOW_9', 'RAINBOW_10']),
    ('rma', {}, 'RMA_10', None),
    ('sinwma', {}, 'SINWMA_14', None),
    ('sma', {}, 'SMA_10', None),
    ('ssf', {'poles': 2}, 'SSF_10_2', None),
    ('ssf', {'poles': 3}, 'SSF_10_3', None),
    ('supertrend', {'length': 10}, 'SUPERT_10_3.0', ['SUPERT_10_3.0', 'SUPERTd_10_3.0', 'SUPERTl_10_3.0', 'SUPERTs_10_3.0']),
    ('supertrend', {'multiplier': 2.0}, 'SUPERT_7_2.0', ['SUPERT_7_2.0', 'SUPERTd_7_2.0', 'SUPERTl_7_2.0', 'SUPERTs_7_2.0']),
    ('supertrend', {}, 'SUPERT_7_3.0', ['SUPERT_7_3.0', 'SUPERTd_7_3.0', 'SUPERTl_7_3.0', 'SUPERTs_7_3.0']),
    ('swma', {}, 'SWMA_10', None),
    ('t3', {}, 'T3_10_0.7', None),
    ('tema', {}, 'TEMA_10', None),
    ('trima', {}, 'TRIMA_10', None),
    ('tsf', {}, 'TSF_14', None),
    ('typprice', {}, 'TYPPRICE', None),
    ('vidya', {}, 'VIDYA_14', None),
    ('vwap', {}, 'VWAP_D', None),
    ('vwma', {}, 'VWMA_10', None),
    ('wcp', {}, 'WCP', None),
    ('wma', {}, 'WMA_10', None),
    ('zlma', {}, 'ZL_EMA_10', None),
    # performance
    ('drawdown', {}, 'DD', ['DD', 'DD_PCT', 'DD_LOG']),
    ('log_return', {'cumulative': True}, 'CUMLOGRET_1', None),
    ('log_return', {}, 'LOGRET_1', None),
    ('percent_return', {'cumulative': True}, 'CUMPCTRET_1', None),
    ('percent_return', {}, 'PCTRET_1', None),
    # statistics
    ('beta', {}, 'BETA_30', None),
    ('correl', {}, 'CORREL_30', None),
    ('entropy', {}, 'ENTP_10', None),
    ('kurtosis', {}, 'KURT_30', None),
    ('mad', {}, 'MAD_30', None),
    ('md', {}, 'MD_30', None),
    ('median', {}, 'MEDIAN_30', None),
    ('quantile', {}, 'QTL_30_0.5', None),
    ('skew', {}, 'SKEW_30', None),
    ('stderr', {}, 'STDERR_14', None),
    ('stdev', {}, 'STDEV_30', None),
    ('tos_stdevall', {}, 'TOS_STDEVALL', ['TOS_STDEVALL_LR', 'TOS_STDEVALL_L_1', 'TOS_STDEVALL_U_1', 'TOS_STDEVALL_L_2', 'TOS_STDEVALL_U_2', 'TOS_STDEVALL_L_3', 'TOS_STDEVALL_U_3']),
    ('variance', {}, 'VAR_30', None),
    ('zscore', {}, 'ZS_30', None),
    # trend
    ('adx', {}, 'ADX_14', ['ADX_14', 'DMP_14', 'DMN_14']),
    ('adxr', {}, 'ADXR_14', ['ADXR_14', 'DMP_14', 'DMN_14']),
    ('amat', {}, 'AMATe_8_21_2', ['AMATe_LR_8_21_2', 'AMATe_SR_8_21_2']),
    ('aroon', {}, 'AROON_14', ['AROOND_14', 'AROONU_14', 'AROONOSC_14']),
    ('chop', {'ln': False}, 'CHOP_14_1_100', None),
    ('chop', {'ln': True}, 'CHOPln_14_1_100', None),
    ('cksp', {'tvmode': False}, 'CKSP_10_3_20', ['CKSPl_10_3_20', 'CKSPs_10_3_20']),
    ('cksp', {'tvmode': True}, 'CKSP_10_1_9', ['CKSPl_10_1_9', 'CKSPs_10_1_9']),
    ('cpr', {'levels': 'basic'}, 'CPR', ['CPR_TC', 'CPR_PIVOT', 'CPR_BC', 'CPR_WIDTH', 'CPR_WIDTH_PCT', 'CPR_WIDTH_CLASS', 'CPR_POSITION']),
    ('cpr', {'method': 'camarilla', 'levels': 'all'}, 'CPR', ['CPR_TC', 'CPR_PIVOT', 'CPR_BC', 'CPR_R1', 'CPR_R2', 'CPR_S1', 'CPR_S2', 'CPR_R3', 'CPR_R4', 'CPR_S3', 'CPR_S4', 'CPR_WIDTH', 'CPR_WIDTH_PCT', 'CPR_WIDTH_CLASS', 'CPR_POSITION']),
    ('cpr', {'method': 'classic', 'levels': 'extended'}, 'CPR', ['CPR_TC', 'CPR_PIVOT', 'CPR_BC', 'CPR_R1', 'CPR_R2', 'CPR_S1', 'CPR_S2', 'CPR_R3', 'CPR_R4', 'CPR_S3', 'CPR_S4', 'CPR_WIDTH', 'CPR_WIDTH_PCT', 'CPR_WIDTH_CLASS', 'CPR_POSITION']),
    ('cpr', {'method': 'classic', 'levels': 'standard'}, 'CPR', ['CPR_TC', 'CPR_PIVOT', 'CPR_BC', 'CPR_R1', 'CPR_R2', 'CPR_S1', 'CPR_S2', 'CPR_WIDTH', 'CPR_WIDTH_PCT', 'CPR_WIDTH_CLASS', 'CPR_POSITION']),
    ('cpr', {'method': 'fibonacci', 'levels': 'all'}, 'CPR', ['CPR_TC', 'CPR_PIVOT', 'CPR_BC', 'CPR_R1', 'CPR_R2', 'CPR_S1', 'CPR_S2', 'CPR_R3', 'CPR_S3', 'CPR_WIDTH', 'CPR_WIDTH_PCT', 'CPR_WIDTH_CLASS', 'CPR_POSITION']),
    ('cpr', {'method': 'woodie', 'levels': 'standard'}, 'CPR', ['CPR_TC', 'CPR_PIVOT', 'CPR_BC', 'CPR_R1', 'CPR_R2', 'CPR_S1', 'CPR_S2', 'CPR_WIDTH', 'CPR_WIDTH_PCT', 'CPR_WIDTH_CLASS', 'CPR_POSITION']),
    ('cpr', {'price_position': True}, 'CPR', ['CPR_TC', 'CPR_PIVOT', 'CPR_BC', 'CPR_R1', 'CPR_R2', 'CPR_S1', 'CPR_S2', 'CPR_WIDTH', 'CPR_WIDTH_PCT', 'CPR_WIDTH_CLASS', 'CPR_POSITION']),
    ('cpr', {'virgin_cpr': False}, 'CPR', ['CPR_TC', 'CPR_PIVOT', 'CPR_BC', 'CPR_R1', 'CPR_R2', 'CPR_S1', 'CPR_S2', 'CPR_WIDTH', 'CPR_WIDTH_PCT', 'CPR_WIDTH_CLASS', 'CPR_POSITION']),
    ('cpr', {'virgin_cpr': True, 'virgin_lookforward': 10}, 'CPR', ['CPR_TC', 'CPR_PIVOT', 'CPR_BC', 'CPR_R1', 'CPR_R2', 'CPR_S1', 'CPR_S2', 'CPR_WIDTH', 'CPR_WIDTH_PCT', 'CPR_WIDTH_CLASS', 'CPR_POSITION', 'CPR_VIRGIN']),
    ('cpr', {'virgin_cpr': True, 'virgin_lookforward': 5}, 'CPR', ['CPR_TC', 'CPR_PIVOT', 'CPR_BC', 'CPR_R1', 'CPR_R2', 'CPR_S1', 'CPR_S2', 'CPR_WIDTH', 'CPR_WIDTH_PCT', 'CPR_WIDTH_CLASS', 'CPR_POSITION', 'CPR_VIRGIN']),
    ('cpr', {'width_analysis': True}, 'CPR', ['CPR_TC', 'CPR_PIVOT', 'CPR_BC', 'CPR_R1', 'CPR_R2', 'CPR_S1', 'CPR_S2', 'CPR_WIDTH', 'CPR_WIDTH_PCT', 'CPR_WIDTH_CLASS', 'CPR_POSITION']),
    ('decay', {'mode': 'exp'}, 'EXPDECAY_5', None),
    ('decay', {}, 'LDECAY_5', None),
    ('decreasing', {'length': 3, 'strict': True}, 'SDEC_3', None),
    ('decreasing', {}, 'DEC_1', None),
    ('dpo', {}, 'DPO_20', None),
    ('dx', {}, 'DX_14', None),
    ('edecay', {}, 'EDECAY_5', None),
    ('increasing', {'length': 3, 'strict': True}, 'SINC_3', None),
    ('increasing', {}, 'INC_1', None),
    ('long_run', {}, 'LR_2', None),
    ('minus_dm', {}, 'MINUS_DM_14', None),
    ('plus_dm', {}, 'PLUS_DM_14', None),
    ('pmax', {}, 'PMAX_E_10_3.0', None),
    ('psar', {}, 'PSAR_0.02_0.2', ['PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSARaf_0.02_0.2', 'PSARr_0.02_0.2']),
    ('qstick', {}, 'QS_10', None),
    ('sarext', {}, 'SAREXT', None),
    ('short_run', {}, 'SR_2', None),
    ('tsignals', {}, 'TS', ['TS_Trends', 'TS_Trades', 'TS_Entries', 'TS_Exits']),
    ('ttm_trend', {}, 'TTMTREND_6', ['TTM_TRND_6']),
    ('vhf', {}, 'VHF_28', None),
    ('vortex', {}, 'VTX_14', ['VTXP_14', 'VTXM_14']),
    ('xsignals', {'xa': 70, 'xb': 30}, 'XS', ['TS_Trends', 'TS_Trades', 'TS_Entries', 'TS_Exits']),
    # volatility
    ('aberration', {}, 'ABER_5_15', ['ABER_ZG_5_15', 'ABER_SG_5_15', 'ABER_XG_5_15', 'ABER_ATR_5_15']),
    ('accbands', {}, 'ACCBANDS_20', ['ACCBL_20', 'ACCBM_20', 'ACCBU_20']),
    ('atr', {}, 'ATRr_14', None),
    ('avolume', {}, 'AVOLUME_20', None),
    ('bbands', {}, 'BBANDS_5_2.0', ['BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0']),
    ('ce', {}, 'CE_22_3.0', ['CE_L_22_3.0', 'CE_S_22_3.0']),
    ('cvi', {}, 'CVI_10', None),
    ('donchian', {}, 'DC_20_20', ['DCL_20_20', 'DCM_20_20', 'DCU_20_20']),
    ('hvol', {}, 'HVOL_20', None),
    ('hwc', {}, 'HWC', ['HWM', 'HWU', 'HWL']),
    ('kc', {}, 'KCe_20_2', ['KCLe_20_2', 'KCBe_20_2', 'KCUe_20_2']),
    ('massi', {}, 'MASSI_9_25', None),
    ('natr', {}, 'NATR_14', None),
    ('pdist', {}, 'PDIST', None),
    ('rvi', {'refined': True}, 'RVIr_14', None),
    ('rvi', {'thirds': True}, 'RVIt_14', None),
    ('rvi', {}, 'RVI_14', None),
    ('thermo', {}, 'THERMO_20_2_0.5', ['THERMO_20_2_0.5', 'THERMOma_20_2_0.5', 'THERMOl_20_2_0.5', 'THERMOs_20_2_0.5']),
    ('true_range', {}, 'TRUERANGE_1', None),
    ('ui', {'everget': True}, 'UIe_14', None),
    ('ui', {}, 'UI_14', None),
    # volume
    ('ad', {'open_': OPEN}, 'ADo', None),
    ('ad', {}, 'AD', None),
    ('adosc', {}, 'ADOSC_3_10', None),
    ('aobv', {}, 'AOBVe_4_12_2_2_2', ['OBV', 'OBV_min_2', 'OBV_max_2', 'OBVe_4', 'OBVe_12', 'AOBV_LR_2', 'AOBV_SR_2']),
    ('cmf', {}, 'CMF_20', None),
    ('efi', {}, 'EFI_13', None),
    ('emv', {}, 'EMV', None),
    ('eom', {}, 'EOM_14_100000000', None),
    ('kvo', {}, 'KVO_34_55_13', ['KVO_34_55_13', 'KVOs_34_55_13']),
    ('marketfi', {}, 'MARKETFI', None),
    ('mfi', {}, 'MFI_14', None),
    ('nvi', {}, 'NVI_1', None),
    ('obv', {}, 'OBV', None),
    ('pvi', {}, 'PVI_1', None),
    ('pvol', {}, 'PVOL', None),
    ('pvr', {}, 'PVR', None),
    ('pvt', {}, 'PVT', None),
    ('vfi', {}, 'VFI_130', None),
    ('vosc', {}, 'VOSC_14_28', None),
    ('vp', {}, 'VP_10', ['low_close', 'mean_close', 'high_close', 'pos_volume', 'neg_volume', 'total_volume']),
    ('wad', {}, 'WAD', None),
]


def _rows():
    return [pytest.param(*row, id="-".join([row[0], *(f"{k}={v}" for k, v in row[1].items())])) for row in EXPECTED_COLUMNS]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return spy(400)


def _function_kwargs(indicator: str, kwargs: dict, df: pd.DataFrame) -> dict:
    """The function's inputs from *df*, without an optional open_ unless the row asks, and the row's options."""
    series = inputs(indicator, df)
    if "open_" in series and inspect.signature(getattr(ta, indicator)).parameters["open_"].default is not inspect.Parameter.empty:
        del series["open_"]
    options = {k: (df[v] if k == "open_" and v == OPEN else v) for k, v in kwargs.items()}
    return {**series, **options}


def _accessor_kwargs(indicator: str, kwargs: dict, df: pd.DataFrame) -> dict:
    """df.ta fetches the OHLCV columns itself; the other inputs (a benchmark, fast/slow, a trend) are passed on."""
    extra = {k: v for k, v in inputs(indicator, df).items() if k not in ("open_", "high", "low", "close", "volume")}
    options = {("open" if k == "open_" else k): v for k, v in kwargs.items()}
    return {**extra, **options}


def _check(result, name: str, columns: list[str] | None, where: str) -> None:
    if columns is None:
        assert isinstance(result, pd.Series), f"{where}: expected a Series, got {type(result).__name__}"
    else:
        assert isinstance(result, pd.DataFrame), f"{where}: expected a DataFrame, got {type(result).__name__}"
        assert list(result.columns) == columns, f"{where}: columns {list(result.columns)} != {columns}"
    assert result.name == name, f"{where}: name {result.name!r} != {name!r}"


@pytest.mark.parametrize(("indicator", "kwargs", "name", "columns"), _rows())
def test_function_names_its_columns(indicator, kwargs, name, columns, frame):
    result = getattr(ta, indicator)(**_function_kwargs(indicator, kwargs, frame))
    _check(result, name, columns, f"{indicator}({kwargs})")


@pytest.mark.parametrize(("indicator", "kwargs", "name", "columns"), _rows())
def test_accessor_appends_those_columns(indicator, kwargs, name, columns, frame):
    df = frame.copy()
    before = list(df.columns)
    result = getattr(df.ta, indicator)(**_accessor_kwargs(indicator, kwargs, df), append=True)
    _check(result, name, columns, f"df.ta.{indicator}({kwargs})")
    appended = [c for c in df.columns if c not in before]
    assert appended == (columns or [name]), f"df.ta.{indicator}({kwargs}) appended {appended}"


def test_every_indicator_has_a_row():
    """A new indicator gets its names written down here."""
    covered = {row[0] for row in EXPECTED_COLUMNS}
    missing = sorted(set(INDICATORS) - covered - {"ma"})
    assert not missing, f"no EXPECTED_COLUMNS row for {missing}"
    # ma is a dispatcher: every moving average it reaches has a row of its own.
    assert set(ta.ma()) <= covered, f"ma() reaches {sorted(set(ta.ma()) - covered)} without a row"
