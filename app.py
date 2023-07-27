import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from heapq import nlargest
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Title and Subheader

st.title("Nifty50 Stocks Performance App")
st.subheader("Web App with Streamlit")

with st.sidebar:
    startDate = st.date_input(
        "Enter Start Date of Simulation Period",  value = dt.date(2020,10,1))

    endDate  = st.date_input("Enter End Date of Simulation Period",value = dt.date.today())


    Investment_time_in_years = (((endDate - startDate).days)/365.242)

#endDate = dt.datetime(2022, 11, 1)
#endDate = dt.date.today()

    n_days_measure_perf  = st.number_input("Enter perforformance measure for sample strategy", value = 100)

    total_investment = st.number_input("Enter initial Equity amount", value = 1000000)


    number_of_top_stockes = st.number_input('Enter number of top stocks', value = 10)


nifty50_shares = pd.read_csv("https://archives.nseindia.com/content/indices/ind_nifty50list.csv") #["Company Name"].values

#startDate = dt.datetime(2020, 10, 1)
#endDate = dt.datetime(2022, 11, 1)

stocks_name_list = []
for i in range(len(nifty50_shares)):
  stock_name =  f'Stock {i+1} ({nifty50_shares.loc[i, "Company Name"]})'
  stocks_name_list.append(stock_name)
#
# GetShareInfo_index = yf.Ticker(nifty50_shares["Symbol"][0]+".NS")
# DateIndex = GetShareInfo_index.history(start=startDate, end=endDate).index


Investment_time_in_years = ((endDate - startDate).days)/365.2425



def Benchmark_strategy(nifty50_shares, startDate, endDate, stocks_name_list, total_investment,
                       Investment_time_in_years):
    Open_Benchmark = {}
    Close_Benchmark = {}

    for i, j in enumerate(nifty50_shares["Symbol"]):
        GetShareInfo_Benchmark = yf.Ticker(j + ".NS")
        ShareInfo_Benchmark = GetShareInfo_Benchmark.history(start=startDate, end=endDate)
        Open_Benchmark[stocks_name_list[i]] = ShareInfo_Benchmark["Open"].values
        Close_Benchmark[stocks_name_list[i]] = ShareInfo_Benchmark["Close"].values

    # share_info_df = pd.DataFrame(share_info_dict, index=stock_name)
    # share_info_df
    Share_Open_Benchmark_df = pd.DataFrame(Open_Benchmark)
    Share_Close_Benchmark_df = pd.DataFrame(Close_Benchmark)

    Data_list_open_Benchmark = []
    Data_list_close_Benchmark = []
    for i in range(len(Share_Open_Benchmark_df)):
        Data_list_open_Benchmark.append(list(Share_Open_Benchmark_df.iloc[i].values))
        Data_list_close_Benchmark.append(list(Share_Close_Benchmark_df.iloc[i].values))

    arrays1 = [["Open Prices" for x in range(len(stocks_name_list))], stocks_name_list]
    cols1 = pd.MultiIndex.from_arrays(arrays1)
    arrays2 = [["Close Prices" for x in range(len(stocks_name_list))], stocks_name_list]
    cols2 = pd.MultiIndex.from_arrays(arrays2)

    Share_Open_df_Benchmark = pd.DataFrame(Data_list_open_Benchmark, columns=cols1)
    Share_Close_df_Benchmark = pd.DataFrame(Data_list_close_Benchmark, columns=cols2)
    stokes_info_df_Benchmark = Share_Open_df_Benchmark.join(Share_Close_df_Benchmark)

    # share_Qty_stocks = buy_value_day1_per_share/ opening_price_of_share     []
    buy_value_day1_per_share_Benchmark = total_investment // len(nifty50_shares)
    share_Qty_stocks_list_Benchmark = []
    for i in range(len(nifty50_shares)):
        share_Qty_stocks_list_Benchmark.append(buy_value_day1_per_share_Benchmark // Data_list_open_Benchmark[0][i])


    stocks_value_full_Benchmark = []
    equity_curve_Benchmark = []

    for s in range(len(Data_list_close_Benchmark)):
        day_wise_stocks_value_Benchmark = []
        for i, j in enumerate(Data_list_close_Benchmark[s]):
            day_wise_stocks_value_Benchmark.append(share_Qty_stocks_list_Benchmark[i] * j)
        stocks_value_full_Benchmark.append(day_wise_stocks_value_Benchmark)
        equity_curve_Benchmark.append(round(sum(day_wise_stocks_value_Benchmark), 2))

    Daily_return_Benchmark = []
    for i in range(1, len(equity_curve_Benchmark)):
        Daily_return_Benchmark.append(equity_curve_Benchmark[i] / equity_curve_Benchmark[i - 1])

    # 2. Benchmark_Allocation

    # 1. CAGR

    Vfinal_for_Benchmark = equity_curve_Benchmark[-1]
    Vbegin_for_Benchmark = equity_curve_Benchmark[0]
    CAGR_Benchmark_percentage = (((Vfinal_for_Benchmark / Vbegin_for_Benchmark) ** (
                1 / (Investment_time_in_years))) - 1) * 100

    # Volatility

    Volatility_Benchmark_percentage = ((np.std(Daily_return_Benchmark)) ** (1 / 252)) * 100

    # Sharpe Ratio

    Sharpe_Ratio_Benchmark = ((np.mean(Daily_return_Benchmark)) ** (1 / 252))

    return CAGR_Benchmark_percentage, Volatility_Benchmark_percentage, Sharpe_Ratio_Benchmark, equity_curve_Benchmark

Equal_Alloc_Buy_Hold = Benchmark_strategy(nifty50_shares, startDate, endDate, stocks_name_list, total_investment, Investment_time_in_years)

def sample_strategy(nifty50_shares, stocks_name_list, endDate, startDate, n_days_measure_perf, number_of_top_stockes,
                    total_investment, Investment_time_in_years):
    Open_sample = {}
    Close_sample = {}

    GetShareInfo_sample = yf.Ticker(nifty50_shares["Symbol"][0] + ".NS")
    total_length = len(GetShareInfo_sample.history(start=startDate, end=endDate))
    period = str(n_days_measure_perf + total_length) + "d"

    for i, j in enumerate(nifty50_shares["Symbol"]):
        GetShareInfo_sample = yf.Ticker(j + ".NS")
        GetShareInfo_sample.history(start=startDate, end=endDate)
        ShareInfo = GetShareInfo_sample.history(period=period)
        Open_sample[stocks_name_list[i]] = ShareInfo["Open"].values
        Close_sample[stocks_name_list[i]] = ShareInfo["Close"].values

    Share_Open_sample_df = pd.DataFrame(Open_sample)
    Share_Close_sample_df = pd.DataFrame(Close_sample)

    # prtformance of stocks
    prtformance_sample = {}
    for i in Open_sample:
        per = (Close_sample[i][n_days_measure_perf - 1] / Close_sample[i][0] - 1)
        prtformance_sample[i] = per

    top_n_stocks_sample_pre = nlargest(number_of_top_stockes, prtformance_sample, key=prtformance_sample.get)

    Share_sample_open_info_df = pd.DataFrame()
    Share_sample_close_info_df = pd.DataFrame()

    for i in top_n_stocks_sample_pre:
        Share_sample_open_info_df[i] = Share_Open_sample_df[i][n_days_measure_perf:]
        Share_sample_close_info_df[i] = Share_Close_sample_df[i][n_days_measure_perf:]

    Data_list_open_sample = []
    Data_list_close_sample = []

    for i in range(len(Share_sample_open_info_df)):
        Data_list_open_sample.append(list(Share_sample_open_info_df.iloc[i].values))
        Data_list_close_sample.append(list(Share_sample_close_info_df.iloc[i].values))

    arrays1_sample = [["Open Prices_sample" for x in range(len(top_n_stocks_sample_pre))], top_n_stocks_sample_pre]
    cols1_sample = pd.MultiIndex.from_arrays(arrays1_sample)
    arrays2_sample = [["Close Prices" for x in range(len(top_n_stocks_sample_pre))], top_n_stocks_sample_pre]
    cols2_sample = pd.MultiIndex.from_arrays(arrays2_sample)

    stokes_Open_df_sample = pd.DataFrame(Data_list_open_sample, columns=cols1_sample)
    stokes_Close_df_sample = pd.DataFrame(Data_list_close_sample, columns=cols2_sample)
    stokes_info_df_sample = stokes_Open_df_sample.join(stokes_Close_df_sample)

    # share_Qty_stocks_sample = buy_value_day1_per_share_sample/ opening_price_of_share_sample

    buy_value_day1_per_share_sample = total_investment // number_of_top_stockes
    share_Qty_stocks_list_sample = []
    for i in range(len(top_n_stocks_sample_pre)):
        share_Qty_stocks_list_sample.append(buy_value_day1_per_share_sample // Data_list_open_sample[0][i])

    stocks_value_full_sample = []
    equity_curve_sample = []

    for s in range(len(Data_list_close_sample)):
        day_wise_stocks_value_sample = []
        for i, j in enumerate(Data_list_close_sample[s]):
            day_wise_stocks_value_sample.append(share_Qty_stocks_list_sample[i] * j)
        stocks_value_full_sample.append(day_wise_stocks_value_sample)
        equity_curve_sample.append(round(sum(day_wise_stocks_value_sample), 2))

    Daily_return_sample = []
    for i in range(1, len(equity_curve_sample)):
        Daily_return_sample.append(equity_curve_sample[i] / equity_curve_sample[i - 1])

    # Sample Strategy

    # 1. CAGR

    Vfinal_for_sample = equity_curve_sample[-1]
    Vbegin_for_sample = equity_curve_sample[0]
    CAGR_sample_percentage = (((Vfinal_for_sample / Vbegin_for_sample) ** (1 / (Investment_time_in_years))) - 1) * 100

    # Volatility

    Volatility_sample_percentage = ((np.std(Daily_return_sample)) ** (1 / 252)) * 100

    # Sharpe Ratio

    Sharpe_Ratio_sample = ((np.mean(Daily_return_sample)) ** (1 / 252))

    Top_Stocks_Selected = []
    for i in top_n_stocks_sample_pre:
        Top_Stocks_Selected.append(i.split("(")[1].split(")")[0])

    return CAGR_sample_percentage, Volatility_sample_percentage, Sharpe_Ratio_sample, equity_curve_sample, Top_Stocks_Selected

performance_Strat = sample_strategy(nifty50_shares, stocks_name_list, endDate, startDate, n_days_measure_perf,  number_of_top_stockes, total_investment, Investment_time_in_years)


def nifty50def(startDate, endDate, total_investment, Investment_time_in_years):
    GetNiftyInfo = yf.Ticker("^NSEI")
    NiftyInfo = GetNiftyInfo.history(start=startDate, end=endDate)
    # first share qty (first day of simulation day)
    Nifty_share_qty = total_investment // NiftyInfo["Open"][0]

    # daily Value
    # Daily return
    Daily_value_closing_Nifty = []
    Daily_return_Nifty = []

    for i in range(len(NiftyInfo)):
        Daily_value_closing_Nifty.append((NiftyInfo["Close"].values[i]) * (Nifty_share_qty))

    for i in range(1, len(Daily_value_closing_Nifty)):
        Daily_return_Nifty.append(Daily_value_closing_Nifty[i] / Daily_value_closing_Nifty[i - 1])

    first_equity_value_Nifty = Daily_value_closing_Nifty[0]

    last_equity_value_Nifty = Daily_value_closing_Nifty[-1]

    # 1. CAGR

    Vfinal_for_Nifty_index = last_equity_value_Nifty
    Vbegin_for_Nifty_index = first_equity_value_Nifty
    CAGR_NiftyIndex_percentage = (((Vfinal_for_Nifty_index / Vbegin_for_Nifty_index) ** (
                1 / (Investment_time_in_years))) - 1) * 100

    # Volatility

    Volatility_NiftyIndex_percentage = ((np.std(Daily_return_Nifty)) ** (1 / 252)) * 100

    # Sharpe Ratio
    Sharpe_Ratio_NiftyIndex = ((np.mean(Daily_return_Nifty)) ** (1 / 252))

    return CAGR_NiftyIndex_percentage, Volatility_NiftyIndex_percentage, Sharpe_Ratio_NiftyIndex, Daily_value_closing_Nifty

Nifty50 = nifty50def(startDate,endDate, total_investment,Investment_time_in_years)

st.write("Sample Input:")
st.write("Input Config", '  \n',"sim_start = '", str(startDate),"'  \n", "end_date = '", str(endDate),"'  \n","n_days_measure_perf = ", n_days_measure_perf, '  \n', "top_n_stocks = ", number_of_top_stockes, "  \n", "in_eq = ", total_investment)


st.write("Sample Output:")
column01, column02 = st.columns(2)
# Group data together

plot_data = [Equal_Alloc_Buy_Hold[3], Nifty50[3], performance_Strat[3]]

df_plot = pd.DataFrame()
df_plot["Equal_Alloc_Buy_Hold"] = Equal_Alloc_Buy_Hold[3][1:]
df_plot["Nifty50"] = Nifty50[3]
df_plot["performance_Strat"] = performance_Strat[3][1:]
df_plot.index = pd.date_range(start = startDate, periods =len(Nifty50[3]) , freq = "1D")

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(df_plot)
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator(3) # every month
yearsFmt = mdates.DateFormatter('%Y-%m')
ax.grid(True)
#ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
#ax.xaxis.set_minor_locator(months)

for label in ax.get_xticklabels(which='major'):
    label.set(rotation=45, horizontalalignment='right')


column02.pyplot(fig)

df_performance = pd.DataFrame()
df_performance["Index"] = ["Equal_Alloc_Buy_Hold", "Nifty50", "Performance_Strat"]
df_performance["CAPR%"] = [Equal_Alloc_Buy_Hold[0], Nifty50[0], performance_Strat[0]]
df_performance["Volatility%"] = [Equal_Alloc_Buy_Hold[1], Nifty50[1], performance_Strat[1]]
df_performance["Sharp%"] = [Equal_Alloc_Buy_Hold[2], Nifty50[2], performance_Strat[2]]
df_performance.set_index("Index")
df_performance.style.background_gradient(axis=None, cmap='RdYlBu')

column01.dataframe(df_performance)


lst_elem = " "

for i in performance_Strat[4]:
    if performance_Strat[4][(len(performance_Strat[4])-1)] == i:
      lst_elem += i
    else:
      lst_elem += i +", "

lst_elem = "[" + lst_elem + "]"

st.write("Top Stocks Selected:")
st.write(lst_elem)



