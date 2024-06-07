import csv
import json
import pandas as pd
from utils import create_folder

benchmark = 'sp500'

class EqualWeightBenchmark:
    def __init__(self):        
        self.total_return = 1.00
        self.timeseries = []

equal_benchmark = EqualWeightBenchmark()

# First get all daily dates and loop over
with open(f"stock_dates/test/{benchmark}_all.csv", 'r', encoding="utf-8") as file:
    reader = csv.DictReader(file)
    all_trading_dates = [row['trading_date'] for row in reader]

# Load ticker list once
with open(f"ticker_lists/{benchmark}.json", 'r', encoding="utf-8") as file:
    ticker_symbol_list = json.load(file)

# Load each ticker's data into memory once
ticker_dfs = {}
for ticker_symbol in ticker_symbol_list:
    with open(f"stock_data/test/{benchmark}/{ticker_symbol}.csv", 'r', encoding="utf-8") as file:
        ticker_dfs[ticker_symbol] = pd.read_csv(file, index_col=0)
        ticker_dfs[ticker_symbol].index = pd.to_datetime(ticker_dfs[ticker_symbol].index)
        ticker_dfs[ticker_symbol] = ticker_dfs[ticker_symbol].loc[all_trading_dates[0]:all_trading_dates[-1]]


for trading_date in all_trading_dates:
    available_stocks = 0
    stock_returns = 0
    total_return = 0

    # If first trading date, skip
    if trading_date == all_trading_dates[0]:
        continue
    
    for ticker_symbol in ticker_symbol_list:
        # Check if ticker_dfs[ticker_symbol] has the trading date
        if trading_date in ticker_dfs[ticker_symbol].index:
            available_stocks += 1
            enter_price = ticker_dfs[ticker_symbol].iloc[0]['Close']
            exit_price = ticker_dfs[ticker_symbol].loc[trading_date]['Close']
            stock_returns += (exit_price - enter_price) / enter_price

    total_return = stock_returns / available_stocks
    equal_benchmark.total_return = total_return

    equal_benchmark.timeseries.append([trading_date, 1 + total_return])

    print(f"{trading_date}: {(equal_benchmark.total_return)}")

# Save timeseries to csv
create_folder(f"portfolio_data/equal_weight_benchmarks")
with open(f"portfolio_data/equal_weight_benchmarks/{benchmark}.csv", 'w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(['Date', 'Close'])
    for row in equal_benchmark.timeseries:
        writer.writerow(row)