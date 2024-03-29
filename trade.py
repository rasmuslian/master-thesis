import json
from model import predict_graph
import setup
import os
import csv
import pandas as pd

class Stock:
    def __init__(self, ticker, prediction, enter_date, exit_date):
        self.ticker = ticker
        self.prediction = prediction
        self.enter_date = enter_date
        self.exit_date = exit_date

class Portfolio:
    def __init__(self):
        self.prev_long = []
        self.prev_short = []
        self.portfolio_percentage = 1.00
        self.total_trades = 0

portfolio = Portfolio()

with open(f"stock_dates/test/{setup.test_tickerslist}.csv", 'r', encoding="utf-8") as file:
    reader = csv.DictReader(file)
    earliest_dates = []
    latest_dates = []
    for row in reader:
        earliest_dates.append(row['earliest_date'])
        latest_dates.append(row['latest_date'])

# Get all files in the stock_graphs folder
stock_graphs_folder = os.listdir(f"stock_graphs/{setup.test_tickerslist}/trade")

def make_trade(ticker, position, enter_date, exit_date):
    # Get exit and enter price
    with open(f"stock_data/test/{setup.test_tickerslist}/{ticker}.csv", 'r', encoding="utf-8") as file:
        # Use pandas dataframe to get the row with the enter date
        data = pd.read_csv(file)
        data['Date'] = pd.to_datetime(data['Date'], utc=True)
        data.set_index('Date', inplace=True)
        enter_price = data.loc[enter_date]['Open']
        exit_price = data.loc[exit_date]['Close']

    # Calculate the trade
    match position:
        case 'long':
            trade_return = (exit_price - enter_price) / enter_price
        case 'short':
            trade_return = (enter_price - exit_price) / enter_price
        
    # Calculate the portfolio percentage
    print(f"{position} | {ticker} | {enter_date} | {exit_date} | {trade_return}")
    return (1 + trade_return)

for index, trading_date in enumerate(earliest_dates[:2]):
    print(trading_date)
    # Get all files inside the stock_graphs folder that start with the trading date
    trading_date_files = [file for file in stock_graphs_folder if file.startswith(trading_date)]

    # All stocks
    stocks = []

    for file in trading_date_files:
        file_path = f"stock_graphs/{setup.test_tickerslist}/trade/{file}"
        ticker = file.split('__')[-1].split('.')[0]

        prediction = predict_graph(file_path)

        enter_date = earliest_dates[index + 1]
        exit_date = latest_dates[index + 1]
        stocks.append(Stock(ticker, prediction, enter_date, exit_date))
        print(f"Predicted {ticker}: {prediction} {enter_date} {exit_date}")
    
    # Sort the stocks by prediction
    stocks.sort(key=lambda x: x.prediction, reverse=True)

    # Get the top decile and bottom decile of the portfolio
    top_decile = stocks[:int(len(stocks) * 0.1)]
    bottom_decile = stocks[-int(len(stocks) * 0.1):]

    for stock in top_decile:
        trade_return = make_trade(stock.ticker, 'long', stock.enter_date, stock.exit_date)
        portfolio.portfolio_percentage *= trade_return
    
    for stock in bottom_decile:
        trade_return = make_trade(stock.ticker, 'short', stock.enter_date, stock.exit_date)
        portfolio.portfolio_percentage *= trade_return
    
print(f"Portfolio percentage: {'{:.2f}'.format((portfolio.portfolio_percentage - 1) * 100)}%, Total trades: {portfolio.total_trades}")


'''--- CREATES BENCHMARKS ---'''
# Create benchmarks folder if it does not exist
if not os.path.exists('benchmarks'):
    os.makedirs('benchmarks')
benchmark_filepath = "benchmarks/trade_benchmarks.csv"

# If the file does not exist, create it
if not os.path.isfile(benchmark_filepath):
    # Create the file
    df = pd.DataFrame(columns=[
        'model_name',
        'portfolio_return',
        'trades',
        'annualised_return',
        'alpha',
        'sharpe',
        'train_data',
        'trade_data',
        'benchmark_date'
        ])
    df.to_csv(benchmark_filepath, index=False)

# Reads the file
df = pd.read_csv(benchmark_filepath)

# Adds the new model to the dataframe
benchmark_date = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S")
new_row = pd.DataFrame({
    'model_name': [setup.test_model_name],
    'portfolio_return': [(portfolio.portfolio_percentage - 1) * 100],
    'trades': [portfolio.total_trades],
    'annualised_return': [''],
    'alpha': [''],
    'sharpe': [''],
    'train_data': [setup.train_tickerslist],
    'trade_data': [setup.test_tickerslist],
    'benchmark_date': [benchmark_date]
    })
df = pd.concat([df, new_row], ignore_index=True)

# Saves the dataframe to the file
df.to_csv(benchmark_filepath, index=False)