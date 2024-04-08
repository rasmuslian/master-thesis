import json
from model import predict_graph
import setup
import os
import csv
import pandas as pd

from utils import create_folder

class Stock:
    def __init__(self, ticker, prediction, enter_date):
        self.ticker = ticker
        self.prediction = prediction
        self.enter_date = enter_date

class Portfolio:
    def __init__(self):
        self.curr_long = []
        self.curr_short = []
        self.prev_long = []
        self.prev_short = []
        self.portfolio_base_percentage = 1.00
        self.portfolio_percentage = 1.00
        self.portfolio_percentage_after_costs = 1.00
        self.long_percentage = 1.00
        self.short_percentage = 1.00
        self.total_trades = 0

portfolio = Portfolio()
trading_costs = 0.002 # 20 basis points

# Gets all relevant dates
with open(f"stock_dates/test/{setup.test_tickerslist}.csv", 'r', encoding="utf-8") as file:
    reader = csv.DictReader(file)
    earliest_dates = []
    latest_dates = []
    for row in reader:
        earliest_dates.append(row['earliest_date'])
        latest_dates.append(row['latest_date'])
with open(f"stock_dates/test/{setup.test_tickerslist}_all.csv", 'r', encoding="utf-8") as file:
    reader = csv.DictReader(file)
    all_trading_dates = []
    for row in reader:
        all_trading_dates.append(row['trading_date'])

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

    # Calculate the trade and check if ticker is included in the current holdings
    match position:
        case 'long':
            trade_return = (exit_price - enter_price) / enter_price
            is_current_holding = any(stock.ticker == ticker for stock in portfolio.prev_long)
        case 'short':
            trade_return = (enter_price - exit_price) / enter_price
            is_current_holding = any(stock.ticker == ticker for stock in portfolio.prev_short)
    
    # Adjust for trading costs
    print(f"Is current holding: {is_current_holding}")
    if is_current_holding:
        trade_return_after_costs = trade_return
    else:
        trade_return_after_costs = trade_return - trading_costs
        # Add the trade to the portfolio turnover
        portfolio.total_trades += 1

    # Calculate the portfolio percentage
    print(f"{position} | {ticker} | {enter_date} | {exit_date} | {trade_return} | {trade_return_after_costs}")
    return (1 + trade_return), (1 + trade_return_after_costs)

def rebalance_portfolio(trading_date):
    # Get all files inside the stock_graphs folder that start with the trading date
    trading_date_files = [file for file in stock_graphs_folder if file.startswith(trading_date)]

    # All stocks
    stocks = []

    for file in trading_date_files:
        file_path = f"stock_graphs/{setup.test_tickerslist}/trade/{file}"
        ticker = file.split('__')[-1].split('.')[0]

        prediction = predict_graph(file_path)

        # enter_date = earliest_dates[index + 1]
        # exit_date = latest_dates[index + 1]
        stocks.append(Stock(ticker, prediction, trading_date))
        print(f"Predicted {ticker}: {prediction} {trading_date}")
    
    # Sort the stocks by prediction
    stocks.sort(key=lambda x: x.prediction, reverse=True)

    # Get the top decile and bottom decile of the portfolio
    top_decile = stocks[:int(len(stocks) * 0.1)]
    bottom_decile = stocks[-int(len(stocks) * 0.1):]

    # for stock in top_decile:
    #     trade_return, trade_return_after_costs = make_trade(stock.ticker, 'long', stock.enter_date, stock.exit_date)
    #     portfolio.portfolio_percentage *= trade_return
    #     portfolio.portfolio_percentage_after_costs *= trade_return_after_costs
    #     portfolio.long_percentage_after_costs *= trade_return_after_costs
    
    # for stock in bottom_decile:
    #     trade_return, trade_return_after_costs = make_trade(stock.ticker, 'short', stock.enter_date, stock.exit_date)
    #     portfolio.portfolio_percentage *= trade_return
        # portfolio.portfolio_percentage_after_costs *= trade_return_after_costs
        # portfolio.short_percentage_after_costs *= trade_return_after_costs

    # Update the portfolio
    portfolio.curr_long = top_decile
    portfolio.curr_short = bottom_decile

def calculate_stock_return(ticker, position, enter_date, current_date, is_trading):
    with open(f"stock_data/test/{setup.test_tickerslist}/{ticker}.csv", 'r', encoding="utf-8") as file:
        # Use pandas dataframe to get the row with the enter date
        data = pd.read_csv(file)
        data['Date'] = pd.to_datetime(data['Date'], utc=True)
        data.set_index('Date', inplace=True)
        enter_price = data.loc[enter_date]['Open']
        exit_price = data.loc[current_date]['Close']

    
    if position == 'long':
        stock_return = (exit_price - enter_price) / enter_price
    else:
        stock_return = (enter_price - exit_price) / enter_price

    if is_trading:
        stock_return_after_costs = stock_return - trading_costs
    else:
        stock_return_after_costs = stock_return

    return stock_return, stock_return_after_costs

# New pandas dataframe with dates as index
portfolio_df = pd.DataFrame(index=all_trading_dates)

for index, date in enumerate(all_trading_dates):
    print(date)

    # Check if date exist in the earliest dates
    if date in earliest_dates:
        # If date in earliest_dates is the last row, break
        if date == earliest_dates[-1]:
            break
        rebalance_portfolio(date)    


    # Check if date exist in the latest dates
    close_trades = False
    if date in latest_dates:
        close_trades = True

    # Update the portfolio percentage
    total_long_return = 0
    total_long_return_after_costs = 0
    for stock in portfolio.curr_long:
        if stock in portfolio.prev_long and not close_trades:
            is_trading = False
        else:
            is_trading = True

        stock_return, stock_return_after_costs = calculate_stock_return(stock.ticker, 'long', stock.enter_date, date, is_trading)
        total_long_return += stock_return
        total_long_return_after_costs += stock_return_after_costs
    total_long_return = total_long_return / len(portfolio.curr_long)
    total_long_return_after_costs = total_long_return_after_costs / len(portfolio.curr_long)

    total_short_return = 0
    total_short_return_after_costs = 0
    for stock in portfolio.curr_short:
        if stock in portfolio.prev_short and not close_trades:
            is_trading = False
        else:
            is_trading = True

        stock_return, stock_return_after_costs = calculate_stock_return(stock.ticker, 'short', stock.enter_date, date, is_trading)
        total_short_return += stock_return
    total_short_return = total_short_return / len(portfolio.curr_short)
    total_short_return_after_costs = total_short_return_after_costs / len(portfolio.curr_short)

    total_return = (total_long_return + total_short_return) / 2
    total_return_after_costs = (total_long_return_after_costs + total_short_return_after_costs) / 2


    if close_trades:
        portfolio.portfolio_percentage = portfolio.portfolio_base_percentage * (1 + total_return)
        portfolio.portfolio_percentage_after_costs = portfolio.portfolio_base_percentage * (1 + total_return_after_costs)
        portfolio.long_percentage = portfolio.portfolio_base_percentage * (1 + total_long_return)
    else:
        portfolio.portfolio_percentage = portfolio.portfolio_base_percentage + total_return
        portfolio.portfolio_percentage_after_costs = portfolio.portfolio_base_percentage + total_return_after_costs
        portfolio.short_percentage = portfolio.portfolio_base_percentage + total_short_return

    # Update the portfolio dataframe
    portfolio_df.loc[date, 'portfolio_percentage'] = portfolio.portfolio_percentage
    portfolio_df.loc[date, 'portfolio_long_percentage'] = portfolio.long_percentage
    portfolio_df.loc[date, 'portfolio_short_percentage'] = portfolio.short_percentage
    portfolio_df.loc[date, 'portfolio_percentage_after_costs'] = portfolio.portfolio_percentage_after_costs

# Save the portfolio dataframe to a csv file
create_folder(f"portfolios")
portfolio_df.to_csv(f"portfolios/{setup.test_tickerslist}.csv")
exit()

print(f"Portfolio percentage after costs: {'{:.2f}'.format((portfolio.portfolio_percentage_after_costs - 1) * 100)}%, Total trades: {portfolio.total_trades}")


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
        'portfolio_return_after_costs',
        'portfolio_long_return_after_costs',
        'portfolio_short_return_after_costs',
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
    'portfolio_return_after_costs': [(portfolio.portfolio_percentage_after_costs - 1) * 100],
    'portfolio_long_return_after_costs': [(portfolio.long_percentage_after_costs - 1) * 100],
    'portfolio_short_return_after_costs': [(portfolio.short_percentage_after_costs - 1) * 100],
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