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
        
        # Bases
        self.base_pct = 1.00
        self.after_costs_base_pct = 1.00
        self.long_base_pct = 1.00
        self.short_base_pct = 1.00

        # Returns
        self.pct = 1.00
        self.pct_after_costs = 1.00
        self.long_pct = 1.00
        self.short_pct = 1.00

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

def rebalance_portfolio(stock_graph_start_date, enter_date):
    # Get all files inside the stock_graphs folder that start with the trading date
    trading_date_files = [file for file in stock_graphs_folder if file.startswith(stock_graph_start_date)]

    # All stocks
    stocks = []

    for file in trading_date_files:
        file_path = f"stock_graphs/{setup.test_tickerslist}/trade/{file}"
        ticker = file.split('__')[-1].split('.png')[0]

        prediction = predict_graph(file_path)

        # enter_date = earliest_dates[index + 1]
        # exit_date = latest_dates[index + 1]
        stocks.append(Stock(ticker, prediction, enter_date))
        print(f"Predicted {ticker}: {prediction} {enter_date}")
    
    # Sort the stocks by prediction
    stocks.sort(key=lambda x: x.prediction, reverse=True)

    # Get the top decile and bottom decile of the portfolio
    top_decile = stocks[:int(len(stocks) * 0.1)]
    bottom_decile = stocks[-int(len(stocks) * 0.1):]

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
        try:
            exit_price = data.loc[current_date]['Close']
        except:
            print(f"Could not find exit price for {ticker} on {current_date}")
            exit_price = enter_price

    
    if position == 'long':
        stock_return = (exit_price - enter_price) / enter_price
    else:
        stock_return = (enter_price - exit_price) / enter_price

    if is_trading:
        stock_return_after_costs = stock_return - trading_costs
        portfolio.total_trades += 1
    else:
        stock_return_after_costs = stock_return

    return stock_return, stock_return_after_costs


def calculate_stocks_return(curr_stocks, prev_stocks, position, close_trades):
    total_return = 0
    total_return_after_costs = 0

    for stock in curr_stocks:
        # Checks if stock is in current portfolio
        is_trading = close_trades
        if stock in prev_stocks:
            is_trading = False

        stock_return, stock_return_after_costs = calculate_stock_return(stock.ticker, position, stock.enter_date, date, is_trading)
        total_return += stock_return
        total_return_after_costs += stock_return_after_costs
        print(f"{position} in {stock.ticker}: {'{:.2f}'.format(stock_return_after_costs * 100)}%)")
    
    total_return /= len(curr_stocks)
    total_return_after_costs /= len(curr_stocks)

    return total_return, total_return_after_costs


# New pandas dataframe with dates as index
portfolio_df = pd.DataFrame(index=all_trading_dates)

for index, date in enumerate(all_trading_dates):
    # print(date)

    # Check if date exist in the earliest dates
    if date in earliest_dates:
        if date == earliest_dates[0]:
            continue
        # If date in earliest_dates is the last row, break
        if date == earliest_dates[-1]:
            break

        # Find index of date in earliest_dates
        stock_graph_date_index = earliest_dates.index(date)
        stock_graph_date = earliest_dates[stock_graph_date_index - 1]
        rebalance_portfolio(stock_graph_date, date)    

    if portfolio.curr_long == [] and portfolio.curr_short == []:
        continue

    # Check if date exist in the latest dates
    close_trades = False
    if date in latest_dates and date != latest_dates[0]:
        close_trades = True

    # Update the portfolio percentage
    long_return, long_return_after_costs = calculate_stocks_return(portfolio.curr_long, portfolio.prev_long, 'long', close_trades)
    short_return, short_return_after_costs = calculate_stocks_return(portfolio.curr_short, portfolio.prev_short, 'short', close_trades)

    total_return = (long_return + short_return) / 2
    total_return_after_costs = (long_return_after_costs + short_return_after_costs) / 2
    

    if close_trades:
        portfolio.pct = portfolio.base_pct * (1 + total_return)
        portfolio.pct_after_costs = portfolio.after_costs_base_pct * (1 + total_return_after_costs)
        portfolio.long_pct = portfolio.long_base_pct * (1 + long_return)
        portfolio.short_pct = portfolio.short_base_pct * (1 + short_return)

        # Sets new bases
        portfolio.base_pct = portfolio.pct
        portfolio.after_costs_base_pct = portfolio.pct_after_costs
        portfolio.long_base_pct = portfolio.long_pct
        portfolio.short_base_pct = portfolio.short_pct
    else:
        portfolio.pct = portfolio.base_pct + total_return
        portfolio.pct_after_costs = portfolio.after_costs_base_pct + total_return_after_costs
        portfolio.long_pct = portfolio.long_base_pct + long_return
        portfolio.short_pct = portfolio.short_base_pct + short_return

    # Update the portfolio dataframe
    portfolio_df.loc[date, 'portfolio_pct'] = portfolio.pct
    portfolio_df.loc[date, 'portfolio_pct_after_costs'] = portfolio.pct_after_costs
    portfolio_df.loc[date, 'portfolio_long_pct'] = portfolio.long_pct
    portfolio_df.loc[date, 'portfolio_short_pct'] = portfolio.short_pct

    print(f"{date} Portfolio (after costs): {'{:.2f}'.format((portfolio.pct_after_costs - 1) * 100)}%")
    

# Save the portfolio dataframe to a csv file
portfolio_df.at[0, 'total_trades'] = portfolio.total_trades

create_folder(f"portfolios")
portfolio_df.to_csv(f"portfolios/{setup.train_tickerslist}_{setup.test_tickerslist}.csv")
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