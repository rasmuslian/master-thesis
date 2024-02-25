from shutil import copyfile
import mplfinance as mpf
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import random
import setup
from utils import clear_and_create_folder
import json

def generate_graph(data, output_path):
    # Create a copy of the DataFrame
    data = data.copy()

    # Adds moving average line to the graph
    ma_dict = mpf.make_addplot(data[f"ma{setup.ma_period}"])

    # Create a candlestick graph with the custom style and volume, and save it as an image
    mpf.plot(data,
            type='candle',
            #  mav=2,
            addplot=ma_dict,
            style='yahoo',
            figratio=(1,1),
            volume=True,
            # tight_layout=True,
            axisoff=True,
            savefig=dict(fname=output_path,bbox_inches="tight"),
            update_width_config=dict(candle_linewidth=5)
            )


def generate_data(type, start_date, end_date, ticker_symbol):
    # Read the CSV data
    data = pd.read_csv(f"stock_data/{type}/{ticker_symbol}.csv")

    # Convert 'Datetime' to datetime format
    match setup.data_interval:
        case '1h' | '90m' | '60m' | '30m' | '15m' | '5m' | '2m' | '1m':
            date_column = 'Datetime'
        case _:
            date_column = 'Date'

    data['Date'] = pd.to_datetime(data[date_column], utc=True)

    # Set 'Datetime' as the index
    data.set_index('Date', inplace=True)
    
    # Add a column with moving average
    data[f"ma{setup.ma_period}"] = data['Adj Close'].rolling(window=setup.ma_period).mean()
    # Remove first n rows, because they don't have a n-period moving average
    data = data.iloc[setup.ma_period:]

    # Filter the data based on the start and end date
    data = data.loc[start_date:end_date]

    # Take the first five rows of the pandas data and add to list, and then repeat
    group_by_chunks = setup.data_groupby
    grouped_data = [data.iloc[i:i+group_by_chunks] for i in range(0, len(data), group_by_chunks)]

    loop_index = 0

    print(grouped_data)

    # Loop through the grouped data
    for group in grouped_data:
        # Get earliest datetime and latest date in the group, include hours and minute
        earliest_date = group.index[0].strftime("%Y-%m-%d_%H%M")
        latest_date = group.index[-1].strftime("%Y-%m-%d_%H%M")
        interval = f"{earliest_date}__{latest_date}"
        print(interval, ticker_symbol)

        
        # Get the Adj close for the current grouped data
        current_adj_close = group.iloc[-1]['Adj Close']

        if loop_index + 1 < len(grouped_data):
            next_adj_close = grouped_data[loop_index + 1].iloc[-1]['Adj Close']
            loop_index += 1
        else:
            break

        # Check if the Adj close for the next grouped data is greater than the Adj close for the current grouped data
        if next_adj_close > current_adj_close:
            # If yes, then set the trend as 'Increase'
            trend = 'increasing'
        else:
            # If no, then set the trend as 'Decrease'
            trend = 'decreasing'

        # Set output type. If train, pick 30% of the data for validation
        output_type = type
        if(type == 'train' and random.random() < 0.3):
            output_type = 'validate'
            
        # Generate the graph
        filename = f"{interval}__{ticker_symbol}.png"
        output_path = f"stock_graphs/{output_type}/{trend}/{filename}"
        generate_graph(group, output_path)
        # If test, copy the output file to the 'trade' folder
        if(type == 'test'):
            trade_output_path = f"stock_graphs/trade/{filename}"
            copyfile(output_path, trade_output_path)


"""--- PREPARES THE FOLDERS ---"""
folders = [
    "stock_graphs/train/increasing",
    "stock_graphs/train/decreasing",
    "stock_graphs/test/increasing",
    "stock_graphs/test/decreasing",
    "stock_graphs/validate/increasing",
    "stock_graphs/validate/decreasing",
    "stock_graphs/trade",
]

for folder in folders:
    clear_and_create_folder(folder)


"""--- GENERATES THE GRAPHS ---"""
# Generates training and validation data
with open(f"ticker_lists/{setup.train_tickerslist}.json", 'r', encoding="utf-8") as file:
    training_ticker_symbol_list = json.load(file)

for ticker_symbol in training_ticker_symbol_list:
    print(f"Generating data for {ticker_symbol}, training")
    generate_data('train', setup.train_start_date, setup.train_end_date, ticker_symbol)

# Generates testing data
with open(f"ticker_lists/{setup.test_tickerslist}.json", 'r', encoding="utf-8") as file:
    test_ticker_symbol_list = json.load(file)

for ticker_symbol in test_ticker_symbol_list:
    print(f"Generating data for {ticker_symbol}, testing")
    generate_data('test', setup.test_start_date, setup.test_end_date, ticker_symbol)