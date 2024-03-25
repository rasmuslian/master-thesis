from shutil import copyfile
import mplfinance as mpf
import pandas as pd
import datetime
import random
import setup
from utils import clear_and_create_folder, create_folder
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
    if type == 'train':
        tickerlist = setup.train_tickerslist
    else:
        tickerlist = setup.test_tickerslist

    # Read the CSV data
    data = pd.read_csv(f"stock_data/{type}/{tickerlist}/{ticker_symbol}.csv")

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

    # Loop through the grouped data
    for group in grouped_data:
        # Get earliest datetime and latest date in the group, include hours and minute
        earliest_date = group.index[0].strftime("%Y-%m-%d")
        latest_date = group.index[-1].strftime("%Y-%m-%d")
        interval = f"{earliest_date}__{latest_date}"
        print(interval, ticker_symbol)

        if loop_index + 1 < len(grouped_data):
            next_open = grouped_data[loop_index + 1].iloc[-1]['Open']
            next_adj_close = grouped_data[loop_index + 1].iloc[-1]['Adj Close']
            loop_index += 1
        else:
            break

        period_return = (next_adj_close - next_open) / next_open
        # Round period return to 2 decimal places. Ensure always two decimals, even if 0
        period_return = "{:.2f}".format(period_return * 100)

        # Set output type, and pick data for validation set.
        output_type = type

        val_cutoff_datetime = datetime.datetime.strptime(setup.val_cutoff_date, '%Y-%m-%d')
        earliest_datetime = datetime.datetime.strptime(earliest_date, '%Y-%m-%d')
        if (type == 'train' and val_cutoff_datetime < earliest_datetime):
            output_type = 'validate'
            print('Validate')
            
        # Generate the graph
        if type == 'train':
            tickerlist = setup.train_tickerslist
        else:
            tickerlist = setup.test_tickerslist

        filename = f"{interval}__{period_return}__{ticker_symbol}.png"
        output_path = f"stock_graphs/{tickerlist}/{output_type}/{filename}"
        generate_graph(group, output_path)
        # If test, copy the output file to the 'trade' folder
        if(type == 'test'):
            trade_output_path = f"stock_graphs/{tickerlist}/trade/{interval}__{ticker_symbol}.png"
            copyfile(output_path, trade_output_path)


"""--- PREPARES THE FOLDERS ---"""
folders = [
    f"stock_graphs/{setup.train_tickerslist}/train",
    f"stock_graphs/{setup.test_tickerslist}/test",
    f"stock_graphs/{setup.train_tickerslist}/validate",
    f"stock_graphs/{setup.test_tickerslist}/trade",
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