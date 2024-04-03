from shutil import copyfile
import mplfinance as mpf
import pandas as pd
import datetime
import random
import setup
from utils import clear_and_create_folder, create_folder, check_if_folder_exists
import json
import csv

def generate_graph(data, output_path):
    # Create a copy of the DataFrame
    data = data.copy()

    # Adds moving average line to the graph
    ma_dict = mpf.make_addplot(data[f"ma{setup.ma_period}"])

    # Define market colors and style
    market_colors = mpf.make_marketcolors(up='green', down='red', edge='white', wick={'up':'green', 'down':'red'}, volume={'up': 'green', 'down': 'red'})
    style = mpf.make_mpf_style(marketcolors=market_colors)

    # Determine the min and max values for the y-axis
    price_min = min(data['Low'].min(), data[f"ma{setup.ma_period}"].min())
    price_max = max(data['High'].max(), data[f"ma{setup.ma_period}"].max())

    # Adjusting x-axis limits by adding a margin
    x_limits = [data.index[0] - pd.Timedelta(hours=14), data.index[-1] + pd.Timedelta(hours=14)]

    # Setup
    width_config = dict(candle_linewidth=5, candle_width=1, volume_linewidth=0, line_width=4)

    # Create a candlestick graph with the custom style and volume, and save it as an image
    mpf.plot(data,
             type='candle',
             addplot=ma_dict,
             style=style,
             yscale='linear',
             ylim=(price_min, price_max),  # Adjust y-axis limits
             xlim=x_limits,  # Setting x-axis limits
             tight_layout=True,  # This will remove whitespace around the figure
             scale_padding=dict(left=0, bottom=0, top=0, right=0),  # This will scale the plot to fit the figure
             figratio=(1,2),
             volume=True,
             axisoff=True,
             savefig=dict(fname=output_path, bbox_inches="tight"),
             scale_width_adjustment=dict(candle=0.8),
             update_width_config=width_config)


def generate_data(type, start_date, end_date, ticker_symbol):
    if type == 'train':
        tickerlist = setup.train_tickerslist
    else:
        tickerlist = setup.test_tickerslist

    # Read the CSV data
    data_path_type = type
    if type == 'validate':
        data_path_type = 'train'
    data = pd.read_csv(f"stock_data/{data_path_type}/{tickerlist}/{ticker_symbol}.csv")

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
    data[f"ma{setup.ma_period}"] = data['Close'].rolling(window=setup.ma_period).mean()
    # Remove first n rows, because they don't have a n-period moving average
    data = data.iloc[setup.ma_period:]

    # Filter the data based on the start and end date
    data = data.loc[start_date:end_date]

    '''--- ENSURES TRADING DATES SYNC, FOR SORTING ---'''
    # Read the trading dates CSV file
    with open(f"stock_dates/{type}/{tickerlist}.csv", 'r', encoding="utf-8") as file:
        reader = csv.DictReader(file)
        earliest_dates = [row['earliest_date'] for row in reader]    
        
    # Check the first rows earliest_date of the data. If it isn't included in the trading_dates, remove the row. Keep doing this until the earliest_date is included in the trading_dates
    while len(data.index) > 0 and data.index[0].strftime("%Y-%m-%d") not in earliest_dates:
        data = data.iloc[1:]
        print(f"Removed row from {ticker_symbol}")

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
            next_open = grouped_data[loop_index + 1].iloc[0]['Open']
            next_adj_close = grouped_data[loop_index + 1].iloc[-1]['Close']
            loop_index += 1
            print(next_open, next_adj_close)
        else:
            break

        period_return = (next_adj_close - next_open) / next_open
        # Round period return to 2 decimal places. Ensure always two decimals, even if 0
        period_return = "{:.2f}".format(period_return * 100)
            
        # Generate the graph
        if type == 'test':
            tickerlist = setup.test_tickerslist
        else:
            tickerlist = setup.train_tickerslist
        
        filename = f"{interval}__{period_return}__{ticker_symbol}.png"
        output_path = f"stock_graphs/{tickerlist}/{type}/{filename}"
        generate_graph(group, output_path)
        # If test, copy the output file to the 'trade' folder
        if(type == 'test'):
            trade_output_path = f"stock_graphs/{tickerlist}/trade/{interval}__{ticker_symbol}.png"
            copyfile(output_path, trade_output_path)


def generate_flow(type, tickerslist, start_date, end_date):
    folder_exist = check_if_folder_exists(f"stock_graphs/{tickerslist}/{type}")
    
    if not folder_exist:
        clear_and_create_folder(f"stock_graphs/{tickerslist}/{type}")
        if type == 'test':
            clear_and_create_folder(f"stock_graphs/{tickerslist}/trade")

        with open(f"ticker_lists/{tickerslist}.json", 'r', encoding="utf-8") as file:
            ticker_symbol_list = json.load(file)

        for ticker_symbol in ticker_symbol_list:
            print(f"Generating data for {ticker_symbol}, {type}")
            generate_data(type, start_date, end_date, ticker_symbol)
    else:
        print(f"{type} stock graphs already exists")


"""--- GENERATES THE GRAPHS ---"""
generate_flow('train', setup.train_tickerslist, setup.train_start_date, setup.train_end_date)
generate_flow('validate', setup.train_tickerslist, setup.validate_start_date, setup.validate_end_date)
generate_flow('test', setup.test_tickerslist, setup.test_start_date, setup.test_end_date)