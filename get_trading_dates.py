import setup
import pandas as pd
import csv
from utils import clear_and_create_folder, check_if_folder_exists

def get_trading_dates(type, start_date, end_date, ticker_symbol):
    # clear_and_create_folder(f"stock_dates/{type}")

    if type == 'test':
        tickerlist = setup.test_tickerslist
    else:
        tickerlist = setup.train_tickerslist

    folder_exist = check_if_folder_exists(f"stock_dates/{type}")
    if not folder_exist:
        clear_and_create_folder(f"stock_dates/{type}")

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

    # Get all trading days variable, in datetime format from dataframe
    all_trading_dates = data.copy()    
    
    # Add a column with moving average
    data[f"ma{setup.ma_period}"] = data['Close'].rolling(window=setup.ma_period).mean()
    # Remove first n rows, because they don't have a n-period moving average
    data = data.iloc[setup.ma_period:]
    all_trading_dates = all_trading_dates.iloc[setup.ma_period:]

    # Filter the data based on the start and end date
    data = data.loc[start_date:end_date]
    all_trading_dates = all_trading_dates.loc[start_date:end_date]

    # Take the first five rows of the pandas data and add to list, and then repeat
    group_by_chunks = setup.data_groupby
    grouped_data = [data.iloc[i:i+group_by_chunks] for i in range(0, len(data), group_by_chunks)]

    loop_index = 0

    # Loop through the grouped data
    dates = []
    for group in grouped_data:
        # Get earliest datetime and latest date in the group
        earliest_date = group.index[0].strftime("%Y-%m-%d")
        latest_date = group.index[-1].strftime("%Y-%m-%d")
        dates.append((earliest_date, latest_date))
    
    # Open the CSV file in write mode
    with open(f"stock_dates/{type}/{tickerlist}.csv", 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(['earliest_date', 'latest_date'])

        # Write the dates to the CSV file
        for date in dates:
            writer.writerow(date)
    
    # Get trading date for the base ticker
    if type == 'test':
        trading_dates = []
        for trading_date_obj in all_trading_dates.groupby(all_trading_dates.index):
            trading_date = trading_date_obj[0].strftime("%Y-%m-%d")
            trading_dates.append(trading_date)

        # Open the CSV file in write mode
        with open(f"stock_dates/{type}/{tickerlist}_all.csv", 'w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(['trading_date'])

            # Write the dates to the CSV file
            for trading_date in trading_dates:
                writer.writerow([trading_date])

# Get training ticker list
get_trading_dates('train', setup.train_start_date, setup.train_end_date, setup.train_ticker_trading_base)
get_trading_dates('validate', setup.validate_start_date, setup.validate_end_date, setup.validate_ticker_trading_base)
get_trading_dates('test', setup.test_start_date, setup.test_end_date, setup.test_ticker_trading_base)