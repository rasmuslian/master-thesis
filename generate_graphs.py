import mplfinance as mpf
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import random
import setup
from utils import clear_and_create_folder

def generate_graph(data, output_path):
    # Create a copy of the DataFrame
    data = data.copy()

    # Adds ma20 column
    ma20dict = mpf.make_addplot(data['ma20'])

    # Create a candlestick graph with the custom style and volume, and save it as an image
    mpf.plot(data,
            type='candle',
            #  mav=2,
            addplot=ma20dict,
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
    data = pd.read_csv(f"stock_data/{ticker_symbol}.csv")


    # Define a custom business day calendar
    bday_us = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    # Convert 'Datetime' to datetime format
    data['Date'] = pd.to_datetime(data['Date'], utc=True)

    # Set 'Datetime' as the index
    data.set_index('Date', inplace=True)
    
    # Add a column with 20-day moving average
    data['ma20'] = data['Adj Close'].rolling(window=20).mean()
    # Remove first 20 rows, because they don't have a 20-day moving average
    data = data.iloc[20:]

    # Filter the data based on the start and end date
    data = data.loc[start_date:end_date]

    # Group the data in chunks of 5 business days
    group_by_days = 5
    grouped_data = data.groupby(pd.Grouper(freq=bday_us * group_by_days))
    compare_list = grouped_data.nth(-1)

    loop_index = 0
    # Print the length of elements in compare_list
    print(len(compare_list))

    # Loop through the grouped data
    for name, group in grouped_data:
        # Get earliest date and latest date in the group
        earliest_date = group.index[0].date()
        latest_date = group.index[-1].date()
        interval = f"{earliest_date}__{latest_date}"
        print(earliest_date, ticker_symbol)

        
        # Get the Adj close for the current grouped data
        current_adj_close = group.iloc[-1]['Adj Close']

        if loop_index + 1 < len(compare_list):
            next_adj_close = compare_list.iloc[loop_index + 1]['Adj Close']
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
        generate_graph(group, f"output/{output_type}/{trend}/{earliest_date}__{ticker_symbol}")


"""--- PREPARES THE FOLDERS ---"""
folders = [
    "output/train/increasing",
    "output/train/decreasing",
    "output/test/increasing",
    "output/test/decreasing",
    "output/validate/increasing",
    "output/validate/decreasing",
]

for folder in folders:
    clear_and_create_folder(folder)


"""--- GENERATES THE GRAPHS ---"""
for ticker_symbol in setup.ticker_symbol_list:
    # Generate training and validation data
    generate_data('train', setup.train_start_date, setup.train_end_date, ticker_symbol)

    # Generate testing data
    generate_data('test', setup.test_start_date, setup.test_end_date, ticker_symbol)