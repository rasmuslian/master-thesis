import os
from model import predict_graph
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import setup

# Get all images in the 'trade' folder and loop over them
trade_folder = f"stock_graphs/{setup.test_tickerslist}/trade/"
trade_images = os.listdir(trade_folder)

entered_position_amount = 0 # USD
exit_position_amount = 0

def extract_info_from_filename(image_name):
    # Split the image name by '__'
    split_name = image_name.split('__')
    # Get the date and ticker symbol from the image name
    date = split_name[0]
    ticker_symbol = split_name[1].split('.')[0]
    print()
    return date, ticker_symbol

def execute_trade(ticker, date, action):
    global entered_position_amount
    global exit_position_amount
    # Find the next trading day that occurs after the date
    enter_trading_date = get_enter_trading_date(date)
    exit_trading_date = get_exit_trading_date(date)

    # Get the price of the stock on the next trading day, from the csv file
    df = pd.read_csv(f"stock_data/test/{ticker}.csv")
    enter_df = df[df['Date'] == enter_trading_date]
    exit_df = df[df['Date'] == exit_trading_date]
    enter_price = enter_df['Open'].values[0]
    exit_price = exit_df['Close'].values[0]

    entered_position_amount += enter_price if action == 'long' else -enter_price
    exit_position_amount += exit_price if action == 'long' else -exit_price

    print(f"{action} in {ticker}, {enter_trading_date} opening price of {enter_price}")
    print(f"Entered position amount: {entered_position_amount}, Exit position amount: {exit_position_amount}")
    print(f"Profit: {exit_position_amount - entered_position_amount}, {(exit_position_amount - entered_position_amount) / entered_position_amount * 100}%")
    
    

def get_enter_trading_date(date):
    # Convert date into pd.Timestamp
    date = pd.Timestamp(date)

    business_day_us = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    # Find closest business day to the date
    next_trading_date = date + business_day_us
    # Convert to format 2023-01-01
    next_trading_date = next_trading_date.strftime('%Y-%m-%d')
    return next_trading_date
    
def get_exit_trading_date(date):
    # Convert date into pd.Timestamp
    date = pd.Timestamp(date)

    business_day_us = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    # Find closest business day to the date
    next_trading_date = date + business_day_us * 5
    # Convert to format 2023-01-01
    next_trading_date = next_trading_date.strftime('%Y-%m-%d')
    return next_trading_date


for graph_image_filename in trade_images:
    # Get the date and ticker symbol from the image name
    date, ticker_symbol = extract_info_from_filename(graph_image_filename)
    
    # Get the prediction for the image
    predicted_class, probabilities = predict_graph(f"{trade_folder}/{graph_image_filename}")
    trade_action = 'long' if predicted_class == 'increasing' else 'short'

    execute_trade(ticker_symbol, date, trade_action)

