import yfinance as yf
import setup
from utils import clear_and_create_folder, create_folder
import json

def download_stock_data(type, start_date, end_date, ticker_symbol):
    # Download stock data
    data = yf.download(ticker_symbol, start=start_date, end=end_date, interval=setup.data_interval)

    if type == 'train':
        tickerlist = setup.train_tickerslist
    else:
        tickerlist = setup.test_tickerslist

    # Save data to a CSV file
    data.to_csv(f"stock_data/{type}/{tickerlist}/{ticker_symbol}.csv")


# Prepares the folders
clear_and_create_folder(f"stock_data/train/{setup.train_tickerslist}")
clear_and_create_folder(f"stock_data/test/{setup.test_tickerslist}")


# Get training ticker list
with open(f"ticker_lists/{setup.train_tickerslist}.json", 'r', encoding="utf-8") as file:
    training_ticker_symbol_list = json.load(file)

for ticker_symbol in training_ticker_symbol_list:
    # Download stock data
    download_stock_data('train', setup.train_start_date, setup.train_end_date, ticker_symbol)

# Get test ticker list
with open(f"ticker_lists/{setup.test_tickerslist}.json", 'r', encoding="utf-8") as file:
    test_ticker_symbol_list = json.load(file)

for ticker_symbol in test_ticker_symbol_list:
    # Download stock data
    download_stock_data('test', setup.test_start_date, setup.test_end_date, ticker_symbol)