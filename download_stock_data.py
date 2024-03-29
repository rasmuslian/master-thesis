import yfinance as yf
import setup
from utils import clear_and_create_folder, create_folder, check_if_folder_exists
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

def download_flow(type, tickerlist, start_date, end_date):
    clear_and_create_folder(f"stock_data/{type}/{tickerlist}")

    with open(f"ticker_lists/{tickerlist}.json", 'r', encoding="utf-8") as file:
        ticker_symbol_list = json.load(file)

    for ticker_symbol in ticker_symbol_list:
        # Download stock data
        download_stock_data(type, start_date, end_date, ticker_symbol)


# Check if folders already exists
train_folder_exist = check_if_folder_exists(f"stock_data/train/{setup.train_tickerslist}")
test_folder_exist = check_if_folder_exists(f"stock_data/test/{setup.test_tickerslist}")

# Get training ticker list
if not train_folder_exist:
    download_flow('train', setup.train_tickerslist, setup.train_start_date, setup.validate_end_date)
else:
    print('Training stock data already exists')

# Get test ticker list
if not test_folder_exist:
    download_flow('test', setup.test_tickerslist, setup.test_start_date, setup.test_end_date)
else:
    print('Test stock data already exists')