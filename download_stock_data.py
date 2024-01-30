import yfinance as yf
import setup
from utils import clear_and_create_folder

def download_stock_data(start_date, end_date, ticker_symbol):
    # Download stock data
    data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d')

    # Save data to a CSV file
    data.to_csv(f"stock_data/{ticker_symbol}.csv")


# Create the 'stock_data' folder if it doesn't exist
clear_and_create_folder("stock_data")

for ticker_symbol in setup.ticker_symbol_list:
    # Download stock data
    download_stock_data(setup.train_start_date, setup.test_end_date, ticker_symbol)