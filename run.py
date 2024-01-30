import pathlib
from download_stock_data import download_stock_data
from utils import clear_folder, prepare_folders
from generate_data import generate_data

train_start_date = '2018-01-01'
train_end_date = '2022-12-31'

test_start_date = '2023-01-01'
test_end_date = '2024-01-28'

ticker_symbol_list = [
    'AAPL',
    'MSFT',
    'AMZN',
    'GOOG',
    'TSLA',
    'NVDA',
    'ADBE',
    'NFLX',
    'CSCO',
    'PEP',
]

prepare_folders()
# pathlib.Path("stock_data").mkdir(parents=True, exist_ok=True)
# clear_folder("stock_data")

for ticker_symbol in ticker_symbol_list:
    # Download stock data    
    # download_stock_data(train_start_date, test_end_date, ticker_symbol)

    # Generate training and validation data
    generate_data('train', train_start_date, train_end_date, ticker_symbol)

    # Generate testing data
    generate_data('test', test_start_date, test_end_date, ticker_symbol)