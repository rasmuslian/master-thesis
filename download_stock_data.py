import yfinance as yf
import setup

def download_stock_data(start_date, end_date, ticker_symbol):
    # Download stock data
    data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d')

    # Save data to a CSV file
    data.to_csv(f"stock_data/{ticker_symbol}.csv")

for ticker_symbol in setup.ticker_symbol_list:
    # Download stock data    
    # download_stock_data(train_start_date, test_end_date, ticker_symbol)

    # Generate training and validation data
    generate_data('train', setup.train_start_date, setup.train_end_date, ticker_symbol)

    # Generate testing data
    generate_data('test', setup.test_start_date, setup.test_end_date, ticker_symbol)