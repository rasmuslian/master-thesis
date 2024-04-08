import yfinance as yf
import setup
from utils import create_folder

create_folder("portfolio_data/benchmarks")

match setup.test_tickerslist:
    case 'test1' | 'test2' | 'test3' | 'sp500':
        benchmark = '^SPX'
    case 'stockallshares' | 'omxlarge' | 'omxmid' | 'omxsmall':
        benchmark = '^OMXSPI'
    case 'firstnorth':
        benchmark = '^FNSESEKPI'

data = yf.download(benchmark, start=setup.test_start_date, end=setup.test_end_date, interval='1d')

# Save data to a CSV file
data.to_csv(f"portfolio_data/benchmarks/{setup.test_tickerslist}.csv")