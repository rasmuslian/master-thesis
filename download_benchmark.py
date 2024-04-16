import yfinance as yf
import setup
from utils import create_folder

create_folder("portfolio_data/benchmarks")

test_tickerslist = setup.test_tickerslist

match test_tickerslist:
    case 'test' | 'sp500':
        benchmark = '^SPX'
    case 'stockallshares':
        benchmark = '^OMXSPI'
    case 'omxlarge':
        benchmark = '^OMXSPI'
    case 'omxmid':
        benchmark = '^OMXSPI'
    case 'omxsmall':
        benchmark = '^OMXSPI'
    case 'firstnorth':
        benchmark = '^FNSESEKPI'

data = yf.download(benchmark, start=setup.test_start_date, end=setup.test_end_date, interval='1d')

# Save data to a CSV file
data.to_csv(f"portfolio_data/benchmarks/{test_tickerslist}.csv")