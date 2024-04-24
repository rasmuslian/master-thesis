import pandas as pd
import setup

test_tickerslist = 'omxsmall'

benchmark_df = pd.read_csv(f"portfolio_data/benchmarks/{test_tickerslist}.csv", index_col=0)
benchmark_df.index = pd.to_datetime(benchmark_df.index)
benchmark_df = benchmark_df.sort_index()

closest_date = benchmark_df.index.asof(setup.test_start_date)
benchmark_df = benchmark_df.loc[closest_date:setup.test_end_date]

# Save
benchmark_df.to_csv(f"portfolio_data/excel/TEST{test_tickerslist}.csv")