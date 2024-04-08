import pandas as pd
import setup

# Import csv from portfolios/{setup.tickeslist}.csv
portfolio_df = pd.read_csv(f"portfolios/{setup.test_tickerslist}.csv", index_col=0)

# Make a new column called daily_return
portfolio_df['portfolio'] = portfolio_df['portfolio_percentage'].pct_change()
portfolio_df['portfolio_long'] = portfolio_df['portfolio_percentage'].pct_change()
portfolio_df['portfolio_short'] = portfolio_df['portfolio_percentage'].pct_change()
portfolio_df['portfolio_after_costs'] = portfolio_df['portfolio_percentage'].pct_change()

# Save portfolio_df, overwrite
portfolio_df.to_csv(f"portfolios/{setup.test_tickerslist}.csv")