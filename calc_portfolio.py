import pandas as pd
import setup

'''--- Portfolio ---'''

# Import csv from portfolios/{setup.tickeslist}.csv
portfolio_df = pd.read_csv(f"portfolios/{setup.test_tickerslist}.csv", index_col=0)

# Make a new column called daily_return
portfolio_df['portfolio'] = portfolio_df['portfolio_percentage'].pct_change()
portfolio_df['portfolio_long'] = portfolio_df['portfolio_percentage'].pct_change()
portfolio_df['portfolio_short'] = portfolio_df['portfolio_percentage'].pct_change()
portfolio_df['portfolio_after_costs'] = portfolio_df['portfolio_percentage'].pct_change()


'''--- Risk free rate ---'''
riskfree_df = pd.read_csv(f"portfolio_data/riskfree.csv", index_col=0)
riskfree_df = riskfree_df.loc[setup.test_start_date:setup.test_end_date]

# Append new column called riskfree from column "3 mån" to portfolio_df. Divide each value by 100 to get the percentage
portfolio_df['riskfree'] = riskfree_df['3 mån']/100

# Add a new column that takes the annualized risk free rate and divides it by 360 to get the daily risk free rate. Append to new column called riskfree_daily
portfolio_df['riskfree_daily'] = (1 + portfolio_df['riskfree']/100)**(1/360) - 1


'''--- Benchmarks ---'''
benchmark_df = pd.read_csv(f"portfolio_data/benchmarks/{setup.test_tickerslist}.csv", index_col=0)
benchmark_df = benchmark_df.loc[setup.test_start_date:setup.test_end_date]

portfolio_df['benchmark'] = benchmark_df['Close'].pct_change()

# Calculate average returns
average_portfolio_return = portfolio_df['portfolio'].mean()
average_risk_free_return = portfolio_df['riskfree_daily'].mean()
average_benchmark_return = portfolio_df['benchmark'].mean()

# Calculate the portfolio's beta
covariance = portfolio_df['portfolio'].cov(portfolio_df['benchmark'])
benchmark_variance = portfolio_df['benchmark'].var()
beta = covariance / benchmark_variance

# Calculate Jensen's Alpha
jensens_alpha = average_portfolio_return - (average_risk_free_return + beta * (average_benchmark_return - average_risk_free_return))

print("Jensen's Alpha:", jensens_alpha)


'''--- Beta, Alpha and Sharpe ---'''



'''--- Save ---'''
portfolio_df.to_csv(f"portfolios/{setup.test_tickerslist}.csv")