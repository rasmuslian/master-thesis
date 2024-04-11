import pandas as pd
import setup

'''--- Portfolio ---'''

# Import csv from portfolios/{setup.tickeslist}.csv
portfolio_df = pd.read_csv(f"portfolios/{setup.test_tickerslist}.csv", index_col=0)

# Make a new column called daily_return
portfolio_df['portfolio'] = portfolio_df['portfolio_pct'].pct_change()
portfolio_df['portfolio_long'] = portfolio_df['portfolio_long_pct'].pct_change()
portfolio_df['portfolio_short'] = portfolio_df['portfolio_short_pct'].pct_change()
portfolio_df['portfolio_after_costs'] = portfolio_df['portfolio_pct_after_costs'].pct_change()


'''--- Risk free rate ---'''
riskfree_df = pd.read_csv(f"portfolio_data/riskfree.csv", index_col=0)
riskfree_df = riskfree_df.loc[setup.test_start_date:setup.test_end_date]

# Append new column called riskfree from column "3 mån" to portfolio_df. Divide each value by 100 to get the percentage
portfolio_df['riskfree'] = riskfree_df['3 mån']/100

# Add a new column that takes the annualized risk free rate and divides it by 360 to get the daily risk free rate. Append to new column called riskfree_daily
portfolio_df['riskfree_daily'] = (1 + portfolio_df['riskfree'])**(1/360) - 1


'''--- Benchmarks ---'''
benchmark_df = pd.read_csv(f"portfolio_data/benchmarks/{setup.test_tickerslist}.csv", index_col=0)
benchmark_df = benchmark_df.loc[setup.test_start_date:setup.test_end_date]

portfolio_df['benchmark'] = benchmark_df['Close'].pct_change()


'''--- Annualised returns ---'''
# Calculate the annualised return of the portfolio

# Calculate average returns
average_daily_portfolio_return = portfolio_df['portfolio'].mean()
avererage_daily_long_return = portfolio_df['portfolio_long'].mean()
average_daily_short_return = portfolio_df['portfolio_short'].mean()

average_daily_risk_free_return = portfolio_df['riskfree_daily'].mean()
average_daily_benchmark_return = portfolio_df['benchmark'].mean()

# Calculate the annualised return
trading_days = 252

annualised_portfolio_return = (1 + average_daily_portfolio_return)**trading_days - 1
annualised_long_return = (1 + avererage_daily_long_return)**trading_days - 1
annualised_short_return = (1 + average_daily_short_return)**trading_days - 1

annualised_risk_free_return = (1 + average_daily_risk_free_return)**trading_days - 1
annualised_benchmark_return = (1 + average_daily_benchmark_return)**trading_days - 1

'''--- Alpha ---'''
# Calculate the portfolio's beta
covariance = portfolio_df['portfolio'].cov(portfolio_df['benchmark'])
benchmark_variance = portfolio_df['benchmark'].var()
beta = covariance / benchmark_variance

# Calculate Jensen's Alpha
jensens_alpha = annualised_portfolio_return - (annualised_risk_free_return + beta * (annualised_benchmark_return - annualised_risk_free_return))

print("Jensen's Alpha:", jensens_alpha)


'''--- Sharpe ---'''
# Calculate the excess returns of the portfolio
excess_returns = portfolio_df['portfolio'] - portfolio_df['riskfree_daily']

# Calculate the standard deviation of the portfolio's returns
std_dev = excess_returns.std()
annualised_std_dev = std_dev * (trading_days)**0.5

# Calculate the Sharpe Ratio
sharpe_ratio = (annualised_portfolio_return - annualised_risk_free_return) / annualised_std_dev

print("Sharpe Ratio:", sharpe_ratio)


'''--- Save ---'''
portfolio_df.to_csv(f"portfolios/{setup.test_tickerslist}.csv")