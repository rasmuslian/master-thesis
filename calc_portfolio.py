import pandas as pd
import setup

'''--- Portfolio ---'''

# Import csv from portfolios/{setup.tickeslist}.csv
portfolio_df = pd.read_csv(f"portfolios/{setup.train_tickerslist}_{setup.test_tickerslist}.csv", index_col=0)

# Make a new column called daily_return
portfolio_df['p'] = portfolio_df['portfolio_pct'].pct_change()
portfolio_df['l'] = portfolio_df['portfolio_long_pct'].pct_change()
portfolio_df['s'] = portfolio_df['portfolio_short_pct'].pct_change()
portfolio_df['p_ac'] = portfolio_df['portfolio_pct_after_costs'].pct_change()


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
average_daily_p_return = portfolio_df['p'].mean()
average_daily_p_ac_return = portfolio_df['p_ac'].mean()
avererage_daily_l_return = portfolio_df['l'].mean()
average_daily_s_return = portfolio_df['s'].mean()

average_daily_risk_free_return = portfolio_df['riskfree_daily'].mean()
average_daily_benchmark_return = portfolio_df['benchmark'].mean()

# Calculate the annualised return
trading_days = 252

annualised_p_return = (1 + average_daily_p_return)**trading_days - 1
annualised_p_ac_return = (1 + average_daily_p_ac_return)**trading_days - 1
annualised_l_return = (1 + avererage_daily_l_return)**trading_days - 1
annualised_s_return = (1 + average_daily_s_return)**trading_days - 1

annualised_risk_free_return = (1 + average_daily_risk_free_return)**trading_days - 1
annualised_benchmark_return = (1 + average_daily_benchmark_return)**trading_days - 1

# Add the annualised returns to the dataframe as new columns with one row
portfolio_df['annualised_p_return'] = annualised_p_return
portfolio_df['annualised_p_ac_return'] = annualised_p_ac_return
portfolio_df['annualised_l_return'] = annualised_l_return
portfolio_df['annualised_s_return'] = annualised_s_return

'''--- Alpha ---'''
def calculate_alpha(portfolio, benchmark, rp):
    # Calculate the portfolio's beta
    covariance = portfolio.cov(benchmark)
    benchmark_variance = portfolio.var()
    beta = covariance / benchmark_variance

    # Calculate Jensen's Alpha
    jensens_alpha = rp - (annualised_risk_free_return + beta * (annualised_benchmark_return - annualised_risk_free_return))

    return jensens_alpha

p_alpha = calculate_alpha(portfolio_df['portfolio'], portfolio_df['benchmark'], annualised_p_return)
l_alpha = calculate_alpha(portfolio_df['portfolio_long'], portfolio_df['benchmark'], annualised_l_return)
s_alpha = calculate_alpha(portfolio_df['portfolio_short'], portfolio_df['benchmark'], annualised_s_return)
p_ac_alpha = calculate_alpha(portfolio_df['p_ac'], portfolio_df['benchmark'], annualised_p_ac_return)

# Add the alpha to the dataframe as new columns with one row
portfolio_df['p_alpha'] = p_alpha
portfolio_df['l_alpha'] = l_alpha
portfolio_df['s_alpha'] = s_alpha
portfolio_df['p_ac_alpha'] = p_ac_alpha


'''--- Sharpe ---'''
def calculate_sharpe(portfolio, rp):
    # Calculate the excess returns of the portfolio
    excess_returns = portfolio - portfolio_df['riskfree_daily']

    # Calculate the standard deviation of the portfolio's returns
    std_dev = excess_returns.std()
    annualised_std_dev = std_dev * (trading_days)**0.5

    # Calculate the Sharpe Ratio
    sharpe_ratio = (rp - annualised_risk_free_return) / annualised_std_dev

    return sharpe_ratio, annualised_std_dev

p_sr, p_std_dev = calculate_sharpe(portfolio_df['portfolio'], annualised_p_return)
l_sr, l_std_dev = calculate_sharpe(portfolio_df['portfolio_long'], annualised_l_return)
s_sr, s_std_dev = calculate_sharpe(portfolio_df['portfolio_short'], annualised_s_return)
p_ac_sr, p_ac_std_dev = calculate_sharpe(portfolio_df['p_ac'], annualised_p_ac_return)

# Add the sharpe ratio to the dataframe as new columns with one row
portfolio_df['p_sr'] = p_sr
portfolio_df['l_sr'] = l_sr
portfolio_df['s_sr'] = s_sr
portfolio_df['p_ac_sr'] = p_ac_sr

portfolio_df['p_std_dev'] = p_std_dev
portfolio_df['l_std_dev'] = l_std_dev
portfolio_df['s_std_dev'] = s_std_dev
portfolio_df['p_ac_std_dev'] = p_ac_std_dev


'''--- Save ---'''
portfolio_df.to_csv(f"portfolios/{setup.train_tickerslist}_{setup.test_tickerslist}_calc.csv")