import pandas as pd
import setup

'''--- Portfolio ---'''

# train_tickerslist = 'sp500'
# test_tickerslist = 'omxmid'
train_tickerslist = setup.train_tickerslist
test_tickerslist = setup.test_tickerslist

# Import csv from portfolios/{setup.tickeslist}.csv
portfolio_df = pd.read_csv(f"portfolios/{train_tickerslist}_{test_tickerslist}.csv", index_col=0)

# Rename the columns to make it easier to work with
portfolio_df = portfolio_df.rename(columns={'portfolio_pct': 'p_pct'})
portfolio_df = portfolio_df.rename(columns={'portfolio_long_pct': 'l_pct'})
portfolio_df = portfolio_df.rename(columns={'portfolio_short_pct': 's_pct'})
portfolio_df = portfolio_df.rename(columns={'portfolio_pct_after_costs': 'pac_pct'})

# Add 'total_trades' column
portfolio_df['total_trades'] = ''

# Move columns to this order > p_pct, l_pct, s_pct, pac_pct
portfolio_df = portfolio_df[['p_pct', 'l_pct', 's_pct', 'pac_pct', 'total_trades']]


# Make a new column called daily_return
portfolio_df['p'] = portfolio_df['p_pct'].pct_change()
portfolio_df['l'] = portfolio_df['l_pct'].pct_change()
portfolio_df['s'] = portfolio_df['s_pct'].pct_change()
portfolio_df['pac'] = portfolio_df['pac_pct'].pct_change()


'''--- Risk free rate ---'''
riskfree_df = pd.read_csv(f"portfolio_data/riskfree.csv", index_col=0)
riskfree_df = riskfree_df.loc[setup.test_start_date:setup.test_end_date]

# Append new column called riskfree from column "3 mån" to portfolio_df. Divide each value by 100 to get the percentage
portfolio_df['riskfree'] = riskfree_df['3 mån']/100

# Add a new column that takes the annualized risk free rate and divides it by 360 to get the daily risk free rate. Append to new column called riskfree_daily
portfolio_df['riskfree_daily'] = (1 + portfolio_df['riskfree'])**(1/360) - 1


'''--- Benchmarks ---'''
benchmark_df = pd.read_csv(f"portfolio_data/benchmarks/{test_tickerslist}.csv", index_col=0)
benchmark_df = benchmark_df.loc[setup.test_start_date:setup.test_end_date]

portfolio_df['benchmark'] = benchmark_df['Close'].pct_change()


'''--- Annualised returns ---'''
# Calculate the ann return of the portfolio

# Calculate avg returns
avg_daily_p_return = portfolio_df['p'].mean()
avererage_daily_l_return = portfolio_df['l'].mean()
avg_daily_s_return = portfolio_df['s'].mean()
avg_daily_pac_return = portfolio_df['pac'].mean()

avg_daily_risk_free_return = portfolio_df['riskfree_daily'].mean()
avg_daily_benchmark_return = portfolio_df['benchmark'].mean()

# Calculate the ann return
trading_days = 252

p_return_ann = (1 + avg_daily_p_return)**trading_days - 1
l_return_ann = (1 + avererage_daily_l_return)**trading_days - 1
s_return_ann = (1 + avg_daily_s_return)**trading_days - 1
pac_return_ann = (1 + avg_daily_pac_return)**trading_days - 1

ann_risk_free_return = (1 + avg_daily_risk_free_return)**trading_days - 1
ann_benchmark_return = (1 + avg_daily_benchmark_return)**trading_days - 1

# Calculate the ann return of the benchmark
benchmark_return_ann = (1 + portfolio_df['benchmark'].mean())**trading_days - 1

# Add the ann returns to the dataframe as new columns with one row
portfolio_df.loc[portfolio_df.index[0], 'p_return_ann'] = p_return_ann
portfolio_df.loc[portfolio_df.index[0], 'l_return_ann'] = l_return_ann
portfolio_df.loc[portfolio_df.index[0], 's_return_ann'] = s_return_ann
portfolio_df.loc[portfolio_df.index[0], 'pac_return_ann'] = pac_return_ann

portfolio_df.loc[portfolio_df.index[0], 'benchmark_return_ann'] = benchmark_return_ann


'''--- Alpha ---'''
def calculate_alpha(portfolio, benchmark, rp):
    # Calculate the portfolio's beta
    covariance = portfolio.cov(benchmark)
    benchmark_variance = portfolio.var()
    beta = covariance / benchmark_variance

    # Calculate Jensen's Alpha
    jensens_alpha = rp - (ann_risk_free_return + beta * (ann_benchmark_return - ann_risk_free_return))

    return jensens_alpha

p_alpha = calculate_alpha(portfolio_df['p'], portfolio_df['benchmark'], p_return_ann)
l_alpha = calculate_alpha(portfolio_df['l'], portfolio_df['benchmark'], l_return_ann)
s_alpha = calculate_alpha(portfolio_df['s'], portfolio_df['benchmark'], s_return_ann)
pac_alpha = calculate_alpha(portfolio_df['pac'], portfolio_df['benchmark'], pac_return_ann)

# Add the alpha to the dataframe as new columns with one row
portfolio_df.loc[portfolio_df.index[0], 'p_alpha'] = p_alpha
portfolio_df.loc[portfolio_df.index[0], 'l_alpha'] = l_alpha
portfolio_df.loc[portfolio_df.index[0], 's_alpha'] = s_alpha
portfolio_df.loc[portfolio_df.index[0], 'pac_alpha'] = pac_alpha


'''--- Sharpe ---'''
def calculate_sharpe(portfolio, rp):
    # Calculate the excess returns of the portfolio
    excess_returns = portfolio - portfolio_df['riskfree_daily']

    # Calculate the standard deviation of the portfolio's returns
    std_dev = excess_returns.std()
    ann_std_dev = std_dev * (trading_days)**0.5

    # Calculate the Sharpe Ratio
    sharpe_ratio = (rp - ann_risk_free_return) / ann_std_dev

    return sharpe_ratio, ann_std_dev

p_sr, p_std_dev = calculate_sharpe(portfolio_df['p'], p_return_ann)
l_sr, l_std_dev = calculate_sharpe(portfolio_df['l'], l_return_ann)
s_sr, s_std_dev = calculate_sharpe(portfolio_df['s'], s_return_ann)
pac_sr, pac_std_dev = calculate_sharpe(portfolio_df['pac'], pac_return_ann)

# Add the sharpe ratio to the dataframe as new columns with one row
portfolio_df.loc[portfolio_df.index[0], 'p_sr'] = p_sr
portfolio_df.loc[portfolio_df.index[0], 'l_sr'] = l_sr
portfolio_df.loc[portfolio_df.index[0], 's_sr'] = s_sr
portfolio_df.loc[portfolio_df.index[0], 'pac_sr'] = pac_sr

portfolio_df.loc[portfolio_df.index[0], 'p_std_dev'] = p_std_dev
portfolio_df.loc[portfolio_df.index[0], 'l_std_dev'] = l_std_dev
portfolio_df.loc[portfolio_df.index[0], 's_std_dev'] = s_std_dev
portfolio_df.loc[portfolio_df.index[0], 'pac_std_dev'] = pac_std_dev


'''--- Save ---'''
# Move column 'total_trades' to the last column
portfolio_df['total_trades'] = portfolio_df.pop('total_trades')

portfolio_df.to_csv(f"portfolios/{train_tickerslist}_{test_tickerslist}_calc.csv")