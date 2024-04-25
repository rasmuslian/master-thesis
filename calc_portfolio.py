import itertools
import os
import pandas as pd
import setup
import numpy as np
import statsmodels.regression.linear_model as sm
import statsmodels.tools.tools as ct
import csv

'''--- Portfolio ---'''

train_tickerslist = 'stockallshares'
test_tickerslist = 'firstnorth'
# train_tickerslist = setup.train_tickerslist
# test_tickerslist = setup.test_tickerslist

# Import csv from portfolios/{setup.tickeslist}.csv
portfolio_df = pd.read_csv(f"portfolios/{train_tickerslist}_{test_tickerslist}.csv", index_col=0)

# Rename the columns to make it easier to work with
portfolio_df = portfolio_df.rename(columns={'portfolio_pct': 'p_pct'})
portfolio_df = portfolio_df.rename(columns={'portfolio_long_pct': 'l_pct'})
portfolio_df = portfolio_df.rename(columns={'portfolio_short_pct': 's_pct'})
portfolio_df = portfolio_df.rename(columns={'portfolio_pct_after_costs': 'pac_pct'})

# Move columns to this order > p_pct, l_pct, s_pct, pac_pct
portfolio_df = portfolio_df[['p_pct', 'l_pct', 's_pct', 'pac_pct', 'total_trades']]
nr_total_trades = portfolio_df['total_trades'].iloc[0] # Save the total trades to a variable

# Drop rows where p_pct is NaN
portfolio_df = portfolio_df.dropna(subset=['p_pct'])
portfolio_df.loc[portfolio_df.index[0], 'total_trades'] = nr_total_trades # Add back the total trades to the dataframe


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
avg_daily_p_return = portfolio_df['p'][1:].mean()
avererage_daily_l_return = portfolio_df['l'][1:].mean()
avg_daily_s_return = portfolio_df['s'][1:].mean()
avg_daily_pac_return = portfolio_df['pac'][1:].mean()

avg_daily_risk_free_return = portfolio_df['riskfree_daily'][1:].mean()
avg_daily_benchmark_return = portfolio_df['benchmark'][1:].mean()

# Calculate the ann return
trading_days = 252

p_return_ann = (1 + avg_daily_p_return)**trading_days - 1
l_return_ann = (1 + avererage_daily_l_return)**trading_days - 1
s_return_ann = (1 + avg_daily_s_return)**trading_days - 1
pac_return_ann = (1 + avg_daily_pac_return)**trading_days - 1

ann_risk_free_return = (1 + avg_daily_risk_free_return)**trading_days - 1
benchmark_return_ann = (1 + avg_daily_benchmark_return)**trading_days - 1

# Add the ann returns to the dataframe as new columns with one row
portfolio_df.loc[portfolio_df.index[0], 'p_return_ann'] = p_return_ann
portfolio_df.loc[portfolio_df.index[0], 'l_return_ann'] = l_return_ann
portfolio_df.loc[portfolio_df.index[0], 's_return_ann'] = s_return_ann
portfolio_df.loc[portfolio_df.index[0], 'pac_return_ann'] = pac_return_ann
portfolio_df.loc[portfolio_df.index[0], 'benchmark_return_ann'] = benchmark_return_ann


# Read the trading dates CSV file
with open(f"stock_dates/test/{test_tickerslist}.csv", 'r', encoding="utf-8") as file:
    reader = csv.DictReader(file)
    latest_dates = [row['latest_date'] for row in reader]
    # Get the second to last latest_dates
    latest_date = latest_dates[-2]

'''--- Alpha ---'''
def alpha_regression(rp_df, name):
    regression_df = pd.DataFrame(columns=['Rp-Rf', 'Rm-Rf'])
    regression_df['Rp-Rf'] = rp_df[1:] - portfolio_df['riskfree_daily'][1:]
    regression_df['Rm-Rf'] = portfolio_df['benchmark'][1:] - portfolio_df['riskfree_daily'][1:]

    regression_df = ct.add_constant(regression_df) # Add a constant to the model

    # Perform the OLS regression
    model = sm.OLS(regression_df['Rp-Rf'], regression_df[['const', 'Rm-Rf']])
    results = model.fit()

    summary = results.summary2(alpha=0.05, float_format='%.8f', title=f"Regression for {name}")

    # Create a new csv file and add a row with the summary title. Do not overwrite but append
    with open(f"portfolios/{train_tickerslist}_{test_tickerslist}_reg.csv", 'a') as file:
        file.write("\n\n")
        file.write(f"{summary.title}\n")

    summary.tables[0].to_csv(f"portfolios/{train_tickerslist}_{test_tickerslist}_reg.csv", mode='a')
    summary.tables[1].to_csv(f"portfolios/{train_tickerslist}_{test_tickerslist}_reg.csv", mode='a')
    summary.tables[2].to_csv(f"portfolios/{train_tickerslist}_{test_tickerslist}_reg.csv", mode='a')
    

try:
    os.remove(f"portfolios/{train_tickerslist}_{test_tickerslist}_reg.csv")
except:
    pass

alpha_regression(portfolio_df['p'], 'p')
alpha_regression(portfolio_df['l'], 'l')
alpha_regression(portfolio_df['s'], 's')
alpha_regression(portfolio_df['pac'], 'pac')


'''--- Sharpe ---'''
def calculate_sharpe(portfolio, rp):
    # Calculate the excess returns of the portfolio
    excess_returns = portfolio[1:] - portfolio_df['riskfree_daily'][1:]

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
benchmark_sr, benchmark_std_dev = calculate_sharpe(portfolio_df['benchmark'], benchmark_return_ann)

# Add the sharpe ratio to the dataframe as new columns with one row
portfolio_df.loc[portfolio_df.index[0], 'p_sr'] = p_sr
portfolio_df.loc[portfolio_df.index[0], 'l_sr'] = l_sr
portfolio_df.loc[portfolio_df.index[0], 's_sr'] = s_sr
portfolio_df.loc[portfolio_df.index[0], 'pac_sr'] = pac_sr
portfolio_df.loc[portfolio_df.index[0], 'benchmark_sr'] = benchmark_sr

portfolio_df.loc[portfolio_df.index[0], 'p_std_dev'] = p_std_dev
portfolio_df.loc[portfolio_df.index[0], 'l_std_dev'] = l_std_dev
portfolio_df.loc[portfolio_df.index[0], 's_std_dev'] = s_std_dev
portfolio_df.loc[portfolio_df.index[0], 'pac_std_dev'] = pac_std_dev
portfolio_df.loc[portfolio_df.index[0], 'benchmark_std_dev'] = benchmark_std_dev


'''--- Save ---'''
# Move column 'total_trades' to the last column
portfolio_df['total_trades'] = portfolio_df.pop('total_trades')

portfolio_df.to_csv(f"portfolios/{train_tickerslist}_{test_tickerslist}_calc.csv")

# Merge all files into a final one
with open(f"portfolios/{train_tickerslist}_{test_tickerslist}_calc.csv", 'r') as calc_file, \
     open(f"portfolios/{train_tickerslist}_{test_tickerslist}_reg.csv", 'r') as reg_file:

    calc_reader = csv.reader(calc_file)
    reg_reader = csv.reader(reg_file)

    # Open the output CSV file and create the CSV writer
    with open(f"portfolios/{train_tickerslist}_{test_tickerslist}_final.csv", 'w', newline='') as merged_file:
        merged_writer = csv.writer(merged_file)
        
        for calc_row, reg_row in itertools.zip_longest(calc_reader, reg_reader, fillvalue=['']):
            # Merge the rows and write them to the output CSV file
            merged_writer.writerow(calc_row + (reg_row if reg_row is not None else ['']))

# Delete all the other files
os.remove(f"portfolios/{train_tickerslist}_{test_tickerslist}_calc.csv")
os.remove(f"portfolios/{train_tickerslist}_{test_tickerslist}_reg.csv")