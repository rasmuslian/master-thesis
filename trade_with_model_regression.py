import json
import os
from model_regression import predict_graph
import pandas as pd
import setup

# Get all images in the 'trade' folder and loop over them
trade_folder = f"stock_graphs_regression/{setup.test_tickerslist}/trade/"

# Metrics
total_trades = 0
portfolio_percentage = 1.00

# Generates testing data
with open(f"ticker_lists/{setup.test_tickerslist}.json", 'r', encoding="utf-8") as file:
    tickers = json.load(file)

for ticker in tickers:
    print(f"Trading {ticker}")
    data = pd.read_csv(f"stock_data/test/{setup.test_tickerslist}/{ticker}.csv")

    # Convert 'Datetime' to datetime format
    match setup.data_interval:
        case '1h' | '90m' | '60m' | '30m' | '15m' | '5m' | '2m' | '1m':
            date_column = 'Datetime'
        case _:
            date_column = 'Date'
    data['Date'] = pd.to_datetime(data[date_column], utc=True)

    # Set 'Datetime' as the index
    data.set_index('Date', inplace=True)
    # Remove first n rows, because they don't have a n-period moving average
    data = data.iloc[setup.ma_period:]
    # Filter the data based on the start and end date
    data = data.loc[setup.test_start_date:setup.test_end_date]

    # Take the first five rows of the pandas data and add to list, and then repeat
    group_by_chunks = setup.data_groupby
    grouped_data = [data.iloc[i:i+group_by_chunks] for i in range(0, len(data), group_by_chunks)]

    # Remove last group
    grouped_data.pop()

    for group in grouped_data:
        # Get earliest datetime and latest date in the group, include hours and minute
        earliest_date = group.index[0].strftime("%Y-%m-%d")
        latest_date = group.index[-1].strftime("%Y-%m-%d")
        interval = f"{earliest_date}__{latest_date}"
    
        enter_price = group.iloc[0]['Open']
        exit_price = group.iloc[-1]['Adj Close']

        # Predict the graph
        prediction = predict_graph(f"{trade_folder}/{interval}__{ticker}.png")

        # Dont trade if prediction is too close to 0
        if abs(prediction) < 0.008:
            print(f"Prediction too close to 0: {prediction}")
            continue

        # Check if long or short
        if prediction == 0: continue
        long_or_short = 'long' if prediction > 0 else 'short'
        
        # Calculate the trade
        if long_or_short == 'long':
            trade_return = (exit_price - enter_price) / enter_price
        else:
            trade_return = (enter_price - exit_price) / enter_price

        # Calculate the portfolio percentage
        portfolio_percentage = portfolio_percentage * (1 + trade_return)
        total_trades += 1

        print(f"{interval} {'{:.2f}'.format(enter_price)} {'{:.2f}'.format(exit_price)}. PREDICTED: {'{:.2f}'.format(prediction)}% - {long_or_short.upper()} TRADE: {'{:.2f}'.format(trade_return * 100)}%")
    
    print(f"Portfolio percentage: {'{:.2f}'.format((portfolio_percentage - 1) * 100)}%, Total trades: {total_trades}")
