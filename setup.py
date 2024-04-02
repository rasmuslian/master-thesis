# Data setup
train_start_date = '2016-01-01'
train_end_date = '2020-06-30'

validate_start_date = '2020-07-01'
validate_end_date = '2021-12-31'

test_start_date = '2022-01-01'
test_end_date = '2023-12-31'

train_tickerslist = 'stockallshares'
test_tickerslist = 'stockallshares'

# Tickers that exist the whole period, to ensure date consistency
train_ticker_trading_base = 'ABB.ST'
validate_ticker_trading_base = 'ABB.ST'
test_ticker_trading_base = 'ABB.ST'

data_interval = '1d'
data_groupby = 5
ma_period = 20

# Training model params
pretrained_model_name = 'tf_efficientnet_b7_ns'
max_epochs = 10
batch_size = 32
learning_rate = 0.001

# Test model
test_model_name = 'model_hhloyb__actep3_mar24-1509_test1_1d_tf_efficientnet_b7_ns_maxep10_bs32_lr0001'

