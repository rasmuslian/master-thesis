train_start_date = '2018-01-01'
train_end_date = '2022-12-31'

test_start_date = '2023-01-01'
test_end_date = '2024-01-28'

ticker_symbol_list = [
    'AAPL',
    'MSFT',
    'AMZN',
    'GOOG',
    'TSLA',
    'NVDA',
    'ADBE',
    'NFLX',
    'CSCO',
    'PEP',
]
training_data_name = 'usa'

# Training model params
pretrained_model_name = 'tf_efficientnet_b7_ns'
num_epochs = 1
batch_size = 32
learning_rate = 0.001

# Test model
test_model_name = 'model_feb01_usa_tf_efficientnet_b7_ns_ep1_bs32_lr0001'