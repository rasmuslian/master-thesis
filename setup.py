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
pretrained_model_name = 'efficientnet_b0'
num_epochs = 2
batch_size = 32
learning_rate = 0.001

# Test model
test_model_name = 'model_jan30_usa_efficientnet_b0_ep2_bs32_lr0001'