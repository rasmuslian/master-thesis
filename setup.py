# Data setup
train_start_date = '2016-01-01'
train_end_date = '2021-12-31'

test_start_date = '2022-01-01'
test_end_date = '2023-12-31'

train_tickerslist = 'usa10'
test_tickerslist = 'usa10'

data_interval = '1d'
data_groupby = 5
ma_period = 20

# Training model params
pretrained_model_name = 'efficientnet_b4'
# pretrained_model_name = 'tf_efficientnet_b7_ns'
max_epochs = 1000
batch_size = 128
learning_rate = 0.001

# Test model
test_model_name = 'model_epkazv__actep2_mar07-2237_usa10_1d_efficientnet_b4_maxep1000_bs128_lr0001'

