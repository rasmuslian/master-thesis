# Data setup
train_start_date = '2016-01-01'
train_end_date = '2022-12-31'

test_start_date = '2023-01-01'
test_end_date = '2024-02-20'

train_tickerslist = 'omx30'
test_tickerslist = 'omx30'

data_interval = '1d'
data_groupby = 5
ma_period = 20

# Training model params
pretrained_model_name = 'tf_efficientnet_b7_ns'
max_epochs = 20
batch_size = 32
learning_rate = 0.001

# Test model
test_model_name = 'model_feb01_usa_tf_efficientnet_b7_ns_act1_bs32_lr0001'

