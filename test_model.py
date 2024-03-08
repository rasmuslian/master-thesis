
import os
from model import test_model
import pandas as pd
import setup

score, rand_score = test_model()

percentage_score = round(score, 4)*100
percentage_rand_score = round(rand_score, 4)*100

print(f"Accuracy: {percentage_score}%")
print(f"Random Accuracy: {percentage_rand_score}%")

# Create benchmarks folder if it does not exist
if not os.path.exists('benchmarks'):
    os.makedirs('benchmarks')
benchmark_filepath = "benchmarks/model_benchmarks.csv"

# If the file does not exist, create it
if not os.path.isfile(benchmark_filepath):
    # Create the file
    df = pd.DataFrame(columns=[
        'model_name',
        'accuracy',
        'random_accuracy',
        'train_data',
        'test_data',
        'train_graph_feat',
        'test_graph_feat',
        'epochs',
        'batch_size',
        'learning_rate',
        'benchmark_date'
        ])
    df.to_csv(benchmark_filepath, index=False)    

# Reads the file
df = pd.read_csv(benchmark_filepath)


# Find the element that starts with 'act' and get the number of epochs
split_name = setup.test_model_name.split('_')
actual_epochs = [s for s in split_name if s.startswith('actep')][0][3:]

# Calculate some metrics
benchmark_date = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S")
train_data_metric = f"{setup.train_tickerslist} {setup.train_start_date}_{setup.train_end_date}"
test_data_metric = f"{setup.test_tickerslist} {setup.test_start_date}_{setup.test_end_date}"
train_graph_feat = f"{setup.data_interval}/grp{setup.data_groupby}/ma{setup.ma_period}"
test_graph_feat = f"{setup.data_interval}/grp{setup.data_groupby}/ma{setup.ma_period}"
epoch_metric = f"max{setup.max_epochs}act{actual_epochs}"

# Adds the new model to the dataframe
new_row = pd.DataFrame({
    'model_name': [setup.test_model_name],
    'accuracy': [percentage_score],
    'random_accuracy': [percentage_rand_score],
    'train_data': [train_data_metric],
    'test_data': [test_data_metric],
    'train_graph_feat': [train_graph_feat],
    'test_graph_feat': [test_graph_feat],
    'epochs': [epoch_metric],
    'batch_size': [setup.batch_size],
    'learning_rate': [setup.learning_rate],
    'benchmark_date': [benchmark_date]
    })
df = pd.concat([df, new_row], ignore_index=True)

# Saves the dataframe to the file
df.to_csv(benchmark_filepath, index=False)