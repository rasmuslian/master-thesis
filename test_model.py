
import os
from model import test_model
import pandas as pd
import setup

score, rand_score = test_model()

percentage_score = round(score, 4)*100
percentage_rand_score = round(rand_score, 4)*100

print(f"Accuracy: {percentage_score}%")
print(f"Random Accuracy: {percentage_rand_score}%")


benchmark_filepath = "model_benchmarks.csv"
# If the file does not exist, create it
if not os.path.isfile(benchmark_filepath):
    # Create the file
    df = pd.DataFrame(columns=['model_name', 'accuracy', 'random_accuracy', 'benchmark_date'])
    df.to_csv(benchmark_filepath, index=False)    

# Reads the file
df = pd.read_csv(benchmark_filepath)

# Todays date in 2018-01-01 12:00:00 format
benchmark_date = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S")

# If a row with the same model_name exists, overwrite
if setup.test_model_name in df['model_name'].values:
    df.loc[df['model_name'] == setup.test_model_name, 'accuracy'] = percentage_score
    df.loc[df['model_name'] == setup.test_model_name, 'random_accuracy'] = percentage_rand_score
    df.loc[df['model_name'] == setup.test_model_name, 'benchmark_date'] = benchmark_date
else:
    # Adds the new model to the dataframe
    new_row = pd.DataFrame({'model_name': [setup.test_model_name], 'accuracy': [percentage_score], 'random_accuracy': [percentage_rand_score], 'benchmark_date': [benchmark_date]})
    df = pd.concat([df, new_row], ignore_index=True)

# Saves the dataframe to the file
df.to_csv(benchmark_filepath, index=False)