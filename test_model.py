
import os
from model import test_model
import pandas as pd
import setup

# score, rand_score = test_model()

# print(f"Accuracy: {round(score, 4)*100}%")
# print(f"Random Accuracy: {round(rand_score, 4)*100}%")


benchmark_filepath = "model_benchmarks.csv"
# If the file does not exist, create it
if not os.path.isfile(benchmark_filepath):
    print("Does not exist")
    # Create the file
    df = pd.DataFrame(columns=['model_name', 'accuracy', 'random_accuracy'])
    df.to_csv(benchmark_filepath, index=False)    
else:
    print("model_benchmarks.csv already exists")

# Reads the file
df = pd.read_csv(benchmark_filepath)

# Adds the new model to the dataframe
new_row = pd.DataFrame({'model_name': [setup.test_model_name], 'accuracy': [22], 'random_accuracy': [23]})
df = pd.concat([df, new_row], ignore_index=True)

# Saves the dataframe to the file
df.to_csv(benchmark_filepath, index=False)