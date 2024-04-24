import pandas as pd
import setup

# Import xlsx to df, open sheet name Sheet1
df = pd.read_excel('portfolio_data/excel/omxlarge.xlsx', sheet_name='Sheet2')

# Rename column Stängn kurs to Close
df = df.rename(columns={'Stängn kurs': 'Close'})

# Reverse the order of the rows
df = df.iloc[::-1]

# df = df.loc[setup.test_start_date:setup.test_end_date]

# Save as csv
df.to_csv('portfolio_data/benchmarks/omxlarge.csv', index=False)