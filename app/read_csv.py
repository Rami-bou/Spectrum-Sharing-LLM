import pandas as pd

df = pd.read_csv('./results/last_csi_for_paper/benchmark.csv')

count = 0

for index, row in df.iterrows():
    if row['TrueInterference'] < row['PredInterference']:
        count += 1

print(f'Constraint violation count: {count/len(df):.2%}')