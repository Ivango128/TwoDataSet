import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('countries of the world.csv')

df = df.dropna()

def get_migration(row):
    row=row.replace(',', '.')
    if float(row) > 0:
        return 1
    else:
        return 0

df['Net migration'] = df['Net migration'].apply(get_migration)

print(df.info())
print(df['Net migration'].value_counts())

