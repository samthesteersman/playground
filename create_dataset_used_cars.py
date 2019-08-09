import numpy as np
import pandas as pd
import sklearn.datasets

n_samples = 5000

feature_cols = ['manufacturer', 'color', 'had_accident', 'resold', 'automatic_transmission']
target_cols = ['price']

n_features = len(feature_cols)

X, y = sklearn.datasets.make_regression(n_samples=n_samples, n_features=n_features)

df1 = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]))

df2 = pd.DataFrame()

df2['price'] = df1[df1.columns[n_features]].abs().astype(int) * 1000 + 1200

cats = {
    0: ['volvo', 'bmd', 'renault', 'ford', 'toyota'],
    1: ['black', 'white', 'red'],
}

for i, vals in cats.items():
    key = feature_cols[i]
    cuts = pd.cut(df1[df1.columns[i]], len(vals))
    cuts_idx = cuts.cat.codes
    df2[key] = cuts_idx.apply(lambda x: vals[x]).astype('category')

i = 2
key = feature_cols[i]
df2[key] = df1[df1.columns[i]].apply(lambda x: int(x < 0)).astype('int8')

i = 3
key = feature_cols[i]
df2[key] = df1[df1.columns[i]].abs().astype(int).apply(lambda x: int(x % 2 == 0)).astype('int8')

i = 4
key = feature_cols[i]
df2[key] = df1[df1.columns[i]].astype(str).apply(lambda x: int('4' in x)).astype('int8')

df2.to_csv('used_cars.csv', index=False)
