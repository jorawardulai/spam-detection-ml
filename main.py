import pandas as pd

data = pd.read_csv('data/spam.csv', sep='\t', header=None, names=['label', 'message'], encoding='latin-1')

print(data.head())

print(data.info())

data['label_number'] = data['label'].map({'ham': '0', 'spam': '1'})

print(data[['label', 'label_number']].head())