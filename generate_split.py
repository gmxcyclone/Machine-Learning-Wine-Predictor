import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

red_wine = pd.read_csv('Data/winequality-red_clean.csv')
white_wine = pd.read_csv('Data/winequality-white_clean.csv')

red_train, red_test = train_test_split(red_wine, test_size=0.2, random_state=42)
white_train, white_test = train_test_split(white_wine, test_size=0.2, random_state=42)

red_train['red_wine'] = 1
red_test['red_wine'] = 1
white_train['red_wine'] = 0
white_test['red_wine'] = 0

combined_train = pd.concat([red_train, white_train], axis=0)
combined_test = pd.concat([red_test, white_test], axis=0)

red_train.drop('red_wine', axis=1, inplace=True)
red_test.drop('red_wine', axis=1, inplace=True)
white_train.drop('red_wine', axis=1, inplace=True)
white_test.drop('red_wine', axis=1, inplace=True)

red_train.to_csv('splits/red_train.csv', index=False)
red_test.to_csv('splits/red_test.csv', index=False)

white_train.to_csv('splits/white_train.csv', index=False)
white_test.to_csv('splits/white_test.csv', index=False)

combined_test.to_csv('splits/combined_test.csv', index=False)
combined_train.to_csv('splits/combined_train.csv', index=False)
