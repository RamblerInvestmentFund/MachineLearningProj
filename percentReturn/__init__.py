import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import talib as ta
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

df = yf.download('SPY', start='2009-01-01', end='2021-10-29')

# Technicals Dataframe
# Daily
#df['Percent Return'] = df['Adj Close'].pct_change(periods=1) * 100
# Weekly
df['Percent Return'] = df['Adj Close'].pct_change(periods=7) * 100

df['Daily Range'] = df['High'] - df['Low']
df['Simple MA'] = ta.SMA(df['Adj Close'], timeperiod=10)
df['Daily Close - SMA Difference'] = df['Adj Close'] - df['Simple MA']
# df['MACD'] = ta.MACD(df['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['RSI'] = ta.RSI(df['Adj Close'], timeperiod=9)
# df['Daily CLose - RSI'] = df['Adj Close'] - df['RSI']
df['Up Band'], df['Middle Band'], df['Down Band'] = ta.BBANDS(df['Adj Close'], timeperiod=10)
# df['Daily Close - Up Band'] = abs(df['Adj Close'] - df['Up Band'])
# df['Daily Close - Middle Band'] = abs(df['Adj Close'] - df['Middle Band'])
# df['Daily Close - Down Band'] = abs(df['Adj Close'] - df['Down Band'])
# df['Daily Close - High'] = abs(df['Adj Close'] - df['High'])
# df['Daily Close - Low'] = abs(df['Adj Close'] - df['Low'])

# Signal for pos/neg percent returns
def signal (x):
    if x['Percent Return'] > 0:
        return 1
    else:
        return 0

# Signal for pos/neg percent returns with shorting
# Percent val can be adjusted
# def signal (x):
#     if x['Percent Return'] > 2 or x['Percent Return'] < -4:
#         return 1
#     else:
#         return 0

df['Signal'] = df.apply(signal, axis=1)

# Remove first 9 rows bc data is not complete; Contains NaN while calculating percent return, etc.
df = df.iloc[9:]
print(df.head())

# Plotting Bands and SMA
df[['Adj Close', 'Up Band', 'Middle Band', 'Down Band', 'Simple MA']].plot(figsize=(12,10))
plt.show()

# Drop Unwanted Columns & creating arrays for SVM
X = np.array(df.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Simple MA', 'Up Band',
                      'Middle Band', 'Down Band', 'Signal'], axis = 1))
# Dropping Date Column
X = X[:, 1:]

Y = np.array(df['Signal'])

# Normalizing Data
X_max = X.max()
X_min = X.min()
X_range = X_max - X_min
X_norm = (X - X_min) / X_range

#DataSplit
X_train, X_test, y_train, y_test = train_test_split(X_norm, Y, test_size=.15, random_state=20)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=.15, random_state=20)

# Find Best Parameters
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)
grid.fit(X_train, y_train)
print(grid.best_params_)

# SVC With Best Parameters
grid_pred = grid.predict(X_dev)
cmatrix = np.array(confusion_matrix(y_dev, grid_pred, labels = [1, 0]))
#INTRODUCE THIRD ROW IN CONFUSION MATRIX FOR SHORTING ACCURACY
confusion_best = pd.DataFrame(cmatrix, index = ['Positive percent', 'Negative percent'],
                              columns = ['Predicted Positive percent', 'Predicted Negative percent'])
print(confusion_best)
print(classification_report(y_dev, grid_pred))