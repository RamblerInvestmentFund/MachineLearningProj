import yfinance as yf
import pandas as pd
import numpy as np

import talib as ta
from talib import MA_Type

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker

def signal(x):
    return 1 if x['Returns'] >= 0 else 0

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''

Downloads Data, creates Dataframe, and splits into train/test

'''

stocks = ["JPM"] #, "MSFT", "SPY", "QQQ", "DIA", "TLT", "GLD", "CVX", "KO", "PEP", "PG", "JNJ", "GSK"]
frames = []
for i in stocks:

    df = yf.download(tickers=i, start='2001-01-01')
    
    df['High Shifted'] = df['High'].shift(1)
    df['Low Shifted'] = df['Low'].shift(1)
    df['Close Shifted'] = df['Close'].shift(1)

    df['Upper BBand'], df['Middle BBand'], df['Lower BBand'] = ta.BBANDS(df['Close Shifted'], timeperiod=20)
    df['RSI'] = ta.RSI(np.array(df['Close Shifted']), timeperiod=14)
    df['Macd'], df['Macd Signal'],df['Macd Hist'] = ta.MACD(df['Close Shifted'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['Momentum'] = ta.MOM(df['Close Shifted'],timeperiod=12)

    df['Returns'] = np.log(df['Open'] / df['Open'].shift(1)) 
    df['Signal'] = df.apply(signal, axis=1)
    df = df[40:]
    frames.append(df)


max_abs_scaler = preprocessing.MaxAbsScaler()
X = np.array(frames[0].drop(columns = ["Signal", "Returns", "Open", "High", "Low", "Close", "Adj Close", "High Shifted", "Low Shifted", "Close Shifted", "Volume"], axis = 1))
X = max_abs_scaler.fit_transform(X)
Y = np.array(frames[0]["Signal"])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''

Grid Search:

SVM: After Running Grid Search, it found rbf, C=1000, and gamma=1 as best hyperparameters
LogisticRegression: After running Grid search it found C=100 and penalty=l2

'''

# param_grid = {'C': [0.1, 1, 10, 100, 1000],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['rbf', 'poly']}

# grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3)
# grid.fit(X_train, y_train)
# print(grid.best_params_)
# print(grid.best_estimator_)

# grid_values = {'penalty': ['l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}
# model_lr = GridSearchCV(LogisticRegression(), param_grid=grid_values)
# model_lr.fit(X_train, y_train)
# print(model_lr.best_params_)
# print(model_lr.best_estimator_)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''

Running Both SVM and Logisitic Regression with optimized hyperparamters

'''

model = svm.SVC(kernel="rbf", C=1000, gamma=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# model = LogisticRegression(C=100, penalty='l2')
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

'''

Logistic Regression
Accuracy: 0.62
Precision: 0:0.59 1:0.63 Avg:0.61
Recall: 0:0.47 1:0.74 Avg: 0.62
f1-score: 0:0.52 1:0.68 Avg: 0.61

'''

'''

SVM
Accuracy: 0.63
Precision: 0:0.61 1:0.64 Avg:0.63
Recall: 0:0.5 1:0.74 Avg:0.63
f1-score: 0:0.55 1:0.69 Avg:0.62

'''
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''

Graphing Buy and Hold returns against our SVM signal Returns
Computing Sharpe Ratio as well

'''

pred_len = len(y_pred)

frames[0]['SVM Signal'] = 0
frames[0]['SVM Returns'] = 0
frames[0]['Total Strat Returns'] = 0
frames[0]['Market Returns'] = 0

Signal_Column = frames[0].columns.get_loc('SVM Signal')
Strat_Column = frames[0].columns.get_loc('SVM Returns')
Return_Column = frames[0].columns.get_loc('Total Strat Returns')
Market_Column = frames[0].columns.get_loc('Market Returns')

frames[0].iloc[-pred_len:,Signal_Column] = list(map(int,y_pred))
frames[0]['SVM Returns'] = frames[0]['SVM Signal'] * frames[0]['Returns'].shift(1)

frames[0].iloc[-pred_len:,Return_Column] = np.nancumsum(frames[0]['SVM Returns'][-pred_len:])
frames[0].iloc[-pred_len:,Market_Column] = np.nancumsum(frames[0]['Returns'][-pred_len:])

sharpe_ratio = (frames[0]['Total Strat Returns'][-1] - frames[0]['Market Returns'][-1])/ \
                    np.nanstd(frames[0]['Total Strat Returns'][-pred_len:])

print(sharpe_ratio)

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(frames[0][-pred_len:].index.values,
        frames[0]['Total Strat Returns'][-pred_len:].values, color='g', label="Strat Returns")

ax.plot(frames[0][-pred_len:].index.values,
        frames[0]['Market Returns'][-pred_len:].values, color='b', label="Market Returns")

ax.set(xlabel= "Date",ylabel="Returns")
plt.title(i,fontsize=15)
ax.xaxis.set_major_locator(ticker.AutoLocator())

plt.legend(loc='best')
plt.savefig("AAPL.png")
plt.show()