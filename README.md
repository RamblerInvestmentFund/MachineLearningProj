# Predicting Market Value
This project attempts to predict the returns of stocks with respect to their previous returns and various technical indicators.

### Contributions and different Implementations
- Anthony Peters -> contributed to the main.py whereby we use the final optimized SVM to predict and plot returns vs a buy and hold strategy
- Matt Hyatt -> 
- Jacob Maffet -> contibuted to the folder jmaffett which holds an SVM implementation on the JPM ticker and graphs of returns and sharpe ratios
- Thomas DiMonte -> contributed to __init__.py in percentReturn. Signalling additions and increasing percent return accuracy
- Yandi Farinango -> Worked on an independent implementation of which he submitted seperately

### Requirements
* run in a venv
    * `virtualenv .venv; deactivate &> /dev/null; source ./.venv/bin/activate;`
* install all requirements
    * `pip install -r requirements.txt`

### Model Selection
- SVM
- RNN

# SVM
### Architecture
An SVM with a rbf kernel is used.
`model = svm.SVC(kernel="rbf", C = 1000, gamma = 1)`

### Dataset
Various stock tickers train the model one at a time.

# RNN
### Architecture
Model: "sequential"

|    Layer (type)          |     Output Shape      |    Param    |
|--------------------------|-----------------------|-------------|
|    lstm (LSTM)           |    (None, 50, 128)    |    73728    |  
|    bidirectional-lstm    |    (None, 50, 64)     |    41216    |
|    dense (Dense)         |    (None, 50, 64)     |    4160     |
|    dense_1 (Dense)       |    (None, 50, 16)     |    1040     |
|    dense_2 (Dense)       |    (None, 50, 1)      |    17       |

* Total params: 120,161
* Trainable params: 120,161
* Non-trainable params: 0

### Dataset
Historical data was pulled from yahoo finance on 10+ different stock tickers from the S&P 500


