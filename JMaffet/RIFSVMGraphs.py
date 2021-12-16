import matplotlib.pyplot as plt
import statistics
import RIFSVM

def sharpe(ticker, numsimulations, timeframe,risk_free_rate):
    percent_returns = RIFSVM.run_model(ticker,numsimulations,timeframe)
    std = statistics.stdev(percent_returns)
    mean = statistics.mean(percent_returns)
    return (mean-risk_free_rate)/std


def general_sharpe(percent_returns,std):
    return (statistics.mean(percent_returns)-3.5)/statistics.stdev(percent_returns)


def sharpegraph(sharpevals1,sharpevals2,timeframe):
    plt.plot(timeframe,sharpevals1, label='Model')
    plt.plot(timeframe,sharpevals2, label='Buy and Hold')
    plt.xlabel("Days")
    plt.ylabel("Sharpe Ratio")
    plt.title('SHARPE Ratios on Different Time Frames For JPM')
    plt.legend()
    plt.show()

def returnsgraph(returns1,returns2,timeframe):
    plt.plot(timeframe, returns1, label='Model')
    plt.plot(timeframe, returns2, label='Buy and Hold')
    plt.xlabel("Days")
    plt.ylabel("Percent Returns")
    plt.title('Percent Returns on Different Time Frames For JPM')
    plt.legend()
    plt.show()

JPM_time_returns = [-.112619,-4.4935,.87192,-2.6524,26.837]
JPM_model_returns = [-2.1601553776476177, -2.7869594795536243, 5.156272498057029, 9.46686275910571, 38.45673066852839]
time_frames = [7,30,90,180,365]
JPMModel_sharpe = [-17.850146424469656, -6.72414728461359, 1.1396281185416481, 1.2251688543706425, 4.986233691852523]
JPMBH_Sharpe =[-0.2802711305788498, -0.6201449093530306, -0.20388946436135766, -0.47731025712185965, 1.8105112590944734]

def main():
    returnsgraph(JPM_model_returns,JPM_time_returns,time_frames)
    sharpegraph(JPMBH_Sharpe,JPMModel_sharpe,time_frames)

if __name__ == "__main__":
    main()




