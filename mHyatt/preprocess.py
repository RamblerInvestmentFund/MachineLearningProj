import numpy as np
import pandas as pd
import talib as ta
import yfinance as yf
from sklearn import preprocessing
from talib import MA_Type
from tqdm import tqdm

"create datasets from yahoo finance"
"preprocesses data for analysis "


def snp():
    snp = [
        "MMM", "ABT", "ABBV", "ABMD", "ACN", "ATVI", "ADBE", "AMD", "AAP", "AES", "AFL", "A", "APD", "AKAM", "ALK", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP", "AXP", "AIG", "AMT", "AWK", "AMP", "ABC", "AME", "AMGN", "APH", "ADI", "ANSS", "ANTM", "AON", "AOS", "APA", "AAPL", "AMAT", "APTV", "ADM", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "BKR", "BLL", "BAC", "BBWI", "BAX", "BDX", "BBY", "BIO", "TECH", "BIIB", "BLK", "BK", "BA", "BKNG", "BWA", "BXP", "BSX", "BMY", "AVGO", "BR", "BRO", "CHRW", "CDNS", "CZR", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CTLT", "CAT", "CBRE", "CDW", "CE", "CNC", "CNP", "CDAY", "CERN", "CF", "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CTXS", "CLX", "CME", "CMS", "KO", "CTSH", "CL", "CMCSA", "CMA", "CAG", "COP", "ED", "STZ", "COO", "CPRT", "GLW", "CTVA", "COST", "CTRA", "CCI", "CSX", "CMI", "CVS", "DHI", "DHR", "DRI", "DVA", "DE", "DAL", "XRAY", "DVN", "DXCM", "FANG", "DLR", "DFS", "DISCA", "DISCK", "DISH", "DG", "DLTR", "D", "DPZ", "DOV", "DOW", "DTE", "DUK", "DRE", "DD", "DXC", "EMN", "ETN", "EBAY", "ECL", "EIX", "EW", "EA", "EMR", "ENPH", "ETR", "EOG", "EFX", "EQIX", "EQR", "ESS", "EL", "ETSY", "EVRG", "ES", "RE", "EXC", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FAST", "FRT", "FDX", "FIS", "FITB", "FE", "FRC", "FISV", "FLT", "FMC", "F", "FTNT", "FTV", "FBHS", "FOXA", "FOX", "BEN", "FCX", "GPS", "GRMN", "IT", "GNRC", "GD", "GE", "GIS", "GM", "GPC", "GILD", "GL", "GPN", "GS", "GWW", "HAL", "HBI", "HIG", "HAS", "HCA", "PEAK", "HSIC", "HSY", "HES", "HPE", "HLT", "HOLX", "HD", "HON", "HRL", "HST", "HWM", "HPQ", "HUM", "HBAN", "HII", "IEX", "IDXX", "INFO", "ITW", "ILMN", "INCY", "IR", "INTC", "ICE", "IBM", "IP", "IPG", "IFF", "INTU", "ISRG", "IVZ", "IPGP", "IQV", "IRM", "JKHY", "J", "JBHT", "SJM", "JNJ", "JCI", "JPM", "JNPR", "KSU", "K", "KEY", "KEYS", "KMB", "KIM", "KMI", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX", "LW", "LVS", "LEG", "LDOS", "LEN", "LLY", "LNC", "LIN", "LYV", "LKQ", "LMT", "L", "LOW", "LUMN", "LYB", "MTB", "MRO", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", "MA", "MTCH", "MKC", "MCD", "MCK", "MDT", "MRK", "FB", "MET", "MTD", "MGM", "MCHP", "MU", "MSFT", "MAA", "MRNA", "MHK", "TAP", "MDLZ", "MPWR", "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NWL", "NEM", "NWSA", "NWS", "NEE", "NLSN", "NKE", "NI", "NSC", "NTRS", "NOC", "NLOK", "NCLH", "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "OKE", "ORCL", "OGN", "OTIS", "PCAR", "PKG", "PH", "PAYX", "PAYC", "PYPL", "PENN", "PNR", "PBCT", "PEP", "PKI", "PFE", "PM", "PSX", "PNW", "PXD", "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PTC", "PEG", "PSA", "PHM", "PVH", "QRVO", "PWR", "QCOM", "DGX", "RL", "RJF", "RTX", "O", "REG", "REGN", "RF", "RSG", "RMD", "RHI", "ROK", "ROL", "ROP", "ROST", "RCL", "SPGI", "CRM", "SBAC", "SLB", "STX", "SEE", "SRE", "NOW", "SHW", "SPG", "SWKS", "SNA", "SO", "LUV", "SWK", "SBUX", "STT", "STE", "SYK", "SIVB", "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO", "TPR", "TGT", "TEL", "TDY", "TFX", "TER", "TSLA", "TXN", "TXT", "TMO", "TJX", "TSCO", "TT", "TDG", "TRV", "TRMB", "TFC", "TWTR", "TYL", "TSN", "UDR", "ULTA", "USB", "UAA", "UA", "UNP", "UAL", "UNH", "UPS", "URI", "UHS", "VLO", "VTR", "VRSN", "VRSK", "VZ", "VRTX", "VFC", "VIAC", "VTRS", "V", "VNO", "VMC", "WRB", "WAB", "WMT", "WBA", "DIS", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC", "WU", "WRK", "WY", "WHR", "WMB", "WLTW", "WYNN", "XEL", "XLNX", "XYL", "YUM", "ZBRA", "ZBH", "ZION", "ZTS",]

    return snp


def get(stock):
    """returns a stock ticker as a dataframe"""
    return yf.download(tickers=stock, progress=False)


def process(df):
    """returns a normalized dataframe with technical indicators...
    ready to be fed into a model
    """

    ## technical indicators
    df["Upper BBand"], df["Middle BBand"], df["Lower BBand"] = ta.BBANDS(
        df["Close"], timeperiod=20
    )

    df["Percent Return"] = df["Adj Close"].pct_change() * 100
    df["Daily Range"] = df["High"] - df["Low"]
    df["Simple MA"] = ta.SMA(df["Adj Close"], timeperiod=10)

    macd, macdsignal, macdhist = ta.MACD(
        df["Adj Close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["MACD"] = macd
    df["RSI"] = ta.RSI(df["Adj Close"], timeperiod=9)

    ## calculate returns and signal
    df["Returns"] = df["Close"] - df["Open"]
    df["Shifted Signal"] = df["Returns"].apply(lambda x: 1 if x >= 0 else 0).shift(-1)

    df.dropna(inplace=True)

    ## normalization
    max_abs_scaler = preprocessing.MaxAbsScaler()
    df.dropna(inplace=True)

    return df


def xy_split(df, timesteps):
    """splits a dataframe into two numpy arrays:

    Params:
    ----------
    df: data to be split
    timesteps: the number of samples in an RNN batch

    Return:
    ----------
    X: feature values
    y: target values
    INPUT_SHAPE: the shape of one batch
    """

    ## split
    c = df.columns[-1]
    y = np.array(df[c])
    X = np.array(df.drop(c, axis=1, inplace=False))

    ## reshape
    timesteps = timesteps
    batch = int(X.shape[0] / timesteps)
    size = timesteps * batch
    feature = X.shape[1]

    INPUT_SHAPE = (timesteps, feature)

    y = np.reshape(y[:size], newshape=(batch, timesteps, 1))
    X = np.reshape(X[:size], newshape=(batch, timesteps, feature))

    return X, y, INPUT_SHAPE


def rnn_data_pipeline(stocks=snp(), timesteps=50):
    """gets data from yahoo finance and prepares it for rnn
    by calling preprocessing methods"""

    ## pull from yahoo finance
    df = []
    print('downloading stocks...')
    for stock in tqdm(stocks):
        df.append(get(stock))
    df = pd.concat(df)

    ## preprocess
    df = process(df)

    return xy_split(df, timesteps)


def save_npz(stocks=snp(), name="snp500"):
    "save X,y as npz file for quick use during testing"

    X, y, SHAPE = rnn_data_pipeline()

    np.savez_compressed(f"{name}", X=X, y=y)


def load_npz(name="snp500"):
    "load npz file and return objects"

    loaded = np.load(f"{name}.npz")
    X = loaded["X"]
    y = loaded["y"]
    return X, y


def main():

    save_npz()


if __name__ == "__main__":
    main()
