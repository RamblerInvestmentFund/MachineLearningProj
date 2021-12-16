import data

import rnn, svm


def main():

    "anything to show Dr. Dligach"
    "final print statements etc"

    # ticker = input('input ticker: ').upper()

    ## runs the svm code
    # svm.main()

    stocks = ['WFC', 'UBS']
    for ticker in stocks:
        svm.plot_simulation(ticker)

    ## runs the rnn code
    # rnn.main()


if __name__ == "__main__":
    main()
