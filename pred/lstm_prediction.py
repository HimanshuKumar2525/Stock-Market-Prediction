import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import json
import datetime

def fetch_stock_data(se, stock_symbol):
    """Fetch stock data using yfinance."""
    if se == 'NSE':
        stock_symbol += ".NS"
    df = yf.download(stock_symbol, period="5y")
    return df

def lstm_prediction(se, stock_symbol):
    """LSTM model for stock price prediction."""
    
    # Fetch stock data
    og_df = fetch_stock_data(se, stock_symbol)
    if og_df.empty:
        return json.dumps({"error": "No data found for the stock symbol."})

    todataframe = og_df.reset_index(inplace=False)

    print("\n<------ Info of the Original Dataset ------>")
    print(todataframe.info())
    print("<------------------------------------------>\n")

    # Prepare dataframe
    seriesdata = todataframe.sort_index(ascending=True)
    new_seriesdata = seriesdata[['Date', 'Close']].copy()
    
    # Setting Date as index
    new_seriesdata.set_index('Date', inplace=True)

    # Convert dataset into training data
    myseriesdataset = new_seriesdata.values
    scalerdata = MinMaxScaler(feature_range=(0, 1))
    scale_data = scalerdata.fit_transform(myseriesdataset)

    x_totrain, y_totrain = [], []
    for i in range(60, len(myseriesdataset)):
        x_totrain.append(scale_data[i - 60:i, 0])
        y_totrain.append(scale_data[i, 0])

    x_totrain, y_totrain = np.array(x_totrain), np.array(y_totrain)
    x_totrain = np.reshape(x_totrain, (x_totrain.shape[0], x_totrain.shape[1], 1))

    # Define LSTM model
    lstm_model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(x_totrain.shape[1], 1)),
        LSTM(units=50),
        Dense(1)
    ])
    
    lstm_model.compile(loss='mean_squared_error', optimizer='adadelta')
    lstm_model.fit(x_totrain, y_totrain, epochs=100, batch_size=32, verbose=2)

    # Prepare test data
    myinputs = new_seriesdata.iloc[-75:].values
    myinputs = myinputs.reshape(-1, 1)
    myinputs = scalerdata.transform(myinputs)

    tostore_test_result = []
    for i in range(60, myinputs.shape[0]):
        tostore_test_result.append(myinputs[i - 60:i, 0])
    
    tostore_test_result = np.array(tostore_test_result)
    tostore_test_result = np.reshape(tostore_test_result, (tostore_test_result.shape[0], tostore_test_result.shape[1], 1))

    # Predict stock price
    myclosing_priceresult = lstm_model.predict(tostore_test_result)
    myclosing_priceresult = scalerdata.inverse_transform(myclosing_priceresult)

    # Create predicted dataset
    datelist = pd.date_range(pd.to_datetime('today').date(), periods=len(myclosing_priceresult) + 1)[1:]
    predicted_df = pd.DataFrame(myclosing_priceresult, columns=['Close'], index=datelist)

    # Combine original and predicted data
    result_df = pd.concat([og_df, predicted_df])[['Close']]
    result_df.reset_index(inplace=True)
    result_df.columns = ['Date', 'Close']

    print("\n<------ Info of the Result Dataset ------>")
    print(result_df.info())
    print("<----------------------------------------->\n")

    # **🔹 Plotting the Results**
    plt.figure(figsize=(14, 7))
    plt.plot(result_df['Date'], result_df['Close'], label="Stock Price", color='blue')
    plt.axvline(x=result_df['Date'].iloc[-len(predicted_df)], color='red', linestyle='--', label="Prediction Start")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"Stock Price Prediction for {stock_symbol}")
    plt.xticks(rotation=45)
    plt.show()

    def get_json(df):
        """Convert DataFrame to JSON with date format 'YYYY-MM-DD'."""
        def convert_timestamp(item_date_object):
            if isinstance(item_date_object, (datetime.date, datetime.datetime)):
                return item_date_object.strftime("%Y-%m-%d")

        return json.dumps(df.to_dict(orient='records'), default=convert_timestamp)

    return get_json(result_df)
