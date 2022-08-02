import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pylab import rcParams
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import tensorflow
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

st.title("Stock Market Forcasting- Amazon")

def get_ticker(name):
    company = yf.Ticker(name)
    return company

company1 = get_ticker("AMZN")

Amazon  = yf.download("AMZN", start="2021-08-01", end="2022-08-01")
data1 = company1.history(period="3mo")
st.write("""
### Amazon
""")
st.write(company1.info['longBusinessSummary'])

st.write(Amazon)

st.line_chart(data1.values)
st.header("Data understanding")

st.table (Amazon.describe())

st. text_input ("Here the Maximum value of share in Open column is 187.199997 and minimum value 14.314000 the Maximum value of share in High column is 188.654007 and minimum value 14.539500. the Maximum value of share in Low column is 184.839493 and minimum value 14.262500. the Maximum value of share in Close column is 186.570496	and minimum value 14.347500.")

new_amazon=Amazon[["Close"]]
series=new_amazon.reset_index()

st.header("Visualization")
fig = px.line(x=series.Date,y=series.Close,labels={"x":"Date","y":"Closing price"})
st.plotly_chart(fig)

st.header("Actual Vs Prediction")

y=np.round((20/100)*len(series),0)
y=int(y)

x=np.round((80/100)*len(series),0)
x=int(x)

train=series.head(x)
test=series.tail(y)

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(series[["Close"]].values)
scaled_train=scaler.fit_transform(train[["Close"]].values)
scaled_test=scaler.fit_transform(test[["Close"]].values)

x_train=[]
y_train=[]

for i in range(20,len(scaled_train)):
  x_train.append(scaled_train[i-20:i])
  y_train.append(scaled_train[i,0])

x_train=np.array(x_train)
y_train=np.array(y_train)


model=Sequential()
model.add(LSTM(units=50,activation="relu",return_sequences=True,input_shape=(x_train.shape[1],1)))  #x_train.shape[1]=20, 1 is for Close column
model.add(Dropout(0.1))
model.add(LSTM(units=60,activation="relu",return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=80,activation="relu",return_sequences=True))  
model.add(Dropout(0.3))
model.add(LSTM(units=120,activation="relu"))  
model.add(Dropout(0.4))
model.add(Dense(units=1))

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(series[["Close"]].values)
scaled_train=scaler.fit_transform(train[["Close"]].values)
scaled_test=scaler.fit_transform(test[["Close"]].values)


X=[]
Y=[]

for i in range(20,len(scaled_data)):
  X.append(scaled_data[i-20:i])
  Y.append(scaled_data[i,0])

X,Y=np.array(X),np.array(Y)

y_pred=model.predict(X)
y_df=pd.DataFrame(Y,columns=["Close"])

Actual_y=scaler.inverse_transform(y_df)

y_pred_df=pd.DataFrame(y_pred,columns=["Predicted_y"])

predicted_y=scaler.inverse_transform(y_pred_df)

s=series[20:]

y_pred_df=pd.DataFrame(predicted_y,columns=["Predictions"])

Y_df=pd.DataFrame(Actual_y,columns=["Actual"])

final_prediction_df=Y_df.join(y_pred_df)

st.write(final_prediction_df)


fig =px.line(final_prediction_df,x=final_prediction_df.index,y=final_prediction_df.columns,labels={"Price":"Numbers"})
st.plotly_chart(fig)


tickers=["AMZN"]

data=yf.download(tickers,start="2015-01-01")
#create new dataframe
new_df = data.filter(['Close'])
#get the last 100 days close price value and covert the data frame into array
last_20_days = new_df[-20:].values
#scale the values between 0 to 1
last_20_days_scaled = scaler.transform(last_20_days)
#create empty list
x_test = []
#append the past 100 days 
x_test.append(last_20_days_scaled)
#convert the x test data into numpy array
x_test = np.array(x_test)
#reshape the data 
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))
#get the predicted scaled price 
pred_price = model.predict(x_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)

new_predictions=new_df.tail(20).values
new_arr = np.append(new_predictions, pred_price)
new_arr_df=pd.DataFrame(new_arr,columns=["Pred"])

future_pred=new_arr_df[20:]

st.header("Prediction of new data points")

fig=plt.figure(figsize=(6,4))
plt.plot(series.Close,label="Actual",color="red")
plt.plot(future_pred,label="Forecasted",color="blue")
plt.legend()
st.plotly_chart(fig)



