#!/usr/bin/env python
# coding: utf-8

# In[80]:


import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import pytz

import shutil
import yfinance as yf
yf.pdr_override()
import matplotlib.pyplot as plt
import talib


# ## NR7 strat

# In[81]:



#valid periods: “1d”, “5d”, “1mo”, “3mo”, “6mo”, “1y”, “2y”, “5y”, “10y”, “ytd”, “max”

def get_data_nr7(ticker, period):
    index_ticker = yf.Ticker(ticker)
    df = index_ticker.history(period)[['Close','Open','High','Low']].reset_index()
    
    df['Range'] = df['High']-df['Low']
    df['Low_range'] = np.nan
    df['Low_range'][6:] = df['Range'].rolling(window=6, min_periods=1).min()[5:-1]
    df['High_yesterday'] = df['High'].shift(1).rolling(window=1).min()
    df['Indicator_buy'] = df['Range']<df['Low_range']
    df['Indicator_sell'] = df['Close']>df['High_yesterday']
    
    return df


# In[82]:


def get_p_l_nr7(df,start_date, end_date,initial_investment):
    df=df.copy()
    df['Date'] = df['Date'].dt.date
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    p_l = 0
    final_position = 0
    current_value=initial_investment
    prices_buy = []
    prices_sell = []
    number_trades = 0
    for i in range(len(df)):
        if df.iloc[i]['Indicator_buy']==True and final_position==0:
            prices_buy.append(df.iloc[i]['Close'])
            final_position += 1
            current_value = current_value/df.iloc[i]['Close']
            number_trades += 1
        if df.iloc[i]['Indicator_sell']==True and final_position>0:
            prices_sell.append(df.iloc[i]['Close'])
            final_position -= 1
            current_value = current_value*df.iloc[i]['Close']
    ## To have a final position = 0
    if final_position>0:
        last_trade=prices_buy.pop()
        current_value=current_value*last_trade
    p_l=current_value-initial_investment
    return p_l, final_position, prices_buy, prices_sell, number_trades


# In[83]:


## Results since 1996

ticker='SPY'

start_date=pd.to_datetime('1996-01-01')
end_date=pd.to_datetime('2024-12-31')
df=get_data_nr7('SPY','30y')
initial_investment=1000
print('number of trades: '+ str(len(get_p_l_nr7(df,start_date,end_date,initial_investment)[2])))


buy_nr7=get_p_l_nr7(df,start_date,end_date,initial_investment)[2]
sell_nr7=get_p_l_nr7(df,start_date,end_date,initial_investment)[3]
pos_returns=[]
neg_returns=[]
for i in range(len(buy_nr7)):
    if (sell_nr7[i]/buy_nr7[i]-1)>0:
        pos_returns.append(sell_nr7[i]/buy_nr7[i]-1)
    else:
        neg_returns.append(sell_nr7[i]/buy_nr7[i]-1)
        
print('win_rate: '+ str(len(pos_returns)/(len(neg_returns)+len(pos_returns))))
print('mean positive returns: ' + str(sum(pos_returns)/len(pos_returns)))
print('mean negative returns: ' + str(sum(neg_returns)/len(neg_returns)))
    
    


# In[84]:


dates = []

for i in range (24):
    if i<10:
        start_date = pd.to_datetime(f'200{i}-01-01')
        end_date = pd.to_datetime(f'200{i}-12-31')
    else:
        start_date = pd.to_datetime(f'20{i}-01-01')
        end_date = pd.to_datetime(f'20{i}-12-31')
    dates.append([start_date,end_date])

ticker='SPY'   
df=get_data_nr7(ticker, "30y")

p_l_per_year = {str(date[0])[:4]:get_p_l_nr7(df,date[0],date[1],initial_investment)[0] 
                for date in dates if get_p_l_nr7(df,date[0],date[1],initial_investment)[0]!=0}
return_per_year = {str(date[0])[:4]:get_p_l_nr7(df,date[0],date[1],initial_investment)[0]/initial_investment 
                   for date in dates}

print(return_per_year)


# In[85]:




years = [key for key in return_per_year.keys()]

plt.bar(years, return_per_year.values(), color='skyblue')
plt.xlabel('Year')
plt.ylabel('Annual Returns')
plt.title('Annual Returns from 2004 to 2024')
x_ticks = [str(year) for year in years[::5]]
plt.xticks(years[::5], x_ticks)
plt.show()


years = [key for key in p_l_per_year.keys()]

plt.bar(years, p_l_per_year.values(), color='skyblue')
plt.xlabel('Year')
plt.ylabel('Annual p&L')
plt.title('Annual p&L from 2004 to 2024')
x_ticks = [str(year) for year in years[::5]]
plt.xticks(years[::5], x_ticks)
plt.show()


# In[86]:


# 1000 dollars invested
values_nr7=[1000]
for i in range(len(buy_nr7)):
    value_now=values_nr7[-1]
    new_value=value_now*sell_nr7[i]/buy_nr7[i]
    values_nr7.append(new_value)

# Plotting the values
plt.plot(values_nr7)

# Adding labels to the axes
plt.xlabel('Number of trades')
plt.ylabel('Values')

# Adding a title to the plot
plt.title('Plot of Values')

# Displaying the plot
plt.show()


# ## Triple RSI

# In[87]:


def get_data_rsi(ticker,period):
    # Get historical data with RSI calculated
    index_ticker=yf.Ticker(ticker)
    df=index_ticker.history(period)[['Close','Open','High','Low']].reset_index()
    df['RSI'] = talib.RSI(df['Close'],timeperiod=5)
    df['200_day_mean_close'] = df['Close'].rolling(window=200).mean()
    df['RSI_shift_3_days']=df['RSI'].shift(3)



    # Calculate the differences between day i and day i-1
    df['rolling_diff'] = df['RSI'].diff(periods=1)

    df['decreasing_3_days'] = 0

    # Check if diff is negative for the past 3 days and set decreasing_3_days accordingly
    for i in range(3, len(df)):
        if all(df.loc[i-2:i, 'rolling_diff'] < 0):
            df.loc[i, 'decreasing_3_days'] = 1
    return df
    


# In[88]:




def get_p_l_rsi(df,start_date, end_date, initial_investment):
    # Compute p&l between two dates
    df=df.copy()
    df['Date']=df['Date'].dt.date
    df=df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    p_l=0
    final_position=0
    current_value=initial_investment
    prices_buy=[]
    prices_sell=[]
    number_trades=0
    for i in range(len(df)):
        if (final_position==0 and
        df.iloc[i]['RSI']<30 and 
        df.iloc[i]['decreasing_3_days']==1 and 
        df.iloc[i]['RSI_shift_3_days']<60 and 
        df.iloc[i]['200_day_mean_close']<df.iloc[i]['Close']):
            #buy
            prices_buy.append(df.iloc[i]['Close'])
            final_position+=1
            current_value=current_value/df.iloc[i]['Close']
            number_trades+=1
        if df.iloc[i]['RSI']>50 and final_position>0:
            #sell
            prices_sell.append(df.iloc[i]['Close'])
            final_position-=1
            current_value=current_value*df.iloc[i]['Close']
        ## To have a final position = 0
    if final_position>0:
        last_trade=prices_buy.pop()
        current_value=current_value*last_trade
    p_l=current_value-initial_investment
            
            
    return p_l, final_position, prices_buy, prices_sell, number_trades


# In[89]:


## Results since 1993

ticker='SPY'
period='30y'

df_rsi=get_data_rsi(ticker,period)

start_date=pd.to_datetime('1993-01-01')
end_date=pd.to_datetime('2024-12-31')
initial_investment=1000
print('number of trades: '+ str(len(get_p_l_rsi(df_rsi,start_date,end_date,initial_investment)[2])))

#83 trades found. Same results than the website

buy_rsi=get_p_l_rsi(df_rsi,start_date,end_date,initial_investment)[2]
sell_rsi=get_p_l_rsi(df_rsi,start_date,end_date,initial_investment)[3]
pos_returns=[]
neg_returns=[]
for i in range(len(buy_rsi)):
    if (sell_rsi[i]/buy_rsi[i]-1)>0:
        pos_returns.append(sell_rsi[i]/buy_rsi[i]-1)
    else:
        neg_returns.append(sell_rsi[i]/buy_rsi[i]-1)
        
print('win_rate: '+ str(len(pos_returns)/(len(neg_returns)+len(pos_returns))))
print('mean positive returns: ' + str(sum(pos_returns)/len(pos_returns)))
print('mean negative returns: ' + str(sum(neg_returns)/len(neg_returns)))
    
    


# In[90]:


## returns per year

dates = []

for i in range (24):
    if i<10:
        start_date=pd.to_datetime(f'200{i}-01-01')
        end_date=pd.to_datetime(f'200{i}-12-31')
    else:
        start_date=pd.to_datetime(f'20{i}-01-01')
        end_date=pd.to_datetime(f'20{i}-12-31')
    dates.append([start_date,end_date])

    


p_l_per_year={str(date[0])[0:4]:get_p_l_rsi(df_rsi,date[0],date[1],initial_investment)[0] for date in dates}
return_per_year={str(date[0])[0:4]:get_p_l_rsi(df_rsi,date[0],date[1],initial_investment)[0]/initial_investment 
                 for date in dates} 

print(p_l_per_year)


# In[92]:




years = [key for key in return_per_year.keys()]

plt.bar(years, return_per_year.values(), color='skyblue')
plt.xlabel('Year')
plt.ylabel('Annual Returns')
plt.title('Annual Returns from 2004 to 2024')
x_ticks = [str(year) for year in years[::5]]
plt.xticks(years[::5], x_ticks)
plt.show()


years = [key for key in p_l_per_year.keys()]

plt.bar(years, p_l_per_year.values(), color='skyblue')
plt.xlabel('Year')
plt.ylabel('Annual p&L')
plt.title('Annual p&L from 2000 to 2024')
x_ticks = [str(year) for year in years[::5]]
print(years[::5])
plt.xticks(years[::5], x_ticks)
plt.show()


# In[93]:


# 1000 dollars invested and compounded
values_rsi=[1000]
for i in range(len(buy_rsi)):
    value_now=values_rsi[-1]
    new_value=value_now*sell_rsi[i]/buy_rsi[i]
    values_rsi.append(new_value)

# Plotting the values
plt.plot(values_rsi)

# Adding labels to the axes
plt.xlabel('Number of trades')
plt.ylabel('Values')

# Adding a title to the plot
plt.title('Plot of Values')

# Displaying the plot
plt.show()


# ## SP500 overnight trading

# In[94]:


def get_data_overnight(ticker,period):
    # Get historical data with RSI calculated
    index_ticker=yf.Ticker(ticker)
    df=index_ticker.history(period)[['Close','Open','High','Low']].reset_index()
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=25)
    df['ATR_25_day_avg'] = df['ATR'].rolling(window=25).mean()
    df['High_10_day'] = df['High'].rolling(window=10).max()
    df['IBS'] = (df['Close']-df['Low'])/(df['High']-df['Low'])
    df['Low_band'] = df['High_10_day'] - 2.5*df['ATR_25_day_avg']
    return df


# In[95]:


def get_p_l_overnight(df,start_date, end_date, initial_investment):
    # Compute p&l between two dates
    df=df.copy()
    df['Date']=df['Date'].dt.date
    df=df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    p_l=0
    current_value=initial_investment
    final_position=0
    prices_buy=[]
    prices_sell=[]
    number_trades=0
    for i in range(len(df)):
        if (final_position==0 and
        df.iloc[i]['Close']<df.iloc[i]['Low_band'] and 
        df.iloc[i]['IBS']<0.5):
            prices_buy.append(df.iloc[i]['Close'])
            final_position+=1
            current_value=current_value/df.iloc[i]['Close']
            number_trades+=1
        if final_position>0 and i<len(df)-1:
            prices_sell.append(df.iloc[i+1]['Open'])
            final_position-=1
            current_value=current_value*df.iloc[i+1]['Open']
    ## To have a final position = 0, remove the last stock bought.
    if final_position>0:
        last_trade=prices_buy.pop()
        current_value=current_value*last_trade
    p_l=current_value-initial_investment
            
            
    return p_l, final_position, prices_buy, prices_sell, number_trades


# In[96]:


## Results 

ticker='SPY'
period='30y'

df=get_data_overnight(ticker,period)

initial_investment=1000
start_date=pd.to_datetime('2005-01-01')
end_date=pd.to_datetime('2024-12-31')
print('number of trades: '+ str(len(get_p_l_overnight(df,start_date,end_date,initial_investment)[2])))


buy_overnight=get_p_l_overnight(df,start_date,end_date,initial_investment)[2]
sell_overnight=get_p_l_overnight(df,start_date,end_date,initial_investment)[3]
pos_returns=[]
neg_returns=[]
for i in range(len(buy)):
    if (sell_overnight[i]/buy_overnight[i]-1)>0:
        pos_returns.append(sell_overnight[i]/buy_overnight[i]-1)
    else:
        neg_returns.append(sell_overnight[i]/buy_overnight[i]-1)
        
print('win_rate: '+ str(round(len(pos_returns)/(len(neg_returns)+len(pos_returns)),4)))
print('mean positive returns: ' + str(round(sum(pos_returns)/len(pos_returns),4)))
print('mean negative returns: ' + str(round(sum(neg_returns)/len(neg_returns),4)))
   
    


# In[97]:


## returns and p&L per year with 1000 dollars initial investment

dates = []

for i in range (30):
    if i<10:
        start_date=pd.to_datetime(f'200{i}-01-01')
        end_date=pd.to_datetime(f'200{i}-12-31')
    else:
        start_date=pd.to_datetime(f'20{i}-01-01')
        end_date=pd.to_datetime(f'20{i}-12-31')
    dates.append([start_date,end_date])

    


p_l_per_year={str(date[0])[0:4]:round(get_p_l_overnight(df,date[0],date[1],initial_investment)[0],5) for date in dates}
return_per_year={str(date[0])[0:4]:round(get_p_l_overnight(df,date[0],date[1],initial_investment)[0]/initial_investment,5)
                 for date in dates} 

print(p_l_per_year)


# In[98]:




years = [key for key in return_per_year.keys()]

plt.bar(years, return_per_year.values(), color='skyblue')
plt.xlabel('Year')
plt.ylabel('Annual Returns')
plt.title('Annual Returns from 2004 to 2024')
x_ticks = [str(year) for year in years[::5]]
plt.xticks(years[::5], x_ticks)
plt.show()


years = [key for key in p_l_per_year.keys()]

plt.bar(years, p_l_per_year.values(), color='skyblue')
plt.xlabel('Year')
plt.ylabel('Annual p&L')
plt.title('Annual p&L from 2000 to 2024')
x_ticks = [str(year) for year in years[::5]]
plt.xticks(years[::5], x_ticks)
plt.show()




# In[99]:


# 1000 dollars invested and compounded
values_overnight=[1000]
for i in range(len(buy_overnight)):
    value_now=values_overnight[-1]
    new_value=value_now*sell_overnight[i]/buy_overnight[i]
    values_overnight.append(new_value)

# Plotting the values
plt.plot(values_overnight)

# Adding labels to the axes
plt.xlabel('Number of trades')
plt.ylabel('Values')

# Adding a title to the plot
plt.title('Plot of Values')

# Displaying the plot
plt.show()


# In[ ]:





# In[ ]:




