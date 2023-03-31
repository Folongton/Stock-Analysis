import requests as r
import json as j
import pandas as pd
import os
import time
import shutil
from datetime import datetime, timedelta
from Scraping import Navigation as Nav

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

class AlphaVantageAPI:
    @staticmethod
    def get_daily_adjusted_data(ticker,apikey,full_or_compact='full'):
        '''
        https://www.alphavantage.co/documentation/#dailyadj
        Returns formatted data from AlphaVantage API
        IN: API Parameters
        OUT: Json Data'''
        url_json = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={apikey}&outputsize={full_or_compact}'
        response = r.get(url_json)
        j_response = response.json()
        formated_response = j.dumps(j_response, indent=4)
        return formated_response
    
    @staticmethod
    def get_daily_adjusted_data_for_list(ticker_list, directory, apikey, full_or_compact, sleep_time):
        '''
        https://www.alphavantage.co/documentation/#dailyadj
        Saves Data from AlphaVantage API to a folder in 1 csv file per ticker
        IN: API Parameters, directory to save data to
        OUT: None - saves data to directory'''
        for ticker in ticker_list:
            if os.path.exists(os.path.join(directory, f'{ticker}-daily-100days.csv')):
                pass
            elif os.path.exists(os.path.join(directory, f'{ticker}.json')):
                pass
            else:
                data = AlphaVantageAPI.get_daily_adjusted_data(ticker,apikey,full_or_compact)
                data = j.loads(data)
                try:
                    df = pd.DataFrame(data['Time Series (Daily)']).T
                    df.index.name = 'Date'
                    df.to_csv(os.path.join(directory, f'{ticker}-daily-100days.csv'))
                    time.sleep(sleep_time)
                    print(f'{ticker_list.index(ticker)/len(ticker_list)*100:.2f}%')
                except KeyError:
                    print(f'No data for {ticker}. Saving Json file')
                    print(data)
                    with open(os.path.join(directory, f'{ticker}.json'), 'w') as f:
                        j.dump(data, f)
                    continue

    @staticmethod
    def get_intraday_data(ticker, apikey, interval='5min', adjusted='false', outputsize='compact', datatype='json'):
        '''
        https://www.alphavantage.co/documentation/#intraday
        Returns json data from AlphaVantage API
        IN: API Parameters
        OUT: Json Data
        '''
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={interval}&apikey={apikey}&adjusted={adjusted}&outputsize={outputsize}&datatype={datatype}'
        response = r.get(url)

        if response.status_code == 200:
            data = j.loads(response.text)
            return data
        else:
            print(f'Error: {response.status_code}')
            return None
    
    @staticmethod
    def get_intraday_data_for_list(tickers_list, directory, apikey, sleep_time, interval='5min', adjusted='false', outputsize='compact', datatype='json'):
        '''
        https://www.alphavantage.co/documentation/#intraday
        This function takes in the following parameters:
        symbol: The name of the equity you want to retrieve intraday data for.
        interval: The time interval between two consecutive data points in the time series (e.g. "1min", "5min", "15min", "30min","60min").
        apikey: Your API key for the Alpha Vantage API.
        adjusted: An optional boolean/string value ('true' or 'false') indicating whether the output time series should be adjusted by historical split and dividend events (default is 'true').
        outputsize: An optional string value indicating the size of the output time series (default is "compact", which returns the latest 100 data points).
        datatype: An optional string value indicating the data format of the output (default is "json").
        '''
        for ticker in tickers_list:
            if os.path.exists(os.path.join(directory, f'{ticker}-1day-5min.csv')):
                pass
            elif os.path.exists(os.path.join(directory, f'{ticker}.json')):
                pass
            else:
                data = AlphaVantageAPI.get_intraday_data(ticker, interval=interval, apikey=apikey, adjusted=adjusted, outputsize=outputsize, datatype=datatype)
                if data:
                    try:
                        df = pd.DataFrame(data['Time Series (5min)']).T
                        df.index = pd.to_datetime(df.index)
                        df.index.name = 'Date-Time'
                        df.to_csv(os.path.join(directory, f'{ticker}-1day-5min.csv'))
                        time.sleep(sleep_time) #to avoid API limit of 75 calls per minute
                        
                    except KeyError:
                        print(f'No data for {ticker}. Saving Json file')
                        print(data)
                        with open(os.path.join(directory, f'{ticker}.json'), 'w') as f:
                            j.dump(data, f)
                        continue
                    
                    print(f'{round(tickers_list.index(ticker)/(len(tickers_list)-1)*100,2)}% completed')

    @staticmethod
    def get_intraday_extended_data(ticker, key, sleep_time, directory, interval='60min',slice='year1month1'):
        '''
        Read here : https://www.alphavantage.co/documentation/#intraday-extended
        Gets the intraday data for a given ticker and saves it as a csv file.
        IN: API parameters
        OUT: None - saves the data as a csv file.
        '''
        if slice == '2yearData':
            months = ['1','2','3','4','5','6','7','8','9','10','11','12']
            years = ['1','2']
            with open(os.path.join(directory, f'{ticker}-2years-hourly.csv'), 'w') as f:
                for year in years:
                    for month in months:
                        slice = f'year{year}month{month}'
                        url_json = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&interval={interval}&slice={slice}&apikey={key}'
                        response = r.get(url_json)
                        f.write(response.text)
                        time.sleep(sleep_time) 
        else:
            url_json = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&interval={interval}&slice={slice}&apikey={key}'
            response = r.get(url_json)
            with open(os.path.join(directory, f'{ticker}-30days-hourly.csv'), 'w') as f:
                f.write(response.text)

    @staticmethod
    def adjust_price_for_splits(df):
        '''
        Adjusts the price for splits in a dataframe from AlphaVantageAPI.
        Reverses date order.
        IN: Dataframe from AlphaVantageAPI per ticker
        OUT: Dataframe with adjusted prices for splits, and a list of the split dates/multipliers
        '''
        splits = df[~df['8. split coefficient'].isin([1])].copy(deep=True)
        splits['8. split coefficient'] = splits['8. split coefficient'].astype('int')
        new_df = df.copy(deep=True)
        for split in splits.index:
            new_df.loc[split-timedelta(days=1):,'1. open'] = new_df.loc[split:,'1. open'] / splits.loc[split,'8. split coefficient']
            new_df.loc[split-timedelta(days=1):,'2. high'] = new_df.loc[split:,'2. high'] / splits.loc[split,'8. split coefficient']
            new_df.loc[split-timedelta(days=1):,'3. low'] = new_df.loc[split:,'3. low'] / splits.loc[split,'8. split coefficient']
            new_df.loc[split-timedelta(days=1):,'4. close'] = new_df.loc[split:,'4. close'] / splits.loc[split,'8. split coefficient']  
        new_df = new_df.rename(columns={'1. open':'Open', '2. high':'High', '3. low':'Low', '4. close':'Close', '6. volume':'Volume'})
        new_df = new_df[::-1]
        return new_df, splits
    
    @staticmethod
    def aggregate_close_prices(directory, intraday=False, day_for_intraday='2023-03-24'):
        '''
        This function takes all the csv files in the directory and aggregates them into one dataframe.
        directory: The directory where the data is stored.
        intraday: If True it will take only points from '10:00' to '13:30'.
        '''
        df_agg = pd.DataFrame()
        for file in os.listdir(directory):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(directory, file), index_col=0, parse_dates=True)
                if df.empty:
                    continue

                if intraday:
                    df = df.loc[day_for_intraday]
                    df = df.between_time('10:00', '13:30')
                    df = df[['4. close']].iloc[::-1] 
                    if df.empty:
                        continue
                    if len(df) < 24: #24 periods of 5 min = 2 hours
                        continue
                    if df['4. close'].nunique() == 1:
                        continue

                else:
                    start_date = datetime.now() - timedelta(days=90) # real 90 days back from today
                    start_date = start_date.strftime('%Y-%m-%d')
                    df = df.loc[df.index > start_date]
                    df = df[['4. close']].iloc[::-1]
                    if df.empty:
                        continue
                    if df['4. close'].nunique() == 1:
                        continue
                
                df.columns = [file.split('-')[0]]
                df_agg = pd.concat([df_agg, df], axis=1)
                    
        return df_agg   

class OtherData:
    @staticmethod
    def get_last_day_data_all_tickers(MARKETS, directory):
        '''
        Scrapes data from www.nasdaq.com/market-activity/stocks/screener,
        in a form of downloading a csv file of the last day data of all stocks in the specified market.
        IN: List of markets to scrape data from.
        OUT: Saves 1 csv file per market, with the last day data of all stocks in that market.'''
        for file in os.listdir(r'C:\Users\Vasyl\Downloads'):
                if file.startswith('nasdaq_screener_'):
                    os.remove(os.path.join(r'C:\Users\Vasyl\Downloads', file))
        for file in os.listdir(directory):
                if file.endswith('last_day.csv'):
                    os.remove(os.path.join(directory, file))

        driver = Nav.create_driver(chromeOptions={})
        for market in MARKETS:
            url = rf'https://www.nasdaq.com/market-activity/stocks/screener?exchange={market}&render=download'
            driver.get(url)
            Nav.click_element(driver,path='//body/div[2]/div/main/div[2]/article/div[3]/div[1]/div/div/div[3]/div[2]/div[2]/div/button',wait_time=5)
            time.sleep(6)

            for file in os.listdir(r'C:\Users\Vasyl\Downloads'):
                if file.startswith('nasdaq_screener_'):
                    newName =  f'{market.lower()}_last_day.csv'
                    shutil.move(os.path.join(r'C:\Users\Vasyl\Downloads',file), rf'{directory}\{newName}')
                    print('Downloaded', newName)

        driver.quit()

class Analysis:
    @staticmethod
    def find_best_linear_stock(df):
        '''
        IN: dataframe of ticker - columns and close price - rows 
        OUT: dataframe with R2, Slope for each ticker.
        '''
        df_stats = pd.DataFrame(columns=['ticker', 'R2', 'slope'])
        for col in df.columns:
            series = df[col].copy().fillna(method='ffill')
            series = series.copy().fillna(method='backfill')

            X = series.reset_index().index.values.reshape(-1,1)
            y = series.values.reshape(-1,1)
            model = LinearRegression().fit(X,y)
        
            df_stats = pd.concat([df_stats, pd.DataFrame({'ticker': col, 'R2': model.score(X,y), 'slope': model.coef_[0]}, index=[0])], ignore_index=True)
        return df_stats
    
    @staticmethod
    def plot_correlation(series, intervals='Days'):
        '''
        This function plots the correlation between the series - Price and the index of series - Time.
        '''
        series = series.fillna(method='ffill')
        series = series.fillna(method='backfill')

        X = series.reset_index().index.values.reshape(-1,1) # 60 days of data in exact same interval - 1 day.
        y = series.values.reshape(-1,1) # prices 

        model = LinearRegression()
        model.fit(X, y)

        plt.style.use('Solarize_Light2')
        plt.figure(figsize=(10,6))
        
        plt.scatter(X, y, s=10, color='#294dba')
        plt.plot(model.predict(X), color='#d9344f')
        
        plt.xlabel(intervals)
        plt.ylabel('Price')
        plt.title(f'Linear Correlation of {series.name} with linear regression line')
        plt.show()