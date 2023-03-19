import os
import shutil
from Scraping import Navigation as Nav
import time

MARKETS = ['NASDAQ', 'NYSE']

def get_last_day_data(MARKETS):
    for file in os.listdir(r'C:\Users\Vasyl\Downloads'):
            if file.startswith('nasdaq_screener_'):
                os.remove(os.path.join(r'C:\Users\Vasyl\Downloads', file))

    driver = Nav.create_driver(chromeOptions={})
    for market in MARKETS:
        url = rf'https://www.nasdaq.com/market-activity/stocks/screener?exchange={market}&render=download'
        driver.get(url)
        Nav.click_element(driver,path='//body/div[2]/div/main/div[2]/article/div[3]/div[1]/div/div/div[3]/div[2]/div[2]/div/button',wait_time=5)
        time.sleep(5)

        for file in os.listdir(r'C:\Users\Vasyl\Downloads'):
            if file.startswith('nasdaq_screener_'):
                newName =  f'{market.lower()}_last_day.csv'
                print(file, newName)
                
                shutil.move(os.path.join(r'C:\Users\Vasyl\Downloads',file), rf'D:\Study 2018 and later\Mignimind Bootcamp\Code\Stock Analysis\P1-Indicators-Chart-Analysis\Data\Last Day\{newName}')
                

        
    driver.quit()

get_last_day_data(MARKETS)
