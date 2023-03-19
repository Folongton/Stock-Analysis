import sys
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import StaleElementReferenceException as ex
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import *
from webdriver_manager.chrome import ChromeDriverManager


class Navigation:

    def create_chrome_options():
        chromeOptions = webdriver.ChromeOptions()
        prefs={ "download.default_directory": r"C:\Users\Vasyl\Desktop",
                "download.prompt_for_download": False,
                "download.directory_upgrade": True, 
                "safebrowsing.enabled": True}
        chromeOptions.add_experimental_option('prefs', prefs)
        chromeOptions.add_argument('--user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"')
        chromeOptions.add_argument('--headless')
        chromeOptions.add_argument('--start-maximized')
        chromeOptions.add_argument('--incognito')
        chromeOptions.add_argument('log-level=3')
        return chromeOptions

    def create_driver(chromeOptions):
        driver = webdriver.Chrome(options=chromeOptions, executable_path='.\seleniumWebdriver\chromedriver.exe')
        return driver

    def click_element(driver,path,wait_time): 
        stale_element = True
        while stale_element == True:
            try:
                element_link = WebDriverWait(driver, wait_time).until(ec.element_to_be_clickable((By.XPATH, path))) 
                element_link.click()
                return True
            except StaleElementReferenceException:
                stale_element = True
            except TimeoutException:
                print('Timeout! Cannot find Path: ' + path + '. Function: ' + sys._getframe().f_code.co_name)
                return False
            
def driver_update_needed(driver):
        browser_version = driver.capabilities['browserVersion']
        driver_version = driver.capabilities['chrome']['chromedriverVersion'].split(' ')[0]
        print('Your Chrome version is : ' + browser_version)
        print('Your driver version is : ' + driver_version)
        if browser_version[0:2] != driver_version[0:2]: 
            print("Please download correct chromedriver version from: 'https://chromedriver.chromium.org/downloads' ")
            print('You can press enter and try to continue (only for testing purposes)')
            input()
            return True
        else:
            return False