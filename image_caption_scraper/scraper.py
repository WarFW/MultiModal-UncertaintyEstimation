
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from loguru import logger
from datetime import datetime
import re
import json
import os
from pathlib import Path
from .helper import *
import uuid
import json
from .expansion import *
import traceback
# print(uuid.uuid4())
def get_public_ip_address():
        """Read the public IP address of the host"""
        #content = requests.get('https://www.whatismyip.org/my-ip-address').content
        #soup = BeautifulSoup(content,'html.parser')
        #print(content)
        #public_ip = soup.find("a",{"href":"https://whatismyip.org/en/my-ip-address"}).string
        #print(public_ip)
        public_ip = requests.get('https://api.ipify.org').content.decode('utf8')
        print(public_ip)
        return public_ip 
class Image_Caption_Scraper():
    public_ip = get_public_ip_address() 
    def __init__(self,engine="all",num_images=100,query="dog chases cat",out_dir="images",headless=True,driver="chromedriver",expand=False,k=3):
        """Initialization is only starting the web driver and getting the public IP address"""
        logger.info("Initializing scraper")
        
        self.google_start_index = 0

        self.cfg = parse_args(engine,num_images,query,out_dir,headless,driver,expand,k)
        self.start_web_driver()

    def start_web_driver(self):
        """Create the webdriver and point it to the specific search engine"""
        logger.info("Starting the engine")
        service = Service()
        chrome_options = Options()
        if self.cfg.headless:
            chrome_options.add_argument("--headless=new")
        self.wd = webdriver.Chrome(service=service, options=chrome_options)
    def close(self):
        self.wd.close()
    def scrape(self,save_images=True):
        """Main function to scrape"""
        img_data = {}
        if self.cfg.expand:
            queries_expanded = generate_synonyms(self.cfg.query,self.cfg.k)
            # queries_expanded = list(set([trans for synonym in synonyms for trans in translate(synonym)]))

            self.cfg.num_images /= len(queries_expanded)
            for i,query in enumerate(queries_expanded):
                logger.info(f"Scraping for query {query} ({i}/{len(queries_expanded)} queries)")
                self.cfg.query = query
                new_data = self.crawl()
                img_data = {**img_data, **new_data}
        else:
            logger.info(f"Scraping for query {self.cfg.query}")
            img_data = self.crawl()

        if save_images:
            self.save_images_and_captions(img_data)
        else:
            self.save_images_data(img_data)

    def crawl(self):
        if self.cfg.engine=='google': img_data = self.get_google_images()
        elif self.cfg.engine=='yahoo': img_data = self.get_yahoo_images()
        elif self.cfg.engine=='flickr': img_data = self.get_flickr_images()
        else: # all 3
            self.cfg.num_images = int(self.cfg.num_images/3) + 1
            img_data1 = self.get_google_images()
            img_data2 = self.get_yahoo_images()
            if not img_data2:
                self.google_start_index += self.cfg.num_images
                img_data2 = self.get_google_images(self.google_start_index)
            img_data3 = self.get_flickr_images()
            if len(img_data3)<self.cfg.num_images:
                self.google_start_index += self.cfg.num_images
                img_data3 = self.get_google_images(self.google_start_index)
            img_data = {**img_data1,**img_data2,**img_data3}
        return img_data

    def set_target_url(self,engine):
        """Given the target engine and query, build the target url"""
        url_index = {
            'google': "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={}&oq={}&gs_l=img".format(self.cfg.query,self.cfg.query),
            'yahoo': "https://images.search.yahoo.com/search/images;?&p={}&ei=UTF-8&iscqry=&fr=sfp".format(self.cfg.query),
            'flickr': "https://www.flickr.com/search/?text={}".format(self.cfg.query)
        }
        if not engine in url_index: 
            logger.error(f"Please choose {' or '.join(k for k in url_index)}.")
            return
        self.target_url = url_index[engine]

    def scroll_to_end(self):
        """Function for Google Images to scroll to new images after finishing all existing images"""
        logger.info("Loading new images")
        self.wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)

    def load_yahoo(self):
        """Function for Yahoo Images to scroll to new images after finishing all existing images"""
        logger.info("Loading new images")
        button = self.wd.find_element(By.NAME, 'more-res')
        button.click()
        time.sleep(3)

    def get_google_images(self,start=0):
        """Retrieve urls for images and captions from Google Images search engine"""
        logger.info("Scraping google images")
        self.set_target_url("google")

        self.wd.get(self.target_url)
        img_data = {}

        # start = 0
        prevLength = 0
        while(len(img_data)<self.cfg.num_images):
            self.scroll_to_end();i=0

            thumbnail_results = self.wd.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")

            if(len(thumbnail_results)==prevLength):
                logger.info("Loaded all images for Google")
                break

            prevLength = len(thumbnail_results)
            # logger.info(f"There are {len(thumbnail_results)} images")

            for i,content in enumerate(thumbnail_results[start:len(thumbnail_results)]):
                try:
                    self.wd.execute_script("arguments[0].click();", content)
                    time.sleep(1)

                    common_path = f'//*[@id="islrg"]/div[1]/div[{i+1}]'

                    caption = self.wd.find_element(By.XPATH, f'{common_path}/a[2]').text

                    # url = self.wd.find_elements_by_css_selector('img.n3VNCb')[0]
                    
                    url = self.wd.find_element(By.XPATH, f'{common_path}/a[1]/div[1]/img')

                    if url.get_attribute('src') and not url.get_attribute('src').endswith('gif') and url.get_attribute('src') not in img_data:

                        now = datetime.now().astimezone()
                        now = now.strftime("%m-%d-%Y %H:%M:%S %z %Z")

                        name = uuid.uuid4() # len(img_data)
                        img_data[f'{i}']={
                            'query':self.cfg.query,
                            'url':url.get_attribute('src'),
                            'caption':caption,
                            'datetime': now,
                            'source': 'google',
                            'public_ip': Image_Caption_Scraper.public_ip
                        }
                        logger.info(f"Finished {len(img_data)}/{self.cfg.num_images} images for Google.")
                except:
                    logger.debug(traceback.format_exc())
                    logger.debug("Couldn't load image and caption for Google")
                
                if(len(img_data)>self.cfg.num_images-1): 
                    logger.info(f"Finished scraping {self.cfg.num_images} for Google!")
                    # logger.info("Loaded all the images and captions!")
                    break
            
            start = len(thumbnail_results)

        return img_data

    def get_yahoo_images(self):
        """Retrieve urls for images and captions from Yahoo Images search engine"""
        logger.info("Scraping yahoo images")
        self.set_target_url("yahoo")

        self.wd.get(self.target_url)

        img_data = {}

        start = 0
        i=0
        while(len(img_data)<self.cfg.num_images):
            # Accept cookie
            try:
                button = self.wd.find_element(By.XPATH, '//*[@id="consent-page"]/div/div/div/form/div[2]/div[2]/button')
                button.click()
            except:
                pass
