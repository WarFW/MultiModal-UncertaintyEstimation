
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