## imports
from sqlalchemy import create_engine
import pandas as pd
import yfinance as yf
import json
import requests
from bs4 import BeautifulSoup




############################
## Yahoo Finance Part
###########################

# According to the DB the following stock types exist
# 1    "Stock"
# 2    "World Index"
# 3    "ETF"
# 4    "Crypto"

# offset is the number that determines when to move to the next page on a table for web scraping

offset = 0

url_data_list = [
   {'Stock' : '''{
        "id": 1,
        "base_url": "",
        "url_type": "single_table"}'''
   }
   ,
    {
        "id": 2,
        "name": "World Index",
        "base_url": "https://finance.yahoo.com/world-indices",
        "url_type": "single_table"
    },
    {
        "id": 3,
        "name": "ETF",
        "base_url": f"https://finance.yahoo.com/etfs/?offset={offset}&count=25",
        "url_type": "multi_table"
    },
    {
        "id": 4,
        "name": "Crypto",
        "base_url": f"https://finance.yahoo.com/crypto/?count=100&offset={offset}",
        "url_type": "multi_table"
    }
]



############################
## Other crypto data
##Mining rigs
#mining profitability
###########################


############################
## Data science things
###########################



############################
### Database connections and operations
############################


############################
## Data Engineering
###########################



############################
## API development
###########################