from newspaper import Article
from constants import global_constants
import pandas as pd
from pickle import dump, load
import requests
from newspaper import fulltext

glob_c = global_constants()

def extract(webpage_url_list: list, text_list: list):
    for url in webpage_url_list:
        html = requests.get(url).text
        text = fulltext(html)
        text_list.append(text)
    
    dump(text_list, open(glob_c.data_dir+'text_list.p', 'wb'))
    
