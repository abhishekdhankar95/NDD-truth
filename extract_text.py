from newspaper import Article
from constants import global_constants
import pandas as pd

glob_c = global_constants()

def extract(webpage_url_list: list, text_list: list):
    for url in webpage_url_list:
        article = Article(webpage_url_list)
        article.download()
        text_list.append(article)
    
    dump(text_list, open(glob_c.data_dir+'text_list.p', 'wb'))
    