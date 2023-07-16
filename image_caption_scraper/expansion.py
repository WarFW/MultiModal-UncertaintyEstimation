from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
import itertools
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from loguru import logger

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def generate_synonyms(phrase,k