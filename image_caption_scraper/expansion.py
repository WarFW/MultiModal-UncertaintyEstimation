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

def generate_synonyms(phrase,k):

    words_pos = pos_tag(word_tokenize(phrase))
    
    synonyms = []
    for (word,pos) in words_pos:
        temp = {}
        wn_pos = get_wordnet_pos(pos)
        if wn_pos:
            for syn in wordnet.synsets(word,pos=wn_pos):
                original = wordnet.synsets(word,pos=wn_pos)