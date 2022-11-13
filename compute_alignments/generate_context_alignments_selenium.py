
#NOTE TO SELF: check natgeo - use the image alt tag, and sometimes it may contain embedded html
import torch
from transformers import AutoTokenizer, AutoModel
from bs4 import BeautifulSoup
from pathlib import Path
import os
import numpy as np
import pandas as pd
import pickle
import nltk.data
import yaml
import argparse

def getText(caption):
    nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    def processText(text):

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        split_arr = text.split('\n')
        res = []
        for text in split_arr:
            res.extend(nltk_tokenizer.tokenize(text))

        return res

    captionText = []

    if bool(BeautifulSoup(caption, "html.parser").find()):
        captionBS = BeautifulSoup(caption, "html.parser")

        # kill all script and style elements
        for script in captionBS(["script", "style"]):
            script.extract()    # rip it out

        captionText = processText(captionBS.get_text())
    else:
        captionText = processText(caption)