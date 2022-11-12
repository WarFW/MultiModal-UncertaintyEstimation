
"""
Principal script for generating context alignments using retriever models. 

Input: a YAML file containing:

    
"""
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


def getText(content, bestTagDict, config):
    nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def processText(text):

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip()
                  for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        split_arr = text.split('\n')
        res = []
        for text in split_arr:
            res.extend(nltk_tokenizer.tokenize(text))

        return res

    bs = BeautifulSoup(content, "html.parser")

    # kill all script and style elements
    for script in bs(["script", "style"]):
        script.extract()    # rip it out

    for tag in bs.select(f"img"):
        if tag.attrs.get("src", None) == bestTagDict['src'] and tag.attrs.get("data-src", None) == bestTagDict['data-src'] and tag.attrs.get("alt", None) == bestTagDict['alt']:
            break
    assert tag.attrs.get("src", None) == bestTagDict['src'] and tag.attrs.get(
        "data-src", None) == bestTagDict['data-src'] and tag.attrs.get("alt", None) == bestTagDict['alt'], f"tag {tag}, best {bestTagDict}"

    textBeforeList, textAfterList = [], []

    for parent in list([tag]) + list(tag.parents):
        for prev_sibling in parent.previous_siblings:
            textBeforeList.extend(processText(prev_sibling.get_text()))
        for next_sibling in parent.next_siblings:
            textAfterList.extend(processText(next_sibling.get_text()))

    captionText = []

    if "alt" in tag.attrs and bool(BeautifulSoup(tag["alt"], "html.parser").find()):
        captionBS = BeautifulSoup(tag["alt"], "html.parser")

        # kill all script and style elements
        for script in captionBS(["script", "style"]):
            script.extract()    # rip it out

        captionText = processText(captionBS.get_text())
    elif "alt" in tag.attrs:
        captionText = processText(tag['alt'])

    textBeforeList.reverse()

    shortenedBeforeList, shortenedAfterList = [], []

    tokenCnt = 0
    idx = len(textBeforeList) - 1
    while (len(textBeforeList) - 1 - idx) < config['sentence_window_length'] and idx >= 0 and tokenCnt <= config['token_window_length']:
        nextTokens = tokenizer([textBeforeList[idx]], padding=True, truncation=True,
                              max_length=config['max_tokens'], return_tensors='pt')['input_ids']
        if (tokenCnt + nextTokens.size(dim=1) > tokenCnt):
            break
        shortenedBeforeList.append(textBeforeList[idx])
        tokenCnt += nextTokens.size(dim=1)
        idx -= 1

    tokenCnt = 0
    idx = 0
    while idx < min(10, len(textAfterList)) and tokenCnt <= config['token_window_length']:
        nextTokens = tokenizer([textAfterList[idx]], padding=True, truncation=True,
                              max_length=config['max_tokens'], return_tensors='pt')['input_ids']
        if (tokenCnt + nextTokens.size(dim=1) > tokenCnt):
            break
        shortenedAfterList.append(textAfterList[idx])
        tokenCnt += nextTokens.size(dim=1)
        idx += 1

    return shortenedBeforeList, captionText, shortenedAfterList


def compute_simscores(context_encoder, query_embedding_dict: dict, tokenized_context):
    scores = {}
    with torch.no_grad():
        ctx_emb = context_encoder(
            **tokenized_context).last_hidden_state[:, 0, :]
    for cls, query_emb in query_embedding_dict.items():
        this_score_arr = query_emb @ ctx_emb.T
        scores[cls] = torch.max(this_score_arr).item()

    return scores


def get_fileidx_list(dataset_subfolder):
    idxSet = set()
    for filePath in dataset_subfolder.iterdir():
        first = str(filePath).rindex("_")+1
        last = str(filePath).rindex(".")
        fileIdx = int(str(str(filePath)[first:last]))
        idxSet.add(fileIdx)

    res = list(idxSet)
    res.sort()