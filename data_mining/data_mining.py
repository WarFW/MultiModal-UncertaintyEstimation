
# # define search params
# # option for commonly used search param are shown below for easy reference.
# # For param marked with '##':
# #   - Multiselect is currently not feasible. Choose ONE option only
# #   - This param can also be omitted from _search_params if you do not wish to define any value
# _search_params = {
#     'q': '...',
#     'num': 10,
#     'fileType': 'jpg|gif|png',
#     'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived',
#     'safe': 'active|high|medium|off|safeUndefined', ##
#     'imgType': 'clipart|face|lineart|stock|photo|animated|imgTypeUndefined', ##
#     'imgSize': 'huge|icon|large|medium|small|xlarge|xxlarge|imgSizeUndefined', ##
#     'imgDominantColor': 'black|blue|brown|gray|green|orange|pink|purple|red|teal|white|yellow|imgDominantColorUndefined', ##
#     'imgColorType': 'color|gray|mono|trans|imgColorTypeUndefined' ##
# }
from math import ceil
import os
import pandas as pd
import numpy as np
from pathlib import Path
from time import sleep
from googleapiclient.errors import HttpError
import json
import shutil
import logging
import sys
from mpi4py import MPI
import requests
import concurrent.futures
import os
import pandas as pd
import numpy as np
from pathlib import Path
from time import sleep
import json
import shutil
import logging
import sys
from mpi4py import MPI
import threading
from PIL import Image
import requests
from io import BytesIO
import concurrent.futures
import traceback
from requests.exceptions import ConnectTimeout
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from urllib.parse import urlparse
from googleapiclient.discovery import build
import pickle
import yaml
import argparse
from data_mining_utils import *
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", help="the path to the yaml config file.", type=str)
    args = parser.parse_args()

    config = {}
    with open(args["config"], "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    for k, v in enumerate(config):
        if (k in ['results_store_dir, calibration_dataset_dir']):
            config[k] = Path(v)

    config['results_store_dir'].mkdir(exist_ok=False)

    class_df = pd.read_csv(config['class_list_csv'])

    cseBuild = build("customsearch", "v1", developerKey=os.environ["GOOGLE_API_KEY"])
    cse = cseBuild.cse()

    READY = np.array([0], dtype='i')
    TERMINATE = np.array([-1], dtype='i')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        config['results_store_dir'].mkdir(exist_ok=False)
    else:
        sleep(10)

    logging.basicConfig(filename=config['results_store_dir']/f"events_{rank}.log", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filemode="w")
    logger = logging.getLogger("my-logger")
    logger.setLevel(logging.INFO)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    API_KEY = os.environ["GOOGLE_API_KEY"]
    PROJECT_CX_KEY = os.environ["GOOGLE_CX_ID"]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) '
                        'AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/27.0.1453.94 '
                        'Safari/537.36'
    }

    if rank == 0:
        logger.info(f"MAIN: using {size} processes.")
        
        class_dict = {}
        class_df = pd.read_csv(config['class_list_csv'])
        
        for i in range(len(class_df)):
            row = class_df.iloc[i, :]
            class_dict[row["Class Index"]] = row["Class"]

        request_list = [(class_id, class_name) for class_id, class_name in class_dict.items()]

        cnt = 0
        while cnt < len(request_list):
            class_id, class_name = request_list[cnt]

            if (config['class_id_start'] is not None and class_id < config['class_id_start']):
                continue
            if (config['class_id_end'] is not None and class_id >= config['class_id_end']):
                continue
            if class_id in config['class_id_exclude_list']:
                continue

            data = np.empty(1, dtype='i')
            comm.Recv([data, MPI.INT], source=MPI.ANY_SOURCE, tag=0)

            dest_rank = data[0]
            comm.send((class_id, class_name), dest=dest_rank, tag=0)