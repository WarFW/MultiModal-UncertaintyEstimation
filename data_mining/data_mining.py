
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

            logger.info(f"MAIN: assigned ({class_id}, {class_name}) to process {dest_rank}")        
            cnt += 1
        
        for i in range(size-1):
            data = np.empty(1, dtype='i')
            comm.Recv([data, MPI.INT], source=MPI.ANY_SOURCE, tag=0)

            dest_rank = data[0]
            comm.send((-1, "NULL"), dest=dest_rank, tag=0)

    else:
        while True:
            comm.Send([np.array([rank], dtype='i'), MPI.INT], dest=0, tag=0)

            data = comm.recv(source=0, tag=0)
            class_id, class_name = data
            if (class_id == -1):
                MPI.Finalize()
                break

            baseIdx = 0
            currPage = 1
            goodFinish = False
            context_url_list, img_url_list = [], []
            while True:
                retry = False

                if baseIdx == 0:
                    this_res_dir = config['results_store_dir'] / f"{class_id}"
                    this_res_dir.mkdir(exist_ok=True)
                    logger.info(f"PROCESS {rank}: Class: {class_id}, {class_name}")
            
                try:
                    while currPage <= config['set_size']: 
                        this_context_list, this_img_list = makeQuery(currPage, class_name)
                        assert len(this_context_list) == 10 and len(this_img_list) == 10
                        currPage += 10
                        context_url_list.extend(this_context_list)
                        img_url_list.extend(this_img_list)

                        if currPage > config['set_size']:
                            #logger.info(f"PROCESS {rank} class {class_id} class name {class_name} page {currPage} image list: {img_url_list} context_url_list: {context_url_list}")
                            logger.info(f"PROCESS {rank} class {class_id} class name {class_name} page {currPage} image list: {img_url_list[0:5]} context_url_list: {context_url_list[0:5]}")
                    if (len(context_url_list) == 0 and sum(1 for x in this_res_dir.glob('*') if x.is_file()) < config['set_size'] * 3):
                        this_context_list, this_img_list = makeQuery(currPage, class_name)
                        assert len(this_context_list) == 10 and len(this_img_list) == 10
                        currPage += 10
                        context_url_list.extend(this_context_list)
                        img_url_list.extend(this_img_list)

                    img_arg_list = [[str(this_res_dir / f"{baseIdx + idx}.image"), img_url] for idx, img_url in enumerate(img_url_list)]
                    context_arg_list = [[str(this_res_dir / f"{baseIdx + idx}.context"), str(this_res_dir / f"{class_name}_{baseIdx + idx}.url"), context_url, img_url] for (idx, (context_url, img_url)) in enumerate(zip(context_url_list, img_url_list))]

                    baseIdx += len(img_url_list)

                    with concurrent.futures.ThreadPoolExecutor(max_workers=config['num_threads']) as executor:
                        executor.map(process_both, img_arg_list, context_arg_list)
                    context_url_list.clear()
                    img_url_list.clear()

                    if (sum(1 for x in this_res_dir.glob('*') if x.is_file()) >= config['set_size'] * 3):
                        goodFinish = True
                    else:
                        retry = True


                except HttpError as e:
                    if e.resp.status == 429:
                        message = ""
                        if e.resp.get('content-type', '').startswith('application/json'):
                            message = json.loads(e.content).get('error').get('errors')[0].get('message')
                        if "Queries per minute per user" in message:
                            logger.info(f"PROCESS {rank}: Queries per minute per user exceeded. Retrying in 15 seconds...")
                            sleep(15)
                            retry = True

                        else:
                            logger.critical(f"PROCESS {rank}" + str(traceback.format_exc()))
                    if e.resp.status == 400:
                        logger.critical(f"PROCESS {rank}" + str(traceback.format_exc()))
                except Exception as e:
                    logger.critical(f"PROCESS {rank}" + str(traceback.format_exc()))
                finally:
                    if goodFinish:
                        for filePath in this_res_dir.iterdir():
                            first = str(filePath).rindex("_")+1
                            last = str(filePath).rindex(".")
                            fileIdx = str(str(filePath)[first:last])
                            if not Path(this_res_dir / f"{class_name}_{fileIdx}.image").exists() or not Path(this_res_dir / f"{class_name}_{fileIdx}.context").exists() or not Path(this_res_dir / f"{class_name}_{fileIdx}.url").exists():
                                logger.critical(f"PROCESS {rank} class {class_id} class name {class_name}: fileindex {fileIdx} IS NOT A TRIPLE")
                        break
                    if not retry:
                        dir_contents = list(this_res_dir.iterdir())
                        if len(dir_contents) == 0:
                            this_res_dir.rmdir()
                        break
                    else:
                        continue

if __name__ == "__main__":
    main()