
# MultiModal-UncertaintyEstimation
This is the codebase for implementing conformal prediction in zero-shot settings.

## Note: as of 10/14/2023 this repository is still a work in progress 
We are currently in the process of migrating from the /main and /dutta branch in our original experimental repository. Code for performing data-mining, context extraction, and context plausibility generation is being ported to this repo.

## Setup

### Required Packages:
NOTE: Need to download different torch version from website instead of 1.13.1+cu117 based on if you have CUDA or if you have a different CUDA version
```
cd MultiModal-UncertaintyEstimation
pip install -r requirements.txt
```

## Experiments

There are 2 included experiments, one for MedMNIST and one for FitzPatrick17k. Their configs are in the experiment_configs folder.

Detailed instructions on running the experiments, the required source data, and how to modify the directories inside the JSON files are provided below.

### Running MedMNIST:

Please have the MedMNIST data folder downloaded and unzipped. Modify experiment_configs/google-hybrid_medmnist_09-01-2023.json with correct directories. Further explanations and commands to run the experiments are given in the readme.

### Running FitzPatrick17k:

Please have the Fitz17k data folder downloaded and unzip. Modify experiment_configs/google-hybrid_fitzpatrick17_09-01-2023.json with correct directories. Further explanations and commands to run the experiments are given in the readme.

## Known Bugs

1. May need to modify code for cpu if your machine doesn't support GPU acceleration via CUDA

## Credits

1. The selenium-based data mining procedure in our library (all files in ./image_caption_scraper) were forked from a different repository and had been modified to suit our purposes.
2. All models including CLIP and BERT variants are sourced from HuggingFace model hub via their transformers library.