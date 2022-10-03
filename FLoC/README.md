# Attacks on FLoC

This repository contains the code for the attack components and evaluation
adapted from the one for the technical report **"Analysing and exploiting Google's FLoC advertising proposal"**.

## Prerequisites

This project makes use of [LeakGAN](https://github.com/CR-Gjx/LeakGAN) by [@CR-Gjx](https://github.com/CR-Gjx) and for compatibility requires the use of an older tensorflow version. 
It was updated to Python 3.6 and TensorFlow 1.8.0.

For convenience we provide a conda environment file (used on Windows 10). Some listed dependencies may not be needed anymore.
```shell
conda env create -f attacks-on-floc.yml
```

A requirements.txt file is also provided. It however does not contain all dependencies that are in the conda env.
```shell
pip install -r requirements.txt
```

## Data
We provide the file used during the FLoC Origin Trial to map SimHashes to FLoC ID.

The Tranco list used can be found [here](https://tranco-list.eu/list/NLKW/1000000).

The MovieLens 25M dataset can be downloaded [here](https://grouplens.org/datasets/movielens/25m/).

## Directory Structure

```
FLoC
├── attack
│   ├── generating_histories.py
│   ├── ...
│   └── README.md
├── chromium_components
│   ├── cityhash.py
│   ├── ...
│   └── sorting_lsh_cluster.py
├── chromium_floc
│   ├── floc_sample.py
│   └── setup.py
├── data
│   ├── Floc
│   │   └── 1.0.6
│   │       └── SortingLshClusters
│   ├── ml-25m
│   │   └── ...
│   ├── host_list.json
│   └── tranco_NLKW.csv
├── evaluation
│   ├── logs
│   ├── saved
│   ├── anonymity_evaluation.py
│   ├── ...
│   └── wasserstein_distance.py
├── GAN
│   ├── chkpts_ml25m
│   ├── save_ml25m
│   ├── dataloader.py
│   ├── ...
│   └── README.md
├── preprocessing
│   ├── movielens_extractor.py
│   ├── ...
│   └── vocab_corpus_builder.py
├── README.md
└── utils.py
```






