# Attacks on FLoC

This folder contains the code for the attacks on FLoC and their evaluation. 
It is adapted from the technical report **"Analysing and exploiting Google's FLoC advertising proposal"**.

## Table of Contents

- [Requirements](#requirements)
- [Data](#data)
- [Directory Structure](#directory-structure)
- [Usage](#usage)


## Requirements

This project makes use of [LeakGAN](https://github.com/CR-Gjx/LeakGAN) by [@CR-Gjx](https://github.com/CR-Gjx).
Note that they do not provide a license for their code.

It requires the use of Python 2.7 and Tensorflow r1.2.1 but we updated the requirements to Python 3.6 and TensorFlow 1.8.0.

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

## Usage

First the LeakGAN needs to be trained, this can be done by running the `ml-25m_main.py` file in the `GAN` folder.
It was trained on a Windows computer using an Nvidia GeForce RTX 2070 Super.

Then once some checkpoints from the GAN training have been saved, it is possible to run `pipeline.py` in the `evaluation` folder.
It runs the GAN-IP attack on FLoC.

Each folder containing code has its own `README.md` which provide additional information on its content. 

