# Attack on the MinHash Hierarchy
## Prerequisites

A requirements.txt file is also provided.
```shell
pip install -r requirements.txt
```

## Dataset

The dataset is available for download [here](https://archive.ics.uci.edu/ml/datasets/Taxi+Service+Trajectory+-+Prediction+Challenge,+ECML+PKDD+2015#)

## Attacks

* `minhash.py` contains preimage attacks for MinHash.
* `porto_trajectories.py` contains the attack on a reproduction of the settings from the MinHash Hierarchy system experiments.

## Usage 

Both file contain a main function and can be run individually.
```shell
python3 minhash.py
python3 porto_trajectories.py
```
It is also possible to play with some of the variable parameters in the main function.