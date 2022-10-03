The code here is mostly taken and adapted from Shigeki's [floc_simulator](https://github.com/shigeki/floc_simulator).
In particular the [main](https://github.com/shigeki/floc_simulator/blob/WIP/demos/floc_sample/main.go) and [setup](https://github.com/shigeki/floc_simulator/blob/WIP/packages/floc/setup.go) files.

To perform the FLoC compuation according to the version used during the Origin Trial, one can run `floc_sample.py`.

It makes use of `../data/host_list.json` for the browsing history and `../data/Floc/1.0.6/SortingLshClusters` to map SimHashes to cohort IDs.

