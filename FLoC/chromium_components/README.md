This directory contains the code for the components from Chromium necessary to perform the FLoC ID computation.

We use the CityHash python implementation made by [@prashnts](https://github.com/prashnts): 
https://github.com/prashnts/Hashes/blob/master/Hashes/cityhash.py.

For the [SimHash](https://github.com/chromium/chromium/blob/d7da0240cae77824d1eda25745c4022757499131/components/federated_learning/sim_hash.cc) and [Sorting LSH](https://github.com/chromium/chromium/blob/d7da0240cae77824d1eda25745c4022757499131/components/federated_learning/floc_sorting_lsh_clusters_service.cc) implementations we take as reference the Chromium source code. We also used the [floc_simulator](https://github.com/shigeki/floc_simulator) made by [@shigeki](https://github.com/shigeki) as additional reference.

None of these files have a main function so they cannot be run direcly.