This directory contains some components of our attacks.

The files `generating_histories.py` and `integer_programming.py` contains parts of our attacks that are mainly used in `../evaluation/pipeline.py`.
* `generating_histories.py` is focused on using the GAN to generate user histories. It is the only file in this folder without a main function. So it cannot be run directly. 
* `integer_programming.py` uses integer programming to perform a preimage attack on SimHash.

The two remaining files use a heuristic-based preimage attack on SimHash and can be run individually. Some parameters may need to be changed.
* `preimage.py` contains the heuristic-based preimage attack.
* `leakGAN_dis.py` applies the GAN's discriminator on the output of the heuristic-based preimage attack on SimHash.

The files making use of the GAN needs it to be trained as they need access to saved checkpoints files to restore the weights of the model.

###### Miscellaneous
Some of the code may have only be run with JetBrains PyCharm's Python Console, since it provides variables preview and inspection features.
However, when disabled some code may not run properly since it seems some variables' scope changed with the use of the Python console.

In addition, when using Python multiprocessing with Pycharm's Python console they may be some problems with function that 
could not be pickled. Nevertheless, disabling the Python console could remove the issue with pickle.