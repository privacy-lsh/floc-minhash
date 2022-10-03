
The files `movielens_extractor.py` and `tranco_preprocessor` do not need to be run individually. The necessary functions will be called when needed elsewhere.

However, if changes to the generation of the training and test data for the GAN are to be made the file `vocab_corpus_builder.py` needs to be run.
Some part of the file `vocab_corpus_builder` are taken and adapted from [TILGAN](https://github.com/shizhediao/TILGAN/blob/main/unconditional_generation/utils.py) by [@shizhediao](https://github.com/shizhediao).
TILGAN was also tried in addition to LeakGAN, however we had better results when experimenting with LeakGAN.