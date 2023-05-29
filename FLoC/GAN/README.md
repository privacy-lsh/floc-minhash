# LeakGAN

Taken from LeakGAN [GitHub](https://github.com/CR-Gjx/LeakGAN) with updated library dependencies and adapted for MovieLens 25M dataset.


## Requirements: 
* **Tensorflow 1.8.0**
* Python 3.6
* CUDA 9.0 (For GPU)

## Introduction
This is the synthetic data experiment of LeakGAN.

## File

LeakGANModel.py : The generator model of LeakGAN including Manager and Worker.

Discriminator.py: The discriminator model of LeakGAN including Feature Extractor and classification.

data_loader.py: Data helpy function for this experiment.

ml-25m_main.py: The Main function of this experiment.

ml-25m_convert.py: The convert one-hot number to real word.

ml-25m_eval_bleu.py: Evaluation the BLEU scores (2-5) between test datatset and generated data.

## Details 
We provide example codes to repeat the synthetic data experiments with oracle evaluation mechanisms in length 20.
To run the experiment with default parameters for length 20:
```
$ python ml-25m_main.py
```

The experiment has two stages. In the first stage, use the positive data provided by the oracle and Maximum Likelihood Estimation to perform supervise learning. In the second stage, use adversarial training to improve the generator.

When running the code, the pre-train model will be stored in folder ``ckpts_ml25m``, if you want to restore the pre-trained discriminator model, you can run:
```
$ python ml-25m_main.py --resD=True --model=leakgan_preD
``` 

if you want to restore all pre-trained model or unsupervised model (store model every 30 epoch named ``leakgan-31`` or other number), you can run:
```
$ python ml-25m_main.py --restore=True --model=leakgan_pre
``` 

After running the experiments, you can run the ``ml-25m_convert.py`` to obtain the real sentence in folder ``speech``.You also can run the ``ml-25m_eval_bleu.py`` to acquire the BLEU score in your command line.
The generated examples store in folder ``save_ml25m`` every 30 epochs, and the file named ``ml25m_31.txt`` or other numbers.

