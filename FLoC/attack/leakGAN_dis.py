from FLoC.GAN.Discriminator import Discriminator

from FLoC.attack.preimage import attack_with_retry_after_timeout
import pickle
from FLoC.preprocessing.movielens_extractor import precompute_cityhash_movielens
from pip._vendor.colorama import Fore, init # colorama always in pip._vendor.colorama
init(autoreset=True) # to avoid reseting color everytime
import numpy as np
# Did not seem to work
# import warnings
# warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
# import logging
# logging.getLogger('tensorflow').disabled = True
import time
from codetiming import Timer
from collections import defaultdict


# Variable that modify with model often:
SEQ_LENGTH = 32
vocab_filepath = f'../GAN/save_ml25m/LeakGAN_ml25m_vocab_5000_32.pkl'
vocab_size = 5002
BATCH_SIZE = 1 # 64 # Want 1 cause will try to classify only 1 at a time

# Others variable
START_TOKEN = 0 # could conflict with what define as 0 in vocab
HIDDEN_DIM = 128

# Discriminator hyper-parameters
dis_embedding_dim = 256
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20,32]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160,160]
GOAL_OUT_SIZE = sum(dis_num_filters)

def print_mean_std(given_list, measure_tag):
    """
    Compute mean and standard deviation of a list of elements
    :param given_list: computes mean and std on this list
    :param measure_tag: tag used for logging
    :return: the mean and standard deviation
    """
    if not given_list: # list is empty
        print(f'list was empty so replaced by 0 for mean, std computation')
        given_list = [0]
    given_list_mean = np.mean(given_list)
    given_list_std = np.std(given_list)
    print( f'{measure_tag} (avg {given_list_mean} std: {given_list_std}) : {sorted(given_list)}')
    return given_list_mean, given_list_std

def compute_confidence_disc_labeling(predictions):
    """
    Compute the confidence of the discriminator for the label that it classified history to
    :param predictions: list of the discriminator output probabilities
    :return: the confidence for the positive and negative labels when they were chosen
    """
    # Compute the confidence for the label that is > 0.5 the assigned label
    confidence_pos_label = []
    confidence_neg_label = []
    for pred in predictions:
        pos_proba = pred[0][0,1]
        neg_proba = pred[0][0,0]
        if neg_proba > 0.5:
            confidence_neg_label.append(neg_proba)
        elif pos_proba > 0.5:
            confidence_pos_label.append(pos_proba)
        else:
            # both equally likely so append to both list
            confidence_neg_label.append(neg_proba)
            confidence_pos_label.append(pos_proba)
    confidence_pos = print_mean_std(confidence_pos_label, 'confidence in positive label')
    confidence_neg = print_mean_std(confidence_neg_label, 'confidence in negative label')
    return confidence_pos, confidence_neg


def init_discriminator(model_path = f'../GAN/ckpts_ml25m'):
    """
    Init the discriminator
    :param model_path: path where stored model weights
    :return: LeakGAN's discriminator restored from latest checkpoint, along with the TensorFlow session
    """
    # could try with different checkpoints instead of default last one
    # used in generating_histories.py to init generator.
    # tf.reset_default_graph() # could be of used
    # might need some session:
    config = tf.ConfigProto()  # can add more param after
    config.gpu_options.allow_growth = True # avoid filling up whole gpu memory
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)
    # sess.run(tf.global_variables_initializer())

    discriminator = Discriminator(SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, dis_emb_dim=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                  batch_size=BATCH_SIZE, hidden_dim=HIDDEN_DIM, start_token=START_TOKEN,
                                  goal_out_size=GOAL_OUT_SIZE, step_size=4)


    # need to load everything or way to load only discriminator ?
    saver_variables = tf.global_variables()
    print(f'saver variables: {saver_variables}')
    saver = tf.train.Saver(saver_variables)
    model = tf.train.latest_checkpoint(model_path)
    print(f'latest checkpoint path: {model}')
    model_path_correct = model.replace(os.sep, '/')
    saver.restore(sess, model_path_correct)

    return sess, discriminator

def discriminate_preimages(iteration, sess, discriminator, seed=32, timeout=20, output_hash_bitcount=10):

    # if want to reinit discriminator each run could call init here

    prediction_list = []
    associated_id_histories = []
    preimage_attack_run_time = []
    discriminator_run_time = []
    common_movie_count = []
    for i in range(iteration):
        print(f'iteration {i}/{iteration}')

        # Not using utils timer as harder to retrieve time values for statistic after
        attack_start_time = time.perf_counter() # perf_counter_ns() (3.7+) like perf_counter() (in fractional seconds) but in nanoseconds
        with Timer(name=f'preimage_attack_{output_hash_bitcount}', text=f"{Fore.LIGHTMAGENTA_EX}Duration of attack {{:.5f}}"): # logger=logging.warning use default logger=print
            title_history, id_history, target_simhash, target_history = \
            attack_with_retry_after_timeout(output_hash_bitcount=output_hash_bitcount, seed=seed, timeout=timeout, dataset='movies')
        attack_duration = time.perf_counter() - attack_start_time
        print(f'attack duration: {attack_duration}')
        preimage_attack_run_time.append(attack_duration)
        associated_id_histories.append(id_history)
        common_movies = [t_movie for t_movie in set(title_history) if t_movie in set(target_history)]
        print(f'common movies ({len(common_movies)}) between target hist and generated hist: {common_movies}')
        common_movie_count.append(len(common_movies))

        # output from preimage attack mapped to vocab:
        vocab_history = [[0] * SEQ_LENGTH] # dimension should be [batch_size, sequence_length]=[1,32]
        for i, movie_id in enumerate(id_history):
            vocab_history[0][i] = vocab_ml25m[movie_id]
        # input_history = tf.convert_to_tensor(vocab_history)
        print(f'Vocab History {vocab_history}')
        input_history = np.array(vocab_history) # accept list and ndarray etc but not tensor for feed from error got

        # check main get_reward for example of code that gets ypred_auc
        feed = { discriminator.D_input_x: input_history, discriminator.dropout_keep_prob: 1.0} # the dropout var was needed
        # Time benchmark
        disc_start_time = time.perf_counter()
        with Timer(name=f'discriminator_{output_hash_bitcount}', text=f"{Fore.LIGHTMAGENTA_EX}Time spent by discriminator {{:.5f}}"):
            preds = sess.run([discriminator.ypred_for_auc], feed)
        disc_end_time = time.perf_counter()
        discriminator_run_time.append(disc_end_time - disc_start_time)

        print(f'Labels used for training: [0,1] for positive (real data) | [1,0] for negative (generated data) see dataloader and discriminator')
        print(f'predictions: {preds}')
        prediction_list.append(preds)
        # predicted_class = tf.argmax(preds, 1) # does not work output not a tensor ?
        predicted_class = np.argmax(preds)
        if predicted_class == 1: # real data
            print(f'{Fore.GREEN}argmax: {predicted_class}, real data: {predicted_class == 1}')
        else: # discriminator label this as generated data
            print(f'{Fore.RED}argmax: {predicted_class}, generated data: {predicted_class == 0}')

        # can print tf.argmax etc

    # Stats on common movies:
    print_mean_std(common_movie_count, f'{Fore.LIGHTCYAN_EX}common movie count')
    print_mean_std(preimage_attack_run_time, 'preimage attack time')
    print_mean_std(discriminator_run_time, 'discriminator time')

    return prediction_list, associated_id_histories, common_movie_count, preimage_attack_run_time, discriminator_run_time

if __name__ == '__main__':
    with Timer(name='Main Func'):
        # Load vocab to change movie_id to word in model vocabulary
        word_ml25m, vocab_ml25m = pickle.load(open(vocab_filepath, mode='rb')) # not passed in arg of func but pycharm console allow it

        np.set_printoptions(precision=5, suppress=True, linewidth=160) # helps the print of numpy array

        # Note it seems that extension to attack code made the new preimage attack runtime slower,
        # hence the reported timeout of 6s used for benchmark is now around 20s
        # it is also possible other configuration changes affected this runtime

        # Some changes of global variables in preimage.py may be needed to run this part

        sess, discriminator = init_discriminator()
        # execute discriminator once to have him cached ?
        rng = np.random.default_rng(None)  # Proper way to set seed without setting the global one
        # size (?,32) tried: (BATCH_SIZE, SEQ_LENGTH) gives BATCH_SIZE output probas, 3 dimension crash
        # (1, SEQ_LENGTH) gives 1 output proba, 2 gives 2, 3 gives 3, Note defined BATCH_SIZE=1
        input_batch = rng.integers(0, vocab_size, (5, SEQ_LENGTH)) # (BATCH_SIZE, SEQ_LENGTH)=(1,32)
        # Input expected to be of size (?, 32) from discriminator code
        feed = {discriminator.D_input_x: input_batch, discriminator.dropout_keep_prob: 1.0}  # the dropout var was needed
        with Timer(name='init_disc', text=f"{Fore.LIGHTMAGENTA_EX}Time spent by discriminator {{:.5f}}"):
            # discriminator gives one predictions for each input history
            rand_batch_pred = sess.run([discriminator.ypred_for_auc], feed)
        print(f'pred on random batch: {rand_batch_pred}')

        saved_per_bitlen_seed = dict()
        saved_preds_per_bitlen = defaultdict(list)
        saved_cmc_per_bitlen = defaultdict(list)  # common movie count

        fast_run = False
        if fast_run: # To test if code run when do modifications without running full evaluation
            seed=32 # 987 seed had a history with 1 bit for this one.
            bitlengths = [20]
            try:
                for bitlen in bitlengths:
                    predictions, histories, common_movie_cnt, attack_timings, disc_timings = discriminate_preimages(5, sess, discriminator, seed=seed, timeout=6, output_hash_bitcount=bitlen)
                    saved_preds_per_bitlen[bitlen].extend(predictions)
                    saved_per_bitlen_seed[(bitlen, seed)] = predictions
                    saved_cmc_per_bitlen[bitlen].extend(common_movie_cnt)
            except KeyboardInterrupt as ki:
                print(f'Interrupted run with keyboard: {ki}')
        else:
            bitlengths = [5, 10, 15, 20]
            seeds = [32, 987, 1331, 4263]


            for bitlen in bitlengths:
                try:
                    for seed in seeds:
                        # For 10 bits knows from previous run that should take less than 6 seconds to finish in optimal case.
                        predictions, histories, common_movie_cnt, attack_timings, disc_timings = discriminate_preimages(25,sess,discriminator,
                                                                                                      seed=seed,timeout=20,output_hash_bitcount=bitlen)
                        saved_preds_per_bitlen[bitlen].extend(predictions)
                        saved_per_bitlen_seed[(bitlen, seed)] = predictions
                        saved_cmc_per_bitlen[bitlen].extend(common_movie_cnt)
                # To be able to cancel if run for too long but at the same time compute statistic for already generated samples
                # Need to press ctrl + C for keyboard interrupt
                except KeyboardInterrupt as ki:
                    print(f'Interrupted run with keyboard: {ki}')

        # In the best case it should take around 5 seconds to solve so timeout of 6 still make sense
        # (can take the time modulo timeout to see real time it took after all rerun)
        # predictions_20b, histories_20b, attack_timings_20b, disc_timings_20b = discriminate_preimages(10, seed=None, timeout=6, output_hash_bitcount=20)

        # could pickle results to a file and choose if want to load or generate results
        # can compute average history length or other metrics with respect to histories

        print(f'Timers: {Timer.timers}')

        # Print stats on bitlengths
        for bitlen in bitlengths:
            print(f'Current bitlength: {bitlen}')
            conf_pos, conf_neg = compute_confidence_disc_labeling(saved_preds_per_bitlen[bitlen])
            print_mean_std(saved_cmc_per_bitlen[bitlen], f'common movie counts')

            # in pipeline got a better way of finding the names of Timers
            dtn = f"discriminator_{bitlen}" # disc timer name
            # Can find over method if ctrl + click on mean etc
            print(f'disc: (mean: {Timer.timers.mean(dtn)}, std: {Timer.timers.stdev(dtn)}, '
                  f'min: {Timer.timers.min(dtn)} max: {Timer.timers.max(dtn)} '
                  f'count: {Timer.timers.count(dtn)} total: {Timer.timers.total(dtn)})')
            atn = f"preimage_attack_{bitlen}"
            print(
                f'preimage_attack: (mean: {Timer.timers.mean(atn)}, std: {Timer.timers.stdev(atn)})'
                f'min: {Timer.timers.min(atn)} max: {Timer.timers.max(atn)} '
                f'count: {Timer.timers.count(atn)} total: {Timer.timers.total(atn)})')

        # codetiming made it so timings cannot be accessed
        # to access them, modified downloaded lib to remove access restriction ._timings -> .timings
        timings = Timer.timers.timings
        print(f'Timings: {timings}')

