from scipy.stats import wasserstein_distance
from attack.generating_histories import init_generator, load_train_test, generate_history, generate_history_rand
from preprocessing.movielens_extractor import precompute_cityhash_movielens
import numpy as np
import logging
import pickle
import utils
import random
import tensorflow as tf
from pip._vendor.colorama import Fore, init # colorama always in pip._vendor.colorama
init(autoreset=True) # to avoid reseting color everytime

# Wasserstein metric [1] aka earth mover distance [2]
# [1] https://en.wikipedia.org/wiki/Wasserstein_metric
# [2] https://en.wikipedia.org/wiki/Earth_mover%27s_distance
# "minimum cost of turning one pile into the other;
# where the cost is assumed to be the amount of dirt moved times the distance by which it is moved"
# Note: with our token space of [0,5001]

# Similar to generating_histories.generate_from_traintest()
def init_reference_data(data_file, sample_size, seed, use_legacy_np=False): # token_to_id

    movie_tokens_hist_ndarray = load_train_test(data_file)

    if use_legacy_np:
        # Old way:
        random_samples = np.random.choice(movie_tokens_hist_ndarray.shape[0], size=sample_size, replace=False)  # uniform distrib as p=None
    else:
        # New code should use
        rng = np.random.default_rng(seed)
        random_samples =rng.choice(movie_tokens_hist_ndarray.shape[0], size=sample_size, replace=False) # uniform distrib as p=None

    random_samples.sort()  # makes it easier to check which line was sampled in logs
    # advanced indexing ? [1] https://numpy.org/doc/stable/reference/arrays.indexing.html
    sampled_token_histories = movie_tokens_hist_ndarray[random_samples, :].tolist()
    logging.info(f'Sample indices from datafile (+1 for line (start at 1) in file)\n{random_samples}')
    logging.info(f'Sample line selected:\n{utils.pretty_format(sampled_token_histories, ppindent=1, ppwidth=160, ppcompact=True)}')

    # Not needed take the 5002 tokens as is
    # sampled_movie_histories = []
    # for history in sampled_token_histories:
    #     cur_hist = []
    #     for token in history:
    #         word_id = token_to_id[token]
    #         if word_id != '<pad>' and word_id != '<oov>':
    #             # Note that token 0 and 1 are the special token
    #             cur_hist.append(token - 2)  # movieid_to_title[word_id
    #     sampled_movie_histories.append(cur_hist)
    return sampled_token_histories

def compute_wasserstein_distance(empirically_observed_values_uv, gan_history_list, ref_history_list):
    corresponding_weights_u, corresponding_weights_v = [0] * VOCABULARY_SIZE, [0] * VOCABULARY_SIZE

    # Compute the weight for u distribution i.e. the GAN samples
    for generated_history in gan_history_list:
        for token in generated_history:
            corresponding_weights_u[token] += 1
    logging.d1bg(f'generated weights:\n{corresponding_weights_u}')

    # Compute the weight for v distribution i.e. the reference empirical samples
    for ref_sample in ref_history_list:
        for token in ref_sample:
            corresponding_weights_v[token] += 1
    logging.d1bg(f'reference weights:\n{corresponding_weights_v}')

    w_dist = wasserstein_distance(empirically_observed_values_uv, empirically_observed_values_uv, corresponding_weights_u, corresponding_weights_v)
    logging.d1bg(f'{Fore.BLUE}wasserstein distance: {w_dist}')
    return w_dist

def set_seed(SEED):
    np.random.seed(SEED)  # for more repoduceability
    random.seed(SEED)
    # Note: can set tensorflow seed too
    # tf.random.set_seed(PARAMETERS['SEED']) # newer tf version
    # only valid for current graph and it is reset for each new checkpoint
    # Also no operation level seed was provided in LeakGAN
    tf.set_random_seed(SEED)

if __name__ == '__main__':
    k = 78 # 5000//64 = 78 # ie if take test set max can get
    PARAMETERS = {
        # Needed for RAND generation
        'SEED': 892735,
        'seed_inc': 0,
        # Needed for init_generator
        'SEQ_LENGTH': 32,
        'BATCH_SIZE': 64,
        'VOCABULARY_SIZE': 5002, # with the 2 special tokens ?
        'CHECKPOINT_FILEPATH': f'../GAN/ckpts_ml25m/leakgan-61',
        # Needed for logger
        'log_filename': 'wasserstein_distance',
        # Needed for others
        # Used in init_reference_data
        'reference_empirical_data': '../GAN/save_ml25m/realtest_ml25m_5000_32.txt', # '../GAN/save_ml25m/realtrain_ml25m_5000_32.txt' train or test
        'sample_size_empirical_data': k*64, # for now same as generated from GAN
        # Needed to report average and stdev
        'number_of_runs': 5,
        # Useful in logs
        'data_gen_source': 'GAN'
    }
    VOCABULARY_SIZE = PARAMETERS['VOCABULARY_SIZE']
    checkpoint_filepaths = [f'../GAN/ckpts_ml25m/leakgan_preD', f'../GAN/ckpts_ml25m/leakgan_pre',
                            f'../GAN/ckpts_ml25m/leakgan-1', f'../GAN/ckpts_ml25m/leakgan-11',
                            f'../GAN/ckpts_ml25m/leakgan-21', f'../GAN/ckpts_ml25m/leakgan-31',
                            f'../GAN/ckpts_ml25m/leakgan-41', f'../GAN/ckpts_ml25m/leakgan-51',
                            f'../GAN/ckpts_ml25m/leakgan-61']
    ##############
    ## Set SEED ##
    ##############
    set_seed(PARAMETERS['SEED'])

    #################
    ## Init logger ##
    #################
    utils.create_logging_levels()  # Need to call it before otherwise logging.d1bg etc crashes as not defined
    # Done later in loop with updated parameters
    # fh, sh = utils.init_loggers(True, PARAMETERS['log_filename'], fh_lvl=logging.DEBUG, sh_lvl=logging.DEBUG) # D2BG
    # logging.info(f'Parameters:\n{utils.pretty_format(PARAMETERS)}')

    # Example wassertein distance usage ?
    u_values = range(VOCABULARY_SIZE)
    v_values = range(VOCABULARY_SIZE)

    u_weights = [1] * VOCABULARY_SIZE
    v_weights = [3] * VOCABULARY_SIZE
    w_d = wasserstein_distance(u_values, v_values, u_weights, v_weights)
    print(f'wasserstein distance test: {w_d}')

    #####################
    ## Precomputations ##
    #####################
    hash_for_movie, title_for_movie, movie_for_title, movie_id_list = precompute_cityhash_movielens()
    # Load vocabulary
    word_ml25m, vocab_ml25m = pickle.load(open('../GAN/save_ml25m/LeakGAN_ml25m_vocab_5000_32.pkl', mode='rb'))

    ## Init reference test set and sample histories
    # Note: can choose train or test
    data_file, SAMPLE_SIZE = PARAMETERS['reference_empirical_data'], PARAMETERS['sample_size_empirical_data']
    sampled_ref_token_histories = init_reference_data(data_file, SAMPLE_SIZE, PARAMETERS['SEED'])  # word_ml25m

    ## Compute for random data generation:
    PARAMETERS['data_gen_source'] = 'RAND'
    SEQ_LENGTH = PARAMETERS['SEQ_LENGTH']
    PARAMETERS['log_filename'] = f'wasserstein_dist_rand'
    fh, sh = utils.init_loggers(log_to_stdout=True, filename=PARAMETERS['log_filename'], fh_lvl=logging.DEBUG, sh_lvl=logging.DEBUG)  # DEBUG, D2BG, INFO
    logging.info(f'Parameters:\n{utils.pretty_format(PARAMETERS)}')
    # Run multiple times and report mean, std

    w_dist_list = []
    for i in range(PARAMETERS['number_of_runs']):
        generated_rand_histories = generate_history_rand(SEQ_LENGTH, SAMPLE_SIZE, VOCABULARY_SIZE, PARAMETERS)
        # Accumulated over k*64 sentences (history)
        logging.info(f'Wasserstein distance for {SAMPLE_SIZE} histories between randomly generated and test data sample')
        w_dist = compute_wasserstein_distance(u_values, generated_rand_histories, sampled_ref_token_histories)
        w_dist_list.append(w_dist)

    logging.info(f'mean, std {np.mean(w_dist_list), np.std(w_dist_list)}\n{w_dist_list}')

    utils.remove_handlers([fh, sh])

    # Run for saved checkpoints of GAN
    PARAMETERS['data_gen_source'] = 'GAN'
    for checkpoint in checkpoint_filepaths:
        # Update parameters:
        PARAMETERS['CHECKPOINT_FILEPATH'] = checkpoint
        checkpoint_tag = checkpoint.split('/')[-1]
        PARAMETERS['log_filename'] = f'wasserstein_dist_{checkpoint_tag}'
        # Update logger
        fh, sh = utils.init_loggers(log_to_stdout=True, filename=PARAMETERS['log_filename'], fh_lvl=logging.D1BG, sh_lvl=logging.D1BG) # INFO D2BG
        logging.info(f'Parameters:\n{utils.pretty_format(PARAMETERS)}')

        ## Init generator
        sess, generator_model, discriminator = init_generator(PARAMETERS)

        # Set seed again because this is a new tensorflow graph ?
        set_seed(PARAMETERS['SEED']) # also reset seed for random and np

        # Run it multiple time and report average standard deviation:
        w_dist_list = []
        for i in range(PARAMETERS['number_of_runs']):

            ## Generate histories
            generated_histories = []
            for _ in range(k):
                generated_histories.extend(generate_history(sess, generator_model))

            # Compute wasserstein distance where probability distribution would just be the counts for the vocabulary
            # Where we removed the special tags from the vocabulary ?

            # For one history
            logging.info(f'Wasserstein distance for one history between generated and test data sample')
            w_dist1 = compute_wasserstein_distance(u_values, [generated_histories[0]], [sampled_ref_token_histories[0]])

            # Accumulated over k*64 sentences (history)
            logging.info(f'Wasserstein distance for {PARAMETERS["sample_size_empirical_data"]} histories between generated and test data sample')
            w_dist = compute_wasserstein_distance(u_values, generated_histories, sampled_ref_token_histories)
            w_dist_list.append(w_dist)

        logging.info(f'mean, std {np.mean(w_dist_list), np.std(w_dist_list)}\n{w_dist_list}')

        # Remove handlers (useful if in a loop with different params)
        utils.remove_handlers([fh, sh])