import tensorflow as tf
import numpy as np
from FLoC.GAN.Discriminator import Discriminator
from FLoC.GAN.LeakGANModel import LeakGAN
import os
import logging
import random
from FLoC.utils import time_func, SeedUpdater, reverse_target_simhash_bits
from FLoC.chromium_components import sim_hash
import pickle
from FLoC.attack.integer_programming import solve_ip_gurobi
from FLoC.preprocessing.movielens_extractor import precompute_cityhash_movielens
from pip._vendor.colorama import Fore, init # colorama always in pip._vendor.colorama
init(autoreset=True) # to avoid reseting color everytime
from codetiming import Timer

#####################################################
## Function related to the generation of histories ##
#####################################################

def init_generator(PARAMETERS):
    """
    Initialize LeakGAN from saved a checkpoint
    :param PARAMETERS: dictionary with required parameters for GAN initialization
    :return: TensorFlow session, LeakGAN generator, LeakGAN discriminator
    """
    # [1] https://github.com/kratzert/finetune_alexnet_with_tensorflow/issues/8
    tf.reset_default_graph() # seems to solve problem of reusing variables
    # Variable that modify with model often:
    SEQ_LENGTH = PARAMETERS['SEQ_LENGTH']
    BATCH_SIZE = PARAMETERS['BATCH_SIZE']

    # SEQ_LENGTH = 32 # placed in outer scope ? main function still accessible in this function ?
    vocab_filepath = f'../GAN/save_ml25m/LeakGAN_ml25m_vocab_5000_32.pkl'
    vocab_size = 5002
    # with batch size 1 had a shape error with: roll_out/while/Merge_5:0 is not an invariant for the loop
    # use the default but could try other values
    # BATCH_SIZE = 64  # 64 # Defined in main function

    # Others variable
    START_TOKEN = 0  # may have a conflict with what define as 0 in vocab
    EMB_DIM = 128  # embedding dimension
    HIDDEN_DIM = 128
    GOAL_SIZE = 16

    # Discriminator hyper-parameters
    dis_embedding_dim = 256
    dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 32]
    dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160, 160]
    GOAL_OUT_SIZE = sum(dis_num_filters)
    # might need some session:
    config = tf.ConfigProto()  # can add more param after
    config.gpu_options.allow_growth = True # so that process does not fill up full memory from start
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5 # not sure
    sess = tf.Session(config=config)
    # sess.run(tf.global_variables_initializer()) # not needed if restore from weights

    discriminator = Discriminator(SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, dis_emb_dim=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                  batch_size=BATCH_SIZE, hidden_dim=HIDDEN_DIM, start_token=START_TOKEN,
                                  goal_out_size=GOAL_OUT_SIZE, step_size=4)
    leakgan = LeakGAN(SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, emb_dim=EMB_DIM, dis_emb_dim=dis_embedding_dim,
                      filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                      batch_size=BATCH_SIZE, hidden_dim=HIDDEN_DIM, start_token=START_TOKEN,
                      goal_out_size=GOAL_OUT_SIZE, goal_size=GOAL_SIZE, step_size=4, D_model=discriminator)

    restore_checkpoint(sess, filepath=PARAMETERS['CHECKPOINT_FILEPATH'])

    return sess, leakgan, discriminator

def restore_checkpoint(sess, filepath=None, model_path = f'../GAN/ckpts_ml25m'):
    """
    Restore the weights for LeakGAN given a checkpoint
    :param sess: TensorFlow session
    :param filepath: checkpoint filepath
    :param model_path: when no filepath is specified restore from latest stored checkpoint
    :return: void
    """
    saver_variables = tf.global_variables()
    logging.debug(f'saver variables: {saver_variables}')  # list all variable restored, might not need them all
    saver = tf.train.Saver(saver_variables)
    if filepath is None:
        model = tf.train.latest_checkpoint(model_path)
        logging.info(f'Loading latest checkpoint path: {model}')
        model_ckpt_path = model.replace(os.sep, '/')
    else:
        logging.info(f'Loading sepecified checkpoint: {filepath}')
        model_ckpt_path = filepath

    saver.restore(sess, model_ckpt_path)


# @time_func
@Timer(name='GAN generator', text=f"{Fore.LIGHTMAGENTA_EX}Time spent for GAN Generation {{:.5f}}", logger=logging.debug) # logging.warning
def generate_history(sess, generator_model):
    """
    Generate histories using the LeakGAN
    :param sess: TensorFlow session
    :param generator_model: LeakGAN generator
    :return: generated histories
    """
    generated_history = generator_model.generate(sess, 1.0)
    # Tensorflow returns a numpy array by default
    return generated_history

# @time_func
@Timer(name='Random generator', text=f"{Fore.LIGHTMAGENTA_EX}Time spent for Random Generation {{:.5f}}", logger=logging.warning)
def generate_history_rand(seq_len, batch_size, vocab_size, PARAMETERS, use_numpy=True):
    """
    Generate histories by randomly sampling from a set (used as baseline)
    :param seq_len: Maximum length of an history
    :param batch_size: Number of history generated in a batch
    :param vocab_size: number of elements in the set sample movies from
    :param PARAMETERS: Dict with parameters, required for incrementing the seed used for sampling
    :param use_numpy: If want faster generation using numpy
    :return: Generated history batch
    """
    if use_numpy:
        # This would produce same generation with each run because of start seed ?
        # with SeedUpdater('gen_rand_hist', PARAMETERS):
        #     # old code using randint
        #     batch = np.random.randint(0, vocab_size, (batch_size, seq_len)) # here [0, vocab_size[

        # Newer code should use integers from default_rng
        rng = np.random.default_rng(None) # Proper way to set seed without setting the global one
        batch = rng.integers(0, vocab_size, (batch_size, seq_len)) # here [0, vocab_size[ as default is endpoint=False
        # To check that the generated batch are not the same
        # logging.warning(f'batch: {batch}') # commented out as logging is slow
    else:
        batch = []
        # vocab_size = len(token_vocab) # no passed as argument
        for i in range(batch_size):
            cur_sentence = []
            for word in range(seq_len):
                # update_seed()
                # logging.debug(f'{Fore.RED}before seed inc: {PARAMETERS["seed_inc"]}')
                with SeedUpdater('gen_rand_hist', PARAMETERS):
                    # improved generation with use_numpy branch above
                    selected_token = random.randint(0, vocab_size-1) # Both endpoints are included
                    logging.debug(f'selected token: {selected_token}') # if fixed seed would always select the same
                # logging.debug(f'{Fore.RED}side effect on seed_inc: {PARAMETERS["seed_inc"]}')
                cur_sentence.append(selected_token)
            batch.append(cur_sentence)

    return batch


# Use a context manager
# def update_seed():
#     # Useful if set a seed and call random.smth multiple time and expect different output
#     if PARAMETERS['SEED'] is not None:
#         PARAMETERS['SEED'] += 1 # Otherwise select same one each time
#         random.seed(PARAMETERS['SEED'])

def generate_target_simhash(movie_histories, movieid_to_title, output_hash_bitcount, PARAMETERS):
    """
    Generate a target SimHash using an history from the complete MovieLens dataset
    :param movie_histories: All movie histories in the MovieLens dataset
    :param movieid_to_title: maps a movieID to its title
    :param output_hash_bitcount: Output length of the SimHash
    :param PARAMETERS: Dict containing parameters, required are the seed increments and sequence length.
    :return: the generated SimHash along with the movie history used to generate it
    """
    # update_seed() # expect different output on each run whp
    with SeedUpdater('gen_target_simhash', PARAMETERS):
        random_history_sample = random.randint(0, len(movie_histories)-1) # Both endpoints are included
        history = movie_histories[random_history_sample]

        # Could match seed use to generate data
        random.shuffle(history) # done in place


    movie_counter = 0
    # 0 is padding symbol
    generated_movie_history = []
    for movie_id in history:
        if movie_counter >= PARAMETERS['SEQ_LENGTH']: # SEQ_LENGTH defined in main as a sort of global variable
            break
        if movie_id in movieid_to_title:
            generated_movie_history.append(movieid_to_title[movie_id])
            movie_counter += 1

    generated_simhash = sim_hash.sim_hash_strings(generated_movie_history, output_dimensions=output_hash_bitcount)

    # if generate for more than one simhash need to return lists
    return generated_simhash, generated_movie_history

def generate_from_traintest(data_file, sample_size, movieid_to_title, output_hash_bitcount, token_to_id, seed=None):
    """
    Generate target SimHashes from movie histories in the `data_file` (e.g., train or test set)
    :param data_file: File containing the training or test movie histories
    :param sample_size: How many SimHashes wants to generate
    :param movieid_to_title: mapping from movieIDs to titles
    :param output_hash_bitcount: output bit length of SimHash
    :param token_to_id: maps token encodding of movieIDs back to movieIDs
    :param seed: used to generate the same target SimHashes on different run
    :return: a list of the generated target SimHashes along with the input histories
    """
    # sample_size represents number of history to sample from train/test data
    movie_tokens_hist = load_train_test(data_file)
    # with SeedUpdater('gen_from_test', PARAMETERS) # not needed here because for now only call func once
    # it should not change default behavior of always making different choice ?
    # This does not work with np global set seed
    rng = np.random.default_rng(seed) # By default seed should be None
    random_samples = rng.choice(movie_tokens_hist.shape[0], size=sample_size, replace=False)
    # This worked with np global set seed
    # random_samples = np.random.choice(movie_tokens_hist.shape[0], size=sample_size, replace=False) # uniform distrib as p=None
    random_samples.sort() # makes it easier to check which line was sampled in logs
    # advanced indexing ? [1] https://numpy.org/doc/stable/reference/arrays.indexing.html
    sampled_token_histories = movie_tokens_hist[random_samples, :].tolist()
    logging.info(f'Sample indices from datafile (+1 for line (start at 1) in file) {np.sort(random_samples)}') # np.sort coppy, .sort() inplace
    logging.info(f'Sample line selected: {sampled_token_histories}')
    sampled_movie_histories = []
    for history in sampled_token_histories:
        cur_hist = []
        for token in history:
            word_id = token_to_id[token]
            if word_id != '<pad>' and word_id != '<oov>':
                cur_hist.append(movieid_to_title[word_id])
        sampled_movie_histories.append(cur_hist)

    generated_simhashes = []
    for movie_hist in sampled_movie_histories:
        # note hist is not a set but should only have unique movies
        generated_simhashes.append(sim_hash.sim_hash_strings(movie_hist, output_dimensions=output_hash_bitcount))
    # Note return lists
    return generated_simhashes, sampled_movie_histories

def load_train_test(data_file):
    """
    Load a file (e.g., train or test data)
    :param data_file: the .txt file to load
    :return: an ndarray containing the data in the file
    """
    # Can check dataloader create_batches() func for GAN:
    # with open(data_file, 'r') as f:
    #     token_stream = []
    #     for line in f:
    #         line = line.strip()
    #         line = line.split()
    #         parse_line = [int(x) for x in line]
    #         if len(parse_line) == SEQ_LENGTH:
    #             token_stream.append(parse_line)
    movie_token_data = np.loadtxt(data_file, dtype=int, delimiter=' ')
    return movie_token_data


def discriminate_computed_hist(movieid_set, movieid_encoder, discriminator, sess, SEQ_LENGTH, data_gen_source):
    """
    Apply LeakGAN's discriminator on a given set of movies
    :param movieid_set: Movie history set
    :param movieid_encoder: maps the movieIDs to their encodings used by LeakGAN
    :param discriminator: LeakGAN discriminators
    :param sess: TensorFlow session
    :param SEQ_LENGTH: Maximum length for a history
    :param data_gen_source: Source of generated history (either GAN or Random) used for runtim benchmarks
    :return: the discriminator's output probabilities
    """
    # output from preimage attack mapped to vocab:
    # note 0 is symbol for padding, could also use vocab tag <pad>
    encoded_movie_history = [[0] * SEQ_LENGTH]  # dimension should be [batch_size, sequence_length]=[1,32]
    for i, movie_id in enumerate(movieid_set):
        encoded_movie_history[0][i] = movieid_encoder[movie_id]

    # check main get_reward for example of code that gets ypred_auc
    with Timer(name=f'Discriminator {data_gen_source}', text=f"{Fore.LIGHTMAGENTA_EX}Time spent for Discriminator {{:.5f}}", logger=logging.warning):
        feed = {discriminator.D_input_x: encoded_movie_history,
                discriminator.dropout_keep_prob: 1.0}  # the dropout var is needed
    preds = sess.run([discriminator.ypred_for_auc], feed)
    logging.debug(f'Labels used for training: [0,1] for positive (real data) | [1,0] for negative (generated data) see dataloader and discriminator')
    logging.debug(f'predictions: {preds}')
    predicted_class = np.argmax(preds)
    # predicted_class == 1 -> real data / predicted_class == 0 -> generated data : labeled by generator
    color = Fore.GREEN if predicted_class == 1 else Fore.RED
    class_label = 'generated' if predicted_class == 0 else 'real'
    logging.debug(f'{color}argmax: {predicted_class}, {class_label} data')

    # counter_stats[f'disc_label_as_{class_label}'] += 1 # var defined in main # currently also done in main func
    return preds


# Returns a tuple
def token_to_id_history(token_history, hash_for_movie=None, word_id_from_token=None):
    """
    Helper function to transform a token movie history into a movieID history, optionally with the associated hash history
    :param token_history: Input history of movie tokens
    :param hash_for_movie: mapping from movieIDs to hash of movie title
    :param word_id_from_token: mapping from movie token to movieID
    :return: the movieID history optionally tupled with the movie title hash history
    """
    if word_id_from_token is None:
        logging.info(f'Load token (encoding) to word (movieid) mapping...')
        # Load vocabulary (word_ml25m, vocab_ml25m)
        word_id_from_token, _ = pickle.load(open('../GAN/save_ml25m/LeakGAN_ml25m_vocab_5000_32.pkl', mode='rb'))

    token_history_set = set(token_history)
    id_history = []
    if hash_for_movie is not None:
        hash_history = []

    # Use a set cause duplicate are not allowed
    for token in token_history_set:
        cur_movieid = word_id_from_token[token] # get the word from token
        if cur_movieid != '<pad>' and cur_movieid != '<oov>':
            # With the history being a set we should not have duplicate movies
            id_history.append(cur_movieid)
            if hash_for_movie is not None:
                hash_history.append(hash_for_movie[cur_movieid])

    return_tuple = (id_history,) # not a tuple if no comma (a,)
    if hash_history is not None:
        return_tuple = return_tuple + (hash_history,)

    return return_tuple


# Use for k-anonymity evaluation
# Returns a cluster (card >= cluster_size) of movie id history matching target_simhash
def generate_movie_simhash_cluster(cluster_size, target_simhash, hash_bitlength,
                                   sess, generator_model, title_for_movie, hash_for_movie, token_decoder=None):
    """
    Generate a cluster of users whose history match the same target SimHash
    :param cluster_size: minimum required size for the cluster
    :param target_simhash: the target SimHash for users in the cluster
    :param hash_bitlength: the output length of the SimHash
    :param sess: TensorFlow session
    :param generator_model: LeakGAN's generator
    :param title_for_movie: mapping from movieIDs to movie titles
    :param hash_for_movie: mapping from movieID to hash of movie title
    :param token_decoder: mapping from movie token to movieID
    :return: a list of the histories of generated users in the cluster
    """
    # Output
    movie_hist_cluster = []

    # Precompute hash for movies
    # do this precomputation outside this function cause reading movie file each time is slow
    # hash_for_movie, title_for_movie, movie_for_title, movie_id_list = precompute_cityhash_movielens()

    # Reverse the target simhash for integer programming constraints:
    r_target_bits = reverse_target_simhash_bits(target_simhash, hash_bitlength)

    history_count = 0
    while history_count < cluster_size:
        gen_history = generate_history(sess, generator_model)
        gen_hist_list = list(gen_history)

        for token_hist in gen_hist_list:
            id_hist, hash_hist = token_to_id_history(token_hist, hash_for_movie=hash_for_movie, word_id_from_token=token_decoder)

            # Integer Programming
            # if need to modify those params, can pass them as args
            gurobi_needed_params = {'find_mult_sol': False, 'gurobi_timelimit': 10, 'timer_nametag': f'cluster bitlen: {hash_bitlength}'}
            # note that some optional params not changed like sol_count_limit etc
            simhash_model_gp, subset_selection_gp, histories_gp = \
                solve_ip_gurobi(hash_hist, hash_bitlength, r_target_bits, id_hist, gurobi_needed_params)

            # No solutions if histories_gp is empty
            if len(histories_gp) > 0:
                for id_history_gp in histories_gp:
                    # Replace movie id in history by associated movie title for simhash computation (expects a set)
                    title_hist_subset = set([title_for_movie[movie_id] for movie_id in id_history_gp])
                    computed_simhash = sim_hash.sim_hash_strings(title_hist_subset, output_dimensions=hash_bitlength)
                    # Note that due to how gurobi constraint works it might happen that the simhash differs
                    if computed_simhash == target_simhash:
                        # Possible extension to apply discriminator on history subsample
                        # ie discriminator to check if subset are labeled real or not

                        # only care about the movieid as can retrieve rest from it
                        movie_hist_cluster.append(id_history_gp)
                        history_count += 1 # separate counter for loop termination
                    else:
                        logging.debug(f'Movie history ({len(title_hist_subset)}) does not match target simhash '
                                      f'(target){target_simhash:b}=={computed_simhash:b}(computed):\n{title_hist_subset}\n')

    return movie_hist_cluster