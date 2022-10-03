import math
from collections import Counter
from FLoC.preprocessing.movielens_extractor import read_ratings
from FLoC.attack.preimage import attack_with_retry_after_timeout
from FLoC.utils import pretty_format, init_loggers, multiprocess_run_release_gpu
from FLoC.attack.generating_histories import *
from FLoC.attack.integer_programming import *
import re
import multiprocessing
from pip._vendor.colorama import Fore, init # colorama always in pip._vendor.colorama
init(autoreset=True) # to avoid reseting color everytime
from tqdm import tqdm
from codetiming import Timer
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

def update_common_movies(name, movieid_history, common_movies_len_stats, target_history, title_for_movie, js_hist_data=None):
    """
    Update common movie statistics
    :param name: name for the common_movies_len_stats string identifier
    :param movieid_history:
    :param common_movies_len_stats: Counter
    :param target_history:
    :param title_for_movie:
    :param js_hist_data: jaccard similarity hist data aggregation, computes jaccard sim if not None
    :return:
    """
    # Update with side effects
    title_history_set = set([title_for_movie[movieid] for movieid in movieid_history])
    common_movies = [t_movie for t_movie in target_history if t_movie in title_history_set]
    if js_hist_data is not None:
        jaccard_sim = jaccard_similarity(title_history_set, set(target_history))
        js_hist_data.append(jaccard_sim)
        logging.d2bg(f'{Fore.BLUE}Jaccard sim between target history and {name} history: {jaccard_sim}')

    logging.d2bg(f'{Fore.BLUE}Common movies between target history and {name} history: {len(common_movies)}\n{common_movies}')
    common_movies_len_stats[f'{len(common_movies)} in common ({name}-target)'] += 1
    # Note: cannot do this in place with sideeffect ?
    # acc_common_movies_set = acc_common_movies_set.union(set(common_movies))
    return common_movies


def compute_min_hamming_distance(cur_title_history, min_hamming_dist_counter, ref_simhashes_dict_for_hamming_dist, out_dim=64):
    """
    Compute the minimum hamming distance between the `cur_title_history` and all the simhashes in `ref_simhashes_dict_for_hamming_dist`
    :param cur_title_history: a history represented as a list of titles
    :param min_hamming_dist_counter: the counter to update in place to compute statistics
    :param ref_simhashes_dict_for_hamming_dist: the simhashes for the test histories used as reference
    :param out_dim: the output dimension for the simhash computation (default 64 bits)
    :return: Nothing
    """
    cur_target_simhash = sim_hash.sim_hash_strings(set(cur_title_history), output_dimensions=out_dim)
    min_ham_dist, closest_ref_simhashes = hamming_closest_simhash(cur_target_simhash, ref_simhashes_dict_for_hamming_dist.keys())
    closest_ref_simhash_titles = [ref_simhashes_dict_for_hamming_dist[ref_simhash] for ref_simhash in closest_ref_simhashes]
    logging.d2bg(f'min hamming distance with reference: {min_ham_dist}')
    msg_to_log = f'closest simhashes ({len(closest_ref_simhashes)}) to:\n{cur_target_simhash:064b} are\n'
    for close_simhash in closest_ref_simhashes:
        msg_to_log += f'{close_simhash:064b}\n'
    logging.d2bg(f'{msg_to_log}')
    logging.debug(f'corresponding histories: {closest_ref_simhash_titles}')
    min_hamming_dist_counter[f'{min_ham_dist} dist.'] += len(closest_ref_simhashes)
    if min_ham_dist <= 8:
        # Note maybe compute common movies for that or sort the print with pprint
        logging.info(
            f'{Fore.BLUE}simhash dist {min_ham_dist} corresponding histories (generated | test ref):')
        # Sorted returns sorted list when .sort() returns None and sort in place here want return sorted list
        logging.info(
            f'Sorted GAN hist:\n{sorted(cur_title_history)}')  # set(title_history) , utils.pretty_format(sorted(title_history), use_json=True)
        for test_history in closest_ref_simhash_titles:
            common_movies_mdist = [t_movie for t_movie in test_history if t_movie in set(cur_title_history)]
            logging.info(f'Sorted closest Hamming dist test hist:\n{sorted(test_history)}\n'
                         f'{Fore.LIGHTCYAN_EX}Common movies between test history and gan history: '
                         f'{len(common_movies_mdist)}\n{sorted(common_movies_mdist)}')


def jaccard_similarity(setA, setB):
    """
    Computes the jaccard similarity between setA and setB.
    J(A,B) = |A ∩ B| / |A ∪ B|
    :param setA: a set
    :param setB: a set
    :return: the jaccard similarity
    """
    A_inter_B = setA.intersection(setB)
    # cardinality: len(setA) + len(setB) - len(A_inter_B)
    # A_union_B = setA.union(setB)
    card_A_union_B = len(setA) + len(setB) - len(A_inter_B)

    # J(A,B) = |A ∩ B| / |A ∪ B|
    # j_similarity = len(A_inter_B) / len(A_union_B)
    j_similarity = len(A_inter_B) / card_A_union_B


    return j_similarity


def compute_and_save_histogram(jsim_data, bin_edges, hist_tag, savepath=None):
    """

    :param jsim_data: the jaccard similarity data
    :param bin_edges: the bin edges for the histogram
    :param target_simhash: the target simhash
    :param hist_tag: the name tag for the histogram saved file
    :param savepath: the path to save the hist plot and data
    :return:
    """
    logging.d2bg(f'jsim_data: {jsim_data}')
    jsim_hist, rbin_edges = np.histogram(jsim_data, bins=bin_edges)
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.hist(jsim_data, bins=bin_edges)
    # plt.axis('off')
    # plt.title(title)
    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{savepath}/{hist_tag}.png')

    plt.close()
    # log_folder may not be accessible (scope) without pycharm python console
    np.savetxt(f"{log_folder}/{hist_tag}_js_hist_data.txt", jsim_data)
    np.savetxt(f"{log_folder}/{hist_tag}_js_hist.txt", jsim_hist)

def aggregate_counter(counter_dict):
    """
    Sum counters, namely sum aggregated counts values with the same key
    :param counter_dict: the counters in a dict (the keys could be the target simhashe of each respective counter)
    :return: the aggregated counter
    """
    out = Counter()
    for cur_cnter in counter_dict.values():
        out += cur_cnter
    return out


def percentage_of_common_movie_cnt_ge_3(common_movie_counter):
    """
    Compute the percentage of common movies with length greater or equal to 3
    :param common_movie_counter: The common movie counter
    :return: the percentage
    """
    extract_starting_number = re.compile(r'\d+')
    key_to_int = lambda key_string: int(extract_starting_number.search(key_string).group(0))
    total_count = 0
    counts_ge_3 = 0
    for k, v in common_movie_counter.items():
        cur_count = key_to_int(k)
        total_count += v
        if cur_count >= 3:
            counts_ge_3 += v

    # total = common_movie_counter.total() # New from python 3.10 (LeakGAN envf on 3.6)
    # if total != total_count:
    #     raise Exception(f'Error wrong count {total} != {total_count}')

    return counts_ge_3/total_count

# Note can move those stats functions to utils statistic section
# Weigthed average
def compute_average_common_movie_counts(common_movie_counter):
    # Compute average of common movie count between source and target
    cur_sum = 0
    total_cnt = 0
    # The start of the string is a number denoting the count:
    extract_starting_number = re.compile(r'\d+')
    for key, value in common_movie_counter.items(): # also .most_common() etc
        cur_sum += int(extract_starting_number.search(key).group(0))*value
        total_cnt += value
    return cur_sum / total_cnt
    # mean = sum(key * count for key, count in counter.items()) / sum(counter.values())
    # https://stackoverflow.com/questions/33695220/calculate-mean-on-values-in-python-collections-counter

# Note: could use same function as above just here use other func from std lib
def compute_average_min_hamming_distance(min_hamming_dist_counter):
    # The start of the string is a number denoting the count:
    extract_starting_number = re.compile(r'\d+')
    key_to_int = lambda key_string: int(extract_starting_number.search(key_string).group(0))
    # [1] https://stackoverflow.com/questions/33695220/calculate-mean-on-values-in-python-collections-counter
    average = sum(key_to_int(key) * value_count for key, value_count in min_hamming_dist_counter.items()) \
              / sum(min_hamming_dist_counter.values())
    return average

# Weighted std
def compute_standard_deviation(counter_collection, mean=None):
    extract_starting_number = re.compile(r'\d+')
    key_to_int = lambda key_string: int(extract_starting_number.search(key_string).group(0))
    if mean is None:
        # mean = compute_average_min_hamming_distance(counter_collection)
        mean = sum(key_to_int(key) * value_count for key, value_count in counter_collection.items()) \
                  / sum(counter_collection.values())
    else:
        mean_2 = sum(key_to_int(key) * value_count for key, value_count in counter_collection.items()) \
               / sum(counter_collection.values())
        if mean != mean_2:
            logging.warning(f'{Fore.RED}computed mean different from provided: {mean} != {mean_2}')

    # Weighted variance [1] https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
    var = sum(value_weight * (key_to_int(key_sample) - mean) ** 2 for key_sample, value_weight in counter_collection.items()) \
          / sum(counter_collection.values())
    std = math.sqrt(var)
    return var, std


def generate_history_matching_target_simhash(PARAMETERS, hash_for_movie, title_for_movie, word_ml25m, vocab_ml25m, ref_simhashes_dict_for_hamming_dist, use_multiprocessing=False):

    # If use multiprocessing need to init_logger again as not inherited from parent process ?
    # Note that multiprocessing did not work with pycharm's python console
    if use_multiprocessing:
        init_loggers(log_to_stdout=True, filename=PARAMETERS['log_filename'])
        logging.info(f'Parameters:\n{pretty_format(PARAMETERS)}') # log parameters for run
        # need to set seed again for each process
        np.random.seed(PARAMETERS['SEED'])
        random.seed(PARAMETERS['SEED'])


    ##########################################################
    ## Generation of target simhash and associated history  ##
    ##########################################################
    # compute simhash of some real history of movies from ML25M output_hash_bit [13, 21]
    # can check if generated history reconstruct movies in the original real history
    # output_hash_bitcount = random.randint(13, 21)
    output_hash_bitcount = random.randint(PARAMETERS['SIMHASH_BITCOUNT_RANGE'][0],
                                          PARAMETERS['SIMHASH_BITCOUNT_RANGE'][1])

    # Generate directly from MovieLens dataset
    if PARAMETERS['TARGET_SIMHASH_ORIGIN'] == 'all':
        # Note: not yet adapted to output lists to test more than one target_simhash in same run
        user_moviehistory = read_ratings(f'../data/ml-25m/ratings.csv')
        target_simhashes, target_histories = [], []
        for i in range(PARAMETERS['TARGET_SIMHASH_COUNT']):
            target_simhash, target_history = generate_target_simhash(user_moviehistory, title_for_movie,
                                                                     output_hash_bitcount, PARAMETERS)
            target_simhashes.append(target_simhash)
            target_histories.append(target_history)
    elif PARAMETERS['TARGET_SIMHASH_ORIGIN'] == 'test':
        # Generate from test dataset
        target_simhashes, target_histories = generate_from_traintest('../GAN/save_ml25m/realtest_ml25m_5000_32.txt',
                                                                     PARAMETERS['TARGET_SIMHASH_COUNT'],
                                                                     title_for_movie, output_hash_bitcount, word_ml25m,
                                                                     seed=PARAMETERS['SEED'])
        # when did not have loop on target simhashes
        # target_simhash, target_history = target_simhashes[0], target_histories[0]
    elif PARAMETERS['TARGET_SIMHASH_ORIGIN'] == 'train':
        # Generate from train dataset
        target_simhashes, target_histories = generate_from_traintest('../GAN/save_ml25m/realtrain_ml25m_5000_32.txt',
                                                                     PARAMETERS['TARGET_SIMHASH_COUNT'],
                                                                     title_for_movie, output_hash_bitcount, word_ml25m,
                                                                     seed=PARAMETERS['SEED'])
    else:
        raise ValueError(f"{PARAMETERS['TARGET_SIMHASH_ORIGIN']} should be 'all', 'test' or 'train'")

    ###################################
    ## Initialize history generation ##
    ###################################
    # With preimage attack as first test:
    use_preimage_attack = False
    if use_preimage_attack:
        # Note this attack already generate a target simhash distinct from one below (can make them the same)
        title_history, id_history, tgt_simhash, tgt_hist = attack_with_retry_after_timeout(output_hash_bitcount=output_hash_bitcount,
                                                                    timeout=20, dataset='movies')

        hash_history = []
        for movie_id in id_history:
            hash_history.append(hash_for_movie[movie_id])

        target_simhash = random.randint((1 << 13), (1 << 21) - 1)  # maybe shorter bitlength for preimage
        target_bits = bin(target_simhash)[2:2 + output_hash_bitcount]
        r_target_bits = target_bits[::-1]  # Reverse target simhash for constraint order
        simhash_problem, subset_select = solve_ip_cvxpy(hash_history, output_hash_bitcount, r_target_bits)
        if simhash_problem[0].value == 0:
            logging.info(f'Failure')
        else:
            logging.info(f'Success')
    elif PARAMETERS['data_gen_source'] == 'GAN':  # outside loop cause do not init more than once
        sess, generator_model, discriminator = init_generator(PARAMETERS)
    elif PARAMETERS['data_gen_source'] == 'RAND':
        # technically we only need the discriminator
        sess, generator_model, discriminator = init_generator(PARAMETERS)
    else:
        raise ValueError(f"{PARAMETERS['data_gen_source']} should be 'GAN' or 'RAND'")

    # Define stats need to keep accross different target simhash run
    common_movies_counter_dicts = {'GAN': dict(), 'GAN if exists IP sol': dict(), 'IP': dict()}
    min_hamming_dist_counter_dicts = {'GAN': dict(), 'GAN if exists IP sol': dict(), 'IP': dict()}
    counter_stats_total = Counter()
    stats_per_target = dict()  # dict with key target_simhash
    avg_gan_over_multitargets = 0
    gan_cmovie_omulti = [] # just accumulate averages and then compute mean, std on them ?
    gan_cmovie_cnt_ot = [] # accumulate counts for each targets (ot for outside or over targets)
    # Min hamming distance multi target stats
    avg_min_dist_gan_multitarget = 0 # DEPRECATED
    gan_mdist_omulti = [] # for every gan generated history
    gan_mdist_omulti_solip_exist = [] # for gan generated history that admit a subset matching target simhash
    gan_mdist_omulti_ip = [] # for the subset of gan history matching target simhash
    avg_ip_over_multitargets = 0
    ip_cmovie_omulti = []
    ip_cmovie_cnt_ot = []
    gan_gen_movie_total_set = set() # accumulate all (over all target hash) unique movie generated by GAN
    gan_history_length_total_list = [] # accumalate all history length for GAN
    dis_history_length_total_list = [] # accumulate all history length for Disc (after integer program)
    # Jaccard similarity multi target
    bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    jsim_hist_data_gan_mt = [] # after gan
    jsim_hist_data_gan_if_ips_mt = [] # after gan only if ip successful (non empty solution)
    jsim_hist_data_ip_mt = [] # after int prog

    if PARAMETERS['apply_disc_on_sol']:
        avg_ipr_over_multitargets = 0
        ipr_cmovie_omulti = []
        ipr_cmovie_cnt_ot = []
        confidence_gen_lbl_total = [] # accumulate confidence over all label
        confidence_real_lbl_total = []

    for target_simhash, target_history in zip(target_simhashes, target_histories):

        # target_simhash = random.randint((1 << 13), (1 << 21) - 1)# 0b101001011100110
        bit_len = target_simhash.bit_length()
        # output_hash_bitcount = min(output_hash_bitcount, target_simhash.bit_length()) # for test useful otherwise crashes
        logging.info(f'Target simhash: {target_simhash:b} hash bitcount: {output_hash_bitcount} bit length: {bit_len}')
        logging.info(f'Associated target history:\n{target_history}')
        # Note made a start of a func for that see utils.reverse_target_simhash_bits
        target_bits = bin(target_simhash)[2:2 + output_hash_bitcount]
        # Reverse target simhash for constraint order
        r_target_bits = target_bits[::-1]
        # If the bit_length of target simhash is not equal to output_hash_bitcount need to manually add leading 0 ?
        # so trailing 0 on the reverse ?
        if len(r_target_bits) < output_hash_bitcount:
            logging.debug('target simhash bit length smaller than output hash bitcount so add leading zeroes')
            while len(r_target_bits) < output_hash_bitcount:
                r_target_bits += '0'
            logging.debug(f'extended len: {len(r_target_bits)} matches output hash len: {output_hash_bitcount}')

        # cvxpy
        matching_hist_count_cp = 0  # this one would also count duplicate generated history ?
        match_hist_set_cp = set()
        acc_common_movies_cp = set()
        hist_common_movie_len_cp = []

        # gurobipy
        matching_hist_count_gp = 0  # this one would also count duplicate generated history ?
        match_hist_set_gp = set()
        # For movies directly from GAN generation
        acc_common_movies_gan = set()  # common movies between history generated by GAN and target one from target simhash
        common_gan_tgt_len_cnt = Counter()  # counts of common movies between gan hist and tgt hist
        common_gan_tgt_len_cnt_if_exist_ip_sol = Counter() # Same above restricted to gan history that admits non empty IP sol
        min_hamming_dist_gan = Counter()
        min_hamming_dist_gan_if_exist_ip_sol = Counter() # Restriction from previous to gan history that admit non empty IP solution
        gan_gen_movie_hash_set = set() # accumulate unique movie generated by GAN (current simhash only)
        gan_history_length_list = [] # accumulate history length for Generator
        dis_history_length_list = [] # accumulate history length for Discriminator (after integer program)
        # For movies from int prog that matches simhash
        acc_common_movies_gp = set()
        common_movie_len_cnt_gp = Counter()  # change from dict to counter
        min_hamming_dist_ip = Counter()
        # Jaccard similarity single target
        jsim_hist_data_gan = []
        jsim_hist_data_gan_if_ips = []
        jsim_hist_data_ip = []
        if PARAMETERS['apply_disc_on_sol']:
            # For movies from int prog that matches simhash and also labeled as real by discriminator
            acc_common_movies_mtch_real_gp = set()
            common_movies_mtch_real_len_cnt = Counter()
            # For other stats
            match_hist_set_real = set()  # Discriminator labeled as real data
            match_hist_set_gen = set()  # Discriminator labeled as generated (fake) data
            confidence_real_lbl = [] # Confidence of prediction similar to leakGAN_dis
            confidence_gen_lbl = []


        counter_stats = Counter()
        # Note due to simhash constraint = 0 not taken into account by integer program can happen that some bit mismatch
        count_sol_not_matching_target_simhash = 0
        multiple_sol_counter = Counter()  # Count how many solution found in satisfiable history

        # Length of hist set:
        hist_count = 0

        # Note prints and logging in main code and function calls will slow it down
        # Default logger is print
        with Timer(name=f'while loop {PARAMETERS["timer_nametag"]}', text=f"{Fore.LIGHTMAGENTA_EX}Time spent in while loop {{:.5f}}", logger=logging.warning):
            while hist_count < PARAMETERS['TARGET_COUNT']:
                counter_stats['generated_batch'] += 1
                logging.info(f'Generate new batch of history ({counter_stats["generated_batch"]}/n)')

                # Generate data
                if PARAMETERS['data_gen_source'] == 'GAN':
                    # Also want a timer with some name parameters here:
                    with Timer(name=f'GAN_gen_{PARAMETERS["timer_nametag"]}', text=f"{Fore.LIGHTMAGENTA_EX}Time spent for GAN generation {PARAMETERS['timer_nametag']} {{:.5f}}", logger=logging.warning):
                        generated_history = generate_history(sess, generator_model)
                elif PARAMETERS['data_gen_source'] == 'RAND':
                    generated_history = generate_history_rand(SEQ_LENGTH, BATCH_SIZE, len(vocab_ml25m), PARAMETERS)

                history_list = list(generated_history)

                # Note made a start of a func for that generating_histories.token_to_id_history
                for history in history_list:
                    id_history = []
                    hash_history = []
                    title_history = [] # added after might be needed for simhash hamming distance computations
                    history_set = set(history)  # To remove duplicates according to simhash specs
                    for encoded_movie in history_set:
                        current_word = word_ml25m[encoded_movie]
                        if current_word != '<pad>' and current_word != '<oov>':
                            # With the history being a set we should not have duplicate movies
                            hash_history.append(hash_for_movie[current_word])
                            id_history.append(current_word)
                            title_history.append(title_for_movie[current_word])

                    # Check if GAN or other able to generate movies in the target movie history
                    common_movies_gan = update_common_movies(PARAMETERS['data_gen_source'], id_history, common_gan_tgt_len_cnt, target_history, title_for_movie, jsim_hist_data_gan)
                    acc_common_movies_gan = acc_common_movies_gan.union(set(common_movies_gan))
                    # Note could compute stats like average history length after GAN ?
                    # Update set containing unique movie generated by for current SimHash
                    gan_gen_movie_hash_set.update(set(title_history)) # Union but inplace ?
                    gan_history_length_list.append(len(title_history))

                    # Compute hamming distance for GAN title movie (set removed duplicate) and test data
                    if PARAMETERS['compute_simhash_distance']:
                        compute_min_hamming_distance(title_history, min_hamming_dist_gan, ref_simhashes_dict_for_hamming_dist)

                    if PARAMETERS['use_cvxpy']:
                        simhash_problems, subset_selections = solve_ip_cvxpy(hash_history, output_hash_bitcount, r_target_bits, multiple_solution=PARAMETERS['find_mult_sol'])
                        # ip_stats = simhash_problem.solver_stats
                        # logging.info(f'solving stats: {ip_stats.solve_time}') # sometimes stats are none
                        # Only take one solution but could find more than one by maybe using Graver's basis ?
                        # if simhash_problem.value == 0:
                        if simhash_problems[0].value == 0:
                            logging.debug(f'{Fore.RED}Failure')
                        else:
                            logging.debug(f'{Fore.GREEN}Success')
                            logging.debug(f'{Fore.LIGHTRED_EX}Solution from same history:')
                            for subset_selection in subset_selections:
                                id_hist_subset = reconstruct_sol(subset_selection.value, id_history)
                                title_hist_subset = [title_for_movie[movie_id] for movie_id in id_hist_subset]
                                # Note sim_hash_string expects a set and set inclusion faster for check of common movies after ?
                                title_hist_subset = set(title_hist_subset)
                                computed_simhash = sim_hash.sim_hash_strings(title_hist_subset, output_dimensions=output_hash_bitcount)
                                if computed_simhash == target_simhash:
                                    logging.debug(f'Movie history:\n{title_hist_subset}\nMatches target simhash (target){target_simhash}=={computed_simhash}(computed)')
                                    # Inneficient way to find matching element with target simhash history
                                    common_movies = [t_movie for t_movie in target_history if t_movie in title_hist_subset]
                                    logging.debug(f'{Fore.BLUE}Common movies between target history and computed history: {len(common_movies)}\n{common_movies}')
                                    # Add the title hist subset to the accumulator set
                                    match_hist_set_cp.add(
                                        frozenset(title_hist_subset))  # otherwise type error unhashable type: 'set'
                                    matching_hist_count_cp += 1  # as do not use set this one could have duplicate history counted
                                    # to know how many movies generated with all history
                                    acc_common_movies_cp = acc_common_movies_cp.union(set(common_movies))
                                    hist_common_movie_len_cp.append(len(common_movies))
                                else:
                                    logging.info(f'{Fore.RED}Error movie history does not match target simhash')
                            logging.debug(f'{Fore.LIGHTRED_EX}End of solution from same history.')

                    if PARAMETERS['use_gurobi']:
                        # Note change upperbound on solution count limit
                        if PARAMETERS['use_dynamic_limit']:
                            # Upperbound on number of subset can form from set of that length
                            sol_count_limit = 2 ** len(hash_history)
                            sol_count_limit = min(sol_count_limit, 2_000_000_000)  # otherwise get overflow error
                            logging.info(f'dynamic limit for int prog sol count: {sol_count_limit}')
                        else:
                            sol_count_limit = PARAMETERS['sol_count_limit']
                        simhash_model_gp, subset_selection_gp, histories_gp = \
                            solve_ip_gurobi(hash_history, output_hash_bitcount, r_target_bits, id_history, PARAMETERS,
                                            counter_stats=counter_stats, max_sol_count=sol_count_limit)

                        if len(histories_gp) == 0:
                            logging.debug(f'{Fore.RED}Failure')
                            # Note count number of failed computation in batch
                            counter_stats['gen_hash_history_cannot_satisfy_simhash'] += 1
                        else:
                            logging.debug(f'{Fore.GREEN}Success')
                            logging.debug(f'{Fore.LIGHTRED_EX}Solution from same history:')
                            counter_stats['gen_hash_history_can_satisfy_simhash'] += 1
                            # can also count one that actually satisfy simhash after check is done
                            multiple_sol_counter[f'{len(histories_gp)} histories'] += len(histories_gp)

                            # To avoid repetition due to the integer program being told to return
                            # any additional solutions found on the way to optimal one
                            # Compute statistics on GAN if IP sol exists here:

                            # Common movies between target history and gan full history when the latter admits a non empty solution to integer program
                            # Do not care about outputs since for now other stats not relevant for experiments
                            _ = update_common_movies(f'{PARAMETERS["data_gen_source"]}_IP_sol_exists', id_history, common_gan_tgt_len_cnt_if_exist_ip_sol, target_history, title_for_movie, jsim_hist_data_gan_if_ips)
                            if PARAMETERS['compute_simhash_distance']:
                                # Min hamming dist stats for gan full history only for case when the history admitted a solution to integer program
                                compute_min_hamming_distance(title_history, min_hamming_dist_gan_if_exist_ip_sol, ref_simhashes_dict_for_hamming_dist)

                            for solution_number, id_history_gp in enumerate(histories_gp):
                                counter_stats['ip_history_subset_can_satisfy_simhash'] += 1
                                # Replace movie id in history by associated movie title for simhash computation
                                title_hist_subset = set([title_for_movie[movie_id] for movie_id in id_history_gp])  # simhash expects a set
                                computed_simhash = sim_hash.sim_hash_strings(title_hist_subset, output_dimensions=output_hash_bitcount)

                                # Due to numerical approximations have to check if the history subset matches the target SimHash
                                if computed_simhash == target_simhash:
                                    counter_stats['ip_history_subset_satisfies_simhash'] += 1
                                    logging.d2bg(f'Movie history ({len(title_hist_subset)}):\n{title_hist_subset}')
                                    logging.d1bg(f'Matches target simhash (target){target_simhash}=={computed_simhash}(computed)')

                                    # Note may want to compute this statistics on the first solution only since
                                    # solution from gurobi are sorted in order of worsening objective value
                                    # (meaning the largest subset is in first history)
                                    # In the case where explicitely look for multiple solutions may want to override this condition
                                    if solution_number == 0 or PARAMETERS['find_mult_sol']:
                                        # To find matching movies with target simhash history
                                        common_movies_mtch_tgt = update_common_movies('matching_simhash', id_history_gp, common_movie_len_cnt_gp, target_history, title_for_movie, jsim_hist_data_ip)
                                        # to know how many movies generated among all histories
                                        acc_common_movies_gp = acc_common_movies_gp.union(set(common_movies_mtch_tgt))
                                        # Update list of history length after integer program (input to discriminator)
                                        dis_history_length_list.append(len(title_hist_subset)) # compute avg std after

                                        # Compute hamming distance for GAN title movie (set removed duplicate) and test data
                                        if PARAMETERS['compute_simhash_distance']:
                                            # Min hamming dist stats for int prog history (subset of GAN history that matches target simhash)
                                            compute_min_hamming_distance(list(title_hist_subset), min_hamming_dist_ip, ref_simhashes_dict_for_hamming_dist)

                                    ## Check with discriminator if current history subset is real or not:
                                    if PARAMETERS['apply_disc_on_sol']:
                                        out_predictions = discriminate_computed_hist(set(id_history_gp), vocab_ml25m, discriminator, sess, PARAMETERS['SEQ_LENGTH'], PARAMETERS['data_gen_source'])

                                        if np.argmax(out_predictions) == 1:

                                            # Same reasoning as above may want to compute this statistics only on the best solution
                                            # since by default suboptimal solution retrieved on way to best solution are smaller subset
                                            if solution_number == 0 or PARAMETERS['find_mult_sol']:
                                                # compute common movie for real histories only (in addition to matching target simhash)
                                                # Can reuse common movie computation above we just accept less movies
                                                # Can do set difference after to know which one were in only one
                                                logging.d1bg(f'{Fore.BLUE}Previous common movies also labeled as real by discriminator')
                                                acc_common_movies_mtch_real_gp = acc_common_movies_mtch_real_gp.union(set(common_movies_mtch_tgt))
                                                common_movies_mtch_real_len_cnt[f'{len(common_movies_mtch_tgt)} in common (match_simhash_label_real-target)'] += 1

                                            match_hist_set_real.add(frozenset(title_hist_subset))
                                            counter_stats['matching_history_lbl_real'] += 1  # could also put as size of set after finish ?
                                            # Report confidence of predicted label
                                            confidence_real_lbl.append(out_predictions[0][0,1]) # append proba of positive label when it is the argmax
                                        else:
                                            match_hist_set_gen.add(frozenset(title_hist_subset))
                                            counter_stats['matching_history_lbl_gen'] += 1
                                            confidence_gen_lbl.append(out_predictions[0][0,0]) # append proba of negative label when it is the argmax

                                    # Add the title hist subset to the accumulator set
                                    match_hist_set_gp.add(frozenset(title_hist_subset))  # otherwise type error unhashable type: 'set'
                                    matching_hist_count_gp += 1  # as do not use set this one could have duplicate history counted

                                else:
                                    logging.debug(f'{Fore.RED}Error movie history does not match target simhash')
                                    logging.d1bg(f'Movie history ({len(title_hist_subset)}) does not match target simhash (target){target_simhash:b}=={computed_simhash:b}(computed):\n{title_hist_subset}\n')

                                    count_sol_not_matching_target_simhash += 1
                            logging.debug(f'{Fore.LIGHTRED_EX}End of solution from same history.')

                if PARAMETERS['use_cvxpy']:
                    hist_count = len(match_hist_set_cp)
                    logging.info(f'{Fore.GREEN}History matching target simhash current count: {hist_count}')
                if PARAMETERS['use_gurobi']:
                    if PARAMETERS['apply_disc_on_sol']:
                        hist_count = len(match_hist_set_real)
                        logging.info(f'{Fore.GREEN}History matching target simhash AND labeled as real by discriminator current count: {hist_count}')
                    else:
                        hist_count = len(match_hist_set_gp)
                        logging.info(f'{Fore.GREEN}History matching target simhash current count: {hist_count}')

        logging.info(f'{Fore.BLUE}End of run for current target simhash: {target_simhash} ({target_simhash:b})')
        if PARAMETERS['use_cvxpy']:
            hist_common_movie_len_cp.sort(reverse=True)  # in place
            logging.info(f'Common movies length of generated history:\n{hist_common_movie_len_cp}')
            logging.info(f'Union of common movies (size: {len(acc_common_movies_cp)}) over all generated histories:\n{acc_common_movies_cp}')

        if PARAMETERS['use_gurobi']:
            # Note stopped printing more as parameters dicts printed at the end
            logging.info(f'Common movies length counts of generated history matching target simhash (grouped by length):\n{common_movie_len_cnt_gp}')
            logging.info(f'Union of common movies (size: {len(acc_common_movies_gp)}) over all generated histories matching target simhash:\n{acc_common_movies_gp}')
            logging.info(f'Number of matching history found: {matching_hist_count_gp} duplicate removed with frozenset: {len(match_hist_set_gp)}')
            logging.info(f'Number of solution found that did not match target simhash (maybe due to wrong condition in the case where constraints = 0): {count_sol_not_matching_target_simhash}')

        logging.info(f'Common movie counts between target simhash history and GAN outputs per history:\n{common_gan_tgt_len_cnt}')
        logging.info(f'{len(acc_common_movies_gan)} movies from the target simhash history also appeared in the GAN generated history (without simhash constraints):\n{acc_common_movies_gan}')
        logging.info(f'Counter stats: {counter_stats}')

        # Compute metric average movie count, min hamming distance
        average_gan_common_movie_cnt = compute_average_common_movie_counts(common_gan_tgt_len_cnt)
        std_gan_cmovie_cnt = compute_standard_deviation(common_gan_tgt_len_cnt, mean=average_gan_common_movie_cnt)
        gan_cmovie_cnt_ot.append(len(acc_common_movies_gan)) # update count of all movie (over all gen. hist.) had in common with target hist
        avg_min_dist_gan = compute_average_min_hamming_distance(min_hamming_dist_gan)
        std_mdist_gan = compute_standard_deviation(min_hamming_dist_gan, mean=avg_min_dist_gan)
        average_ip_common_movie_cnt = compute_average_common_movie_counts(common_movie_len_cnt_gp)
        std_ip_cmovie_cnt = compute_standard_deviation(common_movie_len_cnt_gp, mean=average_ip_common_movie_cnt)
        ip_cmovie_cnt_ot.append(len(acc_common_movies_gp))
        # Min hamming distance GAN history if ip sol exist
        avg_min_dist_gan_if_ip_sol_exists = compute_average_min_hamming_distance(min_hamming_dist_gan_if_exist_ip_sol)
        std_mdist_gan_if_ip_sol_exists = compute_standard_deviation(min_hamming_dist_gan_if_exist_ip_sol, mean=avg_min_dist_gan_if_ip_sol_exists)
        # Min hamming distance IP history
        avg_min_dist_ip = compute_average_min_hamming_distance(min_hamming_dist_ip)
        std_mdist_ip = compute_standard_deviation(min_hamming_dist_ip, mean=avg_min_dist_ip)

        gan_gen_movie_total_set.update(gan_gen_movie_hash_set) # Update set of unique movie generated by GAN over all targets
        gan_history_length_total_list.extend(gan_history_length_list) # Update list of history length for GAN over all targets
        dis_history_length_total_list.extend(dis_history_length_list) # Update list of history length for Disc. over all targets

        # average those averages other all the target histories
        avg_gan_over_multitargets += average_gan_common_movie_cnt
        gan_cmovie_omulti.append(average_gan_common_movie_cnt)
        avg_min_dist_gan_multitarget += avg_min_dist_gan # DEPRECATED
        # Note: Later compute std deviation on the averages could compute mean and std on the per target std  ?
        gan_mdist_omulti.append(avg_min_dist_gan)
        gan_mdist_omulti_ip.append(avg_min_dist_ip)
        gan_mdist_omulti_solip_exist.append(avg_min_dist_gan_if_ip_sol_exists)
        avg_ip_over_multitargets += average_ip_common_movie_cnt
        ip_cmovie_omulti.append(average_ip_common_movie_cnt)
        # Update counter stats
        counter_stats_total += counter_stats # add counters together (also a lot of other operation see doc)

        js_gan_mean, js_gan_std = np.mean(jsim_hist_data_gan), np.std(jsim_hist_data_gan)
        js_gan_if_ips_mean, js_gan_if_ips_std = np.mean(jsim_hist_data_gan_if_ips), np.std(jsim_hist_data_gan_if_ips)
        js_ip_mean, js_ip_std = np.mean(jsim_hist_data_ip), np.std(jsim_hist_data_ip)

        gen_src = PARAMETERS["data_gen_source"].lower()
        # Note if set compute min hamming distance to false this will crash
        save_dict = {'counter_stats': counter_stats, 'target_hist': target_history,
                     'simhash_bitcount': output_hash_bitcount,
                     # Min hamming distance for GAN
                     f'min_dist_{gen_src}_cnt': min_hamming_dist_gan,
                     f'min_dist_{gen_src}_mstd': (avg_min_dist_gan, std_mdist_gan),
                     # f'min_dist_{gen_src}_std': std_mdist_gan,
                     # Min hamming distance fo GAN if ip sol exists
                     f'min_dist_{gen_src}_cnt_ip_sol_exists': min_hamming_dist_gan_if_exist_ip_sol,
                     f'min_dist_{gen_src}_mstd_ip_sol_exists': (avg_min_dist_gan_if_ip_sol_exists, std_mdist_gan_if_ip_sol_exists),
                     # Min hamming distance for IP
                     f'min_dist_{gen_src}_cnt_ip_sol_exists': min_hamming_dist_ip,
                     f'min_dist_{gen_src}_mstd_ip_sol_exists': (avg_min_dist_ip, std_mdist_ip),
                     f'common_movies_{gen_src}': acc_common_movies_gan,
                     f'common_movies_{gen_src}_stats': common_gan_tgt_len_cnt,
                     f'common_movies_{gen_src}_mstd': (average_gan_common_movie_cnt, std_gan_cmovie_cnt),
                     # f'common_movies_{gen_src}_std': std_gan_cmovie_cnt,
                     f'common_movies_{gen_src}_cnt': len(acc_common_movies_gan),
                     'common_movies_ip': acc_common_movies_gp, 'common_movies_ip_stats': common_movie_len_cnt_gp,
                     'common_movies_ip_mstd': (average_ip_common_movie_cnt, std_ip_cmovie_cnt),
                     # 'common_movies_ip_std': std_ip_cmovie_cnt,
                     'common_movies_ip_cnt': len(acc_common_movies_gp),
                     # size of set below give number movies from vocab generated by GAN
                     # f'{gen_src}_gen_movie_set': gan_gen_movie_hash_set, # set of (unique) movie generated by GAN
                     f'{gen_src}_gen_movie_set_size': len(gan_gen_movie_hash_set), # mostly interested in set size
                     f'{gen_src}_gen_hist_len': gan_history_length_list,
                     f'{gen_src}_gen_hist_len_avg_std': (np.mean(gan_history_length_list), np.std(gan_history_length_list)),
                     f'{gen_src}_dis_hist_len': dis_history_length_list,
                     f'{gen_src}_dis_hist_len_avg_std': (np.mean(dis_history_length_list), np.std(dis_history_length_list)),
                     f'js_gan': (js_gan_mean, js_gan_std),
                     f'js_gan_if_ips': (js_gan_if_ips_mean, js_gan_if_ips_std),
                     f'js_ip': (js_ip_mean, js_ip_std),
                     }
        if PARAMETERS['apply_disc_on_sol']:
            # Compute metric average movie count
            average_ipr_common_movie_cnt = compute_average_common_movie_counts(common_movies_mtch_real_len_cnt)
            std_ipr_cmovie_cnt = compute_standard_deviation(common_movies_mtch_real_len_cnt, mean=average_ipr_common_movie_cnt)
            # average those averages other all the target histories
            avg_ipr_over_multitargets += average_ipr_common_movie_cnt
            ipr_cmovie_omulti.append(average_ipr_common_movie_cnt)
            # Fill dict
            save_dict['common_movies_ipreal'] = acc_common_movies_mtch_real_gp
            save_dict['common_movies_ipreal_stats'] = common_movies_mtch_real_len_cnt
            save_dict['common_movies_ipreal_mstd'] = (average_ipr_common_movie_cnt, std_ipr_cmovie_cnt)
            # save_dict['common_movies_ipreal_std'] = std_ipr_cmovie_cnt
            save_dict['common_movies_ipreal_cnt'] = len(acc_common_movies_mtch_real_gp)
            ipr_cmovie_cnt_ot.append(len(acc_common_movies_mtch_real_gp)) # update for multitarget
            # Update confidence for overall confidence over all labels
            confidence_real_lbl_total.extend(confidence_real_lbl)
            confidence_gen_lbl_total.extend(confidence_gen_lbl)
            save_dict[f'{gen_src}_confidence_real'] = confidence_real_lbl
            save_dict[f'{gen_src}_confidence_real_mstd'] = (np.mean(confidence_real_lbl), np.std(confidence_real_lbl))
            save_dict[f'{gen_src}_confidence_gen'] = confidence_gen_lbl
            save_dict[f'{gen_src}_confidence_gen_mstd'] = (np.mean(confidence_gen_lbl), np.std(confidence_gen_lbl))

            # cast to str for json sorted dicts print as other keys are str
        stats_per_target[str(target_simhash)] = save_dict  # note overrides if have same simhash multiple time
        # Update dict for histogram counter aggregation:
        common_movies_counter_dicts['GAN'][str(target_simhash)] = common_gan_tgt_len_cnt
        common_movies_counter_dicts['GAN if exists IP sol'][str(target_simhash)] = common_gan_tgt_len_cnt_if_exist_ip_sol
        common_movies_counter_dicts['IP'][str(target_simhash)] = common_movie_len_cnt_gp
        min_hamming_dist_counter_dicts['GAN'][str(target_simhash)] = min_hamming_dist_gan
        min_hamming_dist_counter_dicts['GAN if exists IP sol'][str(target_simhash)] = min_hamming_dist_gan_if_exist_ip_sol
        min_hamming_dist_counter_dicts['IP'][str(target_simhash)] = min_hamming_dist_ip
        # Jaccard similarity histogram
        # Aggregate over multiple targets
        jsim_hist_data_gan_mt.extend(jsim_hist_data_gan)
        jsim_hist_data_gan_if_ips_mt.extend(jsim_hist_data_gan_if_ips)
        jsim_hist_data_ip_mt.extend(jsim_hist_data_ip)
        # Save intermediate histograms
        # log_folder may not be in scope unless use pycharm python console
        compute_and_save_histogram(jsim_hist_data_gan, bin_edges, f"{PARAMETERS['log_filename']}_{target_simhash}_GAN", savepath=log_folder)
        compute_and_save_histogram(jsim_hist_data_gan_if_ips, bin_edges, f"{PARAMETERS['log_filename']}_{target_simhash}_GAN_IF_IPS", savepath=log_folder)
        compute_and_save_histogram(jsim_hist_data_ip, bin_edges, f"{PARAMETERS['log_filename']}_{target_simhash}_IP", savepath=log_folder)


    # Note if not needed anymore remove avg_gan_over_multitargets and similar
    avg_gan_over_multitargets /= len(target_histories)
    gan_cmovie_mean, gan_cmovie_std = np.mean(gan_cmovie_omulti), np.std(gan_cmovie_omulti)
    # Note sanity check for output similar to manually computed values: (only done for one)
    if avg_gan_over_multitargets != gan_cmovie_mean:
        logging.warning(f"{Fore.RED}computed mean different from np's {avg_gan_over_multitargets} != {gan_cmovie_mean}")
    # Min hamming distances
    gan_mdist_mean, gan_mdist_std = np.mean(gan_mdist_omulti), np.std(gan_mdist_omulti)
    gan_mdist_mean_solip_exist, gan_mdist_std_solip_exist = np.mean(gan_mdist_omulti_solip_exist), np.std(gan_mdist_omulti_solip_exist)
    gan_mdist_mean_ip, gan_mdist_std_ip = np.mean(gan_mdist_omulti_ip), np.std(gan_mdist_omulti_ip)

    avg_ip_over_multitargets /= len(target_histories) # DEPRECATED
    ip_cmovie_mean, ip_cmovie_std = np.mean(ip_cmovie_omulti), np.std(ip_cmovie_omulti)
    avg_min_dist_gan_multitarget /= len(target_histories) # DEPRECATED

    # Average for extended list of jaccard sim
    js_gan_mt_mean, js_gan_mt_std = np.mean(jsim_hist_data_gan_mt), np.std(jsim_hist_data_gan_mt)
    js_gan_if_ips_mt_mean, js_gan_if_ips_mt_std = np.mean(jsim_hist_data_gan_if_ips_mt), np.std(jsim_hist_data_gan_if_ips_mt)
    js_ip_mt_mean, js_ip_mt_std = np.mean(jsim_hist_data_ip_mt), np.std(jsim_hist_data_ip_mt)

    gen_src = PARAMETERS["data_gen_source"].lower()
    stats_over_multitarget = {
                            'counter_stats': counter_stats_total,
                            # Take the mean of the aggregated mean per target
                            f'common_movies_meanstd_of_avgs_{gen_src}': (gan_cmovie_mean, gan_cmovie_std), # avg_gan_over_multitargets
                            # f'common_movies_std_means_{gen_src}': gan_cmovie_std, # tupled with one above
                            f'cmovies_total_distrib_gan': gan_cmovie_cnt_ot,
                            f'cmovies_total_distrib_mstd_gan': (np.mean(gan_cmovie_cnt_ot), np.std(gan_cmovie_cnt_ot)),
                            # Min hamming dist:
                            f'min_dist_meanstd_of_avg_{gen_src}': (gan_mdist_mean, gan_mdist_std), # avg_min_dist_gan_multitarget
                            # f'min_dist_std_means_{gen_src}': gan_mdist_std,
                            f'min_dist_meanstd_of_avg_{gen_src}_solip_exist': (gan_mdist_mean_solip_exist, gan_mdist_std_solip_exist),
                            f'min_dist_meanstd_of_avg_{gen_src}_ip': (gan_mdist_mean_ip, gan_mdist_std_ip),
                            'common_movies_meanstd_avgs_ip': (ip_cmovie_mean, ip_cmovie_std), # avg_ip_over_multitargets
                            # 'common_movies_std_means_ip': ip_cmovie_std,
                            f'cmovies_total_distrib_ip': ip_cmovie_cnt_ot,
                            f'cmovies_total_distrib_mstd_ip': (np.mean(ip_cmovie_cnt_ot), np.std(ip_cmovie_cnt_ot)),
                            # f'{gen_src}_gen_movie_set': gan_gen_movie_total_set, # set big mostly intersted in its size
                            f'{gen_src}_gen_movie_set_size': len(gan_gen_movie_total_set),
                            f'{gen_src}_gen_hist_len': gan_history_length_total_list,
                            f'{gen_src}_gen_hist_len_avg_std_mt': (np.mean(gan_history_length_total_list), np.std(gan_history_length_total_list)),
                            f'{gen_src}_dis_hist_len': dis_history_length_total_list,
                            f'{gen_src}_dis_hist_len_avg_std_mt': (np.mean(dis_history_length_total_list), np.std(dis_history_length_total_list)),
                            f'js_gan_mt': (js_gan_mt_mean, js_gan_mt_std),
                            f'js_gan_if_ips_mt': (js_gan_if_ips_mt_mean, js_gan_if_ips_mt_std),
                            f'js_ip_mt': (js_ip_mt_mean, js_ip_mt_std),
                            }
    if PARAMETERS['apply_disc_on_sol']:
        avg_ipr_over_multitargets /= len(target_histories) # Note remove this if not needed anymore
        ipr_cmovie_mean, ipr_cmovie_std = np.mean(ipr_cmovie_omulti), np.std(ipr_cmovie_omulti)
        stats_over_multitarget['common_movies_meanstd_avgs_ipr'] = (ipr_cmovie_mean, ipr_cmovie_std) # avg_ipr_over_multitargets
        stats_over_multitarget['cmovies_total_distrib_ipr'] = ipr_cmovie_cnt_ot
        stats_over_multitarget['cmovies_total_distrib_mstd_ipr'] = np.mean(ipr_cmovie_cnt_ot), np.std(ipr_cmovie_cnt_ot)
        # stats_over_multitarget['common_movies_std_means_ipr'] = ipr_cmovie_std # included above
        stats_over_multitarget[f'{gen_src}_confidence_real'] = confidence_real_lbl_total
        stats_over_multitarget[f'{gen_src}_confidence_real_mstd'] = (np.mean(confidence_real_lbl_total), np.std(confidence_real_lbl_total))
        stats_over_multitarget[f'{gen_src}_confidence_gen'] = confidence_gen_lbl_total
        stats_over_multitarget[f'{gen_src}_confidence_gen_mstd'] = (np.mean(confidence_gen_lbl_total), np.std(confidence_gen_lbl_total))

    # logging.info(f'stats per target simhash: {stats_per_target}')
    stats_per_target['multitarget'] = stats_over_multitarget
    logging.info(f'stats per target simhash:\n{pretty_format(stats_per_target, use_json=True)}')

    # Histogram computation:
    # pipeline_blocks = ['GAN', 'GAN if exists IP sol','IP']
    aggregated_hist_common_movie = dict()
    for k, v in common_movies_counter_dicts.items():
        aggregated_counts = aggregate_counter(v)
        # Hypothesis metric:
        percentage_cmovie_ge_3 = percentage_of_common_movie_cnt_ge_3(aggregated_counts)
        mean_agg = compute_average_common_movie_counts(aggregated_counts)
        var, std_agg = compute_standard_deviation(aggregated_counts, mean=mean_agg)
        aggregated_hist_common_movie[k] = (percentage_cmovie_ge_3, (mean_agg, std_agg), aggregated_counts)

    aggregated_hist_min_hamming_dist = dict()
    for k, v in min_hamming_dist_counter_dicts.items():
        aggregated_hist_min_hamming_dist[k] = aggregate_counter(v)
    histograms = {'common_movies': aggregated_hist_common_movie, 'min_hamming_dist': aggregated_hist_min_hamming_dist}
    logging.info(f'aggregated histograms:\n{pretty_format(aggregated_hist_common_movie, use_json=True)}')

    logging.d2bg(f'{jsim_hist_data_gan_mt}')
    compute_and_save_histogram(jsim_hist_data_gan_mt, bin_edges, f"{PARAMETERS['log_filename']}_GAN_MT", savepath=log_folder)
    logging.d2bg(f'{jsim_hist_data_gan_if_ips_mt}')
    compute_and_save_histogram(jsim_hist_data_gan_if_ips_mt, bin_edges, f"{PARAMETERS['log_filename']}_GAN_IF_IPS_MT", savepath=log_folder)
    logging.d2bg(f'{jsim_hist_data_ip_mt}')
    compute_and_save_histogram(jsim_hist_data_ip_mt, bin_edges, f"{PARAMETERS['log_filename']}_IP_MT", savepath=log_folder)

    return stats_per_target, histograms

# Note: did not work
@multiprocess_run_release_gpu
def run_kill_release_gpu(PARAMETERS, hash_for_movie, title_for_movie, word_ml25m, vocab_ml25m):
    # attack.utils.FUNC_TO_CALL = generate_history_matching_target_simhash
    out = generate_history_matching_target_simhash(PARAMETERS, hash_for_movie, title_for_movie, word_ml25m, vocab_ml25m)
    return out

# Note: this multiprocessing need the run configuration to disable python console
# In top level of module function can be pickled (problem due to the use of pycharm python console)
def worker_2(output_dict, *args, **kwargs):
        # return_value = func(*args, **kwargs)
        # return_value = FUNC_TO_CALL(*args, **kwargs)
        updated_kwargs = dict(kwargs, use_multiprocessing=True)
        return_value = generate_history_matching_target_simhash(*args, **updated_kwargs)
        if return_value is not None:
            output_dict['return_value'] = return_value

# logging with multiprocessing
# [1] https://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python
def multiprocess_run_release_gpu_2(*args, **kwargs):
    with multiprocessing.Manager() as manager:
        output_dict = manager.dict()
        worker_args = (output_dict, ) + args
        process = multiprocessing.Process(target=worker_2, args=worker_args, kwargs=kwargs)
        process.start()
        process.join() # don't need timeout
        return_value = output_dict.get('return_value', None)

    return return_value

def precompute_simhash_on_data(token_to_id, id_to_title, save_path='./test_set_simhashes_dict.pkl', datafile='../GAN/save_ml25m/realtest_ml25m_5000_32.txt', out_hash_bitcount=64):
    # Note: here could to subsample or not depending on how split train eval test
    #  (ie for now would need a different file with eval data)
    movie_tokens_hist_ndarray = load_train_test(datafile)
    movie_tokens_hist = movie_tokens_hist_ndarray.tolist() # Loops with numpy slow

    movieid_hists, movie_titles_hists = movie_history_decoder(movie_tokens_hist, token_to_id, id_to_title)

    simhash_to_hist = dict()
    # Note for now cur_id_hist not used in loop cause not sure what wanted the value of the dict to be
    # Note Might need to save it to a file (pickle or other) since it is slow to compute
    for cur_id_hist, cur_title_hist in tqdm(zip(movieid_hists, movie_titles_hists), desc='Precomputing test simhashes'):
        # note cur_title_hist is not a set but should only have unique movies
        cur_simhash = sim_hash.sim_hash_strings(cur_title_hist, output_dimensions=out_hash_bitcount)
        simhash_to_hist[cur_simhash] = cur_title_hist

    # May want to create save folder if does not exist
    with open(save_path, 'wb') as f:
        pickle.dump(simhash_to_hist, f)

    return simhash_to_hist

def movie_history_decoder(history_list, token_to_id, id_to_title):
    '''
    Take a list of history encoded as tokens and output movie titles
    :param history_list:
    :param token_to_id:
    :param id_to_title:
    :return:
    '''
    movieid_histories = []
    movie_title_histories = []
    for history in history_list:
        cur_id_hist = []
        cur_title_hist = []
        for token in history:
            word_id = token_to_id[token]
            if word_id != '<pad>' and word_id != '<oov>':
                cur_id_hist.append(word_id)
                cur_title_hist.append(id_to_title[word_id])
        movieid_histories.append(cur_id_hist)
        movie_title_histories.append(cur_title_hist)
    return movieid_histories, movie_title_histories

def hamming_closest_simhash(target_simhash, test_simhashes, prefix_bitlength=64):
    min_hamming_dist = 65 # greater than max hamming distance of 64 for our purposes
    simhash_with_min_ham_dist = []
    for simhash_to_test in test_simhashes:
        cur_hamming_dist = utils.binary_hamming_distance(target_simhash, simhash_to_test, prefix_bitlength=prefix_bitlength)
        if cur_hamming_dist < min_hamming_dist:
            min_hamming_dist = cur_hamming_dist
            simhash_with_min_ham_dist.clear() # remove all subotpimal
            simhash_with_min_ham_dist.append(simhash_to_test)
        elif cur_hamming_dist == min_hamming_dist:
            simhash_with_min_ham_dist.append(simhash_to_test)

    return min_hamming_dist, simhash_with_min_ham_dist


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument()
    # args = parser.parse_args()

    ################
    ## Parameters ##
    ################
    # Params dict easier to save in logs
    # Note at some point can add argparser for useful params
    PARAMETERS = {
        'SEED': 789, # None or int,  948327
        'seed_inc': 0, # seed inc when has to update seed to draw new output, modified with side effect # changed some function that used it before
        'find_mult_sol': False, # If try to find multiple solution for same integer program, note by default gurobi still keep track of sol find on the way
        'apply_disc_on_sol': True, # Use discriminator to check which solution is still labeled as real or not and use it to reach target count NOTE: Used true for disc. confidence
        'use_cvxpy': False,
        'use_gurobi': True,
        'gurobi_timelimit': 10, # timelimit in second for gurobi solver
        'sol_count_limit': 2048, # max number of solution for gurobi int prog
        # Note currently set dynamically as 2^(card(hash_history)) or max of 2000000000 but need to find trade off
        'use_dynamic_limit' : True, # if limit is determined through eg length of history
        # Note instead of boolean could use int between 0 (no computation) and 64 for prefix bit used in distance computation ?
        'compute_simhash_distance': True, # compute hamming distance between simhash from test and generated data
        'TARGET_COUNT': 200,  # repeat attack until get enough unique histories say 2000, NOTE: used 150 for disc. confidence.
        'TARGET_SIMHASH_ORIGIN': 'test', # Where sample real history from 'all' dataset, 'train' or 'test'
        'TARGET_SIMHASH_COUNT': 5, # How many target simhash generates # NOTE: used 10 for disc. confidences # Note might need different seeds so do +1 on prev seed
        'SIMHASH_BITCOUNT_RANGE': (15,15), # both endpoint included
        'data_gen_source': 'RAND', # If use 'GAN' or 'RAND' for generation to make int prog on
        'CHECKPOINT_FILEPATH': f'../GAN/ckpts_ml25m/leakgan-61', # if None then load latest from init_gen model path,
        # 'already_init_gan': False, # Tensorflow crash if redeclare variable, only change the checkpoint # not needed anymore
        'log_filename': 'log',
        'timer_nametag': '',
        'SEQ_LENGTH': 32,
        'BATCH_SIZE': 64
    }

    np.random.seed(PARAMETERS['SEED']) # for more repoduceability
    random.seed(PARAMETERS['SEED'])
    gp.setParam("OutputFlag", 0) # set param for all model
    log_to_stdout = True # if logger also logs to console

    log_folder = f'./logs/run_{datetime.now():%d-%m-%Y_at_%Hh%Mm%Ss}/'
    Path(log_folder).mkdir(parents=True, exist_ok=True)

    # Variable defined in broader scope (should be accessible inside called function scope)
    SEQ_LENGTH = PARAMETERS['SEQ_LENGTH']
    BATCH_SIZE = PARAMETERS['BATCH_SIZE']

    #####################
    ## Precomputations ##
    #####################
    hash_for_movie, title_for_movie, movie_for_title, movie_id_list = precompute_cityhash_movielens()
    # Load vocabulary
    word_ml25m, vocab_ml25m = pickle.load(open('../GAN/save_ml25m/LeakGAN_ml25m_vocab_5000_32.pkl', mode='rb'))

    # Precompute 64 bit simhash (maximum) of test (or eval if used as validation during training)
    save_path = './saved/test_set_simhashes_dict.pkl'
    if not os.path.isfile(save_path):
        testdata_simhash_to_hist = precompute_simhash_on_data(word_ml25m, title_for_movie, datafile='../GAN/save_ml25m/realtest_ml25m_5000_32.txt')
    else:
        testdata_simhash_to_hist = pickle.load(open(save_path, mode='rb'))


    eval55 = False
    eval61 = True # part of evaluation for section 6.1 of thesis
    eval_custom = False

    utils.create_logging_levels()  # Need to call it before otherwise logging.d1bg etc crashes as not defined


    if eval55:
        gen_sources = ['GAN'] # ['RAND', 'GAN']
        multisol_values = [False, True]# [False, True]
        gurobi_tl = [10, 20, 30]
        simhash_bitcount_range = [20] # [5, 10, 15, 20, 25] used for table 5.3, 15 used for table 5.5
        # Note how to stop and restart file logging for different run of a function ? works if re init logger ?
        # found maybe this [1] https://stackoverflow.com/questions/14523996/start-and-stop-logger-in-python-3-2
        out51_stats = dict()

        for data_gen_source in gen_sources:
            for use_multisol in multisol_values:
                    for simhash_bitlen in simhash_bitcount_range:

                        PARAMETERS['data_gen_source'] = data_gen_source
                        PARAMETERS['SIMHASH_BITCOUNT_RANGE'] = (simhash_bitlen, simhash_bitlen)
                        PARAMETERS['find_mult_sol'] = use_multisol

                        # Only want to apply timelimit if use_multisol
                        if use_multisol:
                            for g_time_limit in gurobi_tl:

                                PARAMETERS['gurobi_timelimit'] = g_time_limit
                                PARAMETERS['timer_nametag'] = f'GAN_Multisol_tl{g_time_limit}'
                                # Set meaningful log filename
                                PARAMETERS['log_filename'] = f"{PARAMETERS['timer_nametag']}"

                                # may want to change debug level
                                fh, sh = init_loggers(log_to_stdout=log_to_stdout, filename=PARAMETERS['log_filename'], fh_lvl=logging.D2BG, sh_lvl=logging.D2BG)
                                logging.info(f'Parameters:\n{pretty_format(PARAMETERS)}')
                                out = generate_history_matching_target_simhash(PARAMETERS, hash_for_movie, title_for_movie, word_ml25m, vocab_ml25m, testdata_simhash_to_hist)

                                utils.remove_handlers([fh, sh])
                                out51_stats[PARAMETERS['timer_nametag']] = out
                        else:
                            PARAMETERS['timer_nametag'] = f'GAN_Default' # data_gen_source for disc. confidence # SimHash_{simhash_bitlen} for runtimes
                            # Set meaningful log filename
                            PARAMETERS['log_filename'] = f"{PARAMETERS['timer_nametag']}"  # PARAMETERS["data_gen_source"]

                            # may want to change debug level to d1bg or lower to have generated history
                            fh, sh = init_loggers(log_to_stdout=log_to_stdout, filename=PARAMETERS['log_filename'], fh_lvl=logging.D2BG, sh_lvl=logging.D2BG)
                            logging.info(f'Parameters:\n{pretty_format(PARAMETERS)}')
                            out = generate_history_matching_target_simhash(PARAMETERS, hash_for_movie, title_for_movie, word_ml25m, vocab_ml25m, testdata_simhash_to_hist)

                            utils.remove_handlers([fh, sh])
                            out51_stats[PARAMETERS['timer_nametag']] = out


    if eval61:
        # Redefines fixed parameters for this experiment
        PARAMETERS = {
            # Fixed for a run
            'SEED': 789,
            'apply_disc_on_sol': False,
            'TARGET_COUNT': 200,
            'TARGET_SIMHASH_COUNT': 5,
            'SIMHASH_BITCOUNT_RANGE': (15, 15), 'TARGET_SIMHASH_ORIGIN': 'test',
            'compute_simhash_distance': True,
            # Invariant in this evaluation
            'find_mult_sol': False, 'gurobi_timelimit': 10, 'sol_count_limit': 2048, 'use_dynamic_limit': True,
            'SEQ_LENGTH': 32, 'BATCH_SIZE': 64, 'seed_inc': 0, 'use_gurobi': True, 'use_cvxpy': False,
            # Modified in this evaluation
            'data_gen_source': 'RAND', 'CHECKPOINT_FILEPATH': f'../GAN/ckpts_ml25m/leakgan-61',
            'log_filename': 'log', 'timer_nametag': '',
        }
        # Evaluate for GAN checkpoints
        checkpoint_filepaths = [
            # f'../GAN/ckpts_ml25m/leakgan_preD', f'../GAN/ckpts_ml25m/leakgan_pre', f'../GAN/ckpts_ml25m/leakgan-1',
            # f'../GAN/ckpts_ml25m/leakgan-11', f'../GAN/ckpts_ml25m/leakgan-21', f'../GAN/ckpts_ml25m/leakgan-31',
            f'../GAN/ckpts_ml25m/leakgan-41',
            # f'../GAN/ckpts_ml25m/leakgan-51',
            f'../GAN/ckpts_ml25m/leakgan-61',
            # Note could remove rand from here and run it on all checkpoints too (eg when care about confidence)
            'RAND']

        out61_stats = dict()

        for checkpoint in checkpoint_filepaths:
            if checkpoint == 'RAND':
                PARAMETERS['data_gen_source'] = 'RAND'
                # Note actually for some benchmark could test discriminator for random generator and discriminator on all checkpoints
                PARAMETERS['CHECKPOINT_FILEPATH'] = None # Load latest checkpoint if not specified
                tag = 'RAND'
                PARAMETERS['log_filename'] = f"{tag}"
            else:
                PARAMETERS['CHECKPOINT_FILEPATH'] = checkpoint
                tag = f"{checkpoint.split('/')[-1]}"
                PARAMETERS['data_gen_source'] = 'GAN'

            PARAMETERS['timer_nametag'] = tag
            # Set meaningful log filename
            PARAMETERS['log_filename'] = f"{PARAMETERS['timer_nametag']}"

            # Note might want to change debug level to d1bg or lower to have generated history
            fh, sh = init_loggers(log_to_stdout=log_to_stdout, log_folder=log_folder, filename=PARAMETERS['log_filename'], fh_lvl=logging.D2BG, sh_lvl=logging.D2BG)
            logging.info(f'Parameters:\n{pretty_format(PARAMETERS)}')
            out = generate_history_matching_target_simhash(PARAMETERS, hash_for_movie, title_for_movie, word_ml25m, vocab_ml25m, testdata_simhash_to_hist)

            utils.remove_handlers([fh, sh])
            out61_stats[PARAMETERS['timer_nametag']] = out

        with open(f'{log_folder}/out61_stats.pkl', 'wb') as f:
            pickle.dump(out61_stats, f)
        # out61_stats = pickle.load(open(f'{log_folder}/out61_stats.pkl', mode='rb'))


    if eval_custom:
        checkpoint_filepaths = [
            f'../GAN/ckpts_ml25m/leakgan-41',
            f'../GAN/ckpts_ml25m/leakgan-61',
        ]
        out_custom_stats = dict()
        PARAMETERS['data_gen_source'] = 'GAN'
        for checkpoint in checkpoint_filepaths:
            tag = f"{checkpoint.split('/')[-1]}"
            PARAMETERS['CHECKPOINT_FILEPATH'] = checkpoint
            PARAMETERS['timer_nametag'] = tag
            PARAMETERS['log_filename'] = tag
            fh, sh = init_loggers(log_to_stdout=log_to_stdout, filename=PARAMETERS['log_filename'], fh_lvl=logging.D2BG, sh_lvl=logging.D1BG)
            out = generate_history_matching_target_simhash(PARAMETERS, hash_for_movie, title_for_movie, word_ml25m, vocab_ml25m, testdata_simhash_to_hist)
            utils.remove_handlers([fh, sh])
            out_custom_stats[PARAMETERS['timer_nametag']] = out


    # Legacy:
    # IF multiprocessing is True need to disable pycharm python console in run configurations
    use_multiprocessing = False
    if use_multiprocessing:
        # PARAMETERS to vary
        checkpoint_filepaths = [#f'../GAN/ckpts_ml25m/leakgan_preD', f'../GAN/ckpts_ml25m/leakgan_pre', f'../GAN/ckpts_ml25m/leakgan-1', f'../GAN/ckpts_ml25m/leakgan-11',
                                #f'../GAN/ckpts_ml25m/leakgan-21', f'../GAN/ckpts_ml25m/leakgan-31',
                                f'../GAN/ckpts_ml25m/leakgan-41',
                                #f'../GAN/ckpts_ml25m/leakgan-51',
                                f'../GAN/ckpts_ml25m/leakgan-61',
                                'RAND']
        out_stats = []
        for checkpoint in checkpoint_filepaths:
            if checkpoint == 'RAND':
                PARAMETERS['data_gen_source'] = 'RAND'
                PARAMETERS['CHECKPOINT_FILEPATH'] = None
                tag = 'RAND'
                PARAMETERS['log_filename'] = f"{tag}"
            else:
                PARAMETERS['CHECKPOINT_FILEPATH'] = checkpoint
                tag = checkpoint.split('/')[-1]
                PARAMETERS['data_gen_source'] = 'GAN'

            PARAMETERS['log_filename'] = f"checkpoint {tag}"  # f'checkpoint {checkpoint[-2:]}'
            # One way to free gpu memory with tensorflow is to kill the process
            # cur_out = run_kill_release_gpu(PARAMETERS, hash_for_movie, title_for_movie, word_ml25m, vocab_ml25m)
            # Note: if run without pycharm python console need to add args after PARAMETERS and program crash with logging.d5bg not existing
            # Also as init logging again for each process get multiple logging files
            # Note this seems to not be needed anymore if when change checkpoint file calls tf.reset_default_graph()
            cur_out = multiprocess_run_release_gpu_2(PARAMETERS, hash_for_movie, title_for_movie, word_ml25m, vocab_ml25m, testdata_simhash_to_hist) # this one worked without python console
            # cur_out = multiprocess_run_release_gpu(PARAMETERS)
            cur_stats = (PARAMETERS, cur_out)
            out_stats.append(cur_stats)

    # Can call stats on certain timers
    # https://pypi.org/project/codetiming/
    timers = Timer.timers
    logging.info(f'Timers: {timers}')
    # Timer.timers.mean("example") # can check LeakGAN_dis.py for example
    # Note that to access Timer.timers.timings remove the _ restriction in _timings in associated file
    timings = Timer.timers.timings
    for timer_name in timings.keys():
        min, max = Timer.timers.min(timer_name), Timer.timers.max(timer_name)
        mean, median, std = Timer.timers.mean(timer_name), Timer.timers.median(timer_name), Timer.timers.stdev(timer_name)
        logging.info(f'Timer {timer_name} stats: [min,max]=[{min}, {max}] median={median} (mean,std)=({mean}, {std})')
