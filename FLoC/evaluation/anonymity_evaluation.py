from FLoC.chromium_components import sim_hash
from FLoC.preprocessing.movielens_extractor import feature_extraction_floc_whitepaper, user_id_count as FULL_USER_ID_COUNT, \
    feature_extractor_generated_movies, read_ratings, read_movies, precompute_cityhash_movielens
import numpy as np
import os
import logging
from FLoC.utils import init_loggers, pretty_format, create_logging_levels
from collections import defaultdict
import matplotlib.pyplot as plt
# To save for latex.
# https://timodenk.com/blog/exporting-matplotlib-plots-to-latex/
import matplotlib
# Note this causes bad key messages, better to reset it after with
# `matplotlib.rcParams.update(matplotlib.rcParamsDefault)`
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import pickle
from tqdm import trange
from pip._vendor.colorama import Fore, init
init(autoreset=True) # to avoid reseting color everytime
from FLoC.attack.generating_histories import generate_movie_simhash_cluster, init_generator, load_train_test
import multiprocessing
from pathlib import Path
from datetime import datetime
from codetiming import Timer


# Taken from [1]
# [1] https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)

    # Checks added when found NaN (most likely from normalization and avg cosine)
    np.set_printoptions(precision=5, threshold=1200, suppress=True, linewidth=160)
    # np.interp assumes xp (here weighted_quantiles) is strictly increasing
    if not np.all(np.diff(weighted_quantiles) > 0):
        print(f'{Fore.RED}Weighted quantiles {weighted_quantiles.shape} sequence is not strictly increasing,'
              f' interpolation results are meaningless ?\n{weighted_quantiles}')
    if np.isnan(weighted_quantiles).any():
        print(f'{Fore.RED}weighted_quantiles array contain nan values')
    if np.isnan(values).any():
        print(f'{Fore.RED}values {values.shape} array contain nan values\n{values}')

    return np.interp(quantiles, weighted_quantiles, values)

def cosine_similarity(centroid_of_cluster, user_in_cluster):
    # Computing cosine similarity (normalized dot product)
    # [1] https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    # from np: cos_sim = dot(a, b)/(norm(a)*norm(b))
    # from scipy: 1 - spatial.distance.cosine(dataSetI, dataSetII)
    # sklearn too for matrices
    n_centroid = np.linalg.norm(centroid_of_cluster)
    n_user = np.linalg.norm(user_in_cluster)
    if n_centroid == 0 or n_user == 0:
        cos_sim = np.dot(centroid_of_cluster, user_in_cluster)
        # https://stackoverflow.com/questions/18395725/test-if-numpy-array-contains-only-zeros
        # test if all values are zeros (false) any() return true if one value is not 0 (false)
        # so when np.any(cos_sim) is false cos_sim only contains 0 if that is not the case it is not as expected
        # !np.any(cos_sim)=True is expected so want to print warning when np.any(cos_sim) is True
        if np.any(cos_sim):
            print(f'{Fore.RED}Expected all zero vector as one norm was 0: {cos_sim}')
    else:
        cos_sim = np.dot(centroid_of_cluster, user_in_cluster) / (n_centroid * n_user)
    # can check should be between -1 and 1
    return cos_sim

def average_cosine_similarity(cluster_indices, cluster_centroid, data_matrix, use_for_loop=False, compare_both=False):
    def compute_avg_with_loop():
        cos_sim_list = []
        for cur_id in cluster_indices:
            cos_sim_list.append(cosine_similarity(cluster_centroid, data_matrix[cur_id, :]))

        # Compute average cosine similarity between all users in the cluster and the centroid
        avg_cos_sim = np.mean(cos_sim_list)
        return avg_cos_sim

    def compute_avg_with_np():
        # Note: if use the np.linalg.norm it outputs different results than the code found,
        # so may need to use different parameters for the np.linalg.norm
        normalized_cluster_features = normalize_matrix_for_cosine_similarity(data_matrix[cluster_indices, :])
        centroid_norm = np.linalg.norm(cluster_centroid)
        if centroid_norm == 0:
            # Case where norm of vector is 0 (cannot divide by 0) also means original vector is 0
            normalized_centroid = cluster_centroid
        else:
            normalized_centroid = cluster_centroid / centroid_norm
        avg_cos_sim = np.dot(normalized_cluster_features, normalized_centroid).mean()
        return avg_cos_sim

    if compare_both:
        a = compute_avg_with_loop()
        b = compute_avg_with_np()
        # Not symmetric assume b reference value
        # absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))
        # differ on 17 decimal return false with rtol=0 and atol=1e-17 for Random
        # differ on 16 for some computation for simhash
        isclose = np.isclose(a,b, rtol=0, atol=1e-15)
        logging.debug(f'loop {a}=?={b} np loop') #
        if not isclose:
            logging.warning(f'{Fore.RED} computations differ: loop {a}!={b} np loop')

        if use_for_loop:
            return a
        else:
            return b

    if use_for_loop:
        return compute_avg_with_loop()
    else:
        return compute_avg_with_np()



# Taken from [1]
# [1] https://stackoverflow.com/questions/52030945/python-cosine-similarity-between-two-large-numpy-arrays
def normalize_matrix_for_cosine_similarity(matrix):
    if np.isnan(matrix).any():
        print(f'{Fore.RED}Matrix {matrix.shape} contain nan values (careful they are replaced by 0 in this method)\n{matrix}')
    # Note: can normalize wrt different norms
    # Here we have the frobenius norm (equivalent to L2 norm on each vector in matrix)
    # np.linalg.norm() does frobenius norm for matrix by default
    # different norms (matrix, vector) can output different value and output size
    norms = (matrix**2).sum(axis=1, keepdims=True)**.5
    # equivalent with np norm seems to be the following:
    # norms = np.linalg.norm(matrix, ord=2, axis=1, keepdims=True)

    # Note: other computation
    # norms = np.linalg.norm(matrix) # computes 2 norm of matrix.ravel (flattened 1D array) as ord=axis=None
    # norms = np.linalg.norm(matrix, ord='fro', axis=None, keepdims=True)
    # if matrix.ndim == 2:
    #     # Ord: 'fro' (default) 'nuc', 2 (2-norm)
    #     # norms = np.linalg.norm(matrix, ord='fro', axis=None, keepdims=True)
    #     norms = np.linalg.norm(matrix, ord=2, axis=1, keepdims=True)
    # elif matrix.ndim == 1:
    #     # Ord:
    #     print(f'{Fore.BLUE}matrix dim=1') # not printed most of the time ?
    #     norms = np.linalg.norm(matrix, ord=2, axis=None, keepdims=False)
    # else:
    #     print(f'{Fore.RED}Error unexpected matrix dimension')

    # Matrix norm give one norm for a matrix but want one norm for each vector
    logging.debug(f'{Fore.GREEN}norms shape: {norms.shape}')
    # it could happen that a norm is zero and we cannot divide by zero we could replace the 0 values in the norm vector
    #  by 1 (any other number would work because if a norm is 0 the original vector is the zero vector) ?
    # norms[norms <= 0] = 1 # TypeError: 'numpy.float64' object does not support item assignment
    # this produce error (NaN division by 0) in true divide when do not apply centering (from python RuntimeWarning)
    normalized_matrix = matrix/norms
    # Set the NaN values to 0 (most likely occured due to division by 0 as there were none before)
    normalized_matrix[np.isnan(normalized_matrix)] = 0
    # np.nan_to_num(normalized_matrix) also exists but by default also replace posinf and neginf by representable values
    return normalized_matrix

def random_cluster_assignment(feature_data, cluster_size, seed=None):
    # From quickstart guide this is the new way to do it [1]
    # [1] https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
    rng = np.random.default_rng(seed)
    # Use choice to sample all the data and then just segments it ?
    reordered_indices = rng.choice(range(feature_data.shape[0]), size=feature_data.shape[0], replace=False)
    logging.debug(f'sampled indices order:\n{list(reordered_indices)}')
    # Sanity check
    # unique_id = np.unique(reordered_indices)
    # logging.debug(f'unique ids total: {len(unique_id)} should be {user_id_count}')
    cluster_indices = []
    for i in range(0, feature_data.shape[0], cluster_size):
        logging.debug(f'slice:{i, i+cluster_size}')
        # Note that last cluster size consist of remaining samples so can have arbitrary size
        cluster_indices.append(reordered_indices[i:i+cluster_size])

    return cluster_indices


def compute_cluster_centroid(list_of_ids, centered_feature_data):
    features_of_user_in_cluster = centered_feature_data[list_of_ids, :] # Use advanced indexing
    # logging.d1bg(f'features of user in cluster:\n{features_of_user_in_cluster}\nids of user in cluster:\n{list_of_ids}')
    centroid = np.mean(features_of_user_in_cluster, axis=0) # want a mean feature vector for user in cluster as centroid
    return centroid



# input_vector is an ndarray
def simhash_floc_whitepaper(input_vector, output_bitlength, seed=None):
    # Note: can make the seed depends on value in vector and positions or something similar to chromium code
    # w_i is a random unit-norm vector
    # Note: logging takes time even if not printed anyway and already slow so commented out
    #  from 30 seconds to 5 min in some array prints
    # logging.debug(f'shape of input vector: {input_vector.shape}') # (20,)
    w_list = generate_random_unitnorm_vectors(output_bitlength, input_vector.shape[0], seed=seed)
    # logging.debug(f'w list:\n{w_list}') # to check different vector in list and same across calls
    output_hash_list = []
    out_hash = 0
    # logging.debug(f'dot product between input vector and random vectors with seed {seed}')
    for i in range(output_bitlength): # [0, output_bitlength-1]
        # Scalar/inner/dot product
        scalar_product_res = np.dot(w_list[i], input_vector)
        # logging.debug(f'\n{list(w_list[i])}.\n{list(input_vector)}\n={scalar_product_res}')
        current_bit = int(scalar_product_res > 0) # True as int is 1 and False 0
        output_hash_list.append(current_bit)
        if current_bit:
            # in chromium code but when say vector coordinate start first one has index 1 ?
            # out_hash |= (1 << i)
            # In other order
            out_hash |= (1 << (output_bitlength - 1 - i)) # bitwise or

    return out_hash, output_hash_list


# Inspired by [1]
# [1] https://towardsdatascience.com/the-best-way-to-pick-a-unit-vector-7bd0cc54f9b
def generate_random_unitnorm_vectors(vector_count, coordinate_count, seed=None):
    # if seed is not None: # seed=None is valid
    np.random.seed(seed)

    rand_unit_vector_list = []
    for _ in range(vector_count):
        # Note that seed is not set in random_unit_vector() as want different outputs
        #  though could use seed increments with a set seed
        rand_unit_vector_list.append(random_unit_vector(coordinate_count))

    return rand_unit_vector_list

def random_unit_vector(coordinate_count):
    # Sample standard gaussian
    # components = [np.random.normal() for i in range(coordinate_count)]
    # if set seed before this call it will send same value for each vector (not wanted)
    vector = np.random.normal(size=coordinate_count)
    # Compute norm
    # r = math.sqrt(sum(x * x for x in components))
    unit_vector = vector / np.linalg.norm(vector)
    return unit_vector

def plot_results(xydata_dict, filename='98percentile', save_path=None, title_tag='full'):
    # [1] https://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib
    k = 4
    plt.rcParams["figure.figsize"] = (1.2 * k, 1.4 * k)
    plt.rcParams["figure.dpi"] = 300

    fig, ax = plt.subplots()

    for k, v in xydata_dict.items():
        # x, y
        print(f"{k}: len ({len(v['x_cluster_size']), len(v['y_98wquantile'])}) {v['x_cluster_size']} {v['y_98wquantile']}")
        ax.plot(v['x_cluster_size'], v['y_98wquantile'], v['fmt'], linewidth=2.0, label=k)

    ax.set(xlim=(0, 5000), xlabel='98-percentile anonymity', xticks=np.arange(0, 5001, 1000),
           ylim=(0, 1), ylabel='Cosine similarity', yticks=np.arange(0, 1, 0.2))

    plt.grid(visible=True, linestyle='--')
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    plt.legend(title=f'Clustering method ({title_tag})', loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.25), borderaxespad=0)
    plt.tight_layout()
    # plt.title('Clustering method')
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        # Can specify format of file
        filename += f"_{datetime.now().strftime('%d-%m-%Y_%Hh%Mm%Ss')}"
        # added the PNG but saved a pdf by default with new setting near matplotlib import
        # plt.savefig(f'{save_path}/{filename}.png')
        plt.savefig(f'{save_path}/{filename}.pdf')
        plt.savefig(f'{save_path}/{filename}.pgf')

    plt.show()


def simhash_precomputation(feature_matrix, max_simhash_bitlength, seed):
    id_to_simhash_dict = {}
    for i in trange(user_id_count, desc='simhash precomputation'):
        # for i in range(5):
        current_user_features = feature_matrix[i, ...]

        # Note: might wanna make the seed somehow depend on features as done for chromium implementation
        out_hash, out_list = simhash_floc_whitepaper(current_user_features, max_simhash_bitlength, seed=seed)
        logging.debug(f'{out_hash} {out_hash:0{max_simhash_bitlength}b} {out_list}')
        id_to_simhash_dict[i] = out_hash

    return id_to_simhash_dict


def chromium_simhash_precomputation(userid_2_movieidrating, id_to_title, max_simhash_bitlength):
    userid_to_simhash_dict = {}
    # userid_2_movieidrating has userids from [1 to user_id_count]
    for i in trange(1, user_id_count+1, desc='chromium simhash precomputation'):
    # for i in range(1652, 1653): # userid 1652 had problematic movie `Начальник` that cityhash64 to 67 bits so overflow
        movie_title_history = [id_to_title[movie_id] for movie_id, _ in userid_2_movieidrating[i]]
        # print(f'history {i} ({len(userid_2_movieidrating[i])}): {userid_2_movieidrating[i]}')
        cur_simhash = sim_hash.sim_hash_strings(set(movie_title_history), max_simhash_bitlength)
        userid_to_simhash_dict[i] = cur_simhash

    return userid_to_simhash_dict


# Might need pycharm python console disabled to run as need this to be pickle-able
def worker_simhash_precomputation(start, stop, usr_2_movies, id_to_title, max_bitlength):
    userid_to_simhash_dict = {}
    # userid_2_movieidrating has userids from [1 to user_id_count]

    for i in trange(start, stop, desc=f'chromium simhash precomputation [{start},{stop}['):
        movie_title_history = [id_to_title[movie_id] for movie_id, _ in usr_2_movies[i]]
        cur_simhash = sim_hash.sim_hash_strings(set(movie_title_history), max_bitlength)
        userid_to_simhash_dict[i] = cur_simhash

    return userid_to_simhash_dict


def chromium_simhash_precomputation_parallelized(userid_2_movieidrating, id_to_title, max_simhash_bitlength):
    user_id_count = len(userid_2_movieidrating) # should be the same as user_id_count in main
    step = user_id_count // os.cpu_count() # should return number of thread (e.g. 16)

    penultimate = user_id_count - user_id_count % step
    ranges = [(start, start+step) for start in range(1, penultimate, step)]
    ranges[-1] = (ranges[-1][0], user_id_count+1) # extend length of last
    # ranges.append((penultimate+1, user_id_count+1))
    logging.info(f'ranges ({len(ranges)}): {ranges}')
    params_iterable = []
    for start, stop in ranges:
        useridrange_2_movies = {i: userid_2_movieidrating[i] for i in range(start, stop)}
        params_iterable.append((start, stop, useridrange_2_movies, id_to_title, max_simhash_bitlength))

    # Look for Pool( in [1]
    # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.imap
    # processes=None will take os.cpu_count()
    with multiprocessing.Pool(processes=None) as pool:

        # out_dict_list = pool.map(worker, range(1, 163_000, step)) # will go till 162_000 and step will finish ?

        # same in arbirtrary output order (does not return a list ?)
        # out_dicts = pool.imap_unordered(worker, range(1, 163_000, step))

        # If want multiple argument use [(a,b), (c,d)] will have func(a,b) func(c,d)
        out_dict_list = pool.starmap(worker_simhash_precomputation, params_iterable)

    # https://stackoverflow.com/questions/1781571/how-to-concatenate-two-dictionaries-to-create-a-new-one-in-python
    # id_to_simhash_dict = dict(out_dict_list[0], **out_dict_list[1]) # keyword arguments must be strings
    id_to_simhash_dict = dict(out_dict_list[0])
    for simhash_dict in out_dict_list[1:]:
        id_to_simhash_dict.update(simhash_dict)
    return id_to_simhash_dict

# Return (movieid, rating=1) per user id in compatibility with how read ml25m data
def reconstruct_training_user_movie_rating(training_histories, word_ml25m):
    userid_2_movieidrating = defaultdict(list)  # instead of init all value to list
    cur_userid = 1
    for user_history in training_histories:
        # to make sure create the 2 histories with no movies ? otherwise set user_id_count = 120_000 - 2
        # If want modification to be taken into account need to delete saved file
        userid_2_movieidrating[cur_userid]
        for movie_token in user_history:
            movieid_word = word_ml25m[movie_token]
            if movieid_word != '<pad>' and movieid_word != '<oov>':
                # Note for function reuse pipe a rating of 1 as it won't change any weight
                userid_2_movieidrating[cur_userid].append((movieid_word, 1))
        cur_userid += 1

    return userid_2_movieidrating


if __name__ == '__main__':
    folder_path = '../data/ml-25m'
    movies_file = f'{folder_path}/movies.csv'
    ratings_file = f'{folder_path}/ratings.csv'
    PARAMETERS = {
        'SEED': 32, # Could be None, 42 # Note: with seed None it is almost the same line as Random
        # Note: with 5 user_counts/simhash_values=162541/2^5=5079,4...
        'hash_bitlength': 5, # with 5 can get cohorts of size 5 (modified while running)
        'data_source': 'train', # 'full', 'train'
        'include_gan_simhash_cluster': True,
        'include_chromium_simhash_cluster': True, # Simhash computation according to chromium source code
        'include_wp_simhash_cluster': True, # Whitepaper inspired simhash
        'include_random_cluster': True,
        'CHECKPOINT_FILEPATH': f'../GAN/ckpts_ml25m/leakgan-61',
        'run_all_checkpoints': True,
        'used_already_generated_cluster': True, # If reuse GAN saved cluster assignments ?
        'apply_centering': True,
        'use_rating': False, # Only effective for Full data, ie did not save userid when sampled training so use ratings of 1
        'compare_avg_cos_sim_computation': True, # average_cosine_similarity compare_both= arg
    }
    # Note: floc whitepaper did not specify simhash bitlength used
    # np_cos_sim_computation = True # not needed anymore default value of avg cos sim function

    create_logging_levels() # so that can pass logging.D1BG to args of init_loggers
    # fh might not be initialized so by default None ?
    fh, sh = init_loggers(log_to_stdout=True, log_to_file=True, filename='evaluation', sh_lvl=logging.INFO) # INFO, D1BG
    logging.info(f'Parameters: {pretty_format(PARAMETERS)}')

    if PARAMETERS['data_source'] == 'full':
        # Note: not yet tested 'use_rating' a good way to check is to inspect feature data and see if there are values greater than 1 (when not centered)
        # 'use_rating' only useful for data_source='full' as for train data we did not save userid to recover user ratings and took ratings of 1.
        use_rating_tag = ''
        if not PARAMETERS['use_rating']:
            use_rating_tag = '-wo_rating' # without rating
        # As this is slow need to only load it again when needed otherwise just save the files ?
        if PARAMETERS['apply_centering']:
            # Default: in floc whitepaper centering is applied
            save_precomputed_features_path = f'./saved/floc-whitepaper-features{use_rating_tag}.npy'
        else:
            # Without centering as it could alter the angles for cosine similarity (ie centering conserves euclidian distances)
            save_precomputed_features_path = f'./saved/floc-whitepaper-features-not-centered{use_rating_tag}.npy'
        save_precomputed_chromium_simhash_path = './saved/precomputed_chromium_simhash.pkl'
        user_id_count = FULL_USER_ID_COUNT # use a name alias as now have 120000 users for training ?

    elif PARAMETERS['data_source'] == 'train':
        if PARAMETERS['apply_centering']:
            save_precomputed_features_path = './saved/floc-whitepaper-features-train.npy'
        else:
            save_precomputed_features_path = './saved/floc-whitepaper-features-train-not-centered.npy'
        save_precomputed_chromium_simhash_path = './saved/precomputed_chromium_simhash_train.pkl'
        user_id_count = 120_000 # might wanna define it later when load training ? also 2 empty histories (21053, 43986)

    file_exists = (os.path.isfile(save_precomputed_features_path) and os.path.isfile(save_precomputed_chromium_simhash_path))
    if not file_exists or PARAMETERS['include_gan_simhash_cluster']:
        logging.info(f'Load movies...')
        # Note: id_to_genres needed for GAN feature extraction (could pickle save and load though)
        id_to_title, id_to_genres, id_to_year, id_list = read_movies(movies_file)
        if not file_exists:
            if PARAMETERS['data_source'] == 'full':
                logging.info(f'Load ratings...')
                userid_2_movieidrating = read_ratings(ratings_file, need_rating=True)
            elif PARAMETERS['data_source'] == 'train':
                logging.info(f'Load training data...')
                training_histories = load_train_test('../GAN/save_ml25m/realtrain_ml25m_5000_32.txt')
                logging.info(f'Load token (encoding) to word (movieid) mapping...')
                word_ml25m, vocab_ml25m = pickle.load(open('../GAN/save_ml25m/LeakGAN_ml25m_vocab_5000_32.pkl', mode='rb'))
                # Note for training data as did not keep users all the ratings are taken as 1
                userid_2_movieidrating = reconstruct_training_user_movie_rating(training_histories, word_ml25m)

    # Precompute features
    if not os.path.isfile(save_precomputed_features_path):
        # Here no need to specify the path depending on `apply_centering` param
        # as already dealt with earlier with `save_precomputed_features_path`
        # Note that the 'use_rating' param only useful for 'full' data_source as for training did not save userid to recover ratings and used rating of 1
        centered_feature_matrix = feature_extraction_floc_whitepaper(userid_2_movieidrating, id_to_genres, apply_centering=PARAMETERS['apply_centering'], use_rating=PARAMETERS['use_rating'])
        # can put allow_pickle=False as security issue if untrusted sources
        np.save(save_precomputed_features_path, centered_feature_matrix)
    else:
        # can put allow_pickle=False as security issue if untrusted sources
        centered_feature_matrix = np.load(save_precomputed_features_path)


    plot_xydata_dict = {}

    if PARAMETERS['include_chromium_simhash_cluster']:
        chromium_simhash_wquantiles_list = []
        chromium_simhash_98_wquantile_list = []
        # To make the cluster assignment vary we make the output hash bitlength vary:
        chromium_simhash_bitlength = [10, 9, 8, 7, 6, 5]

        # compute this cluster assignment once outside of the bitlength loop
        chromium_max_simhash_bitlength = 64 # np.max(chromium_simhash_bitlength)
        # can save this (precompute max of 64 bits or less) as simhash is not changed if computed again
        if not os.path.isfile(save_precomputed_chromium_simhash_path):
            # chromium_simhash_user = chromium_simhash_precomputation(userid_2_movieidrating, id_to_title, chromium_max_simhash_bitlength)
            # Note cannot pickle if use pycharm's python console
            chromium_simhash_user = chromium_simhash_precomputation_parallelized(userid_2_movieidrating, id_to_title, chromium_max_simhash_bitlength)
            with open(save_precomputed_chromium_simhash_path, 'wb') as f:
                pickle.dump(chromium_simhash_user, f)
        else:
            chromium_simhash_user = pickle.load(open(save_precomputed_chromium_simhash_path, mode='rb'))

        # Perform cluster assignment etc depending on simhash bitlength etc

        for cur_bitlength in chromium_simhash_bitlength:
            PARAMETERS['hash_bitlength'] = cur_bitlength
            # Full PARAMETERS not logged here
            logging.info(f'Chromium String Simhash bitlengh {cur_bitlength}=={PARAMETERS["hash_bitlength"]}')

            ## Cluster assignment:
            simhash_cluster_assignment = defaultdict(list)
            mask = (1 << cur_bitlength) - 1
            # This time range goes from [1 to user_id_count] and not [0 to user_id_count-1]
            for i in trange(1, user_id_count+1, desc='StrSimHash Cluster Assignment'):
                # Note: iirc smaller simhash is only LSB but could be wrong so do sanity check some time
                #  for sanity check need access to title history derived from userid_2_movieidrating ? eg for one user ?
                cur_simhash = chromium_simhash_user[i] & mask
                # Note that for centered_feature_matrix and others need ids in range [0, COUNT[ and not [1, COUNT]
                simhash_cluster_assignment[cur_simhash].append(i-1)

            # Save cluster assignment for later comparisons:
            with open(f'./saved/cluster_assignments/real/chromium_simhash_hist_cluster_{PARAMETERS["data_source"]}_{cur_bitlength}.pkl', 'wb') as f:
                pickle.dump(simhash_cluster_assignment, f)

            # Compute some cohort size statistics
            str_simhash_cohort_size_dict = {}
            total_count = 0
            for key_simhash, value_id_list in simhash_cluster_assignment.items():
                # cast as int because below pretty format did not work somehow after cur_simhash bit manipulation
                total_count += len(value_id_list)
                str_simhash_cohort_size_dict[int(key_simhash)] = len(value_id_list)

            logging.info(f'total: {total_count}, distribution:\n{pretty_format(str_simhash_cohort_size_dict, use_json=True)}')
            if total_count != user_id_count: logging.warning(f'{Fore.RED}More users than there should {total_count}!={user_id_count}')

            ## Compute average cosine similarity between all users in the cluster and its centroid
            avg_cos_sim_by_simhash_value_list = []
            for str_simhash_value in trange(1 << PARAMETERS['hash_bitlength'], desc='StrSimHash avg cos sim'):
                current_id_list = simhash_cluster_assignment.get(str_simhash_value, None)
                if current_id_list is not None:
                    current_centroid = compute_cluster_centroid(current_id_list, centered_feature_matrix)
                    logging.d1bg(f'centroid: {current_centroid}\ncluster ids:\n{simhash_cluster_assignment[str_simhash_value]}')

                    cur_avg_cosine_similarity = average_cosine_similarity(current_id_list, current_centroid, centered_feature_matrix,
                                                                          compare_both=PARAMETERS['compare_avg_cos_sim_computation'])
                    avg_cos_sim_by_simhash_value_list.append(cur_avg_cosine_similarity)
                else:
                    # Without this get different results but no error is thrown from eg custom weighted quantile ?
                    # if leave pass statement uncommented what is below still gets executed
                    # pass # Note: check which one is more suited
                    avg_cos_sim_by_simhash_value_list.append(0) # So that get matching number of elements for weighted quantile ?

            # Compute weighted quantile
            # ssh for string simhash
            ssh_cluster_size_weights = [str_simhash_cohort_size_dict.get(i, 0) for i in range(1 << PARAMETERS['hash_bitlength'])]
            ssh_cohort_cos_sim_wquantiles = weighted_quantile(avg_cos_sim_by_simhash_value_list, [0.02 * q for q in range(51)],
                                                          sample_weight=ssh_cluster_size_weights)
            chromium_simhash_wquantiles_list.append(ssh_cohort_cos_sim_wquantiles)

            logging.info(f'98-percentile (chromium simhash): {ssh_cohort_cos_sim_wquantiles[49]}')
            chromium_simhash_98_wquantile_list.append(ssh_cohort_cos_sim_wquantiles[49])

        # Fill dict for plot:
        str_simhash_cluster_size = []
        for bitlength in chromium_simhash_bitlength:
            str_simhash_cluster_size.append(user_id_count // (1 << bitlength))
        plot_xydata_dict['StrSimHash'] = {'x_cluster_size':str_simhash_cluster_size, 'y_98wquantile':chromium_simhash_98_wquantile_list, 'fmt':'o-g'}


    if PARAMETERS['include_wp_simhash_cluster']:
        ## Compute the cluster assignment according to equal simhashes

        simhash_wquantiles_list = []
        simhash_98_wquantile_list = []
        # To make the cluster assignment vary we make the output hash bitlength vary:
        simhash_bitlength = [10,9,8,7,6,5]

        # compute this cluster assignment once outside of the bitlength loop
        max_simhash_bitlength = np.max(simhash_bitlength)
        precomputed_simhash_user = simhash_precomputation(centered_feature_matrix, max_simhash_bitlength, PARAMETERS['SEED'])

        for cur_hash_bitlength in simhash_bitlength:
            PARAMETERS['hash_bitlength'] = cur_hash_bitlength
            logging.info(f'Whitepaper simhash bitlen: {PARAMETERS["hash_bitlength"]}')

            # Replace the default value of None to list to create a dictionary of lists
            # (this way do not have to init every possible simhash value)
            # the id is the index in the centered_feature_matrix (do +1 if want user id)
            simhash_to_idList_dict = defaultdict(list)

            # Inefficient way by iterating on ndarray
            # for i in range(user_id_count):
            # Note technically could do it only once and compute the simhash with the highest bitlength
            mask = (1 << cur_hash_bitlength) - 1
            shift = (max_simhash_bitlength - cur_hash_bitlength)
            for i in trange(user_id_count, desc='simhash cluster assignment'):

                # Precomputed simhash
                cur_simhash = (precomputed_simhash_user[i] & (mask << shift)) >> shift

                compare_both_cluster_assignment = False
                if compare_both_cluster_assignment and max_simhash_bitlength != cur_hash_bitlength:
                    current_user_features = centered_feature_matrix[i, ...]
                    # might wanna make the seed somehow depend on features as done for chromium implementation
                    out_hash, out_list = simhash_floc_whitepaper(current_user_features, PARAMETERS['hash_bitlength'], seed=PARAMETERS['SEED'])
                    logging.debug(f'{out_hash} {out_hash:0{PARAMETERS["hash_bitlength"]}b} {out_list}')
                    if out_hash != cur_simhash:
                        fmt = f'0{max_simhash_bitlength}b' # PARAMETERS["hash_bitlength"]
                        logging.warning(f'simhash computation different: {cur_simhash:{fmt}} != {out_hash:{fmt}}, full simhash: {precomputed_simhash_user[i]:{fmt}}')

                simhash_to_idList_dict[cur_simhash].append(i)

            ## Cohort size statistics
            simhash_cohortsize_dict = {}
            total_count = 0
            for key_simhash, value_ids in simhash_to_idList_dict.items():
                total_count += len(value_ids)
                # cast as int because below pretty format did not work somehow after cur_simhash bit manipulation
                simhash_cohortsize_dict[int(key_simhash)] = len(value_ids)

            logging.info(f'total: {total_count}, distribution:\n{pretty_format(simhash_cohortsize_dict, use_json=True)}')
            if total_count != user_id_count: logging.warning(f'{Fore.RED}More users than there should {total_count}!={user_id_count}')

            # Compute average cosine similarity (normalized dot product) between all users in the cluster
            # and the centroid of the cluster (average of features of users in same cluster?)

            # as iterate through possible simhash values in increasing order index of list should be simhash value
            avg_cos_sim_list = []
            # for simhash_value in range(1 << PARAMETERS['hash_bitlength']): # 2 ** bitlength
            for simhash_value in trange(1 << PARAMETERS['hash_bitlength'], desc='simhash avg cos sim'):
                current_id_list = simhash_to_idList_dict.get(simhash_value, None)
                if current_id_list is not None:
                    current_centroid = compute_cluster_centroid(current_id_list, centered_feature_matrix)
                    logging.d1bg(f'centroid: {current_centroid}\ncluster ids:\n{simhash_to_idList_dict[simhash_value]}')

                    cur_avg_cosine_similarity = average_cosine_similarity(current_id_list, current_centroid, centered_feature_matrix,
                                                                          compare_both=PARAMETERS['compare_avg_cos_sim_computation'])
                    avg_cos_sim_list.append(cur_avg_cosine_similarity)
                else:
                    # Without this get different results but no error is thrown from eg custom weighted quantile ?
                    pass # Note: check which one is more suited
                    avg_cos_sim_list.append(0) # So that get matching number of elements for weighted quantile ?

            # "To measure privacy we look at the 2% quantile of the cohort size distribution
            #   weighted by the number of users in the cohort."
            # Note: this means that compute the weighted average of the 2% quantile of the cohort size distribution ?
            # np.average(array, axis=, weights=) # can compute weighted average
            # np.quantile() # np.percentile equivalent but write 2 instead of 0.02 ?)
            # Note: there exist a way to compute weighted quantiles can check [1]
            #  weighted quantile is similar to weighted average could just repeat the values by their weight number of time
            # [1] https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
            # [2] https://en.wikipedia.org/wiki/Percentile#The_weighted_percentile_method
            cohort_size_distrib_wquantiles = weighted_quantile(list(simhash_cohortsize_dict.keys()), [0.02*q for q in range(51)], sample_weight=list(simhash_cohortsize_dict.values()))
            # cohort_cos_sim_wquantiles = weighted_quantile(avg_cos_sim_list, [0.02*q for q in range(51)], sample_weight=list(simhash_cohortsize_dict.values()))
            # Note to ensure that the order is the same between the two lists (result not the same as with above statement)
            #  the parameters values_sorted of wquantile is still false as we did not sort the values which are the avg cos sim score
            # need the get as some simhash value might have no user matching them
            cluster_size_weights = [simhash_cohortsize_dict.get(i, 0) for i in range(1 << PARAMETERS['hash_bitlength'])]
            cohort_cos_sim_wquantiles = weighted_quantile(avg_cos_sim_list, [0.02 * q for q in range(51)], sample_weight=cluster_size_weights)
            simhash_wquantiles_list.append(cohort_cos_sim_wquantiles)

            logging.info(f'98% quantile (whitepaper simhash): {cohort_cos_sim_wquantiles[49]}') # 49*0.02 quantile=98 percentile
            simhash_98_wquantile_list.append(cohort_cos_sim_wquantiles[49])

        # Fill dict for plot:
        simhash_cluster_size = []
        for bitlength in simhash_bitlength:
            simhash_cluster_size.append(user_id_count // (1 << bitlength))
        plot_xydata_dict['SimHash'] = {'x_cluster_size':simhash_cluster_size, 'y_98wquantile':simhash_98_wquantile_list, 'fmt':'o-m'}


    if PARAMETERS['include_random_cluster']:
        ## Same thing for random cluster assignement
        # Iterate over different cluster size

        rand_wquantiles_list = []
        rand_98_wquantile_list = []
        cluster_size_rand = [100, 200, 500, 1000, 2000, 5000]
        # cur_cluster_size = 5000
        for cur_cluster_size in cluster_size_rand:
            # Compute cluster assignment
            random_cluster_indices = random_cluster_assignment(centered_feature_matrix, cur_cluster_size, seed=PARAMETERS['SEED'])

            # Compute average cosine similarity for element in the cluster
            rand_avg_cos_sim_list = []
            for ids_in_cluster in random_cluster_indices:
                cur_centroid = compute_cluster_centroid(ids_in_cluster, centered_feature_matrix)
                rand_cur_avg_cos_sim = average_cosine_similarity(ids_in_cluster, cur_centroid, centered_feature_matrix,
                                                                 compare_both=PARAMETERS['compare_avg_cos_sim_computation'])
                # Accumulate those average cosine similarity per cluster sizes
                rand_avg_cos_sim_list.append(rand_cur_avg_cos_sim)

            # Compute weighted quantile
            rand_cluster_weights = [len(id_list) for id_list in random_cluster_indices]
            total_count = sum(rand_cluster_weights)
            if total_count != user_id_count: logging.warning(f'{Fore.RED}More users than there should {total_count}!={user_id_count}')
            logging.info(f'cluster sizes (weights) for random assignment (total={total_count}):\n{rand_cluster_weights}')

            rand_cohort_cos_sim_wquantiles = weighted_quantile(rand_avg_cos_sim_list, [0.02 * q for q in range(51)], sample_weight=rand_cluster_weights)
            rand_wquantiles_list.append(rand_cohort_cos_sim_wquantiles)
            rand_98_wquantile_list.append(rand_cohort_cos_sim_wquantiles[49])
            logging.info(f'98% quantile (random clusters): {rand_cohort_cos_sim_wquantiles[49]}')

        # Fill data for plot
        plot_xydata_dict['Random'] = {'x_cluster_size': cluster_size_rand, 'y_98wquantile': rand_98_wquantile_list,
                                       'fmt': 'o-b'}

    # Moved to last to rerun plot with each other data already computed
    if PARAMETERS['include_gan_simhash_cluster']:
        # otherwise might be loaded a lot:
        logging.info(f'Load token (encoding) to word (movieid) mapping...')
        # Load vocabulary (word_ml25m, vocab_ml25m)
        word_ml25m, vocab_ml25m = pickle.load(open('../GAN/save_ml25m/LeakGAN_ml25m_vocab_5000_32.pkl', mode='rb'))
        # Precompute cityhashes (only need hash_for_movie and title_for_movie currently ?) for movies in training vocabulary
        hash_for_movie, title_for_movie, movie_for_title, movie_id_list = precompute_cityhash_movielens()

        # Load once the gan model (for each checkpoint run on)
        needed_parameters = {'SEQ_LENGTH': 32, 'BATCH_SIZE': 64}
        # 'CHECKPOINT_FILEPATH': f'../GAN/ckpts_ml25m/leakgan-61'

        # If wanna run for all checkpoints
        run4all_checkpoints = PARAMETERS['run_all_checkpoints']
        if run4all_checkpoints:
            checkpoint_filepaths = [
                f'../GAN/ckpts_ml25m/leakgan_preD', f'../GAN/ckpts_ml25m/leakgan_pre',
                f'../GAN/ckpts_ml25m/leakgan-1', f'../GAN/ckpts_ml25m/leakgan-11',
                f'../GAN/ckpts_ml25m/leakgan-21', f'../GAN/ckpts_ml25m/leakgan-31',
                f'../GAN/ckpts_ml25m/leakgan-41', f'../GAN/ckpts_ml25m/leakgan-51',
                f'../GAN/ckpts_ml25m/leakgan-61',
            ]
        else:
            checkpoint_filepaths = PARAMETERS['CHECKPOINT_FILEPATH']

        for current_checkpoint in checkpoint_filepaths:
            PARAMETERS['CHECKPOINT_FILEPATH'] = current_checkpoint
            ckpt_nbr = current_checkpoint[-2:]

            if not PARAMETERS['used_already_generated_cluster']:
                # Note: somehow froze running if loaded model without doing anything with it ?
                needed_parameters['CHECKPOINT_FILEPATH'] = PARAMETERS['CHECKPOINT_FILEPATH']
                logging.info(f'Init LeakGAN with params: {needed_parameters}')
                sess, generator_model, discriminator = init_generator(needed_parameters)


            gan_wquantiles_list = []
            gan_98_wquantile_list = []
            gan_simhash_bitlength = [10, 9, 8, 7, 6, 5]  # [5, 5, 5, 5, 5, 5]
            # Note: with GAN can decide how many generate per cluster as sample them one by one ?
            # Note: user count can differ a lot per sample
            cluster_size_gan = [100, 200, 500, 1000, 2000, 5000]  #
            # if take ceil (or floor) of useridcount / simhashvaluecount: [159, 318, 635, 1270, 2540, 5080]

            for cur_bitlength, cluster_size in zip(gan_simhash_bitlength, cluster_size_gan):
                PARAMETERS['hash_bitlength'] = cur_bitlength
                # Full PARAMETERS not logged here
                logging.info(f'GAN String Simhash bitlengh {cur_bitlength} cluster size {cluster_size}')

                # Note: as this step is dumped by pickle could load an already saved state
                gan_simhash_cluster_assignment = {}
                gan_gen_save_folderpath = f'./saved/cluster_assignments/generated/checkpoint_{ckpt_nbr}'
                if PARAMETERS['used_already_generated_cluster']:
                    # Note that assumes already generated so does not check if file not there
                    gan_saved_cluster_path = f'{gan_gen_save_folderpath}/LeakGAN_simhash_hist_cluster_{cur_bitlength}_{cluster_size}.pkl'
                    gan_simhash_cluster_histories = pickle.load(open(gan_saved_cluster_path, mode='rb'))
                else:
                    gan_simhash_cluster_histories = {}
                    start = 0
                    for target_simhash_value in trange(1 << PARAMETERS['hash_bitlength'],
                                                       desc=f'GAN generate clusters {cur_bitlength}, {cluster_size}'):
                        # Note: how many movie want to generate from GAN in how many clusters
                        # Note: could also decide to cut out the histories above the desired count for more comparable results ?
                        gan_simhash_cluster_histories[target_simhash_value] = \
                            generate_movie_simhash_cluster(cluster_size, target_simhash_value, cur_bitlength,
                                                           sess, generator_model, title_for_movie, hash_for_movie,
                                                           token_decoder=word_ml25m)

                        # Note that due to the batch generation can have more than cluster size solutions
                        # (also gurobi code can returns more than one solution per problem)
                        cur_cluster_size = len(gan_simhash_cluster_histories[target_simhash_value])
                        # For compatibility with other methods keep track of range for later numpy advanced indexing
                        # also could use slices as indexing so only keep start and end because of continuous range ?
                        gan_simhash_cluster_assignment[target_simhash_value] = list(range(start, start + cur_cluster_size))
                        start += cur_cluster_size

                    # save the generated clusters
                    Path(gan_gen_save_folderpath).mkdir(parents=True, exist_ok=True)
                    with open(f'{gan_gen_save_folderpath}/LeakGAN_simhash_hist_cluster_{cur_bitlength}_{cluster_size}.pkl', 'wb') as f:
                        pickle.dump(gan_simhash_cluster_histories, f)

                # Compute features with genres (also centered all features)
                # Note: where/when apply centering (once have all features most likely but features different from other metric)
                centered_gen_feature_matrix, userid_per_cluster = feature_extractor_generated_movies(gan_simhash_cluster_histories,
                                                                                                     id_to_genres,
                                                                                                     PARAMETERS['hash_bitlength'], apply_centering=PARAMETERS['apply_centering'])

                # Compute cohort size statistic: (also sanity check)
                gan_simhash_cohort_size_dict = {}
                total_count = 0
                if PARAMETERS['used_already_generated_cluster']:
                    for simhash_value in trange(1 << PARAMETERS['hash_bitlength'], desc='compute cohort size stat GAN'):
                        # userid start at 1
                        cluster_userid_start, cluster_userid_end = userid_per_cluster[simhash_value]
                        cluster_size = cluster_userid_end - cluster_userid_start + 1
                        total_count += cluster_size
                        gan_simhash_cohort_size_dict[int(simhash_value)] = cluster_size
                        # Fill in the not yet created gan_simhash_cluster_assignment (note switch to 0 based from 1 based index)
                        # also both indices included ie [start, end] so as range exclude end we do not subtract 1 ?
                        gan_simhash_cluster_assignment[simhash_value] = list(range(cluster_userid_start-1, cluster_userid_end))

                else:

                    for key_simhash, value_id_list in gan_simhash_cluster_assignment.items():
                        # cast as int because below pretty format did not work somehow after cur_simhash bit manipulation
                        cohort_card = len(gan_simhash_cluster_histories[key_simhash])
                        if cohort_card != len(value_id_list): logging.warning(f'{Fore.RED}Different sizes not expected {cohort_card} != {len(value_id_list)}')
                        # Cluster user id range (1-based)
                        cluster_userid_start, cluster_userid_end = userid_per_cluster[key_simhash]
                        if not (cluster_userid_start-1 == value_id_list[0] and cluster_userid_end-1 == value_id_list[-1]):
                            print(f'{Fore.RED}Different userid unexptected: (1-based) {cluster_userid_start, cluster_userid_end} != {value_id_list[0], value_id_list[1]} (0-based)')
                        total_count += len(value_id_list)
                        gan_simhash_cohort_size_dict[int(key_simhash)] = len(value_id_list)

                logging.info(f'total: {total_count}, distribution:\n{pretty_format(gan_simhash_cohort_size_dict, use_json=True)}')
                expected_total = (1 << cur_bitlength) * cluster_size
                # Note: so size of cluster varies quite a bit unless discard some of them
                if total_count != expected_total: logging.warning(f'{Fore.RED}More users than there should {total_count}!={expected_total}')

                # Note: need order in feature matrix to be somehow maintained wrt to index if want to use same methods for computing centroid etc
                #  Compute centroid and avg cosine similarity on centered full feature matrix in other loop

                ## Compute average cosine similarity between all users in the cluster and its centroid
                gan_avg_cos_sim_by_simhash_value_list = []
                for str_simhash_value in trange(1 << PARAMETERS['hash_bitlength'], desc='GAN StrSimHash avg cos sim'):
                    # here should never return None by construction ?
                    current_id_list = gan_simhash_cluster_assignment.get(str_simhash_value, None)
                    if current_id_list is not None:
                        # Note that here use a different feature matrix
                        current_centroid = compute_cluster_centroid(current_id_list, centered_gen_feature_matrix)
                        logging.d1bg(f'centroid: {current_centroid}\ncluster ids:\n{gan_simhash_cluster_assignment[str_simhash_value]}')

                        cur_avg_cosine_similarity = average_cosine_similarity(current_id_list, current_centroid, centered_gen_feature_matrix,
                                                                              compare_both=PARAMETERS['compare_avg_cos_sim_computation'])
                        gan_avg_cos_sim_by_simhash_value_list.append(cur_avg_cosine_similarity)
                    else:
                        gan_avg_cos_sim_by_simhash_value_list.append(0)

                # Compute weighted quantile
                gan_cluster_size_weights = [gan_simhash_cohort_size_dict.get(i, 0) for i in
                                            range(1 << PARAMETERS['hash_bitlength'])]
                gan_cohort_cos_sim_wquantiles = weighted_quantile(gan_avg_cos_sim_by_simhash_value_list,
                                                                  [0.02 * q for q in range(51)],
                                                                  sample_weight=gan_cluster_size_weights)
                gan_wquantiles_list.append(gan_cohort_cos_sim_wquantiles)

                logging.info(f'98-percentile (gan simhash): {gan_cohort_cos_sim_wquantiles[49]}')
                gan_98_wquantile_list.append(gan_cohort_cos_sim_wquantiles[49])

            # Fill dict for plot:
            # Note that for GAN we already have another parameters with cluster size
            # so the total user count is already divisible by simhash bitlength
            # gan_simhash_cluster_size = []
            # for bitlength in gan_simhash_bitlength:
            #     gan_simhash_cluster_size.append(expected_total // (1 << bitlength))
            plot_xydata_dict['GANStrSimHash'] = {'x_cluster_size': cluster_size_gan,
                                                 'y_98wquantile': gan_98_wquantile_list, 'fmt': 'o-r'}

            feature_centered_tag = ''
            if not PARAMETERS['apply_centering']:
                feature_centered_tag = '_not_centered'

            with open(f'{gan_gen_save_folderpath}/plot_data{feature_centered_tag}.pkl', 'wb') as f:
                pickle.dump(plot_xydata_dict, f)


            plot_results(plot_xydata_dict, filename=f'98percentile_ckpt_{ckpt_nbr}',
                         save_path=f'./saved/figs', title_tag=PARAMETERS['data_source'])


    # Plot results (legacy one in the end before multiple checkpoint run for GAN)
    plot_results(plot_xydata_dict, save_path=f'./saved/figs', title_tag=PARAMETERS['data_source'])

    timers = Timer.timers

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
