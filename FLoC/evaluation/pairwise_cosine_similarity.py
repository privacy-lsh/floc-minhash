import pickle
import os
import glob
import re
from tqdm import tqdm, trange
from FLoC.preprocessing.movielens_extractor import feature_extractor_generated_movies, read_movies, read_ratings, \
    user_id_count as FULL_USER_ID_COUNT
import numpy as np
from anonymity_evaluation import compute_cluster_centroid, average_cosine_similarity, normalize_matrix_for_cosine_similarity, reconstruct_training_user_movie_rating
from pip._vendor.colorama import Fore, init
init(autoreset=True) # to avoid reseting color everytime
from FLoC.utils import init_loggers, create_logging_levels
import logging
from FLoC.attack.generating_histories import load_train_test
from collections import namedtuple, Counter
from FLoC.pipeline import compute_standard_deviation, compute_average_common_movie_counts
from FLoC.utils import pretty_format
from FLoC.evaluation.anonymity_evaluation import random_cluster_assignment

# can check online and sklearn source from [1] (different matrix norms used) similar to [2]
# [1] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
# [2] https://stackoverflow.com/questions/52030945/python-cosine-similarity-between-two-large-numpy-arrays
def matrix_cosine_similarity(X, Y):
    # Normalize (note could use different matrix norm)
    X_norm = normalize_matrix_for_cosine_similarity(X) # [n_samples_X, n_features]
    Y_norm = normalize_matrix_for_cosine_similarity(Y) # [n_samples_Y, n_features]
    # Need to transpose T for matrix multiplication dimension match
    Pairwise_cos_sims = X_norm.dot(Y_norm.T) # if both 2D matrix matmul (X@Y) is preferred

    return Pairwise_cos_sims

# This one would work for matrices and vectors but for matrix it would use matrix norm (1 value)
# and want vector norm along axis of matrix (n_samples values ie normalize each feature vector individually)?
def cosine_similarity(X, Y):
    if np.isnan(X).any() or np.isnan(Y).any():
        print(f'{Fore.RED}One of the input matrix to cosine similarity contains NaN values')
    X = np.array(X)
    Y = np.array(Y)
    # Normalize (note could use different matrix norm)
    # Note: check normalize_matrix_for_cosine_similarity might not do same computations
    # ie for matrices if use matrix norm will get 1 dimension output but want each feature vector normalized individually
    # So could compute (as in normalize_matrix_for_cosine_similarity):
    # norms = (matrix ** 2).sum(axis=1, keepdims=True) ** .5
    # equivalent with np norm seems to be the following:
    # norms = np.linalg.norm(matrix, ord=2, axis=1, keepdims=True)

    X_norm = X/np.linalg.norm(X)  # [n_samples_X, n_features]
    Y_norm = Y/np.linalg.norm(Y)  # [n_samples_Y, n_features]

    # It could happen that some norm is zero and divide by zero so replace nan values by 0 after the operation
    if np.isnan(X_norm).any() or np.isnan(Y_norm).any():
        print(f'{Fore.RED}After normalization X and Y contains NaN values => replaced by 0')
        X_norm[np.isnan(X_norm)] = 0
        Y_norm[np.isnan(Y_norm)] = 0

    # Need to transpose T for matrix multiplication dimension match
    Pairwise_cos_sims = X_norm.dot(Y_norm.T)  # if both 2D matrix matmul (X@Y) is preferred

    return Pairwise_cos_sims

def recover_2Dindex_from_flattened(index, shape):
    # assumes x has shape[0] rows and y has shape[1] columns
    x = index // shape[1]
    y = index % shape[1]
    return x, y


def binary_search_cluster_ids(id_range_per_cluster, target_id):
    low = 0
    high = max(id_range_per_cluster, key=id_range_per_cluster.get) # get the maximum key in dict
    while low <= high:
        mid = (high + low) // 2 # should do it otherwise if high+low can overflow

        # If target_id is greater than current cluster upperbound, ignore left half
        if id_range_per_cluster[mid][1] < target_id:
            low = mid + 1

        # If target_id is smaller than current cluster lowerbound, ignore right half
        elif id_range_per_cluster[mid][0] > target_id:
            high = mid - 1

        # means target_id is in current cluster with simhash value mid
        else:
            return mid

    # If we reach here, then the element was not present
    raise Exception(f'Error should have found a solution')
    # return -1

def movieid_rating_to_title_genres(movierating_history, id_to_title, id_to_genres):
    title_history = []
    associated_genres = []
    genres_stats = Counter()
    for movieid_2 in movierating_history:
        if isinstance(movieid_2, tuple):
            movieid = movieid_2[0] # for real have rating with the movieid
        else:
            movieid = movieid_2
        title_history.append(id_to_title[movieid])
        genre_set = id_to_genres[movieid]
        associated_genres.append(genre_set)
        for genre in genre_set:
            genres_stats[genre] += 1

    # Note: could perform genre representation stat Counter for each genre
    return title_history, associated_genres, genres_stats

# Stats about cross (pairwise) cosine similarity extremum entry
def recover_extremum_stats(extr_under_consideration, cross_cos_sim, common_movie_ctr, common_genre_ctr, gan_clusters_history_data, userid_per_cluster_gan, cur_gen_cluster_ids,
                           userid_2_movieidrating, cur_real_cluster_ids, simhash_value, id_to_title, id_to_genres, gan_clusters_history_data_for_random=None, perform_sanity_check=True):
    # Extremum under consideration
    if extr_under_consideration == 'max':
        pcos_sim_extr = np.argmax(cross_cos_sim)  # max
    elif extr_under_consideration == 'min':
        pcos_sim_extr = np.argmin(cross_cos_sim)
    elif extr_under_consideration == 'absmin':
        pcos_sim_extr = np.argmin(np.abs(cross_cos_sim))  # absolute value closest to 0

    # index is in flattened array so need to recover real index:
    pcos_aextr_gen, pcos_aextr_real = recover_2Dindex_from_flattened(pcos_sim_extr, cross_cos_sim.shape)
    # If want to recover features need to use cluster list mapping:
    # Note: cur_gen_cluster_ids not in function parameter so would not work without pycharm console run ?
    feature_aextr_gen, feature_aextr_real = cur_gen_cluster_ids[pcos_aextr_gen], cur_real_cluster_ids[pcos_aextr_real]


    print(f'{Fore.GREEN}{extr_under_consideration}{Fore.RESET} cos sim {cross_cos_sim[pcos_aextr_gen, pcos_aextr_real]} '
          f'{Fore.BLUE}(gen [cluster:{pcos_aextr_gen} feature:{feature_aextr_gen}] | real [cluster:{pcos_aextr_real} feature:{feature_aextr_real}]):{Fore.RESET}')
    # Need to make this print aware of self cosine similarity otherwise out of bound on feature matrix (not passed in argument of func)
    # f'{list(centered_gen_feature_matrix[feature_aextr_gen])}')
    # f'{list(centered_real_feature_matrix[feature_aextr_real])}'

    # Adaptation for self cosine similarity usage:
    if userid_2_movieidrating is not None: # Would be None in case were explicitly pass None
        # For real user from training data or full ml25m data:
        real_tag = 'Real'
        print(f'Features: {list(centered_real_feature_matrix[feature_aextr_real])}')
        # Note: need a +1 on feature_aextr_real ? as features matrix index is 0 based while userid is not by construction ?
        #  anyway had a lot more common movies with the +1 than without (and should get key error if feature_aextr_real = 0)
        extr_real_history = userid_2_movieidrating[feature_aextr_real+1] # would need to check if the index matches eg no +-1 needed ?
        real_title_hist, real_genres, real_genre_stats = movieid_rating_to_title_genres(extr_real_history, id_to_title, id_to_genres)

    else:
        real_tag = 'Gen'
        print(f'Features: {list(centered_gen_feature_matrix[feature_aextr_real])}')
        if PARAMETERS['use_random']:
            feature_index = gan_clusters_history_data[simhash_value][pcos_aextr_real]
            if feature_index != feature_aextr_real:
                print(f'{Fore.RED}{feature_index} == {feature_aextr_real} should be true')
            if PARAMETERS['random_use_gen']:
                extr_real_history = gan_clusters_history_data_for_random[simhash_value][pcos_aextr_real]
            else:
                extr_real_history = userid_2_movieidrating[feature_aextr_real + 1]  # Note: +1 cause 0-based to 1-based indexing ?
        else:
            extr_real_history = gan_clusters_history_data[simhash_value][pcos_aextr_real]
        real_title_hist, real_genres, real_genre_stats = movieid_rating_to_title_genres(extr_real_history, id_to_title, id_to_genres)

    print(f'{real_tag} movie history ({len(real_title_hist)}):{real_title_hist}\nsorted: {sorted(real_title_hist)}\n{real_genres} not sorted as movies')
    first_in_pair = lambda pair: pair[0]
    # sort by highest count real_genre_stats.most_common(), sorted(real_genre_stats.items(), key=first_in_pair, reverse=False)
    # by key is better for comparison of distribution as will have same order for real and gen
    print(f'{Fore.CYAN}{real_tag} genre stats (total {sum(real_genre_stats.values())}): {sorted(real_genre_stats.items(), key=first_in_pair)}')

    if gan_clusters_history_data is not None: # With RANDOM added other could  be None:  userid_per_cluster_gan is not None
        # For generated users
        gen_tag = 'Generated'
        print(f'Features: {list(centered_gen_feature_matrix[feature_aextr_gen])}')
        if PARAMETERS['use_random']:
            feature_index = gan_clusters_history_data[simhash_value][pcos_aextr_gen]
            if PARAMETERS['random_use_gen']:
                extr_gen_history = gan_clusters_history_data_for_random[simhash_value][pcos_aextr_gen]
            else:
                if feature_index != feature_aextr_gen:
                    print(f'{Fore.RED}{feature_index} == {feature_aextr_gen} should be true')
                extr_gen_history = userid_2_movieidrating[feature_aextr_gen + 1] # +1 cause 0-based to 1-based indexing ?

        else: # Note: currently PARAMETERS not passed to function (works with pycharm console)
            # actually target cluster id is not needed as simhash value already gives us the cluster id
            # Added a +1 on feature_aextr_gen as userid_per_cluster_gan starts at 1 but cur_gen_cluster_ids starts at 0
            target_cluster_id = binary_search_cluster_ids(userid_per_cluster_gan, feature_aextr_gen+1) # simhash value of cluster of interest
            # With following index should be 0 based as subtract start of cluster user ids
            # should already be given by pcos_aextr_gen
            index_in_target_cluster = feature_aextr_gen+1 - userid_per_cluster_gan[target_cluster_id][0]
            if not (index_in_target_cluster == pcos_aextr_gen) and (target_cluster_id == simhash_value):
                print(f'{Fore.RED}Error in computation (check something RHS should be the correct ones)')
                print(f'index: {target_cluster_id} {simhash_value} {index_in_target_cluster} {pcos_aextr_gen} {len(gan_clusters_history_data[simhash_value])}')
            extr_gen_history = gan_clusters_history_data[simhash_value][pcos_aextr_gen] # before used index_in_target_cluster that should equal pcos_aextr_gen
        gen_title_hist, gen_genres, gen_genre_stats = movieid_rating_to_title_genres(extr_gen_history, id_to_title, id_to_genres)


    else: # The case were have self similarity so twice same dataset (here real data again)
        # reuse variable name for compatibility with common movies computation below etc:
        gen_tag = 'Real'
        print(f'Features: {list(centered_real_feature_matrix[feature_aextr_gen])}')
        extr_gen_history = userid_2_movieidrating[feature_aextr_gen+1] # here the name should be real but gen for compat.
        gen_title_hist, gen_genres, gen_genre_stats = movieid_rating_to_title_genres(extr_gen_history, id_to_title, id_to_genres)

    # Note: here sorted the list so the genres do not match sorted movies would need to sort index to retrieve them
    print(f'{gen_tag} movie history ({len(gen_title_hist)}):{gen_title_hist}\nsorted: {sorted(gen_title_hist)}\n{gen_genres} not sorted as movies')
    # if want to sort by most common: gen_genre_stats.most_common(), by key sorted(gen_genre_stats.items(), key=first_in_pair)
    print(f'{Fore.CYAN}{gen_tag} genre stats (total {sum(gen_genre_stats.values())}): {sorted(gen_genre_stats.items(), key=first_in_pair)}')

    # Computation over both Real and Gen:
    # computation common movies etc as done in pipeline
    common_movies = [t_movie for t_movie in set(real_title_hist) if t_movie in set(gen_title_hist)]
    # Note: might want to find a way eg with a log and ctrl+f or in code to find max common movies in run and mean, std as done for pipeline
    print(f'{Fore.LIGHTYELLOW_EX}Common movies ({len(common_movies)}): {common_movies}')
    # As a sanity check could try to recompute features but the problem is centering wrt every other userid ?
    common_movie_ctr[f'{len(common_movies)} common movies ({real_tag}-{gen_tag})'] += 1

    # Compute common genres
    # can check if there is zero common genres in user from same cluster (for self cosine similarity)
    # Note: as keys() return a set-like data structure the `in` operator should be fast and no need to cast as set
    common_genres = [c_genre for c_genre in real_genre_stats.keys() if c_genre in gen_genre_stats.keys()]
    print(f'{Fore.LIGHTYELLOW_EX}Common genres ({len(common_genres)}): {common_genres}')
    if common_genre_ctr is not None: # Did not add counter for every call of function
        common_genre_ctr[f'{len(common_genres)} common genres ({real_tag}-{gen_tag})'] += 1

    if perform_sanity_check: # Note: Not adapted for self cosine similarity
        if extr_under_consideration == 'max':
            ref_extr = np.max(cross_cos_sim)
        elif extr_under_consideration == 'min':
            ref_extr = np.min(cross_cos_sim)
        elif extr_under_consideration == 'absmin':
            ref_extr = np.min(np.abs(cross_cos_sim))  # absolute value closest to 0

        ref_extr2 = cosine_similarity(centered_gen_feature_matrix[feature_aextr_gen], centered_real_feature_matrix[feature_aextr_real])
        retrieved_extr = cross_cos_sim[pcos_aextr_gen, pcos_aextr_real]
        if extr_under_consideration == 'absmin':
            ref_extr2 = np.abs(ref_extr2)
            retrieved_extr = np.abs(retrieved_extr)

        # check code for average_cosine_similarity in anonymity_evaluation
        # print(f'{extr_under_consideration}: {ref_extr:.5f} from arg{extr_under_consideration}: {retrieved_extr:.5f}')
        # Check if the extremum is the one retrieved index for in the pairwise cosine similarity matrix
        if not np.isclose(retrieved_extr, ref_extr):
            print(f'{Fore.RED}Error {extr_under_consideration} {ref_extr} got {retrieved_extr}')

        # Check index in feature data retrieved lead to same extremum
        if not np.isclose(ref_extr2, ref_extr):
            print(f'{Fore.RED}Error vectors {centered_gen_feature_matrix[feature_aextr_gen], centered_real_feature_matrix[feature_aextr_real]} cos sim is not {extr_under_consideration}')

    Indices = namedtuple('Indices', [f'pcos_sim_{extr_under_consideration}', f'common_movie_ctr', f'common_genre_ctr',
                                     f'pcos_a{extr_under_consideration}_gen', f'pcos_a{extr_under_consideration}_real',
                                     f'feature_a{extr_under_consideration}_gen', f'feature_a{extr_under_consideration}_real'])

    print() # print blank line for easier readability
    output = Indices(pcos_sim_extr, common_movie_ctr, common_genre_ctr, pcos_aextr_gen, pcos_aextr_real, feature_aextr_gen, feature_aextr_real)
    return output
    # pcos_sim_extr, pcos_aextr_gen, pcos_aextr_real, feature_aextr_gen, feature_aextr_real
    # pcos_sim_max, pcos_amax_gen, pcos_amax_real, feature_amax_gen, feature_amax_real
    # pcos_sim_min, pcos_amin_gen, pcos_amin_real , feature_amin_gen, feature_amin_real
    # pcos_sim_absmin, pcos_absmin_gen, pcos_absmin_real, feature_absmin_gen, feature_absmin_real

# More for pycharm structure to arrive at main function faster
def main():
    pass

if __name__ == '__main__':

    # Logging
    create_logging_levels()  # so that can pass logging.D1BG to args of init_loggers
    # fh might not be initialized so by default None ?
    fh, sh = init_loggers(log_to_stdout=True, log_to_file=False, filename='cosine_similarity_inspection',
                          sh_lvl=logging.INFO)  # INFO, DEBUG D1BG

    ## Params
    data_source = ''
    PARAMETERS = {
                  'SEED': 32, # used for random cluster assignment (32 same as anonymity evaluation)
                  'apply_centering': False,
                  'use_rating': False, # Note that for train data did not save userid to recover rating so took it as 1
                  'use_random': False, # compare with random clustering instead of SimHash
                  'random_use_gen': False,
                  'clusters_checkpoint_subfolder': 'checkpoint_61', # 'checkpoint_41', 'checkpoint_61' checkpoint_62 ,'checkpoint_-1' etc
                   }

    ## Precomputations
    folder_path = '../data/ml-25m'
    movies_file = f'{folder_path}/movies.csv'
    ratings_file = f'{folder_path}/ratings.csv'
    print(f'Read movie files')
    id_to_title, id_to_genres, id_to_year, id_list = read_movies(movies_file)

    # List all files in a directory [1] os.listdir, os.walk, glob.glob
    # [1] https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    root_path = f'./saved/cluster_assignments'

    # if not PARAMETERS['use_random']: # Random can use gen
    # Note: have some clusters (most likely generated by checkpoints 61) in /generated/ (moved to checkpoint62)
    #  the other generated later are in subfolders
    gen_cluster_folder = f'{root_path}/generated/{PARAMETERS["clusters_checkpoint_subfolder"]}/'
    # gen_cluster_folder = f'{root_path}/generated/' # if want absolute path append {os.getcwd()}
    # gan_cluster_folder = gan_cluster_folder.replace('/', os.sep) # not needed


    # filenames_gen = os.listdir(gen_cluster_folder) # [OS] returns file names also include folders
    # from [1] with this one can use pattern matching
    # filenames_gen = glob.glob(f'{gen_cluster_folder}/*.pkl') # with glob its relative path not only file names
    # had other .pkl file related to plots in checkpoint folders
    filenames_gen = glob.glob(f'{gen_cluster_folder}/LeakGAN*.pkl')
    print(f'Files to process: {filenames_gen}')

    # extract parameters number before file extension if present
    # [2] https://docs.python.org/3/library/re.html
    # extract_params = re.compile(r'(\d+)_(\d+)[.a-zA-Z]*$') # without the named group syntax
    extract_params_gen = re.compile(r'(?P<bitlength>\d+)_(?P<cluster_size>\d+)[.a-zA-Z]*$')

    # Match by simhash bitlength
    filepaths_gen_by_bitlength = {}
    for file in filenames_gen:
        matched_params = extract_params_gen.search(file)
        # bitlength, cluster_size = matched_params.group(1,2) # without the name group syntax also works either way
        bitlength, cluster_size = matched_params.group('bitlength','cluster_size')
        print(f'{file}, bitlength: {bitlength}, cluster_size: {cluster_size}')
        # filepath = f'{gen_cluster_folder}/{file}' # works with [OS] os.listdir that only gave filenames ?
        filepath = f'{file}' # works with glob which has relative path
        filepaths_gen_by_bitlength[int(bitlength)] = filepath

    # Real user StrSimHash
    real_cluster_folder = f'{root_path}/real/'
    filenames_real = os.listdir(real_cluster_folder)
    print(f'Files to process: {filenames_real}')
    extract_params_real = re.compile(r'_(?P<data_source>[a-zA-Z]+)_(?P<bitlength>\d+)[.a-zA-Z]*$')
    filepaths_real_by_bitlength = {}
    for file in filenames_real:
        matched_param = extract_params_real.search(file)
        bitlength, data_source = matched_param.group('bitlength', 'data_source')
        print(f'{file}, bitlength: {bitlength}, data_source:{data_source}')
        filepath = f'{real_cluster_folder}/{file}'
        filepaths_real_by_bitlength[int(bitlength)] = filepath

    # Load precomputed centered features according to data_source
    if data_source == 'full':
        use_rating_tag = ''
        if not PARAMETERS['use_rating']:
            use_rating_tag = '-wo_rating' # without rating

        if PARAMETERS['apply_centering']:
            save_precomputed_features_path = f'./saved/floc-whitepaper-features{use_rating_tag}.npy'
        else:
            save_precomputed_features_path = f'./saved/floc-whitepaper-features-not-centered{use_rating_tag}.npy'
        logging.info(f'Load ratings...')
        userid_2_movieidrating = read_ratings(ratings_file, need_rating=True)
        user_id_count = FULL_USER_ID_COUNT
    elif data_source == 'train':
        if PARAMETERS['apply_centering']:
            save_precomputed_features_path = './saved/floc-whitepaper-features-train.npy'
        else:
            save_precomputed_features_path = './saved/floc-whitepaper-features-train-not-centered.npy'
        # Load trainning data histories so can get movies and genres from it for real data
        logging.info(f'Load training data...')
        training_histories = load_train_test('../GAN/save_ml25m/realtrain_ml25m_5000_32.txt')
        logging.info(f'Load token (encoding) to word (movieid) mapping...')
        word_ml25m, vocab_ml25m = pickle.load(open('../GAN/save_ml25m/LeakGAN_ml25m_vocab_5000_32.pkl', mode='rb'))
        userid_2_movieidrating = reconstruct_training_user_movie_rating(training_histories, word_ml25m)
        user_id_count = 120000

    # Precompute features
    if os.path.isfile(save_precomputed_features_path):
        print(f'Loading saved train features: {save_precomputed_features_path}')
        centered_real_feature_matrix = np.load(save_precomputed_features_path)
    else:
        raise Exception(f'The file should already exists, otherwise need to run anonymity_evaluation.py first to generate it')

    # Set print option for numpy:
    np.set_printoptions(precision=5, suppress=True, linewidth=160)

    # Store for inspection key would be (bitlength -> dict with key simhash_cluster_id)
    pairwise_cos_sim_bitlength = {}
    self_cos_sim_bitlen_real = {}
    self_cos_sim_bitlen_gen = {}
    # Note: would have to check both dict have same keys ?
    #  8 had 32 common movies twice for checkpoint 61
    bitlength2check = [8] # [5,6,7,8,9,10] filepaths_gen_by_bitlength.keys() (not always init now)

    # Variable for stats per simhash bitlength
    pcosim_stats_per_bitlen = dict()
    pcosim_max_distrib_bitlen = dict()
    pcosim_max_genre_distrib_bitlen = dict()
    pcosim_amin_genre_distrib_bitlen = dict()
    # Self (pairwise) cosine similarity for real data
    scosimr_stats_per_bitlen = dict()
    scosimr_max_distrib_bitlen = dict()
    scosimr_max_genre_distrib_bitlen = dict()
    scosimr_amin_genre_distrib_bitlen = dict()
    # Self (pairwise) cosine similarity for generated data
    scosimg_stats_per_bitlen = dict()
    scosimg_max_distrib_bitlen = dict()
    scosimg_max_genre_distrib_bitlen = dict()
    scosimg_amin_genre_distrib_bitlen = dict()

    for bitlength in bitlength2check:
        print(f'bitlength under consideration: {bitlength}')
        # Random case:
        if PARAMETERS['use_random']:

            # RANDOM clustering uses real user data
            # Note: metric too high as use the same users, so try random assignment on generated users ?

            if PARAMETERS['random_use_gen']:
                # Discard the userid_per_cluster_gan as set to None after
                # Extract clusters of generated histories
                gen_filepath = filepaths_gen_by_bitlength[bitlength]
                # Load clusters containing each user movie id history
                gan_clusters_history_data_for_feature = pickle.load(open(gen_filepath, mode='rb'))
                centered_gen_feature_matrix, _ = feature_extractor_generated_movies(gan_clusters_history_data_for_feature, id_to_genres, bitlength, apply_centering=PARAMETERS['apply_centering'])
            else:
                centered_gen_feature_matrix = centered_real_feature_matrix

            cur_cluster_size = user_id_count // (1 << bitlength)
            random_cluster_indices = random_cluster_assignment(centered_gen_feature_matrix, cur_cluster_size, seed=PARAMETERS['SEED'])
            gan_clusters_history_data = {}
            for i in range(1 << bitlength):
                gan_clusters_history_data[i] = random_cluster_indices[i]
            # Cannot process random_cluster_indices as userid_per_cluster_gan becaue range not continuous ? but not really needed (except code refactoring)
            userid_per_cluster_gan = None # NOT USED for RANDOM
        else:
            # GAN case
            # Extract clusters of generated histories
            gen_filepath = filepaths_gen_by_bitlength[bitlength]
            # Load clusters containing each user movie id history
            gan_clusters_history_data = pickle.load(open(gen_filepath, mode='rb'))
            # Compute centered features, centered over all users (irrespective of cluster) as in other file:
            centered_gen_feature_matrix, userid_per_cluster_gan = feature_extractor_generated_movies(gan_clusters_history_data, id_to_genres, bitlength, apply_centering=PARAMETERS['apply_centering'])

        # Extract clusters of real histories
        real_filepath = filepaths_real_by_bitlength[bitlength]
        real_clusters_userid_data = pickle.load(open(real_filepath, mode='rb'))

        # Variable for stat per cluster (simhash value)
        pcosim_max_counter = Counter()
        pcosim_max_genre_ctr = Counter()
        pcosim_min_counter = Counter()
        pcosim_absmin_counter = Counter()
        pcosim_amin_genre_ctr = Counter()
        # Self cosine similarity for real data
        scosimr_max_counter = Counter()
        scosimr_max_genre_ctr = Counter()
        scosimr_min_counter = Counter()
        scosimr_amin_counter = Counter()
        scosimr_amin_genre_ctr = Counter()
        # Self cosine similarity for gen data
        scosimg_max_counter = Counter()
        scosimg_max_genre_ctr = Counter()
        scosimg_amin_counter = Counter()
        scosimg_amin_genre_ctr = Counter()

        # Iterate over clusters
        start_cluster_id_gen = 0
        pairwise_cos_sim_simhashids = {}
        self_cos_sim_simhash_real = {}
        self_cos_sim_simhash_gen = {}
        for simhash_value in trange(1 << bitlength, desc='iterate over clusters'):
            print(f'Cluster associated to simhash: {simhash_value}')
            # recompute cluster ids as in anonymity_evaluation.py
            cur_gen_cluster_size = len(gan_clusters_history_data[simhash_value])
            if PARAMETERS['use_random']:
                cur_gen_cluster_ids = random_cluster_indices[simhash_value]
            else:
                # Sanity check since now also have userid_per_cluster
                # index start at 1 and both indices included ie [start, end]
                cluster_id_start, cluster_id_end = userid_per_cluster_gan[simhash_value]
                card_cluster_range = cluster_id_end - cluster_id_start + 1
                if card_cluster_range != cur_gen_cluster_size:
                    print(f'Error check why range not equal {card_cluster_range} != {cur_gen_cluster_size}')
                cur_gen_cluster_ids = list(range(start_cluster_id_gen, start_cluster_id_gen + cur_gen_cluster_size))
                start_cluster_id_gen += cur_gen_cluster_size

            cur_real_cluster_ids = real_clusters_userid_data[simhash_value]
            print(f'Current cluster sizes (gen, real): {cur_gen_cluster_size, len(cur_real_cluster_ids)}')

            # compute here too the cluster centroid and average cosine similarity of users and centroid
            cur_gen_centroid = compute_cluster_centroid(cur_gen_cluster_ids, centered_gen_feature_matrix)
            cur_real_centroid = compute_cluster_centroid(cur_real_cluster_ids, centered_real_feature_matrix)
            # as an order param default L2 norm (vector) frobenius
            # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
            euclidian_dist = np.linalg.norm(cur_gen_centroid - cur_real_centroid)
            print(f'Centroid of cluster (gen | real) dist={euclidian_dist}:\n{list(cur_gen_centroid)}\n{list(cur_real_centroid)}')

            # Try
            cross_cos_sim = matrix_cosine_similarity(centered_gen_feature_matrix[cur_gen_cluster_ids, :], centered_real_feature_matrix[cur_real_cluster_ids, :])
            pairwise_cos_sim_simhashids[simhash_value] = cross_cos_sim
            # Mean seems to be close to 0 (centered data, normalized matrices)
            # mean std do not seemv ery meaningful
            # print(f'{Fore.BLUE}average pairwise cosine similarity: {cross_cos_sim.mean()}')
            # print(f'{Fore.BLUE}standard deviation pairwise cosine similarity: {cross_cos_sim.std()}')
            # print(cross_cos_sim)
            # More stats:
            # Could do boolean indexing to filter values [1] https://numpy.org/devdocs/user/basics.indexing.html#boolean-array-indexing
            # cross_cos_sim[cross_cos_sim > 0.95] # but then would want to get back index where this happenned in original array ?
            pcos_sim_gt_95 = np.argwhere(cross_cos_sim > 0.95) # should give indices where condition satisfied

            # Counters for stats here would have one element
            # defining counter here would have only one element if count common movies for max entry etc

            # As only defined under certain conditions
            if 'gan_clusters_history_data_for_feature' not in locals(): # variable could be undefined
                gan_clusters_history_data_for_feature = None

            print(f'Pairwise Cosine Similarity:')
            stats_max = recover_extremum_stats('max', cross_cos_sim, pcosim_max_counter, pcosim_max_genre_ctr, gan_clusters_history_data, userid_per_cluster_gan, cur_gen_cluster_ids, userid_2_movieidrating, cur_real_cluster_ids, simhash_value, id_to_title, id_to_genres, gan_clusters_history_data_for_random=gan_clusters_history_data_for_feature, perform_sanity_check=True)
            stats_min = recover_extremum_stats('min', cross_cos_sim, pcosim_min_counter, None, gan_clusters_history_data, userid_per_cluster_gan, cur_gen_cluster_ids, userid_2_movieidrating, cur_real_cluster_ids, simhash_value, id_to_title, id_to_genres, gan_clusters_history_data_for_random=gan_clusters_history_data_for_feature,perform_sanity_check=True)
            stats_absmin = recover_extremum_stats('absmin', cross_cos_sim, pcosim_absmin_counter, pcosim_amin_genre_ctr, gan_clusters_history_data, userid_per_cluster_gan, cur_gen_cluster_ids, userid_2_movieidrating, cur_real_cluster_ids, simhash_value, id_to_title, id_to_genres, gan_clusters_history_data_for_random=gan_clusters_history_data_for_feature, perform_sanity_check=True)
            # would have one element if compute average here
            pcosim_max_counter = stats_max.common_movie_ctr
            pcosim_max_genre_ctr = stats_max.common_genre_ctr
            pcosim_min_counter = stats_min.common_movie_ctr
            pcosim_absmin_counter = stats_absmin.common_movie_ctr
            pcosim_amin_genre_ctr = stats_absmin.common_genre_ctr

            self_cos_sim_real = matrix_cosine_similarity(centered_real_feature_matrix[cur_real_cluster_ids, :], centered_real_feature_matrix[cur_real_cluster_ids, :])
            self_cos_sim_simhash_real[simhash_value] = self_cos_sim_real
            # do not seem that meaningful across sample so not printed anymore
            # print(f'{Fore.BLUE}average self cosine similarity real: {self_cos_sim_real.mean()}')
            # print(f'{Fore.BLUE}std self cosine similarity real: {self_cos_sim_real.std()}')

            print(f'Self Cosine Similarity (real):')
            # Extremums stats for real data (extremum_real_
            stats_rmax = recover_extremum_stats('max', self_cos_sim_real, scosimr_max_counter, scosimr_max_genre_ctr,
                                                 None, None,
                                                 cur_real_cluster_ids, userid_2_movieidrating, cur_real_cluster_ids,
                                                 simhash_value, id_to_title, id_to_genres, perform_sanity_check=False)
            stats_rmin = recover_extremum_stats('min', self_cos_sim_real, scosimr_min_counter, None, None, None,
                                                 cur_real_cluster_ids, userid_2_movieidrating, cur_real_cluster_ids,
                                                 simhash_value, id_to_title, id_to_genres, perform_sanity_check=False)
            stats_ramin = recover_extremum_stats('absmin', self_cos_sim_real, scosimr_amin_counter, scosimr_amin_genre_ctr,
                                                 None, None,
                                                 cur_real_cluster_ids, userid_2_movieidrating, cur_real_cluster_ids,
                                                 simhash_value, id_to_title, id_to_genres, perform_sanity_check=False)
            # would have one element if compute average here
            scosimr_max_counter = stats_rmax.common_movie_ctr
            scosimr_max_genre_ctr = stats_rmax.common_genre_ctr
            scosimr_min_counter = stats_rmin.common_movie_ctr
            scosimr_amin_counter = stats_ramin.common_movie_ctr
            scosimr_amin_genre_ctr = stats_ramin.common_genre_ctr

            self_cos_sim_gen = matrix_cosine_similarity(centered_gen_feature_matrix[cur_gen_cluster_ids, :], centered_gen_feature_matrix[cur_gen_cluster_ids, :])
            self_cos_sim_simhash_gen[simhash_value] = self_cos_sim_gen
            # do not seem that meaningful across sample so not printed anymore
            # print(f'{Fore.BLUE}average self cosine similarity gen: {self_cos_sim_gen.mean()}')
            # print(f'{Fore.BLUE}std self cosine similarity gen: {self_cos_sim_gen.std()}')

            if PARAMETERS['use_random'] and not PARAMETERS['random_use_gen']:
                print(f'Self Cosine Similarity (random):')
                # Can do same for random only clusters:
                # But it needs the userid_2_movieidrating that is set to None for GAN
                # Also the userid_per_cluster_gan=None so it is not passed and None is passed instead for clarity
                stats_gmax = recover_extremum_stats('max', self_cos_sim_gen, scosimg_max_counter, scosimg_max_genre_ctr,
                                                    gan_clusters_history_data, None,
                                                    cur_gen_cluster_ids, userid_2_movieidrating, cur_gen_cluster_ids,
                                                    simhash_value, id_to_title, id_to_genres,
                                                    perform_sanity_check=False)

                stats_gamin = recover_extremum_stats('absmin', self_cos_sim_gen, scosimg_amin_counter,
                                                     scosimg_amin_genre_ctr,
                                                     gan_clusters_history_data, None,
                                                     cur_gen_cluster_ids, userid_2_movieidrating, cur_gen_cluster_ids,
                                                     simhash_value, id_to_title, id_to_genres,
                                                     perform_sanity_check=False)
            else:
                print(f'Self Cosine Similarity (generated):')
                # Can do same for gen only clusters:
                stats_gmax = recover_extremum_stats('max', self_cos_sim_gen, scosimg_max_counter, scosimg_max_genre_ctr,
                                                    gan_clusters_history_data, userid_per_cluster_gan,
                                                    cur_gen_cluster_ids, None, cur_gen_cluster_ids,
                                                    simhash_value, id_to_title, id_to_genres, perform_sanity_check=False,
                                                    gan_clusters_history_data_for_random=gan_clusters_history_data_for_feature)

                stats_gamin = recover_extremum_stats('absmin', self_cos_sim_gen, scosimg_amin_counter, scosimg_amin_genre_ctr,
                                                     gan_clusters_history_data, userid_per_cluster_gan,
                                                     cur_gen_cluster_ids, None, cur_gen_cluster_ids,
                                                     simhash_value, id_to_title, id_to_genres, perform_sanity_check=False,
                                                     gan_clusters_history_data_for_random=gan_clusters_history_data_for_feature)

            scosimg_max_counter = stats_gmax.common_movie_ctr
            scosimg_max_genre_ctr = stats_gmax.common_genre_ctr
            # Not done min currently as same as absmin without centering
            scosimg_amin_counter = stats_gamin.common_movie_ctr
            scosimg_amin_genre_ctr = stats_gamin.common_genre_ctr


        # common_movie_counters = [stats_max.common_movie_ctr, stats_min.common_movie_ctr, stats_absmin.common_movie_ctr]
        common_counters = [('max', 'movies', pcosim_max_counter), ('min','movies', pcosim_min_counter), ('absmin', 'movies',pcosim_absmin_counter),
                                 ('max', 'genres', pcosim_max_genre_ctr), ('absmin', 'genres', pcosim_amin_genre_ctr)]
        stats_to_save = {}
        for extr, common, counter in common_counters:
            mean = compute_average_common_movie_counts(counter)
            var, std = compute_standard_deviation(counter, mean=mean)
            stats_to_save[f'pcosim {extr} entry common {common} (mean, std)'] = (mean, std)

        common_counters = [('max','movies',scosimr_max_counter), ('min','movies',scosimr_min_counter), ('amin','movies',scosimr_amin_counter),
                                 ('max','genres',scosimr_max_genre_ctr), ('absmin','genres',scosimr_amin_genre_ctr)]
        self_rstats_to_save = {}
        for extr, common, counter in common_counters:
            mean = compute_average_common_movie_counts(counter)
            var, std = compute_standard_deviation(counter, mean=mean)
            self_rstats_to_save[f'scosim real {extr} entry common {common} (mean, std)'] = (mean, std)

        common_counters = [('max', 'movies', scosimg_max_counter), # ('min', 'movies', scosimg_min_counter),
                           ('amin', 'movies', scosimg_amin_counter),
                           ('max', 'genres', scosimg_max_genre_ctr), ('absmin', 'genres', scosimg_amin_genre_ctr)]
        self_gstats_to_save = {}
        for extr, common, counter in common_counters:
            mean = compute_average_common_movie_counts(counter)
            var, std = compute_standard_deviation(counter, mean=mean)
            self_gstats_to_save[f'scosim gen {extr} entry common {common} (mean, std)'] = (mean, std)


        # Save the stats of the common movie counts
        # pcosim_stats_per_bitlen[bitlength] = pcosim_stats_per_simhash
        pcosim_stats_per_bitlen[bitlength] = stats_to_save
        scosimr_stats_per_bitlen[bitlength] = self_rstats_to_save
        scosimg_stats_per_bitlen[bitlength] = self_gstats_to_save
        # Save the distribution of common movie counts:
        pcosim_max_distrib_bitlen[bitlength] = pcosim_max_counter
        pcosim_max_genre_distrib_bitlen[bitlength] = pcosim_max_genre_ctr
        pcosim_amin_genre_distrib_bitlen[bitlength] = pcosim_amin_genre_ctr
        # Save for self cosine similarity real
        # Note: should only contains 32 common movies so want the min actually ?
        # interestingly does not only contains 32 maybe because the max length of history was not 32
        # ie should have the full diagonal with 1s
        scosimr_max_distrib_bitlen[bitlength] = scosimr_max_counter
        scosimr_max_genre_distrib_bitlen[bitlength] = scosimr_max_genre_ctr
        scosimr_amin_genre_distrib_bitlen[bitlength] = scosimr_amin_genre_ctr
        # Note could also add distribution for min and abs min but without centering seems to be mostly zeros
        # Save for self cosine similarity Generated
        scosimg_max_distrib_bitlen[bitlength] = scosimg_max_counter
        scosimg_max_genre_distrib_bitlen[bitlength] = scosimg_max_genre_ctr
        scosimg_amin_genre_distrib_bitlen[bitlength] = scosimg_amin_genre_ctr
        # Save the cosine similarity matrices
        pairwise_cos_sim_bitlength[bitlength] = pairwise_cos_sim_simhashids
        self_cos_sim_bitlen_real[bitlength] = self_cos_sim_simhash_real
        self_cos_sim_bitlen_gen[bitlength] = self_cos_sim_simhash_gen

    logging.info(f'Parameters: {pretty_format(PARAMETERS)}')

    logging.info(f'{Fore.BLUE}Pairwise cosine similarity:')
    logging.info(f'stats on pcosim:{pretty_format(pcosim_stats_per_bitlen, use_json=True)}')
    logging.info(f'pcosim max entry (common movies) distribution per bitlength:{pretty_format(pcosim_max_distrib_bitlen, use_json=True)}')
    logging.info(f'pcosim absmin entry common genre distribution per bitlength:{pretty_format(pcosim_amin_genre_distrib_bitlen, use_json=True)}')

    # Self cosine similarity for real data (max should be 1 as it is each entry in diagonal of matrix so not interesting metric)
    logging.info(f'{Fore.BLUE}Self cosine similarity (real):')
    logging.info(f'stats on scosimr:{pretty_format(scosimr_stats_per_bitlen, use_json=True)}')
    # This one not very useful should be max length by default (except when history smaller than 32)
    # logging.info(f'scosimr max entry common movie distribution per bitlength:{pretty_format(scosimr_max_distrib_bitlen, use_json=True)}')
    logging.info(f'scosimr absmin entry common genre distribution per bitlength:{pretty_format(scosimr_amin_genre_distrib_bitlen, use_json=True)}')

    # Self cosine similarity for generated data
    logging.info(f'{Fore.BLUE}Self cosine similarity (generated):')
    logging.info(f'stats on scosimg:{pretty_format(scosimg_stats_per_bitlen, use_json=True)}')
    logging.info(f'scosimg absmin entry common genre distribution per bitlength:{pretty_format(scosimg_amin_genre_distrib_bitlen, use_json=True)}')

    # Note could save files


