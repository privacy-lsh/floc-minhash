import pickle
import random
import numpy as np
import csv # The movie titles contains , so .split(',') does not work as is
# import pandas as pd
from FLoC.preprocessing.vocab_corpus_builder import Dictionary
from FLoC.chromium_components import cityhash
from tqdm import trange

user_id_count = 162541
movie_id_count = 62423

# Analysis and exploration with pandas:
# [1] https://www.kaggle.com/jneupane12/analysis-of-movielens-dataset-beginner-sanalysis
# [2] https://analyticsindiamag.com/how-to-build-your-first-recommender-system-using-python-movielens-dataset/
# For [1] the plots in the end are nice


# custom file reading
def read_ratings(file_path, need_rating=False):
    """
    Reads the ratings file from the MovieLens dataset
    :param file_path: path to the ratings file
    :param need_rating: if wants to also store the ratings
    :return: a dict or a list of list depending if saved the ratings or not
    """
    # file reading basics [1] https://www.pythontutorial.net/python-basics/python-read-text-file/

    # want a dict ? with index being userId (a list of list could work if first index is userId?)
    if need_rating:
        # Init each key (user id) to an empty list (not the same (mutable) for each value)
        userid_to_ratings = {k: [] for k in range(1, user_id_count+1)} # [1, user_id_count+1[
    else:
        # First method not modified for legacy reason
        # Note that [ [] ] * N, result in the list containing the same list object N times (so get side effects when append to list)
        # every user created should have an history if we are to believe there are 162451 users and user id range is [1,162451]
        userId_movieId_history = [[] for _ in range(user_id_count)]

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # do not care about first line which is `userId,movieId,rating,timestamp`
        for line in lines[1:]:
            # could put , _ instead of , rating cause discard those values rating and timestamp for now
            splitted = line.split(',')
            userId, movieId = int(splitted[0]), int(splitted[1])
            # userId,movieId = line.split(',', maxsplit=1) # stops at first , and return first and remaining of string
            if need_rating:
                rating, timestamp = float(splitted[2]), splitted[3]
                # Append the tuple to the history for this user_id
                userid_to_ratings[userId].append((movieId, rating))

            else:
                # Note the -1 cause list indexing start at 0
                userId_movieId_history[userId-1].append(movieId) # Note movieId range start from 1

    # can serialize object (e.g., using pickle_ or maybe once extracted features
    if need_rating:
        return userid_to_ratings
    else:
        return userId_movieId_history


def read_movies(file_path):
    """
    Read the movies file from MovieLens dataset
    :param file_path: path to the movies file
    :return: 4-tuple: mapping from movieID to title, movieID to genres, movieID to year and a list of movieIDs
    """
    movieId_list = []
    movieTitle_from_Id = {}
    genres_from_Id = {}
    year_from_Id = {} # could be extracted from title as they are defined

    # if encoding not specified might get errors like UnicodeDecodeError: 'charmap' codec can't decode byte
    with open(file_path, 'r', encoding='utf-8') as f:
        # old method did not take into account , delimiter is ignored in ""
        # lines = f.readlines()
        line_count = 0
        # [1] https://realpython.com/python-csv/ # DictReader(f) also exists
        csv_reader = csv.reader(f, delimiter=',') # also have quotechar, escapechar see [1] above
        for row in csv_reader:
        # for line in lines[1:]:
            # do not care about first line which is `movieId,title,genres`
            if line_count == 0:
                # print(f'Column names of csv file are {", ".join(row)}')
                pass
            # splitted = line.split(',') # if want all 3 (could also split subgenre substring by |)
            # movieId, title = int(splitted[0]), splitted[1]
            else:
                movieId, title = int(row[0]), row[1]
                # some movies have their date left unspecified even if they have one
                # title == "Millions Game, The (Das Millionenspiel)" date is missing in movies.csv but wiki says 1970
                start_incl, end_excl = -5, -1
                if title[-1] == ' ': # some have spaces or other as last character # or title[-1] == ')'
                    start_incl, end_excl = -6, -2
                try:
                    release_year = int(title[start_incl:end_excl])
                except ValueError as ve:
                    release_year = 3000 # to make them last in the range
                    # print(f'{title}')
                    # print(f'{title} {title[-6:-2], title[-5:-1]} {start_incl, end_excl}')
                    # print(ve)
                # need genres for preprocessing sorting
                # genres = splitted[2].rstrip() # remove trailing \n
                genres = row[2].rstrip() # remove trailing \n
                genres_splitted = genres.split('|')
                genres = set() # note that reassign already created genres var
                for elem in genres_splitted:
                    genres.add(elem)
                # movieId index starts from 1 (62423 movies but discontinuous ids)
                movieId_list.append(movieId)
                movieTitle_from_Id[movieId] = title
                genres_from_Id[movieId] = genres
                year_from_Id[movieId] = release_year
            line_count += 1
    # Mapping from movieId to title
    return movieTitle_from_Id, genres_from_Id, year_from_Id, movieId_list


##############################################################################
## Precomputations for pre-image attack on a subset of the movie histories  ##
##############################################################################

# Only precomputes it for movie in LeakGAN vocabulary
def precompute_cityhash_movielens(movies_path=f'../data/ml-25m/movies.csv',vocab_filepath=f'../GAN/save_ml25m/LeakGAN_ml25m_vocab_5000_32.pkl'):
    """
    Precompute the CityHash for the movieIDs
    :param movies_path: path to the movies file
    :param vocab_filepath: path to vocabulary file for LeakGAN
    :return: 4-tuple, mapping from movieID to CityHash, for movieID to title, for title to movieID and the movie_id_list
    """
    # Read movies and extract id to title mapping with id_list
    id_to_title, _, _, id_list = read_movies(movies_path) # id_to_genres, id_to_year
    # Load vocab to know which movie id are kept
    word_ml25m, vocab_ml25m = pickle.load(open(vocab_filepath, mode='rb'))
    # Precomputed hash lookup for vocab size:
    hash_for_movie = dict()
    title_for_movie = dict()
    movie_for_title = dict()
    movie_id_list = []
    for movie_id in id_list:
        if movie_id in vocab_ml25m:
            movie_id_list.append(movie_id)
            title_for_movie[movie_id] = id_to_title[movie_id]
            movie_for_title[title_for_movie[movie_id]] = movie_id
            hash = cityhash.hash64(title_for_movie[movie_id])
            hash_for_movie[movie_id] = hash
    return hash_for_movie, title_for_movie, movie_for_title, movie_id_list

########################################################################
## Feature extraction from FLoC whitepaper for k-anonymity evaluation ##
########################################################################

# Code that test function results is in this file's main() but was only copied here
# ratings_file, movies_file
def feature_extraction_floc_whitepaper(userid_2_movieidrating, id_to_genres, use_rating=True, apply_centering=True):
    """
    Perform the feature extraction as explained in Google FLoC's whitepaper
    :param userid_2_movieidrating: contains a mapping from userID to movieIDs and ratings for real users
    :param id_to_genres: mapping from movieIDs to genres
    :param use_rating: if use the ratings for the feature extraction
    :param apply_centering: if apply centering to the extracted feature matrix
    :return: the feature matrix
    """
    feature_matrix = [] # will contains weight vector (aggregated avg) of each users
    # more general if use dict.items or len(dict), also wants to ensure order
    # for user_id in range(1, user_id_count+1):
    for user_id in range(1, len(userid_2_movieidrating)+1):
        aggregated_vector = [0] * 20
        movieids_ratings = userid_2_movieidrating[user_id]
        count = 0 # for average later

        # Extract movie features of user history and aggregate them into one by taking the average
        for movie_id, rating in movieids_ratings:
            weight_vector = [0] * 20
            # returns a set cause tried subset comparison at some point for sorting wrt genres etc
            genre_set = id_to_genres[movie_id]
            if not use_rating:
                rating = 1

            for genre in genre_set:
                # Set the weight (1 if genre mentioned 0 otherwise) for current vectors times rating
                # genres_popularity gives the indexing of which genre has what index (-1 cause want from 0 to 19)
                weight_vector[genres_popularity[genre] - 1] = 1 * rating

            # Aggregate each movie feature of same user
            # As use python list and not numpy
            # [1] https://stackoverflow.com/questions/18713321/element-wise-addition-of-2-lists
            aggregated_vector = list(map(lambda a, b: a + b, aggregated_vector, weight_vector))
            count += 1

        # Division by 0 could happen (with training data userid 21053 and 43986)
        if count == 0:
            print(f'division by 0 userid {user_id}')
            # Put the default zero vector as no movie is present also no genre is present
            average_weight_vector = aggregated_vector
        else:
            average_weight_vector = [e / count for e in aggregated_vector]

        feature_matrix.append(average_weight_vector)

    if apply_centering:
        features = np.array(feature_matrix)
        # substract mean from should be the genre (feature) so axis 0 which subtracts the mean per columns
        # other possibility is axis 1 the row where take mean of each vector but would be done differently for each users
        features_mean = features.mean(axis=0)
        centered_features = features - features_mean

        return centered_features # ndarray

    return np.array(feature_matrix) # for compatibility create ndarray from list


def feature_extractor_generated_movies(gen_cluster_histories, id_to_genres, hash_bitlength, apply_centering=True):
    """
    Similar to `feature_extraction_floc_whitepaper` but applied to our generated users and not real users
    :param gen_cluster_histories: contains the movie histories of generated users
    :param id_to_genres: mapping from movieIDs to genres
    :param hash_bitlength: SimHash output length
    :param apply_centering: if apply centering to the feature matrix
    :return: the feature matrix and the generated userIDs in each cluster
    """
    gen_feature_matrix = []
    # Build a reverse index like a list with created userid in each cluster
    userid_per_cluster = {} # could use dict or nested list and contain all users or [min, max]
    cur_user_id = 1 # could change start of user id but it is also the one used in ml25m data
    for target_simhash_value in trange(1 << hash_bitlength, desc='GAN StrSimHash Feature Extraction'):
        cur_user_history_cluster = gen_cluster_histories[target_simhash_value]
        starting_user_id = cur_user_id
        for cur_user_history in cur_user_history_cluster:
            aggregated_vector = [0] * 20
            count = 0  # for average later
            # Extract movie features of user history and aggregate them into one by taking the average
            for movieid in cur_user_history:
                weight_vector = [0] * 20
                genre_set = id_to_genres[movieid]
                rating = 1  # Note: we do not generate ratings with GAN
                for genre in genre_set:
                    # Set the weight (1 if genre mentioned 0 otherwise) for current vectors times rating
                    # genres_popularity gives the indexing of which genre has what index (-1 cause want from 0 to 19)
                    weight_vector[genres_popularity[genre] - 1] = 1 * rating

                # aggregate each movie feature of one user
                aggregated_vector = list(map(lambda a, b: a + b, aggregated_vector, weight_vector))
                count += 1

            # Take the average of all movie feature vector of one user
            average_weight_vector = [e / count for e in aggregated_vector]
            gen_feature_matrix.append(average_weight_vector)
            # Update user id in cluster
            cur_user_id += 1

        # Update userid_per_cluster
        userid_per_cluster[target_simhash_value] = (starting_user_id, cur_user_id-1)

    if apply_centering:
        # does creating the numpy array should not change the order of element
        features = np.array(gen_feature_matrix)
        # axis want to substract mean from should be the genre (feature) so axis 0 which takes the 0 mean per columns ?
        # other possibility is axis 1 the row where take mean of each vector but would not make sense here ?
        features_mean = features.mean(axis=0)
        centered_features = features - features_mean

        return centered_features, userid_per_cluster # ndarray, dict of pair

    return np.array(gen_feature_matrix), userid_per_cluster # ndarray from list, dict of pair



#######################################################################################
## (Legacy) First kind of preprocessing of the data to use for training with LeakGAN ##
#######################################################################################


# would want to sort movies by genres first then year so that get more meaningful index range for encoding ?
# also which genres prioritize when multiple ? the least popular one ?
def sort_movieId_by_genres_then_year(id_list, id_to_genres, id_to_year):
    """
    Sorts the movieIDs first by genres and then by year if the genres were equal
    :param id_list: list of movieIDs
    :param id_to_genres: mapping from movieIDs to genres
    :param id_to_year: mapping from movieIDs to year
    :return: the sorted list of movieIDs
    """
    # A tuple comparison operator should first compare first element and on equal the second element
    # First made used of set comparison maybe not the best ? it uses subset of as <= operator
    #
    # Python might use Timsort for sorting
    # sorted_id_list = sorted(id_list, key=lambda movieId: (id_to_genres[movieId], id_to_year[movieId]))
    sorted_id_list = sorted(id_list, key=lambda movieId: (encode_genre(id_to_genres[movieId]), id_to_year[movieId]))
    return sorted_id_list

# Check https://www.kaggle.com/jneupane12/analysis-of-movielens-dataset-beginner-sanalysis not run for data downloaded
# on [1] they list 19 genres the same without IMAX and Children does not have the 's
# [1] https://files.grouplens.org/datasets/movielens/ml-25m-README.html
genres_popularity = {'Drama':1, 'Comedy':2, 'Thriller':3, 'Romance':4, 'Action':5, 'Crime':6,'Horror':7,'Documentary':8,
                     'Adventure':9,'Sci-Fi':10,'Mystery':11,'Fantasy':12,'War':13,'Children':14,'Musical':15,
                     'Animation':16, 'Western':17,'Film-Noir':18,'(no genres listed)':19,'IMAX':20}

def encode_genre(genres):
    """
    Encode the genres as a bit vector (1 if the genre is present 0 otherwise)
    :param genres: the genre set to encode
    :return: the encoded bit vector (represented as an integer)
    """
    # seems there is no bitset implementation in standard python lib but do something similar here
    # basically encode the set as an integer (binary) where 1 means the genre is present
    genre_bit_encoded = 0
    for genre_name in genres:
        # We want the least popular genres at the LSB so 0 ?
        pos_to_set = 20 - genres_popularity[genre_name]
        genre_bit_encoded = genre_bit_encoded | (1 << pos_to_set)
    return genre_bit_encoded

def filter_movies_history(movie_histories, movie_id_list, max_hist_len = 77, movie_count_thresh = 20):
    """
    Filter the movie history to reduce the computational ressources necessary to train LeakGAN on it
    :param movie_histories: The movie histories whish to filter
    :param movie_id_list: list of movieIDs
    :param max_hist_len: the maximum length a postprocessed movie history should have
    :param movie_count_thresh: the minimum number of times a movie should appear in the histories to be considered relevant
    :return: the movieID and userIDs kept after filtering along with the number of history kept
    """
    # Done in caller :
    # movie_histories = read_rating(ratings_filepath)
    # _, _, _, movie_id_list = read_movies(movies_filepath)
    index_from_movie_id = dict()
    count = 0
    for elem in movie_id_list:
        # list traversed from start to end so no need to create inverse mapping (it already is the list)
        index_from_movie_id[elem] = count
        count += 1

    filtered_usr_hist_count = 0
    movie_count_in_filtered_hist = [0] * movie_id_count
    user_id_kept = dict()
    for user_id_0, history in enumerate(movie_histories):
        if len(history) < max_hist_len:
            user_id_kept[user_id_0] = True # user_id_0 means indexing is zero based so add 1 for user id in dataset
            filtered_usr_hist_count += 1
            for movie_id in history:
                movie_count_in_filtered_hist[index_from_movie_id[movie_id]] += 1

    print(f'Number of history retained {filtered_usr_hist_count}')
    # Vocab size change if remove movies with lower than threshold appearance in histories of interest
    # from data exploration in main decided to keep movies that appeared more than 20 times
    movie_id_kept = dict()
    for i in range(movie_id_count):
        if movie_count_in_filtered_hist[i] > movie_count_thresh:
            # movie_id_list should reverse index_from_movie_id
            movie_id_kept[movie_id_list[i]] = True
        else:
            movie_id_kept[movie_id_list[i]] = False

    return movie_id_kept, user_id_kept, filtered_usr_hist_count


def generate_vocab(movies_filepath, ratings_filepath, apply_filtering=True, save_path=f'../data/ml-25m/vocab_ml25m.pkl'):
    """
    Generate a vocabulary for LeakGAN
    :param movies_filepath: path to the movies file from MovieLens dataset
    :param ratings_filepath: path to the ratings file from MovieLens dataset
    :param apply_filtering: if apply filtering to the histories
    :param save_path: path where save the generated vocabulary
    :return: the two mapping defining the vocabulary, word to token, token to work
    """
    movieId_to_title_mapping, id_to_genres, id_to_year, movieId_list = read_movies(movies_filepath)
    sorted_movieId = sort_movieId_by_genres_then_year(movieId_list, id_to_genres, id_to_year)

    if apply_filtering:
        movie_histories = read_ratings(ratings_filepath)
        keep_movie_id, _, _ = filter_movies_history(movie_histories, movieId_list)

    vocab = dict()
    word = dict()
    # to build vocab first want the value for padding with EMPTY when no movie in history ?
    index = 0
    vocab[0] = 0 # movieId start at 1 so if get 0 can translate it to [EMPTY] or blank space ?
    word[0] = 0
    index += 1
    for movieId in sorted_movieId:
        # apply filtering here or when sort movieId ?
        if apply_filtering:
            if keep_movie_id[movieId]:
                # note that in coco caption movieID was actually a word (string)
                word[index] = movieId
                vocab[movieId] = index
                index += 1
        else:
            word[index] = movieId
            vocab[movieId] = index
            index += 1
    # save pickle vocab for LeakGAN code
    with open(save_path, 'wb') as f:
        # if not compatible with GAN maybe change protocol (default None)
        # pickle.dump((word, vocab), f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump((word, vocab), f) # protocol 5 not supported with python 3.6 from LeakGAN
    # Returns index to word (movieId) mapping, word (movieId) to index mapping
    return word, vocab

def generate_train_test_data(movies_filepath, ratings_filepath, seed=1337, val_size=5000, apply_filtering=True,
                             max_hist_len = 77, save_folder=f'../data/ml-25m',
                             vocab_file=f'../data/ml-25m/vocab_ml-25m.pkl'):
    """
    Generating training and test data for LeakGAN
    :param movies_filepath: path to the movies file
    :param ratings_filepath: path to the ratings file
    :param seed: seed used for generation of the train test split
    :param val_size: size of the test data
    :param apply_filtering: if filters movie histories to reduce computational ressources needed to train LeakGAN
    :param max_hist_len: maximum length of a postprocessed history
    :param save_folder: where save the train and test data files
    :param vocab_file: path to the vocabulary file
    :return: void (writes the train/test files to the specified destination)
    """

    movieId_to_title_mapping, id_to_genres, id_to_year, movieId_list = read_movies(movies_filepath)
    movie_histories = read_ratings(ratings_filepath)

    if apply_filtering:
        keep_movie_id, keep_user_id, usr_hist_count = filter_movies_history(movie_histories, movieId_list)
        filtered_data = [[0] * max_hist_len for _ in range(usr_hist_count)]

    filtered_count = 0
    for user_id_0, history in enumerate(movie_histories):
        if apply_filtering:
            if keep_user_id.get(user_id_0, False): # if key is not there return False (different from keep_movie_id)
                for i, movie_id in enumerate(movie_histories[user_id_0]):
                    if keep_movie_id[movie_id]:
                        # filtered_data already init with zeros for absence of movie
                        filtered_data[filtered_count][i] = movie_id
                filtered_count += 1
        else:
            # for now always filter
            pass

    # Generate train test split
    # indices_list = list(range(usr_hist_count))
    print(f'{filtered_data[:5]}')
    # this should shuffle the order of sublist (can use list comprehension to shuffle sublist but not needed)
    random.Random(seed).shuffle(filtered_data)
    print(f'after shuffle: \n{filtered_data[:5]}')
    train_histories = filtered_data[:-val_size]
    test_histories = filtered_data[-val_size:]

    # Load vocab
    with open(vocab_file, 'rb') as f:
        word, vocab = pickle.load(f)

    # Write training data
    with open(f'{save_folder}/realtrain_ml25m.txt', 'w') as fout:
        for history in train_histories:
            # remove left and right blank space and add newline
            to_write = ' '.join([str(vocab[movie_id]) for movie_id in history]).strip() + '\n'
            fout.write(to_write)

    # Write test data
    with open(f'{save_folder}/realtest_ml25m.txt', 'w') as fout:
        for history in test_histories:
            to_write = ' '.join([str(vocab[movie_id]) for movie_id in history]).strip() + '\n'
            fout.write(to_write)



if __name__ == '__main__':
    folder_path = '../data/ml-25m'
    movies_file = f'{folder_path}/movies.csv'
    ratings_file = f'{folder_path}/ratings.csv'
    #extract_data('../data/ml-25m')

    explore_data = True
    vocab_pruning = False
    generate_word_vocab_mapping = False
    generate_train_test_split_data = False
    test_encode_genre = False
    test_sorting_movie_id = False
    test_filter_movie = False
    test_floc_feature_extrac = False # FLOC whitepaper method

    print(f'Load ratings...')
    movie_history_by_user = read_ratings(ratings_file)
    print(f'Load movies...')
    movieId_to_title_mapping, id_to_genres, id_to_year, movieId_list = read_movies(movies_file)

    if generate_word_vocab_mapping:
        word, vocab = generate_vocab(movies_file, ratings_file)

    if generate_train_test_split_data:
        generate_train_test_data(movies_file, ratings_file)


    if vocab_pruning:
        vocab_dict = Dictionary()
        # iterate over all histories
        for history in movie_history_by_user:
            for movie_id in history:
                # it already generate an index for word with no meaning (ie sorting of indices)
                vocab_dict.add_word(movie_id)
        sorted_word_counts = sorted(vocab_dict.wordcounts.items(), key=lambda x: (x[1], x[0]), reverse=True)
        vocab_dict.prune_vocab(k=11_000, cnt=False)

    # exploratory analysis
    if explore_data:
        index_from_movieId = dict()
        count = 0
        for elem in movieId_list:
            index_from_movieId[elem] = count
            count += 1

        target_max_len = 77
        target_usr_hist_count = 0
        movie_count_in_target_hist = [0] * movie_id_count

        # min history length of movie ratings is 20 ?
        max_hist_len_stat = 5000
        history_length_stat = [0] * max_hist_len_stat # one user has 32202 ratings closest below is at 9178 maybe
        for user_history in movie_history_by_user:
            # to find the length above the max
            # try:
            #     history_length_stat[len(user_history)] += 1
            # except IndexError as ie:
            #     print(len(user_history))
            if len(user_history) < max_hist_len_stat:
                history_length_stat[len(user_history)] += 1
            if len(user_history) < target_max_len:
                # with that could filter movie that are not present from vocab
                # and also decide if want to remove movie that are not present much from histories
                # or the entire history that contains them (not needed as we do not care about order and whole in histories)
                target_usr_hist_count += 1
                for movieId in user_history:
                    movie_count_in_target_hist[index_from_movieId[movieId]] += 1

        print(history_length_stat)
        # Compute average history length:
        # take target_max_len and discard value above
        total = sum(history_length_stat[i]*i for i in range(target_max_len))
        denom = sum(history_length_stat[:target_max_len])
        hist_len_average = total/denom
        print(f'History length average: {hist_len_average} = {total}/{denom} for {target_max_len}={len(history_length_stat[:target_max_len])}')
        # print(movie_count_in_target_hist)
        print(f'Number of history retained {target_usr_hist_count}')
        # Vocab size change if remove movies with lower than threshold appearance in histories of interest
        for threshold in range(0, 50, 5):
            # Sanity check appearance_count >= 0 gives movie count of 62423,
            # in Leakgan paper for long sentences had vocab size of 5,742 words
            vocab_size = sum(appearance_count > threshold for appearance_count in movie_count_in_target_hist)
            print(f'Vocab size for threshold (>) {threshold}: {vocab_size}')


        # Maybe want to filter users that have to little history as this is harder to find simhash collision with ?
        # actually if add simhash condition in discriminator it could learn to generate longer history ?
        cumulative_count_per_hist_len = [0] * (max_hist_len_stat + 1)
        for index, elem in enumerate(history_length_stat):
            cumulative_count_per_hist_len[index + 1] = cumulative_count_per_hist_len[index] + elem
        # With that can filter to see where approximately have enough samples with a not so high length eg 75
        # print(cumulative_count_per_hist_len[1:])


    # Test sorting movie_id for preprocessing
    if test_sorting_movie_id:
        sorted_movieId = sort_movieId_by_genres_then_year(movieId_list, id_to_genres, id_to_year)
        for i in range(movie_id_count):
            movie_id = sorted_movieId[i]
            print(f'{movie_id}: {id_to_genres[movie_id]} {encode_genre(id_to_genres[movie_id]):>020b} {id_to_year[movie_id]}')

    # Test encode_genre
    if test_encode_genre:
        for i in range(20):
            movie_id = movieId_list[i]
            genre_encoding = encode_genre(id_to_genres[movie_id])
            print(f'{movie_id}, {bin(genre_encoding)}, {id_to_genres[movie_id]}')

    if test_filter_movie:
        # keep_movie_id = filter_movies_history(movies_file, ratings_file)
        keep_movie_id = filter_movies_history(movie_history_by_user, movieId_list)

    if test_floc_feature_extrac:
        first_test = False
        print(f'load ratings...')
        userid_2_movieidrating = read_ratings(ratings_file, need_rating=True)

        if first_test:
            # read_movies already called above and only care about id_to_genres ?
            weight_matrix = [] # Not needed as technically we would aggregate with the average
            aggregated_vector = [0] * 20
            selected_id = 88 # 88 has 21 movies
            movieids_ratings = userid_2_movieidrating[selected_id]
            count = 0
            for movie_id, rating in movieids_ratings:
                weight_vector = [0] * 20
                # returns a set cause tried subset comparison at some point for sorting wrt genres etc
                genre_set = id_to_genres[movie_id]
                for genre in genre_set:
                    # Set the weight (1 if genre mentioned 0 otherwise) for current vectors times rating
                    # genres_popularity gives the indexing of which genre has what index (-1 cause want from 0 to 19)
                    weight_vector[genres_popularity[genre]-1] = 1 * rating

                # As use python list and not numpy
                # [1] https://stackoverflow.com/questions/18713321/element-wise-addition-of-2-lists
                aggregated_vector = list(map(lambda a,b: a+b, aggregated_vector, weight_vector))
                count += 1
                weight_matrix.append(weight_vector)

            if aggregated_vector[0] != 25:
                print(f'change something average not the same')
            average_weight_vector = [e / count for e in aggregated_vector]
            print(f'{average_weight_vector}')
        else:
            # feature_matrix = feature_extraction_floc_whitepaper(ratings_file, movies_file, apply_centering=False)
            feature_matrix = feature_extraction_floc_whitepaper(userid_2_movieidrating, id_to_genres, apply_centering=False)
            # need centering to ensure dataset (ie feature matrix) has zero mean ?
            # Could use sklearn or numpy in this case
            features = np.array(feature_matrix)
            # which axis want to substract mean from i would say the genre so axis 0 which mean the 0 mean per columns ?
            # other possibility is axis 1 the row where take mean of each vector but would be done differently for each users ?
            features_mean = features.mean(axis=0)
            features_centered = features - features_mean
            print(features)
            print(features_centered)
