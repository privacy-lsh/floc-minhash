from FLoC.chromium_components import sim_hash
import random
import operator
from pip._vendor.colorama import Fore, init # colorama always in pip._vendor.colorama
init(autoreset=True) # to avoid reseting color everytime
import multiprocessing
from FLoC.preprocessing.tranco_preprocessor import extract_top_n, precompute_cityhash
from FLoC.preprocessing.movielens_extractor import precompute_cityhash_movielens
import pickle
from FLoC.attack.generating_histories import generate_from_traintest
from codetiming import Timer
# import utils
# import logging

def find_preimage(target_sim_hash: int, prefix_bit_length, reverse_hash=False, dataset='movies', min_threshold=3,
                  domain_sample_size=None):
    """
    Want to find preimage for simhash with equal prefix of a given length
    :param target_sim_hash: the target SimHash
    :param prefix_bit_length: the output bit length of SimHash
    :param reverse_hash: if sample the random hash so that know the input
    :param dataset: if reverse_hash=True one of 'domains', 'movies'
    :param min_threshold: positive minimum threshold to use for sampling outliers
    :param domain_sample_size: reduce the number of domain can sample from (works for 'domains' dataset)
    :return: the preimage, its hash list and domain list
    """
    if target_sim_hash.bit_length() > 64:
        return ValueError(f'sim_hash too big {target_sim_hash.bit_length()}')

    prefix_bits = bin(target_sim_hash)[2:2+prefix_bit_length]
    # we go through the bits in reverse order, we should take the prefix from the end
    prefix_bits = prefix_bits[::-1]
    print(f'prefix {prefix_bit_length} bits: {prefix_bits}')

    # Ensure that if SimHash (integer value) has leading 0 bits, they will not be ignored
    if len(prefix_bits) < prefix_bit_length:
        print(f'target simhash bit length {prefix_bits} smaller than prefix bit length {prefix_bit_length} so add leading zeroes')
        while len(prefix_bits) < prefix_bit_length:
            prefix_bits += '0'
        # print(f'extended len: {len(prefix_bits)} matches simhash prefix bit length: {prefix_bit_length}')

    # for nested list similar init if not careful could have side effects
    # i.e. if modify an element other elements also modified
    current_gaussian_sums = [0] * prefix_bit_length
    domain_hash_list = []
    domain_list = [] # If available the domain associated with the hashes

    # Attempt to find input domain hash that match output sim_hash (bit by bit)
    for cur_bit_id in range(prefix_bit_length):
        target_bit = int(prefix_bits[cur_bit_id])
        print(f'{cur_bit_id}-th target bit: {target_bit}')
        # Need to check if the current target bit is satisfied with the current domain hash list
        # and also add the corresponding gaussian sums to our `current_gaussian_sums` accumulator
        if cur_bit_id > 0: # we do not do this if we have not added any domain
            for domain_hash in domain_hash_list:
                rand_sample = sim_hash.random_gaussian(cur_bit_id, domain_hash)
                current_gaussian_sums[cur_bit_id] += rand_sample
            if target_bit == 1:
                if not (current_gaussian_sums[cur_bit_id] > 0):
                    prefix_bits_mismatch = True # with current domain list target bit is not 1
                else:
                    prefix_bits_mismatch = False
            else:
                if current_gaussian_sums[cur_bit_id] > 0:
                    prefix_bits_mismatch = True
                else:
                    prefix_bits_mismatch = False
        else: # if cur_bit_id == 0 we trivially need a domain for the target bit
            prefix_bits_mismatch = True

        while prefix_bits_mismatch:
            # print(f'Bit mismatch attempt to fix with a new domain')
            # Note (less readable) instead of `target_bit == 1` could put `target_bit`
            # as int 0 == False, 1 == True

            if abs(current_gaussian_sums[cur_bit_id]) > 4:
                # finding value greater than 4.5 seems slow (usually timeout)
                # when this print appears again (same gaussian sum) it means found a value greater:
                # but constraints on previous bits were not satisfied
                print(f'Current gaussian sum ({current_gaussian_sums[cur_bit_id]}) greater than threshold {4}')
            # use dynamic threshold
            var_pos_thresh = max(min_threshold, abs(current_gaussian_sums[cur_bit_id]))
            # could speed up retries after timeout if terminate when judge it would be too long to find appropriate
            # gaussian sample extremum

            cur_hash, cur_sample, cur_domain_name = \
                find_high_value_gaussian_sample(cur_bit_id, target_bit == 1, hash_reversibility=reverse_hash,
                                                dataset=dataset, positive_thresh=var_pos_thresh,
                                                domain_sample_size=domain_sample_size)
            # Check that for the current bit the target is obtained with the current cumulative gaussian sum
            if target_bit == 1:
                if not (current_gaussian_sums[cur_bit_id] + cur_sample > 0):
                    prefix_bits_mismatch = True
                    continue # if this is not the case continue
            else:
                if current_gaussian_sums[cur_bit_id] + cur_sample > 0:
                    prefix_bits_mismatch = True
                    continue

            # Check that for all previous bit this new hash does not change the sign of the sum of gaussians
            temp_gaussian_samples = []
            for previous_bit_id in range(cur_bit_id):
                # print('check if prefix bits still matched with new hash')
                target_prev_bit = int(prefix_bits[previous_bit_id])
                new_sample_to_add = sim_hash.random_gaussian(previous_bit_id, cur_hash)
                if target_prev_bit == 1:
                    if current_gaussian_sums[previous_bit_id] + new_sample_to_add > 0:
                        temp_gaussian_samples.append(new_sample_to_add)
                    else:
                        # one of the previous bits has its value changed if we add this domain hash
                        prefix_bits_mismatch = True
                        break
                else:
                    if not (current_gaussian_sums[previous_bit_id] + new_sample_to_add > 0):
                        temp_gaussian_samples.append(new_sample_to_add)
                    else:
                        # one of the previous bits has its value changed if we add this domain hash
                        prefix_bits_mismatch = True
                        break

            # special case if domain list empty before
            if cur_bit_id == 0:
                prefix_bits_mismatch = False
                current_gaussian_sums[0] += cur_sample # sim_hash.random_gaussian(0, cur_hash)
                domain_hash_list.append(cur_hash)
                domain_list.append(cur_domain_name)
            # we do not run this for first bit of index 0 ?
            elif len(temp_gaussian_samples) == cur_bit_id:
                # print(f'adding suitable domain {cur_hash}')
                prefix_bits_mismatch = False
                # if prefix bits still match add temp_gaussian_samples to current_gaussian_sums
                for i in range(cur_bit_id):
                    current_gaussian_sums[i] += temp_gaussian_samples[i]
                # increment the cur_bit_id
                current_gaussian_sums[cur_bit_id] += cur_sample
                # add the current domain to the list
                domain_hash_list.append(cur_hash)
                domain_list.append(cur_domain_name)

    return domain_hash_list, domain_list





def find_high_value_gaussian_sample(curr_bit, is_positive, hash_reversibility=False, dataset='movies', positive_thresh=4, domain_sample_size=None):
    """
    Repeatedly sample from a Gaussian until a large enough outlier is found
    :param curr_bit: the current target bit position
    :param is_positive: if the current target bit is 1 or 0, it defines the sign of the Gaussian outlier
    :param hash_reversibility: restrict the sampling to hashes for which have a domain name can recover
    :param dataset: dataset to use to sample elements (e.g., 'domains' or 'movies')
    :param positive_thresh: absolute value of minimum threshold to define a sample as outlier
    :param domain_sample_size: reduce the number of domains can sample from (works for 'domains' dataset)
    :return: the domain hash along with the outlier value and the domain name
    """
    if is_positive:
        # domain hash associated with the extremum gaussian sample ?
        extremum, domain_hash = -100, -1
        op, threshold = operator.lt, positive_thresh
    else:
        extremum, domain_hash = 100, -1
        op, threshold = operator.gt, -positive_thresh
    domain_name = None # initialization

    while op(extremum, threshold): # max < thresh or min > thresh

        if hash_reversibility:
            # Option 2 which try reversal of cityhash reversibility
            if dataset == 'domains':
                # id_domain_drawn = random.randint(0, N - 1)  # both extremity included [a,b] and list index from 0 to N-1
                if domain_sample_size is not None:
                    id_domain_drawn = random.randint(0, domain_sample_size - 1)
                else:
                    id_domain_drawn = random.randint(0, len(top_domains) - 1)
                rand_domain_name = top_domains[id_domain_drawn]
                # use lookup table from domain names to hashes to speed up avoid recomputing hash each time sample same domain name
                rand_domain_hash = hash_for_domain[rand_domain_name]
                # rand_domain_hash = cityhash.hash64(rand_domain_name)
            elif dataset == 'movies':
                id_domain_drawn = movie_id_list[random.randint(0, len(movie_id_list) - 1)]
                rand_domain_name = title_for_movie[id_domain_drawn]
                # use lookup table from [movie title] to hashes
                rand_domain_hash = hash_for_movie[id_domain_drawn]
            else:
                raise Exception(f'Wrong dataset selected')
        else:
            # Option 1 without cityhash reversibility
            rand_domain_name = None # Not available
            rand_domain_hash = random.randint(0, (1 << 64) - 1)

        rand_val = sim_hash.random_gaussian(curr_bit, rand_domain_hash)
        if op(extremum, rand_val): # max < rand_val or min > rand_val
            extremum, domain_hash, domain_name = rand_val, rand_domain_hash, rand_domain_name

    return domain_hash, extremum, domain_name


# If function inside main could not use multiprocessing
def check_attack_success(target_simhash, out_hash_bitlen=None, dataset='movies', domain_sample_size=None):
    """
    Check if the preimage attack was successful, i.e. the SimHash of the preimage matches the target
    :param target_simhash: the target SimHash
    :param out_hash_bitlen: the output length of the SimHash
    :param dataset: the dataset 'domains' or 'movies'
    :param domain_sample_size: reduce the number of domains can sample from (works for 'domains' dataset)
    :return: boolean for success of attack along with preimage if found
    """
    # Note: add this parts otherwise leading 0 bits would be ignored (added first in pipeline.py)
    if out_hash_bitlen is not None:
        prefix_bit_len = out_hash_bitlen
    else:
        prefix_bit_len = target_simhash.bit_length()
    domain_hash_list, domain_list_reversed = find_preimage(target_simhash, prefix_bit_len, reverse_hash=True,
                                                           dataset=dataset, domain_sample_size=domain_sample_size)
    print(domain_hash_list)
    print(f'reversed cityhash if available (count: {len(domain_hash_list)}):\n{domain_list_reversed}')

    # try to see if can find hash for which know domain
    # not needed anymore if keep using the reverse_hash=True parameter of find preimage
    if domain_list_reversed[0] is None: # if used hash_reversibility don't need it
        reverse_hash_list = []
        # global cityhash_lookup
        for domain_hash in domain_hash_list:
            # cityhash_lookup[domain_hash] # crash on keyerror
            reversed_hash = cityhash_lookup.get(domain_hash, None)
            # if reversed_hash is not None:
            #     print(f'{Fore.RED}Reversed hash {domain_hash} into {reversed_hash}')
            reverse_hash_list.append(reversed_hash)
            # second parameter is default return value when key does not exist
            # reverse_hash_list.append(cityhash_lookup.get(domain_hash, domain_hash))
        print(f'cityhash reversal lookup:\n{reverse_hash_list}')

    # sanity check run in other direction
    features = dict()
    for hash in domain_hash_list:
        features[hash] = 1
    computed_simhash_from_preimage = sim_hash.sim_hash_weighted_features(features, prefix_bit_len)
    if domain_list_reversed[0] is not None:
        computed_from_reversed_domain_name = sim_hash.sim_hash_strings(set(domain_list_reversed), prefix_bit_len)
        if computed_simhash_from_preimage != computed_from_reversed_domain_name:
            print(f'domain hash inconsistent with domain name {computed_simhash_from_preimage} != {computed_from_reversed_domain_name}')
    if target_simhash != computed_simhash_from_preimage:
        print(f'{Fore.RED}sanity check target failed (target, computed):{target_simhash, computed_simhash_from_preimage}')
        print(f'{bin(target_simhash)}\n{bin(computed_simhash_from_preimage)}')
        return False, domain_list_reversed
    else:
        print(f'{Fore.GREEN}Success')
        return True, domain_list_reversed


def generate_target_simhash(output_hash_bitcount=20, seed=None, real_user_movies=True, dataset='domains', domain_sample_size=None):
    """
    Generate a target SimHash
    :param output_hash_bitcount: SimHash output bit length
    :param seed: use to generate the same target on multiple run
    :param real_user_movies: if use real user movie history
    :param dataset: 'domains' or 'movies' history
    :param domain_sample_size: reduce the number of domains can sample from (works for 'domains' dataset)
    :return: the target SimHash and associated history
    """
    random.seed(seed) # default is None so do not change anything ?
    rand_domain_count = random.randint(1,20)
    domain_list = set()
    for i in range(rand_domain_count):
        if dataset == 'domains':
            if domain_sample_size is not None:
                id_domain_drawn = random.randint(0, domain_sample_size-1)
            else:
                id_domain_drawn = random.randint(0, N-1) # both extremity included [a,b] and list index from 0 to N-1
            domain_list.add(top_domains[id_domain_drawn]) # use a set
        elif dataset == 'movies':
            if real_user_movies:
                target_simhashes, target_histories = generate_from_traintest('../GAN/save_ml25m/realtest_ml25m_5000_32.txt', 1,
                                                                            title_for_movie, output_hash_bitcount, word_ml25m, seed=seed)
                print(f'Generated target SimHash {target_simhashes} from real user history {target_histories}')
                # It already computes the SimHash (also stop for loop at first iteration)
                return target_simhashes[0], target_histories[0]
            else:
                id_domain_drawn = movie_id_list[random.randint(0, len(movie_id_list) - 1)]
                rand_domain_name = title_for_movie[id_domain_drawn]
                domain_list.add(rand_domain_name)
        else:
            raise Exception(f'Wrong dataset selected')
    print(f'randomly drawn {rand_domain_count} domains: {domain_list}')
    target_simhash = sim_hash.sim_hash_strings(domain_list, output_hash_bitcount) # second param is output dim or kMaxNumberOfBitsInFloc
    return target_simhash, domain_list

# need a dict as return value for reference use cause otherwise value not modified
# function need to be pickled [1]
# [1] https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
# def worker_man(input_target_hash, out_hash_bitlen, dataset, return_dict):
#     success, generated_history = check_attack_success(input_target_hash, out_hash_bitlen=out_hash_bitlen, dataset=dataset)
def worker_man(input_target_hash, out_hash_bitlen, dataset, domain_sample_size, return_dict):
    """
    Multiprocessing using a manager
    :param input_target_hash: target SimHash
    :param out_hash_bitlen: output length of SimHash
    :param dataset: chosen dataset 'movies' or 'domains'
    :param domain_sample_size: reduce the number of domains can sample from (works for 'domains' dataset)
    :param return_dict: a dict containing the 'success' of the attack and the preimage 'history'
    :return: the updated return_dict
    """
    success, generated_history = check_attack_success(input_target_hash, out_hash_bitlen=out_hash_bitlen, dataset=dataset, domain_sample_size=domain_sample_size)
    return_dict['success'] = success
    return_dict['history'] = generated_history

def worker_queue(input_target_hash, queue):
    """
    Multiprocessing using a queue
    :param input_target_hash: target SimHash
    :param queue: queue
    :return: a dict with the 'success' of the attack
    """
    return_dict = queue.get()
    return_dict['success'] = check_attack_success(input_target_hash)
    queue.put(return_dict)

def attack_with_retry_after_timeout(timeout, output_hash_bitcount, seed=32, use_manager=True, dataset='movies', domain_sample_size=None): # use_queue=not use_manager
    '''
    Perfrom preimage attack and restart it if did not find a solution after a timeout
    :param timeout: seconds after which the process is stopped and restarted
    :param output_hash_bitcount: SimHash output bit length
    :param seed: seed only used for generating target SimHash from random sampling of movie in vocabulary
    :param use_manager: Uses Manager from multiprocessing otherwise uses Queue.
    :param dataset: 'movies' or 'domains' dataset from which to sample CithyHashes on
    :param domain_sample_size: reduce the number of domains can sample from (works for 'domains' dataset)
    :return: Target SimHash preimage : list of movieIDs and list of movie titles and TargetSimHash value and its original preimage
    '''
    # As attack may not work due to interdependencies not taken into account
    # the process should be killed after a timeout and restarted

    # Option 1 when generated simhash randomly in a range (interval)
    # target_simhash = random.randint((1 << 13), (1 << 21) - 1)
    target_simhash, target_history = generate_target_simhash(output_hash_bitcount=output_hash_bitcount, seed=seed,
                                                             dataset=dataset, domain_sample_size=domain_sample_size) # Can use seed to generate the same simhash each time

    print(f'target simhash integer: {target_simhash}')

    # Source: https://stackoverflow.com/questions/10415028/how-can-i-recover-the-return-value-of-a-function-passed-to-multiprocessing-proce/10415215#10415215
    # With a manager # can cause EOFError: Ran out of input while using pickle when use pycharm's python console
    if use_manager:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        return_dict['success'] = False # need init in case function does not terminate
        # process1 = multiprocessing.Process(target=worker_man, args=(target_simhash, output_hash_bitcount, dataset, return_dict))
        process1 = multiprocessing.Process(target=worker_man, args=(target_simhash, output_hash_bitcount, dataset, domain_sample_size, return_dict))
    # With a queue:
    else:
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.JoinableQueue.task_done
        queue = multiprocessing.Queue() # might need joinableQueue (manager works but spawn new process each time ?)
        return_dict = {'success': False}
        queue.put(return_dict)
        # did not implement modification to take leading 0 into account with simhash bitlen
        process1 = multiprocessing.Process(target=worker_queue, args=(target_simhash, queue))

    # src: https://stackoverflow.com/questions/492519/timeout-on-a-function-call
    process1.start()

    # wait for timeout seconds or until process finishes
    process1.join(timeout)

    if process1.is_alive():
        print(f'{Fore.RED}process still running so kill it')
        process1.terminate()
        # if terminate does not work can do
        # process1.kill()
        process1.join()
        print(f'{Fore.RED}problematic hash is {target_simhash}')
    if not use_manager:
        # might have problem with queue here
        return_dict = queue.get() # here does not need to place anything back in queue as no more process start ?
        queue.put(return_dict) # only in the case were did not terminate process ?


    while not return_dict['success']: # can use same return dict
        if use_manager:
            # process2 = multiprocessing.Process(target=worker_man, args=(target_simhash, output_hash_bitcount, dataset, return_dict))
            process2 = multiprocessing.Process(target=worker_man, args=(target_simhash, output_hash_bitcount, dataset, domain_sample_size, return_dict))
        else:
            # this does not work with queue after it fails
            process2 = multiprocessing.Process(target=worker_queue, args=(target_simhash, queue))
            return_dict = queue.get() # for the while condition ?
            queue.put(return_dict) # need something in queue for next worker_queue call ?
        process2.start()
        process2.join(timeout)
        if process2.is_alive():
            print(f'process still running so kill it')
            process2.terminate()
            # if terminate does not work can do
            # process1.kill()
            process2.join()
            print(f'{Fore.RED}problematic hash is {target_simhash}')

    movie_id_history = []
    if dataset == 'movies':
        for movie_title in return_dict['history']:
            movie_id_history.append(movie_for_title[movie_title])
    # at this point the history should match target simhash
    return return_dict['history'], movie_id_history, target_simhash, target_history


# Top N domains
# Since put reversal of cityhash into sampling process in find_high_value_gaussian_sample
# might not need those global variables anymore but did not refactor code to remove part that made used of it
# chosen_dataset = 'movies'
# Do not need legacy chosen dataset separation here so load everything
# if chosen_dataset == 'domains':
N = 1000000 # 100_000
top_domains = extract_top_n(N, f'../data/tranco_NLKW.csv')
cityhash_lookup, hash_for_domain = precompute_cityhash(top_domains)
# elif chosen_dataset == 'movies':
hash_for_movie, title_for_movie, movie_for_title, movie_id_list = precompute_cityhash_movielens()
# Movies does not need it anymore
# N = len(movie_id_list) # for legacy reason and avoid having more if else at other code locations
# Load vocab (used in func generate_target_simhash)
vocab_filepath = f'../GAN/save_ml25m/LeakGAN_ml25m_vocab_5000_32.pkl'
word_ml25m, vocab_ml25m = pickle.load(open(vocab_filepath, mode='rb'))


if __name__ == '__main__':
    # h1, h2 = find_gaussian_outlier(0)

    # Try global lookup table for easy access across functions for now ?
    # precompute_top_domain_hash(100_000)

    # timeout = 20
    # for i in range(100):
    #     print(f'current iteration: {i+1}')
    #     title_history, id_history, target_simhash, target_history = attack_with_retry_after_timeout(timeout, output_hash_bitcount=10, dataset=chosen_dataset)
    #     print(f'generated history:\n{title_history}\n{id_history}')
    #     print(f'original history:\n{target_history}')


    # Similar to settings in GAN disc pipeline
    # Ran without python console
    # print(hash_for_domain['google.com'])
    chosen_dataset = 'movies' # 'movies' or 'domains'
    bitlengths = [8] # [5, 10, 15, 20]
    seeds = [32, 987, 1331, 4263]
    iteration = 25
    samples_counts = [1000000, 100000, 10000]

    for sample_count in samples_counts:
        # Problem with current logging and multiprocessing
        # fh, sh = utils.init_loggers(log_to_stdout=True, filename=f'preimage_atck_{sample_count}', fh_lvl=logging.DEBUG, sh_lvl=logging.DEBUG)
        for bitlen in bitlengths:
            try:
                for seed in seeds:
                    for i in range(iteration):
                        with Timer(name=f'preimage_attack_{sample_count}', text=f"{Fore.LIGHTMAGENTA_EX}Duration of attack {{:.5f}}"):
                            title_history, id_history, target_simhash, target_history = \
                                attack_with_retry_after_timeout(output_hash_bitcount=bitlen, seed=seed, timeout=20,
                                                                dataset=chosen_dataset, domain_sample_size=sample_count)

            # To be able to cancel if run for too long but at the same time compute statistic for already generated samples
            # Need to press ctrl + C for keyboard interrupt
            except KeyboardInterrupt as ki:
                print(f'Interrupted run with keyboard: {ki}')

        # utils.remove_handlers([fh, sh])

    timers = Timer.timers
    print(timers)
    timings = Timer.timers.timings
    print(timings)
    for timer_name in timings.keys():
        min, max = Timer.timers.min(timer_name), Timer.timers.max(timer_name)
        mean, median, std = Timer.timers.mean(timer_name), Timer.timers.median(timer_name), Timer.timers.stdev(timer_name)
        print(f'Timer {timer_name} stats: [min,max]=[{min}, {max}] median={median} (mean,std)=({mean}, {std})')