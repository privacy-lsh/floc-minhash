import setup
from FLoC.chromium_components import sorting_lsh_cluster, sim_hash, cityhash

kMaxNumberOfBitsInFloc = 50 # uint8

# Taken and adapted from [1]
# [1] https://github.com/shigeki/floc_simulator/blob/WIP/demos/floc_sample/main.go

def compute_floc(domain_history_set, simhash_out_size, sorting_lsh_cluster_data, check_sensiveness=True):
    """
    Computes the FLoC ID
    :param domain_history_set: input history to the FLoC computations
    :param simhash_out_size: SimHash output bit length
    :param sorting_lsh_cluster_data: data distributed by a Chrome server to browser to map SimHashes to cohortID
    :param check_sensiveness: if checks when a cohort is sensitive
    :return: the FLoC ID
    """
    print(f'history: {domain_history_set}')
    hashed_domain_list = sim_hash.sim_hash_strings(domain_history_set, simhash_out_size)
    print(f'simhash: {hashed_domain_list} {hex(hashed_domain_list)} {bin(hashed_domain_list)}')
    cohort_id = sorting_lsh_cluster.apply_sorting_lsh(hashed_domain_list, sorting_lsh_cluster_data, simhash_out_size, check_sensiveness)
    print(f'cohort id: {cohort_id}')

def compute_fingerprint_vector(domain_name, output_size):
    """
    Computes the fingerprint vector for an element in the history
    :param domain_name: the name of the domain
    :param output_size: the SimHash output bit length
    :return: the fingerpring vector associated to that domain
    """
    domain_hash = cityhash.hash64(domain_name)
    print(f'domain: {domain_name} hash: {domain_hash}')
    fingerprint_vector = []
    for d in range(output_size):
        gaussian_coordinate = sim_hash.random_gaussian(d, domain_hash)
        fingerprint_vector.append(gaussian_coordinate)

    # Reverse list as index 0 is LSB
    print(f'{fingerprint_vector[::-1]}')
    return fingerprint_vector


# Filepath are relative to directory where run code
cluster_file = "../data/Floc/1.0.6/SortingLshClusters"
domain_file = '../data/host_list.json'
domain_list, sorting_lsh_cluster_data = setup.setup(cluster_fp=cluster_file, domain_fp=domain_file)
print(sorting_lsh_cluster_data)

compute_floc(domain_list, kMaxNumberOfBitsInFloc, sorting_lsh_cluster_data)

'''
domain_list: [nikkei.com hatenablog.com nikkansports.com yahoo.co.jp sponichi.co.jp cnn.co.jp floc.glitch.me ohtsu.org]
sim_hash: 779363756518407
cohortId: 21454
'''
# for i in range(0):
#     print(i)

# Test while debugging chromium with 'floc.glitch.me'
out_hash_len = 5
domain_list_set = {'google.com', 'youtube.com', 'facebook.com'} # 'netflix.com', 'microsoft.com'
compute_floc(domain_list_set, out_hash_len, sorting_lsh_cluster_data)

# Test while debugging chromium
domain_list_set = {'google.com', 'youtube.com', 'netflix.com'} # 'netflix.com', 'microsoft.com'
compute_floc(domain_list_set, out_hash_len, sorting_lsh_cluster_data)

compute_fingerprint_vector('google.com', out_hash_len)
compute_fingerprint_vector('youtube.com', out_hash_len)
compute_fingerprint_vector('facebook.com', out_hash_len)
compute_fingerprint_vector('netflix.com', out_hash_len)

domain_list = ['microsoft.com', 'zoom.us', 'github.com', 'wikipedia.org', 'mozilla.org', 'amazon.com', 'twitter.com']
for domain in domain_list:
    compute_fingerprint_vector(domain, out_hash_len)