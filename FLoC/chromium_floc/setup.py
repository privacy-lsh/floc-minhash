import json
import publicsuffixlist
import argparse
# Taken and adapted from [1]
# [1] https://github.com/shigeki/floc_simulator/blob/WIP/packages/floc/setup.go

# cluster data from chrome files
# (\AppData\Local\Google\Chrome\User Data\Floc\1.0.6 on windows)
cluster_file = "../data/Floc/1.0.6/SortingLshClusters"

kFlocIdMinimumHistoryDomainSizeRequired = 7

def setup_domain_list(domain_file='data/host_list.json'):
    domain_list = [] # list of strings

    with open(domain_file, 'r') as f:
        host_list = list(json.load(f))

    if len(host_list) < kFlocIdMinimumHistoryDomainSizeRequired:
        raise Exception(f'History should contain more than {kFlocIdMinimumHistoryDomainSizeRequired} domains')

    psl = publicsuffixlist.PublicSuffixList()
    for url in host_list:
        domain_list.append(psl.privatesuffix(url))

    return domain_list

def setup_cluster_data(file_path=cluster_file):
    with open(file_path, 'rb') as f:
        bytes_of_f = f.read()
    # For readability can use the byte in a list of char (uint8)
    # sorting_lsh_cluster_data = [bytes_of_f[i] for i in range(len(bytes_of_f))]
    # return sorting_lsh_cluster_data
    return bytes_of_f

def setup(domain_fp='data/host_list.json', cluster_fp=cluster_file):
    domain_list = setup_domain_list(domain_fp)
    sorting_lsh_cluster_data = setup_cluster_data(cluster_fp)
    return domain_list, sorting_lsh_cluster_data