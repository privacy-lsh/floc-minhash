
# Inspired from https://github.com/shigeki/floc_simulator/blob/WIP/packages/floc/sorting_lsh_clusters.go
# and https://github.com/chromium/chromium/blob/d7da0240cae77824d1eda25745c4022757499131/components/federated_learning/floc_sorting_lsh_clusters_service.cc
def apply_sorting_lsh(sim_hash, cluster_data, kMaxNumberOfBitsInFloc, check_sensiveness):
    """
    Sorting LSH defines the mapping from SimHash ranges to cohort IDs
    :param sim_hash: the SimHash for which wants to know the cohort ID
    :type sim_hash: uint64
    :param cluster_data: data file distributed by a Chrome server defining the mapping from SimHash to cohort ID
    :type cluster_data: byte array
    :param kMaxNumberOfBitsInFloc: Maximum number of bits used in SimHash computations (theoretical maximum is 64 bits)
     FLoC used 50 bits for SimHash computations.
    :type kMaxNumberOfBitsInFloc: uint8
    :param check_sensiveness: checks if the cohort is sensitive
    :type check_sensiveness: bool
    :return: the FLoC cohort ID
    :rtype: uint64
    """
    kExpectedFinalCumulativeSum = (1 << kMaxNumberOfBitsInFloc)
    kSortingLshMaxBits = 7
    kSortingLshBlockedMask = 0b1000000
    kSortingLshSizeMask = 0b0111111
    cumulative_sum = 0

    for index in range(len(cluster_data)):
        # might not cover all case in chromium implementation from comment in go simulator
        next_combined = cluster_data[index] # cluster_data is byte it will already be a uint8
        if (next_combined >> kSortingLshMaxBits) > 0:
            print('CodedInputStream::ReadVarint32 not implemented')
            return 0

        is_blocked = next_combined & kSortingLshBlockedMask
        next = next_combined & kSortingLshSizeMask

        if next > kMaxNumberOfBitsInFloc:
            print('Invalid cluster data')
            return 0

        cumulative_sum += (1 << next)

        if cumulative_sum > kExpectedFinalCumulativeSum:
            print('Overflow on cumulative_sum')
            return 0

        if cumulative_sum > sim_hash:
            if check_sensiveness and (is_blocked != 0):
                print('Blocked')
                return 0
            return index

    print('Index not found')
    return 0