import math

# Taken from [1]
# [1] https://github.com/chromium/chromium/blob/d7da0240cae77824d1eda25745c4022757499131/components/federated_learning/sim_hash.cc
from typing import Dict, Set
from FLoC.chromium_components import cityhash

g_seed1 = 1 # ULL (unsigned long long int)
g_seed2 = 2
kTwoPi = 2.0 * 3.141592653589793
MAX_UINT64 = 18446744073709551615


def random_uniform(i: int, j: int, seed: int) -> float:
    """[From Chromium] Hashes i and j to create a uniform RV in [0,1].
        :param i: a parameter from which to derive the random uniform sample
        :type i: uint64
        :param j: a parameter from which to derive the random uniform sample
        :type j: uint64
        :param seed: seed used for reproduceability
        :type seed: uint64
        :return: A sample from a random uniform
        :rtype: double
    """
    # Makes use of  base::legacy::CityHash64WithSeed [chromium/base/hash/legacy_hash.cc] which uses
    # internal::cityhash_v103::CityHash64WithSeed [chromium/base/third_party/cityhash_v103/src/]
    byte_arr = i.to_bytes(8, 'little') + j.to_bytes(8, 'little')

    # Note: when ran anonymity_evaluation.chromium_simhash_precomputation() run into overflow here
    # try:
    #     byte_arr = i.to_bytes(8, 'little') + j.to_bytes(8, 'little')
    # except OverflowError:
    #     print(f'OverflowError: i: {i:0b} ({i.bit_length()}), j: {j:0b} ({j.bit_length()})')
    #     raise

    # if take an index of a byte array get an int not a byte in python
    byte_list = ''.join([chr(byte_arr[i]) for i in range(len(byte_arr))])
    # It did not work as intended with str(byte_array) but with byte_list it did
    # cityhash python implementation taken seems to expect string as input
    hashed = cityhash.hash64WithSeed(byte_list, seed)
    # hashed = cityhash.hash64WithSeed(str(byte_arr), seed)

    return hashed/MAX_UINT64

# Interesting blog post [1] on why using the box-muller transform is a good way to generate
# unbiased random directions in an n-dimensional space
# [1] https://towardsdatascience.com/the-best-way-to-pick-a-unit-vector-7bd0cc54f9b
def random_gaussian(i: int, j: int) -> float:
    """ [From Chromium] Uses the Box-Muller transform to generate a Gaussian from two uniform RVs in [0,1] derived from i and j.
        :param i: In the SimHash computation this represents the current bit position in the SimHash
        :type i: uint64
        :param j: In the SimHash computation this represent the hash of the current domain
        :type j: uint64
        :return: The sampled Gaussian coordinate
        :rtype: double
    """
    rv1 = random_uniform(i, j, g_seed1) # type should be double (aka float)
    rv2 = random_uniform(j, i, g_seed2)

    # Sanity check
    if rv1 > 1 or rv1 < 0 or rv2 > 1 or rv2 < 0:
        raise ValueError(f'random variable not from a uniform distribution {rv1}, {rv2}')

    # BoxMuller
    return math.sqrt(-2.0 * math.log(rv1)) * math.cos(kTwoPi * rv2)

def sim_hash_weighted_features(features: Dict[int, int], output_dimensions: int) -> int:
    """
    Computes the SimHash given the features
    :param features: Weighted Features
    :type features: dict[int, int], map<FeatureEncoding, FeatureWeight>
    :param output_dimensions: Output length of the SimHash
    :type output_dimensions: uint8
    :return: The SimHash
    :rtype: uint64
    """
    # sanity check
    if not (0 <= output_dimensions <= 64):
        raise ValueError(f'output dimension not in range {output_dimensions}')

    result: int = 0 # type uint64
    # print(f'Features {features}')
    for d in range(0,output_dimensions):
        acc: float = 0 # type double
        for domain_hash, weight in features.items(): # key is the hash, value the weight
            acc += random_gaussian(d, domain_hash) * weight

        # i-th bit indicates the sign of the sum of i-th coordinate floating-point vector
        # derived from the domain names
        if acc > 0:
            result |= (1 << d) # bitwise or

    return result # simhash

# set in python unordered by default [2]
# [2] https://docs.python.org/3.9/library/stdtypes.html#set-types-set-frozenset
def sim_hash_strings(input: Set[str], output_dimensions: int) -> int:
    """
    Computes the SimHash given an input history
    :param input: Browsing history set
    :type input: unordered set of string
    :param output_dimensions: Output length of the SimHash
    :type output_dimensions: uint8
    :return: the SimHash
    """
    # sanity check
    if not (0 < output_dimensions <= 64):
        raise ValueError(f'output dimension not in range {output_dimensions}')

    # features: Dict[int, int] = dict() # WeightedFeatures type: map<FeatureEncoding, FeatureWeight>
    features = dict()

    for s in input: # s would be a domain (part of website url of browser history)
        # Hash function expects string as input it would seem
        string_hash = cityhash.hash64(s) # note the hash is an int
        # Note: it seems hash function returns output with more than 64 bits (e.g. input Начальни of movielens)
        if string_hash.bit_length() > 64:
            print(f'hash > 64 bits input: {s} out_hash: {string_hash}, {string_hash:b}, {string_hash.bit_length()}')
            # Solution Truncate to 64 bits (LSB)
            mask = (1 << 64) - 1
            string_hash = string_hash & mask
            print(f'64 LSB hash: {string_hash}, {string_hash:b}, {string_hash.bit_length()} bits')
        features[string_hash] = 1 # assign a weight of 1 to that feature
        # Order of map used to iterate seems random in run of go simulator
    return sim_hash_weighted_features(features, output_dimensions)