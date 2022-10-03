import hashlib
import math
import random
from random import shuffle
import numpy as np
from tqdm import tqdm, trange
import FLoC.utils as utils
from codetiming import Timer

# Some part taken from or inspired by https://www.pinecone.io/learn/locality-sensitive-hashing/
# https://github.com/pinecone-io/examples/blob/master/locality_sensitive_hashing_traditional/sparse_implementation.ipynb
# other notebook has some speed up of operation using numpy
# https://raw.githubusercontent.com/pinecone-io/examples/master/locality_sensitive_hashing_traditional/testing_lsh.ipynb

## Utils

def int_to_bytes(integer):
    return integer.to_bytes((integer.bit_length() // 8) + 1, byteorder='big')

def shingle(text: str, k: int):
    shingle_set = set()
    for i in range(len(text) - k + 1):
        shingle_set.add(text[i:i + k])
    return shingle_set


def onehot_encoder(shingles_set: set, vocab: set):
    onehot_encoded = [1 if shingle in shingles_set else 0 for shingle in vocab]
    return onehot_encoded


# Permutation based approach from MMDS Book Sec. 3.3.2 Minhashing
def create_permutation_minhash(vocab_size: int):
    # function for creating the hash vector/function
    minhash = list(range(1, vocab_size + 1))
    shuffle(minhash)
    return minhash


def build_permutation_minhash_list(vocab_size: int, out_bitlen: int):
    # function for building multiple minhash vectors
    hashes = []
    for _ in range(out_bitlen):
        hashes.append(create_permutation_minhash(vocab_size))
    return hashes


def create_permutation_minhash_signature(vector: list, minhash_list: list, vocab_size: int):
    # use this function for creating our signatures (eg the matching)
    signature = []
    for minhash in minhash_list:
        for i in range(1, vocab_size + 1):
            idx = minhash.index(i)
            signature_val = vector[idx]
            # MMDS Book Sec. 3.3.2 method with `n` random permutations
            # Sec 3.3.5 says in practice use n randomly chosen hash functions and perform different approach
            if signature_val == 1:
                signature.append(idx)
                break  # Only care about first value where onehot is 1
    return signature


# MMDS Book Sec. 3.3.5 Computing Minhash Signatures in Practice
# ``instead of picking n random permutations of rows, we pick n randomly
#   chosen hash functions h1, h2, . . . , hn on the rows.''

def compute_hashfunc_minhash_signature(target_set: set, hash_list: list):
    """
    Compute the minhash signature according to the procedure in Section 3.3.5 of the Mining of Massive Datasets book (3rd edition).
    :param target_set: the set for which want to compute the minhash signature
    :param hash_list: the n hash function, n determines the number of element in the signature.
    :return: the signature as a list of minimum elements
    """
    # We compute one signature of size len(hash_list)
    signature = []
    for hash_func in hash_list:
        min_elem = math.inf
        for s in target_set:
            cur_elem = hash_func(s)
            min_elem = min(min_elem, cur_elem)
        signature.append(min_elem)
    return signature

#################
## Brute Force ##
#################

def precompute_hashes_of_vocab(hash_list: list, vocab_set: set):
    """
    Precompute a hash table for each hash in hash_list.
    :param hash_list: he hash used to generate the signature
    :param vocab_set: a vocabulary of elements under consideration, it can cover all values of the hash or not
    :return: the list of mapping of minhash values to set elements
    """
    # Precompute the hashes of the elements under consideration
    minhash_to_vocab_list = [dict() for _ in range(len(hash_list))]
    for elem in vocab_set:
        # Note: Can improve memory efficiency since only need one hash table at a time
        # here compute and store every hash table at the same time
        for i, hash_func in enumerate(hash_list):
            minhash_to_vocab_list[i][hash_func(elem)] = elem
    return minhash_to_vocab_list


def precompute_hash_of_vocab(hash_func_i, vocab):
    """
    Precompute a single hash table. Save memory since only need one hash table at a time.
    :param hash_func_i: the current hash function
    :param vocab: an iterable containing set elements under consideration
    :return: hash table (dictionnary)
    """
    minhash_to_id = dict()
    for elem in vocab:
        minhash_to_id[hash_func_i(elem)] = elem

    return minhash_to_id


def recover_set_from_minhash_sig(signature, minhash_to_vocab_list: list):
    """
    Attempts to recover an input set from the minhash signature
    :param signature: the minimum element defining the signature (there are as many element as hash functions)
    :param minhash_to_vocab_list: the list of mapping from minhash values to set elements.
    (Memory inefficient since store all of them at the same time)
    :return: the recovered set
    """

    if len(minhash_to_vocab_list) != len(signature):
        raise ValueError("The signature should have the number of hash as length")

    # We have the signature which gives us the minimum element of the target set
    # this means that all the minhash values below this value are not part of the target set but its complement
    # all the other elements can be part of target set

    potential_set_elements = []
    elements_not_in_set, elements_in_set, elements_maybe_in_set = set(), set(), set()
    for hash_index, cur_minhash_to_vocab_dict in enumerate(minhash_to_vocab_list):
        sorted_minhash_values = sorted(list(cur_minhash_to_vocab_dict.keys()))  # if call .sort() (in place) so return None
        # Want the index of the minhash value in the given signature
        cur_set_min_elem_index = sorted_minhash_values.index(signature[hash_index])
        # print(f'DEBUG:index sig minhash value: {cur_set_min_elem_index}')
        not_in_set, in_set, maybe_in_set = set(), set(), set()
        # may not need to store individual sets but only union of respective sets
        for i in range(len(sorted_minhash_values)):
            # iterate over the sorted minhash values
            # and retrieve the corresponding set element in increasing order of minhash values
            cur_set_elem = cur_minhash_to_vocab_dict[sorted_minhash_values[i]]
            if i < cur_set_min_elem_index:
                not_in_set.add(cur_set_elem)
            elif i == cur_set_min_elem_index:
                in_set.add(cur_set_elem)
            else:  # i > cur_set_min_elem_index
                maybe_in_set.add(cur_set_elem)
        potential_set_elements.append((not_in_set, in_set, maybe_in_set))
        # Update (union) set in place
        elements_not_in_set.update(not_in_set)
        elements_in_set.update(in_set)
        elements_maybe_in_set.update(maybe_in_set)

    print(f'DEBUG:potential set elements: {utils.pretty_format(potential_set_elements, ppindent=1, ppwidth=140, ppcompact=True)}')

    if len(elements_not_in_set.intersection(elements_in_set)) != 0:
        print(f'elems not in set {elements_not_in_set} intersection {elements_in_set} elems in set')
        raise Exception(
            "Error: the intersection of the set containing elements in the target set and the one with elements not in the target should be the empty set")

    possible_additional_elements = elements_maybe_in_set.difference(elements_not_in_set, elements_in_set)

    return elements_in_set, possible_additional_elements


def preimages_of_minhash_sig(minhash_sig, hash_list, vocab):
    """

    :param minhash_sig: the minhash signature
    :param hash_list: the list of hash function used
    :param vocab: (iterable) restriction of elements that can be in the set under consideration
    :return:
    """
    if len(hash_list) != len(minhash_sig):
        raise ValueError("The signature should have the number of hash as length")

    element_not_in_target = set()
    element_in_target = set() # Note that collision could still happen with a very low probability

    for hash_index, hash_func in enumerate(tqdm(hash_list)):
        # Precompute hash table for current hash
        minhash_to_vocab = precompute_hash_of_vocab(hash_func, vocab)
        sorted_hash_values = sorted(list(minhash_to_vocab.keys()))
        cur_minhash = minhash_sig[hash_index]
        # DEBUG
        # print(f'sorted hash value: {sorted_hash_values}')
        # print(f'hash values count: {len(sorted_hash_values)}') # +1 more than needed
        cur_minhash_index = sorted_hash_values.index(cur_minhash)

        # Iterate over hash values list sorted in increasing order
        count = 0
        while count < cur_minhash_index:
            # retrieve the set elements associated to the current hash value
            cur_elem = minhash_to_vocab[sorted_hash_values[count]]
            # since iterate over sorted hash values the current element is smaller than the minhash signature element
            # therefore it cannot belong to the target set that generated this signature
            element_not_in_target.add(cur_elem)
            count += 1

        # now count == cur_minhash_index
        if count != cur_minhash_index or sorted_hash_values[count] != cur_minhash:
            raise Exception(f'Error property not verified')

        element_in_target.add(minhash_to_vocab[sorted_hash_values[count]])
        # print(f'DEBUG:not in target: {utils.pretty_format(element_not_in_target, ppindent=1, ppwidth=140, ppcompact=True)}')
        # print(f'DEBUG:in target: {utils.pretty_format(element_in_target, ppindent=1, ppwidth=140, ppcompact=True)}')

    return element_not_in_target, element_in_target


###############
## Heuristic ##
###############

def recover_target_minhash_set(signature, hash_list, max_x_can_hash):
    """
    Heuristic to recover a set resulting in the minhash signature
    :param signature: the minhash signature
    :param hash_list: he hash functions used to compute the minhash the signature
    :param max_x_can_hash: the maximum value used as input for the hash
    :return: the preimage set
    """

    preimage_set = set()
    for hash_index, hash_func in enumerate(hash_list):

        # sample an element (integer) from chosen range (e.g. 32 bit unsigned int)
        # The heuristic being that if find a value below a threshold there should be lower probability of other
        # values being inferior for the current hash function
        # Currently retain one sampled element per hash function
        target_set_elem = sample_minhash_value(hash_func, max_x_can_hash, signature[hash_index])

        # There should be no need to check that the target set element is not lower than other simhash values
        # since we assume uniqueness (hash functions are bijection)
        preimage_set.add(target_set_elem)

    return preimage_set



def recover_set_not_contradicting_minhash_sig(signature, hash_list, max_x_can_hash):
    """
    Heuristic to recover a set not contradicting minhash signature, meaning that all set elements have a hash
    greater or equal to each of the minhash signature values
    :param signature: the minhash signature
    :param hash_list: the hash functions used to compute the minhash the signature
    :param max_x_can_hash: the maximum value used as input for the hash
    :return: a preimage for the given minhash signature
    """

    potential_set = set()

    for hash_index, hash_func in enumerate(hash_list):

        minhash_condition_not_satisfied = True
        while minhash_condition_not_satisfied:
            # sample an element (integer) from chosen range (e.g. 32 bit unsigned int)
            # The heuristic being that if find a value below a threshold there should be lower probability of other
            # values being inferior for the current hash function
            # Currently retain one sampled element per hash function
            lower_thresh = signature[hash_index]
            upper_thresh = min(max_x_can_hash, signature[hash_index] + max_x_can_hash//len(hash_list))
            if lower_thresh > upper_thresh:
                upper_thresh = lower_thresh
            target_set_elem, hashed_elem = sample_hash_value_in_range(hash_func, max_x_can_hash, lower_thresh, upper_thresh)

            # Need to check that current set element does is not lower than any of the minhash signature values
            for i in range(len(hash_list)):
                if hash_list[i](target_set_elem) < signature[i]:
                    break
                else: # ensure sig_i <= hash_i(target_elem)
                    if i == len(hash_list)-1: # checked condition satisfied for all minhash values
                        # the current set element had its hashes greater or equal to all minhash signature values
                        minhash_condition_not_satisfied = False
                        potential_set.add(target_set_elem)

    return potential_set


def sample_minhash_value(hash_func_i, max_sample, minhash_sig_i):
    """
    Recover the element that gives minhash_sig_i w.r.t. hash_func_i
    :param hash_func_i: the i-th hash with which computed current minhash signature i-th element
    :type hash_func_i: ModuloPrimeHashFunction
    :param max_sample: the maximum uint can sample to be in the vocabulary set
    :param minhash_sig_i: the minhash signature's i-th element want to find preimage for
    :return: the element that gives minhash_sig_i w.r.t. hash_func_i
    """
    # Slow iterative approach:
    # for elem in range(max_sample + 1):
    # for elem in trange(max_sample + 1, desc='find minhash', leave=False):
    #     if hash_func_i(elem) == minhash_sig_i:
    #         return elem
    # raise Exception(f'Did not find minhash in range')

    # Modular inverse
    # h(x) = r*x+c mod p (have access to r,c and p with HashFunction class
    # r_inv = modinv(r, p)
    r_inv = modular_inverse(hash_func_i.r, hash_func_i.p)
    # x = (h(x)-c) * r_inv % p
    x = (((int(minhash_sig_i) - hash_func_i.c) % hash_func_i.p) * r_inv) % hash_func_i.p

    # verification:
    if hash_func_i(x) % hash_func_i.p != minhash_sig_i % hash_func_i.p:
        raise ArithmeticError(f'Wrong calculation {hash_func_i(x)} != {minhash_sig_i} mod {hash_func_i.p}')
        # print(f'Wrong calculation {hash_func_i(x)} != {minhash_sig_i} mod {hash_func_i.p}')
    return x




def sample_hash_value_in_range(hash_func_i, max_sample, minhash_sig_i, upper_threshold):
    """
    The sampled element needs to be low enough but not lower than the minhash otherwise the minhash would be different.
    :param hash_func_i: the i-th hash with which computed current minhash signature i-th element
    :param max_sample: the maximum uint can sample to be in the vocabulary set
    :param minhash_sig_i: the minhash signature's i-th element which is also the lower threshold
    :param upper_threshold: the sampled value should be less or equal to the threshold to be considered low enough
    :return: the sampled value in range [minhash_sig_i, upper_threshold]
    """

    hashed_sample = upper_threshold + 1

    while not (minhash_sig_i <= hashed_sample <= upper_threshold):
        set_elem_sample = random.randint(0, max_sample)
        hashed_sample = hash_func_i(set_elem_sample)

    return set_elem_sample, hashed_sample





# example 3.8 from Book
def mmds335_example():
    # Sets
    set1 = {0, 3}
    set2 = {2}
    set3 = {1, 3, 4}
    set4 = {0, 2, 3}
    vocab_set = {0, 1, 2, 3, 4}  # set1.union(set2).union(set3).union(set4)
    # Hash function
    h1 = ModuloPrimeHashFunction(1, 1, 5) # lambda x: (x + 1) % 5
    h2 = ModuloPrimeHashFunction(3, 1, 5) # lambda x: (3 * x + 1) % 5
    hash_list = [h1, h2]
    sig1 = compute_hashfunc_minhash_signature(set1, hash_list)
    sig2 = compute_hashfunc_minhash_signature(set2, hash_list)
    sig3 = compute_hashfunc_minhash_signature(set3, hash_list)
    sig4 = compute_hashfunc_minhash_signature(set4, hash_list)
    print(f'{sig1}\n{sig2}\n{sig3}\n{sig4}')
    # should be [1,0] [3,2] [0,0] [1,0]
    minhash_to_vocab_list = precompute_hashes_of_vocab(hash_list, vocab_set)
    print(f'DEBUG:minhash_to_vocab_list{minhash_to_vocab_list}')
    elems_in_set1, elems_maybe_in_set1 = recover_set_from_minhash_sig(sig1, minhash_to_vocab_list)
    print(f'sig {sig1}: real set {set1} recovered: in set {elems_in_set1} + maybe {elems_maybe_in_set1}')
    heuristic_preimage_set1 = recover_target_minhash_set(sig1, hash_list, 4)
    heuristic_potential_set1 = recover_set_not_contradicting_minhash_sig(sig1, hash_list, 4)
    verify_heur1 = compute_hashfunc_minhash_signature(heuristic_preimage_set1, hash_list)
    verify_heur1plus = compute_hashfunc_minhash_signature(heuristic_preimage_set1.union(heuristic_potential_set1), hash_list)
    print(f'heuristic preimage: {heuristic_preimage_set1} potential: {heuristic_potential_set1}, [target] {sig1} == {verify_heur1} [recovered] == {verify_heur1plus}')
    elems_in_set2, elems_maybe_in_set2 = recover_set_from_minhash_sig(sig2, minhash_to_vocab_list)
    print(f'sig {sig2}: real set {set2} recovered: in set {elems_in_set2} + maybe {elems_maybe_in_set2}')
    heuristic_preimage_set2 = recover_set_not_contradicting_minhash_sig(sig2, hash_list, 4)
    verify_heur2 = compute_hashfunc_minhash_signature(heuristic_preimage_set1, hash_list)
    print(f'heuristic preimage: {heuristic_preimage_set2}, [target] {sig2} == {verify_heur2} [recovered]')
    elems_in_set3, elems_maybe_in_set3 = recover_set_from_minhash_sig(sig3, minhash_to_vocab_list)
    print(f'sig {sig3}: real set {set3} recovered: in set {elems_in_set3} + maybe {elems_maybe_in_set3}')
    heuristic_preimage_set3 = recover_set_not_contradicting_minhash_sig(sig3, hash_list, 4)
    verify_heur3 = compute_hashfunc_minhash_signature(heuristic_preimage_set3, hash_list)
    print(f'heuristic preimage: {heuristic_preimage_set3}, [target] {sig3} == {verify_heur3} [recovered]')
    elems_in_set4, elems_maybe_in_set4 = recover_set_from_minhash_sig(sig4, minhash_to_vocab_list)
    print(f'sig {sig4}: real set {set4} recovered: in set {elems_in_set4} + maybe {elems_maybe_in_set4}')
    heuristic_preimage_set4 = recover_set_not_contradicting_minhash_sig(sig4, hash_list, 4)
    verify_heur4 = compute_hashfunc_minhash_signature(heuristic_preimage_set4, hash_list)
    print(f'heuristic preimage: {heuristic_preimage_set4}, [target] {sig4} == {verify_heur4} [recovered]')


def sample_random_coefficients(coeff_count, low, high, seed=None, use_numpy=False):
    """
    Sample coefficient r,c for function h(x) = r*x+c mod p
    :param coeff_count: number of coefficients r,c to samples
    :param low: lowest element (included) in sample range
    :param high: highest element (excluded) in sample range
    :param seed: seed to use for rng sampling
    :param use_numpy: if use numpy (limited to 64 bit integers) or standard python
    :return: the sampled integer coefficients as two ndarrays
    """
    # Numpy limited to 64 bit integers, but may want to experiment with bigger integers
    if use_numpy:
        rng = np.random.default_rng(seed)
        use_rng_integers=True
        if use_rng_integers:
            # can have repetitions
            # note: Numpy is limited to support of 64 bit integers
            r_coeffs = rng.integers(low,high,coeff_count, dtype=np.uint64)
            c_coeffs = rng.integers(low,high,coeff_count, dtype=np.uint64)
            def ensure_unique_values(ndarray):
                # Unique values used as coeffs
                new_ucoeffs = np.unique(ndarray) # can also return indices (inverse) and counts
                if new_ucoeffs.size < coeff_count:
                    print(f'DEBUG:unique coeffs ({new_ucoeffs.size}):{new_ucoeffs}')
                    for i in range(coeff_count - new_ucoeffs.size):
                        drawn_coeff = rng.integers(low,high)
                        while drawn_coeff in new_ucoeffs:
                            drawn_coeff = rng.integers(low, high)
                        new_ucoeffs = np.append(new_ucoeffs, drawn_coeff)
                return new_ucoeffs
            r_coeffs = ensure_unique_values(r_coeffs)
            c_coeffs = ensure_unique_values(c_coeffs)

        else:
            # can also use rng.choice with np.arrange (allow to sample without replacement)
            # (though enumerate all values in range so much slower)
            uint_range = np.arange(low, high) # dtype=np.uint32
            r_coeffs = rng.choice(uint_range, coeff_count, replace=False)
            c_coeffs = rng.choice(uint_range, coeff_count, replace=False)

        # print(f'DEBUG:coeffs:\n{r_coeffs}\n{c_coeffs}')
        # return r_coeffs, c_coeffs
    else:

        def sample_random_coeffs(count):
            rand_coeffs = []
            while count > 0:
                rand_int = random.randint(low, high)

                # uniqueness is required
                while rand_int in rand_coeffs:
                    rand_int = random.randint(low, high)

                rand_coeffs.append(rand_int)
                count -= 1

            return rand_coeffs

        r_coeffs = sample_random_coeffs(coeff_count)
        c_coeffs = sample_random_coeffs(coeff_count)

    print(f'DEBUG:coeffs:\n{r_coeffs}\n{c_coeffs}')
    return r_coeffs, c_coeffs


# It seemed to not work with lambda expression (same values for each hash functions) so create a class
class ModuloPrimeHashFunction:

    # __new__ is called before __init__
    def __init__(self, r, c, p):
        """
        Hash function of the form h(x) = r*x+c mod p
        :param r: r coefficient
        :param c: c coefficient
        :param p: prime p
        """
        # Due to overflow warning cast to python int which have unlimited size
        # Generated using numpy which uses limited precision integers
        self.r = int(r)
        self.c = int(c)
        self.p = int(p)

    # allows to call class instance inst as inst(x) instead of inst.func_name(x)
    def __call__(self, x): # , *args, **kwargs
        """
        Hash the input value x according to h(x) = r*x+c mod p
        :param x: the input to hash
        :return: the hashed value h(x)
        """
        # Due to overflow warning cast to python int which have unlimited size
        # (limited size may be due to numpy or other using a different type than python int)
        return (self.r * int(x) + self.c) % self.p


class CryptographicHashFunction:
    # Since need multiple hash functions, it may be preferable to used keyed hash function
    # where the key can be the index of the hash function
    # https://en.wikipedia.org/wiki/List_of_hash_functions#Keyed_cryptographic_hash_functions

    # https://docs.python.org/3/library/hashlib.html#module-hashlib
    # hashlib.sha256('test'.encode('utf-8')).digest() # sha3_256()
    # Option 2:
    # m = hashlib.sha256(); m.update(b"test"); m.digest()
    # Slower with hashlib.new('sha256')

    def __init__(self, hash_object):
        """

        """
        # no need to init hash object since have to create new hash object each time hash new message
        # or could use hash_obj.copy()
        self.root_hash_object = hash_object.copy() # the common initial substring is empty

    def __call__(self, input_to_hash: int):
        """
        Return the hash of the input value
        :param input_to_hash: integer to hash
        :return: the hash of input_to_hash as an integer
        """
        cur_hash_obj = self.root_hash_object.copy()
        input_bytes = int_to_bytes(input_to_hash)
        cur_hash_obj.update(input_bytes)
        out_hash = int(cur_hash_obj.hexdigest(), 16)
        return out_hash


# Modular inverse:
# https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm
def egcd(a,b):
    """
    Extended euclidian algorithm for computing GCD(a,b)
    :param a: an integer
    :param b: an integer
    :return: (g, x, y) such that a*x + b*y = g = gcd(a, b)
    """
    if a == 0:
        return b, 0, 1
    else:
        b_div_a, b_mod_a = divmod(b,a)  # b//a, b%a
        g, y, x = egcd(b_mod_a, a)      # egcd(b % a, a)
        return g, x - b_div_a * y, y    # g, x - (b // a) * y, y

# Python 3.8+ may do y = pow(x, -1, p) ?
def modular_inverse(a, p):
    """
    Compute the inverse of a modulo p using the extended euclidian algorithm
    :param a:
    :param p:
    :return: x such that (x * a) % p == 1
    """
    g, x, y = egcd(a, p) # y is not needed can replace by _
    if g != 1:
        raise Exception('gcd(a,p) != 1, modular inverse does not exist')
    else:
        return x % p



def hash_list_from_coeffs(r_coeffs: list, c_coeffs: list, p: int) -> list:
    """
    Generate hash functions of the form h(x) = r*x+c mod p
    :param r_coeffs: r coefficient in the hash function
    :param c_coeffs: c coefficient in the hash function
    :param p: prime number greater than max value for range under consideration
    :return: a list of hash function
    """
    if len(r_coeffs) != len(c_coeffs):
        raise Exception(f'coeffs list not the same size')
    hash_list = []
    for i in range(len(r_coeffs)):
        # it seemed to have same hash function for every element in list with lambda function
        # hash_list.append(lambda x: (r_coeffs[i]*x + c_coeffs[i]) % p)
        hash_list.append(ModuloPrimeHashFunction(r_coeffs[i], c_coeffs[i], p))

    return hash_list


def generate_cryptographic_hash_list(hash_counts: int, hash_len=32):
    """

    :param hash_counts: number of hash function in the list
    :param hash_len: output hash length in bytes (default 32 bytes = 256 bits)
    :return: list of hash function
    """
    hash_list = []
    for i in range(1, hash_counts+1):
        key_i = int_to_bytes(i)
        hash_object = hashlib.blake2b(key=key_i, digest_size=hash_len)
        hash_func = CryptographicHashFunction(hash_object)
        hash_list.append(hash_func)

    return hash_list


def generate_target_set(set_size: int, sample_range=None, chosen_population=None, seed=None) -> set:
    """

    :param set_size:
    :param sample_range: tuple of lower upper values included
    :param chosen_population: if enumerate all elements and make a choice from the list like object
    :param seed:
    :return:
    """
    if chosen_population is not None:
        # random.choices(chosen_population, k=set_size)
        rng = np.random.default_rng(seed)
        target = rng.choice(chosen_population, set_size, replace=False)
        return set(target)
    elif sample_range is not None:
        random.seed(seed)
        target = set()
        while len(target) < set_size:
            sample = random.randint(sample_range[0], sample_range[1])
            target.add(sample)

        return target
    else:
        raise Exception('One of sample_range or chosen_population must be specified')


def verify_valid_preimage(target_sig, preimage, hash_list, verbose=0):
    """

    :param target_sig: target minhash signature
    :param preimage: preimage to verify if signature is valid
    :param hash_list: list of hash used to compute signature
    :param verbose: if print results, default 0 no print
    :return:
    """

    sig_of_preimg = compute_hashfunc_minhash_signature(preimage, hash_list)

    if verbose > 0:
        if sig_of_preimg != target_sig:
            print(f'ERROR: {sig_of_preimg} != {target_sig}')
        else:
            print(f'{sig_of_preimg == target_sig}: {sig_of_preimg} == {target_sig}')

    return sig_of_preimg == target_sig



def main():
    pinecone_example = False
    mmds_335_example = True
    if pinecone_example:
        a = "flying fish flew by the space station"
        b = "he will not allow you to bring your sticks of dynamite and pet armadillo along"
        c = "he figured a few sticks of dynamite were easier than a fishing pole to catch an armadillo"

        k = 2

        shingle_set_a = shingle(a, k)
        shingle_set_b = shingle(b, k)
        shingle_set_c = shingle(c, k)

        vocab = shingle_set_a.union(shingle_set_b).union(shingle_set_c)
        print(vocab)

        onehot_a = onehot_encoder(shingle_set_a, vocab)
        onehot_b = onehot_encoder(shingle_set_b, vocab)
        onehot_c = onehot_encoder(shingle_set_c, vocab)
        print(onehot_a)

        # we create 20 minhash vectors
        minhash_list = build_permutation_minhash_list(len(vocab), 20)
        sig_a = create_permutation_minhash_signature(onehot_a, minhash_list, len(vocab))
        sig_b = create_permutation_minhash_signature(onehot_b, minhash_list, len(vocab))
        sig_c = create_permutation_minhash_signature(onehot_c, minhash_list, len(vocab))

        print(f'{sig_a}\n{sig_b}\n{sig_c}')

    if mmds_335_example:
        mmds335_example()



if __name__ == '__main__':
    main()

    recover_all_preimages = False
    heuristic = False
    minhash_hierarchy = True

    # Some inspiration taken from [1] and linked code [2]
    # [1] https://mccormickml.com/2015/06/12/minhash-tutorial-with-python-code/
    # [2] https://github.com/chrisjmccormick/MinHash/blob/master/runMinHashExample.py
    hash_count = 10
    MAX_POSSIBLE_VALUE = (1 << 128) - 160
    print(f'sampling random coefficients')
    r_coeffs, c_coeffs = sample_random_coefficients(hash_count, 0, MAX_POSSIBLE_VALUE, seed=None)
    # Prime higher than max value in range under consideration (2^32=4294967296)
    # can check http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php (from McCormick's tutorial code)
    # The prime p should be close to the max value in the population
    # Prime just less than a power of 2: https://primes.utm.edu/lists/2small/100bit.html
    prime_p = (1 << 128) - 159 # 20 < 23; 1000 < ; 10000 < 10007; 2^32 < 4294967311, (1<<128)-160 < 2^128 - 159 (prime)
    print(f'creating hash functions')
    hash_list = hash_list_from_coeffs(r_coeffs, c_coeffs, prime_p)

    print(f'generating target set')
    if recover_all_preimages:
        # Note: Run out of memory for large MAX_POSSIBLE_VALUE when creating chosen population
        chosen_population = np.arange(MAX_POSSIBLE_VALUE)  # vocab considered (integer range)
        target_set = generate_target_set(10000, chosen_population=chosen_population, seed=None)
    else:
        target_set = generate_target_set(10000, sample_range=(0, MAX_POSSIBLE_VALUE), seed=None)

    print(f'compute signature on target set')
    sig = compute_hashfunc_minhash_signature(target_set, hash_list)

    print(f'sig {len(sig)} {sig}')
    print(f'Target set:\n{sorted(target_set)}')

    ## Find all preimages:
    if recover_all_preimages:
        minhash_to_vocab_list = precompute_hashes_of_vocab(hash_list, set(chosen_population))
        in_set, could_be_in_set = recover_set_from_minhash_sig(sig, minhash_to_vocab_list)
        print(f'Recovered preimages:\n{sorted(in_set)}')
        # print(f'Possibly in set: {utils.pretty_format(could_be_in_set, ppindent=1, ppwidth=140, ppcompact=True)}')
        print(f'Recovered from target set: {len(in_set)}\nPossible additional inputs: {len(could_be_in_set)}\nNot in set: {MAX_POSSIBLE_VALUE - len(in_set) - len(could_be_in_set)}')
        # n=10 (hash functions) m=20 (target set size) would eliminate around n/m elements ?

    ## Heuristic based
    if heuristic:
        heur_recovered_potential_set = recover_set_not_contradicting_minhash_sig(sig, hash_list, MAX_POSSIBLE_VALUE)
        for i in range(5):
            # update is union in-place
            heur_recovered_potential_set.update(recover_set_not_contradicting_minhash_sig(sig, hash_list, MAX_POSSIBLE_VALUE))
        verify_recovered = compute_hashfunc_minhash_signature(heur_recovered_potential_set, hash_list)
        print(f'Heuristic ONLY potential verification {verify_recovered==sig} {verify_recovered} == {sig}\n{heur_recovered_potential_set} ')
        print(f'No hash lower than minhash: {all([h >= minhash for h, minhash in zip(verify_recovered, sig)])}')

        heur_recovered_target_set = recover_target_minhash_set(sig, hash_list, MAX_POSSIBLE_VALUE)
        verify_recovered_target = compute_hashfunc_minhash_signature(heur_recovered_target_set, hash_list)
        print(f'Heuristic verification {verify_recovered_target==sig} [recovered sig] {verify_recovered_target} == {sig} [real sig]\n{heur_recovered_target_set}')
        # Add target and recovered
        heur_recovered_potential_set = heur_recovered_potential_set.union(heur_recovered_target_set)
        verify_recovered = compute_hashfunc_minhash_signature(heur_recovered_potential_set, hash_list)
        print(f'Heuristic target+potential verification {verify_recovered==sig} {verify_recovered} == {sig}\n{heur_recovered_potential_set} ')

    if minhash_hierarchy:
        '''
        In current setup here assume that a number in range [0,n] is an id for a vehicle (mobile entity)
        And the input set is built by sampling vehicle id from this range
        It may be similar to what used in given paper's code, where they may have use a list of such ids for checkpoint signatures
        '''
        # Paper test: n=29_639 trajectories, checkpoints m in [0, 5000], hash k in [10,50,100,200]
        # n >> m
        n = 30000 # int(1e6) # takes around 30 min ?
        k = 200 # 1000 # Hash counts
        c_vocab = range(n+1) # +1 since generate_target_set sampling include both endpoints
        target_set_size = 3000
        print(f'Params: {n} elements, {k} hash function, {len(c_vocab)} vocab elements')
        # Generate target set
        crypto_target_set = generate_target_set(target_set_size, sample_range=(0, n), seed=None)
        print(f'Target set:\n{sorted(crypto_target_set)}')
        # Generate hash functions
        crypto_hash_list = generate_cryptographic_hash_list(k)
        # Reuse previous function that computes signature
        c_sig = compute_hashfunc_minhash_signature(crypto_target_set, crypto_hash_list)
        print(f'c_sig {len(sig)}: {sig}')
        with Timer(name=f'Minhash hierarchy', text=f"Preimage attack time {{:.5f}}", logger=print):
            element_not_in_target, element_in_target = preimages_of_minhash_sig(c_sig, crypto_hash_list, c_vocab)
        print(f'in target {len(element_in_target)}: {element_in_target}')
        print(f'not in target {len(element_not_in_target)}: {element_not_in_target}')
        print(f'Eliminated %: {len(element_not_in_target)}/{n}={len(element_not_in_target)/n}')

        # Verification:
        verify = False
        if verify:
            if not verify_valid_preimage(c_sig, element_in_target, crypto_hash_list, verbose=0):
                raise Exception(f'Invalid preimage found')


            # add each element that should not be in target and check signature:
            for elem in tqdm(element_not_in_target):
                new_set = set()
                new_set.update(element_in_target)
                new_set.add(elem)
                verif = verify_valid_preimage(c_sig, new_set, crypto_hash_list, verbose=0)
                if verif:
                    print(f'ERROR elem should not be in set of elements that cannot be in target')

    print(f'End')