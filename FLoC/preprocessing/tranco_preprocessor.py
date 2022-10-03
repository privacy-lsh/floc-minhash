from FLoC.chromium_components import cityhash

def extract_top_n(n, filepath):
    top_n_domains = []
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        # f.readlines(), readline(), read()
        for line in f: # also reads line from file
            splitted = line.split(',')
            rank, domain = int(splitted[0]), splitted[1].rstrip() # remove trailing \n for cityhash
            # for now do not need rank, its given as the index in the list too
            top_n_domains.append(domain)
            count += 1
            if count >= n:
                break
    return top_n_domains

def precompute_cityhash(domain_list):
    cityhash_lookup = dict() # for hash table in python can use dict
    hash_for_domain = dict()
    # can save in a file the hashes and then load them instead of recomputing cityhash
    for domain in domain_list:
        hash = cityhash.hash64(domain) # should be an int (64 bits ?)
        if hash in cityhash_lookup:
            print(f'Found collision: {domain} and {cityhash_lookup[hash]} give {hash}')
        cityhash_lookup[hash] = domain # update hash table ?
        hash_for_domain[domain] = hash

    return cityhash_lookup, hash_for_domain

if __name__ == '__main__':
    n = 1_000_000
    top_domains = extract_top_n(n, f'../data/tranco_NLKW.csv')
    cityhash_lookup = precompute_cityhash(top_domains)