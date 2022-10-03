import os
import random
import pickle

# Taken and adapted from https://github.com/shizhediao/TILGAN/blob/main/unconditional_generation/utils.py

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        # If use sentence of 20 words (movies) every histories will have that so no need for padding ?
        # don't need this for movie make sentence longer for no reason do not care about order
        self.word2idx['<pad>'] = 0 # TAG for padding when sequence length is not filled with words
        # self.word2idx['<sos>'] = 1
        # self.word2idx['<eos>'] = 2
        self.word2idx['<oov>'] = 1 # 3 TAG for when do not have the word (movie) due to vocab pruning (oov= out of vocab.)
        # so when see in train or test a 1 followed by 0s know that could not fill with movie in vocab (min movie history len = 20 ?)
        self.wordcounts = {}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
        else:
            self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5, cnt=False):
        # get all words and their respective counts
        vocab_list = [(word, count) for word, count in self.wordcounts.items()]
        if cnt:
            # prune by count # Namely does not have an expected output vocabulary size ?
            # self.pruned_vocab = \
            #         {pair[0]: pair[1] for pair in vocab_list if pair[1] > k}
            self.pruned_vocab = \
                    [pair[0] for pair in vocab_list if pair[1] > k]
        else:
            # prune by most frequently seen words
            vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]
         # (as idx are assign first to most frequent movies like this)
        # so will sort by movieID in increasing order ?
        #  here could add same sorting done in movielens_extractor ?

        # sort to make vocabulary determistic
        # Note: How would it not be deterministic given that vocab_list was sorted before
        # Maybe this is only needed if use cnt=True ? So could move it to corresponding if branch ?
        self.pruned_vocab.sort()

        # add all chosen words to new vocabulary/dict
        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        print("original vocab {}; pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)


# One used in TILGAN source code
class Corpus(object):
    def __init__(self, path, maxlen, vocab_size=11000, lowercase=False):
        self.dictionary = Dictionary()
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.train_path = os.path.join(path, 'train.txt')
        self.test_path = os.path.join(path, 'test.txt')

        # make the vocabulary from training set
        self.make_vocab()

        self.train = self.tokenize(self.train_path)
        self.test = self.tokenize(self.test_path)

    def make_vocab(self):
        assert os.path.exists(self.train_path)
        # Add words to the dictionary
        with open(self.train_path, 'r') as f:
            for line in f:
                if self.lowercase:
                    # -1 to get rid of \n character # could use rstrip() too ?
                    words = line[:-1].lower().split(" ")
                else:
                    words = line[:-1].split(" ")
                for word in words:
                    self.dictionary.add_word(word)

        # prune the vocabulary
        self.dictionary.prune_vocab(k=self.vocab_size, cnt=True) # As cnt=True just filter cnt > vocab_size

    def tokenize(self, path):
        """Tokenizes a text file."""
        dropped = 0
        with open(path, 'r') as f:
            linecount = 0
            lines = []
            for line in f:
                linecount += 1
                if self.lowercase:
                    words = line[:-1].lower().strip().split(" ")
                else:
                    words = line[:-1].strip().split(" ")
                if len(words) > self.maxlen: # Note: do not want to drop all those with size > max len
                    dropped += 1
                    continue
                words = ['<sos>'] + words
                words += ['<eos>']
                # vectorize
                vocab = self.dictionary.word2idx # Note: why reassigned at every iteration ?
                unk_idx = vocab['<oov>']
                indices = [vocab[w] if w in vocab else unk_idx for w in words]
                lines.append(indices)

        print("Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return lines


class MovieLensCorpus(object):
    def __init__(self, vocab_size=11000, maxlen=20, train_size=120_000, test_size=5_000, seed=1337,
                 data_path='../data/ml-25m/', tilgan_gen=False, lowercase=False):
        # vocab_size has a different meaning if count=True for dictionary.prune_vocab where it removes count lower than vocab_size
        self.my_dictionary = Dictionary()  # For my filtering and usage with LeakGAN
        self.old_dictionary = Dictionary()  # For compatibility with TILGAN
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.train_size = train_size
        self.test_size = test_size
        self.seed = seed
        self.movie_path = os.path.join(data_path, 'movies.csv')
        self.rating_path = os.path.join(data_path, 'ratings.csv')

        print(f'Parsing data...')
        self.my_train_idx, self.my_test_idx, train_words, test_words = self.parse_data()

        # For LeakGAN compatibility
        # save the vocab for LeakGAN
        print(f'saving data for LeakGAN...')
        self.dump_to_file(f'{data_path}LeakGAN_ml25m_vocab_{vocab_size}_{maxlen}.pkl')
        # This file should directly used my_dictionary vocab encoding according (sorted by most frequent words ?)
        self.write_to_file(self.my_train_idx, f'{data_path}realtrain_ml25m_{vocab_size}_{maxlen}.txt', use_vocab=False)
        # Note: this file is only used for eval_bleu.py
        self.write_to_file(self.my_test_idx, f'{data_path}realtest_ml25m_{vocab_size}_{maxlen}.txt', use_vocab=False)

        if tilgan_gen:
            print(f'generating file for TILGAN...')
            # Generate the file needed
            # This one should directly use movie ids
            self.write_to_file(train_words, f'{data_path}train.txt', use_vocab=False)
            self.write_to_file(test_words, f'{data_path}test.txt', use_vocab=False)

            # reuse original code to parse for tests
            self.train_path = os.path.join(data_path, 'train.txt')
            self.test_path = os.path.join(data_path, 'test.txt')

            # make the vocabulary from training set
            self.make_vocab()

            print('Tokenizing the data...')
            self.train = self.tokenize(self.train_path)
            self.test = self.tokenize(self.test_path)

    def parse_data(self):
        # Import here due to import error maybe caused by circular dependencies (ie movielens_extractor import Dictionary for tests) ?
        # [1] https://stackoverflow.com/questions/9252543/importerror-cannot-import-name-x
        from FLoC.preprocessing.movielens_extractor import read_ratings, read_movies
        movie_histories = read_ratings(self.rating_path)
        # id_to_title, id_to_genres, id_to_year, movie_id_list = read_movies(self.movie_path)

        # Parse all histories then randomly shuffle everything inside ?
        # 25_000_095 ratings; 162_541 users; 62_423 movies
        # One user has 32202 ratings closest below is at 9178 maybe and not a lot are above 5000
        for history in movie_histories:
            for movie_id in history:
                self.my_dictionary.add_word(movie_id)

        # Prune vocabulary
        # Note: if want to use different filtering eg cnt > 0 need cnt=True and vocab_size=0
        self.my_dictionary.prune_vocab(k=self.vocab_size, cnt=False)

        # Go over shuffled histories again and this time build train/test data
        # seed set for all future call to random ?
        # Note: may have conflicts if already set different seed in train.py e.g. for TILGAN ?
        random.seed(self.seed)
        random.shuffle(movie_histories) # done in place use random.sample() otherwise ?

        # May want to do vocab pruning after this
        vocab = self.my_dictionary.word2idx
        UNK_IDX = vocab['<oov>']
        PAD_IDX = vocab['<pad>']
        out_filtered_lines_idx = [[PAD_IDX] * self.maxlen for _ in range(self.train_size+self.test_size)]
        out_filtered_lines_word = []
        for line_count, history in enumerate(movie_histories, start=0):
            if line_count >= (self.train_size + self.test_size):
                break  # we have enough sample

            # done in place, not using sample cause want to take other movies if some not in vocab
            random.shuffle(history)

            # If uses words only with no dict yet (for TILGAN for example)
            # could do that but then vocab would be bigger and different between training of leakgan and TILGAN
            # out_filtered_lines_word.append(history[:self.maxlen])
            cur_words_builder = []

            # if try to fill data at idx with next word if in vocab if previous were not
            cur_id_in_line = 0
            for movie_id in history:
                if cur_id_in_line >= self.maxlen:
                    break
                if movie_id in vocab:
                    out_filtered_lines_idx[line_count][cur_id_in_line] = vocab[movie_id]
                    cur_id_in_line += 1
                    # case with words
                    cur_words_builder.append(movie_id)
                else:
                    out_filtered_lines_idx[line_count][cur_id_in_line] = UNK_IDX
                    # Do not increment cur_id_in_line as want to replace UNK_IDX with following movie_id in vocab if possible

            # case with words:
            out_filtered_lines_word.append(cur_words_builder)

        # Split train / test
        train_idx, test_idx = out_filtered_lines_idx[:-self.test_size], out_filtered_lines_idx[-self.test_size:]
        train_words, test_words = out_filtered_lines_word[:-self.test_size], out_filtered_lines_word[-self.test_size:]
        return train_idx, test_idx, train_words, test_words

    def write_to_file(self, data_to_save, save_path, use_vocab):
        # Write training data
        with open(save_path, 'w') as fout:
            for line in data_to_save:
                # remove left and right blank space and add newline
                if use_vocab:
                    to_write = ' '.join([str(self.my_dictionary.word2idx[token]) for token in line]).strip() + '\n'
                else:
                    to_write = ' '.join([str(token) for token in line]).strip() + '\n'
                fout.write(to_write)

    def dump_to_file(self, save_path):
        with open(save_path, 'wb') as f:
            # pickle.dump((word, vocab), f, protocol=pickle.HIGHEST_PROTOCOL)
            # as not compatible with LeakGAN change protocol (default None)
            pickle.dump((self.my_dictionary.idx2word, self.my_dictionary.word2idx), f)

    def make_vocab(self):
        assert os.path.exists(self.train_path)
        # Add words to the dictionary
        with open(self.train_path, 'r') as f:
            for line in f:
                if self.lowercase:
                    # -1 to get rid of \n character # could use rstrip() too ?
                    words = line[:-1].lower().split(" ")
                else:
                    words = line[:-1].split(" ")
                for word in words:
                    self.old_dictionary.add_word(word)

        # prune the vocabulary
        # need to change that line with current training data as otherwise vocab too big if remove cnt > 0
        # self.old_dictionary.prune_vocab(k=self.vocab_size, cnt=True) # As cnt=True just filter cnt > vocab_size
        self.old_dictionary.prune_vocab(k=self.vocab_size, cnt=False) # As cnt=True just filter cnt > vocab_size

    def tokenize(self, path):
        """Tokenizes a text file."""
        dropped = 0
        with open(path, 'r') as f:
            linecount = 0
            lines = []
            for line in f:
                linecount += 1
                if self.lowercase:
                    words = line[:-1].lower().strip().split(" ")
                else:
                    words = line[:-1].strip().split(" ")
                if len(words) > self.maxlen: # Note: do not want to drop all those with size > max len
                    dropped += 1
                    continue
                # Note: if left like this both start of sentence and end of sentence token will be oov index
                # words = ['<sos>'] + words
                # words += ['<eos>']
                # vectorize
                vocab = self.old_dictionary.word2idx  # Note: why reassigned at every iteration ?
                unk_idx = vocab['<oov>']
                indices = [vocab[w] if w in vocab else unk_idx for w in words]
                lines.append(indices)

        print("Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return lines


if __name__ == '__main__':
    # can check [1] for a usage of corpus
    # [1] https://github.com/shizhediao/TILGAN/blob/main/unconditional_generation/train.py
    data_path = f'../data/ml-25m/' # need train.txt and test.txt
    # corpus = Corpus(data_path, maxlen=20, vocab_size=11000, lowercase=False)
    ml25m_corpus = MovieLensCorpus(data_path=data_path, maxlen=32, vocab_size=5000, tilgan_gen=False, lowercase=False)