import nltk
import random
from scipy import stats
import pickle
# from tqdm import tqdm

vocab_file = 'save_ml25m/LeakGAN_ml25m_vocab_5000_32.pkl'
word_ml25m, vocab_ml25m = pickle.load(open(vocab_file, mode='rb'))

pad = vocab_ml25m['<pad>']
oov =  vocab_ml25m['<oov>']
print(pad, oov)

reference_file = 'save_ml25m/realtest_ml25m_5000_32.txt' # Test histories never used during training
hypothesis_file_leakgan = 'save_ml25m/generator_sample.txt'
# hypothesis_file_leakgan = 'save_ml25m/ml25m_41.txt'
#################################################
reference = []
with open(reference_file) as fin:
    for line in fin:
        candidate = []
        line = line.split() # Note: default sep=None sep=whitespace empty string removed
        for i in line:
            # Note: do not want that in our case cause we can have blank space inbetween ?
            #  actually do this cause knows that can have random word appearing after the end of sentence dot or ?
            if i == str(pad):
                # break
                continue # want to go to next element in history
            candidate.append(i)

        reference.append(candidate)
#################################################
hypothesis_list_leakgan = []
with open(hypothesis_file_leakgan) as fin:
    for line in fin:
        line = line.split()
        # Note: might want to also remove other special tag which is 1 (Out of vocabulary tag)
        while line[-1] == str(pad):
            line.remove(str(pad))
        # by training data construction if present oov appears only once followed by padding
        if line[-1] == str(oov):
            line.remove(str(oov))
        hypothesis_list_leakgan.append(line)
#################################################
#################################################
random.shuffle(hypothesis_list_leakgan)
#################################################

for ngram in range(2,6):
    weight = tuple((1. / ngram for _ in range(ngram)))
    bleu_leakgan = []
    bleu_supervise = []
    bleu_base2 = []
    num = 0
    for h in hypothesis_list_leakgan[:2000]:
        BLEUscore = nltk.translate.bleu_score.sentence_bleu(reference, h, weight)
        # print (num, BLEUscore)
        num += 1
        bleu_leakgan.append(BLEUscore)
    print('leakgan')
    print(len(weight), '-gram BLEU score : ', 1.0 * sum(bleu_leakgan) / len(bleu_leakgan))

pickle.dump([hypothesis_list_leakgan], open('save_ml25m/significance_test_sample.pkl', 'wb'))
