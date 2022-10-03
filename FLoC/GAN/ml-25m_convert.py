# import cPickle
# import _pickle as cPickle
import pickle
from tqdm import tqdm, trange
from preprocessing.movielens_extractor import read_movies, encode_genre

## Initialize index to word mapping
# data_Name = "cotra"
# vocab_file = "vocab_" + data_Name + ".pkl"
# # word, vocab = cPickle.load(open('save/'+vocab_file))
# word, vocab = pickle.load(open('save/'+vocab_file, mode='rb')) # Need mode byte for python3
# print(len(word))

# word_ml25m, vocab_ml25m = pickle.load(open('save_ml25m/vocab_ml-25m.pkl', mode='rb')) # Note: the one that generated the data
# word_ml25m, vocab_ml25m = pickle.load(open('save_ml25m/vocab_ml25m.pkl', mode='rb')) # cross compatibility with older python
word_ml25m, vocab_ml25m = pickle.load(open('save_ml25m/LeakGAN_ml25m_vocab_5000_32.pkl', mode='rb')) # other vocab file
print(len(word_ml25m))

# def decode_to_file(input_file, output_file):
#     # Input file generated from Main has 9984 line
#     # input_file = 'save/generator_sample.txt'
#     # input_file = 'save/coco_451.txt'
#     # output_file = 'speech/' + data_Name + '_' + input_file.split('_')[-1]
#     with open(output_file, 'w') as fout:
#         with open(input_file)as fin:
#             count = 0
#             for line in tqdm(fin):
#                 #line.decode('utf-8')
#                 line = line.split()
#                 #line.pop()
#                 #line.pop()
#                 line = [int(x) for x in line]
#                 line = [word[x] for x in line]
#                 # if 'OTHERPAD' not in line:
#                 line = ' '.join(line) + '\n'
#                 count += 1
#                 if count % 100 == 0:
#                     print(line)
#                 fout.write(line)#.encode('utf-8'))

# def decode_to_console(input_file, freq=1):
#     with open(input_file)as fin:
#         count = 0
#         for line in tqdm(fin):
#             #line.decode('utf-8')
#             line = line.split()
#             #line.pop()
#             #line.pop()
#             line = [int(x) for x in line]
#             line = [word[x] for x in line]
#             # if 'OTHERPAD' not in line:
#             line = ' '.join(line) + '\n'
#             count += 1
#             if count % freq == 0: # By default mod 1 always equal 0
#                 print(line) #.encode('utf-8'))


def movie_decode_to_console(input_file, freq=1, movies_filepath=f'../data/ml-25m/movies.csv'):
    print(f"Input file displayed: {input_file}")
    id_to_title, id_to_genres, id_to_year, id_list = read_movies(movies_filepath)
    with open(input_file, 'r', encoding='utf-8') as fin:
        count = 0
        for line in tqdm(fin):
            if count % freq == 0:  # By default mod 1 always equal 0
                print(line, end='') # already \n at end of line
                line = line.split()
                line = [int(x) for x in line]
                line = [word_ml25m[x] for x in line]
                # Note: careful depending on vocab used have special tag to remove like 0,1 or more ?
                line_str = ' '.join([str(movie_id) for movie_id in line]) # + '\n'
                print(line_str)
                for movie_id in line:
                    # movie_id starts from 1, 0 is used for EMPTY Note: is it as expected ?
                    if movie_id != 0 and movie_id != '<oov>' and movie_id != '<pad>':
                        genres, encoded_genres = id_to_genres[movie_id], encode_genre(id_to_genres[movie_id])
                        print(f'{movie_id}: {id_to_title[movie_id]} ({genres}, {id_to_year[movie_id]}) '
                              f'[{encoded_genres}, {encoded_genres:>020b}]',sep=' ', end=';\n')
                print() # blank line
            count += 1

def movie_decode_to_file(input_file, output_file, freq=1, movies_filepath=f'../data/ml-25m/movies.csv'):
    id_to_title, id_to_genres, id_to_year, id_list = read_movies(movies_filepath)
    with open(output_file, 'w', encoding='utf-8') as fout:
        with open(input_file, 'r', encoding='utf-8') as fin:
            count = 0
            for line in tqdm(fin):
                if count % freq == 0:  # By default mod 1 always equal 0
                    fout.write(line) # line already as \n in the end
                    line = line.split() # removes the trailing \n
                    line = [int(x) for x in line]
                    line = [word_ml25m[x] for x in line]
                    line_str = ' '.join([str(movie_id) for movie_id in line]) + '\n'
                    fout.write(line_str)
                    for movie_id in line:
                        # movie_id starts from 1, 0 is used for EMPTY
                        if movie_id != 0 and movie_id != '<oov>' and movie_id != '<pad>':
                            genres, encoded_genres = id_to_genres[movie_id], encode_genre(id_to_genres[movie_id])
                            fout.write(f'{movie_id}: {id_to_title[movie_id]} ({genres}, {id_to_year[movie_id]}) '
                                       f'[{encoded_genres}, {encoded_genres:>020b}]\n')
                    fout.write('\n')  # blank line
                count += 1

if __name__ == '__main__':
    coco_caption = False
    ml25m = True

    if coco_caption:
        # generator_sample is the one used every epoch for training ?
        input_file = 'save/coco_21.txt'
        # output_file = f"speech/{data_Name}_{input_file.split('_')[-1]}"
        # decode_to_file(input_file, output_file)
        # could also decode the other one ie realtest and realtrain ?
        filename = f'coco_11' # coco_101 realtest_coco realtrain_cotra generator_sample
        input_to_console = f'save/{filename}.txt' #
        # decode_to_console(input_to_console, freq=1)

    if ml25m:
        filename = f'generator_sample' # f'ml25m_21+21' generator_sample ml25m_61
        # input_to_print = f'save_ml25m/older/seq_len77_vocab_5847/{filename}.txt'
        input_to_print = f'save_ml25m/{filename}.txt'
        # movie_decode_to_console(input_to_print, freq=1)
        out_file = f'save_ml25m/{filename}_speech.txt' # f'save_ml25m/older/seq_len77_vocab_5847/{filename}_speech.txt'
        movie_decode_to_file(input_to_print, out_file)



