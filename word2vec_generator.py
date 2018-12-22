from gensim.models import Word2Vec
import pickle
import multiprocessing

infile = open('n_gram_model', 'rb')
n_grams_list = pickle.load(infile)
infile.close()

print(n_grams_list)

model = Word2Vec(n_grams_list, size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())

model.save('data/word2vec_model')
