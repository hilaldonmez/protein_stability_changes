import pandas as pd
from nltk import trigrams
import pickle

dataframe = pd.read_csv('./data/pdb_data_seq.csv')
dataframe = dataframe.loc[dataframe['macromoleculeType'] == 'Protein']
all_sequences = dataframe['sequence'].dropna().tolist()

n = 3

n_grams_list = [list(trigrams(x)) for x in all_sequences]
n_grams_list = [[y[0]+y[1]+y[2] for y in x] for x in n_grams_list]

outfile = open('n_gram_model', 'wb')
pickle.dump(list(n_grams_list), outfile)
outfile.close()
