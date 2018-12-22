import pandas as pd
from nltk import trigrams

dataframe = pd.read_csv('./data/pdb_data_seq.csv')
dataframe = dataframe.loc[dataframe['macromoleculeType'] == 'Protein']
all_sequences = dataframe['sequence'].dropna().tolist()
one_line_seq = ''.join(all_sequences)
n = 3

n_grams = trigrams(one_line_seq)

with open('n_gram_model.txt', 'a') as file:
    for gram in n_grams:
        for g in gram:
            file.write(g)
        file.write('\n')
