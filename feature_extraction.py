import numpy as np
from Bio.SubsMat import MatrixInfo

pam250 = MatrixInfo.pam250
b62 = MatrixInfo.blosum62


# %%

def generate_dic(bio_dict):
    d = {}
    for key in bio_dict:
        d[(key[0], key[1])] = bio_dict.get(key)
        d[(key[1], key[0])] = bio_dict.get(key)
    return d


# %%
# generate a vector with 20 input for each mutations
# return vectors for each mutations 
# the row which corresponds to a vector for a mutation
# column which corresponds to an aminoacid id    
def generate_mutation_info(mutations, aa_dict, len_aa):
    len_mutations = len(mutations)
    mut_info = np.zeros((len_mutations, len_aa))
    for index, mut in zip(range(len_mutations), mutations):
        original = aa_dict[mut[-1][1]]
        new = aa_dict[mut[-1][3]]
        mut_info[index][original] = -1
        mut_info[index][new] = 1
    return mut_info


# %%
# add neighbor vectors 
# add mutation info vector
# add pam250 and blosu62 vector at the end
# create a vector for each mutation     
def generate_SO_vector(mutations, mutation_info, aa_dict, window_size, len_aa, bio_dic, expand=False):
    extention = int(window_size / 2)
    SO_vectors = []

    for mut in mutations:
        temp_mut_seq = mut[2]  # sequence of the mutations
        temp_position = mut[-1][2] - 1  # the position of the mutations
        original = mut[-1][1]
        substitue = mut[-1][3]
        temp = mut[-1][5]
        pH = mut[-1][6]

        neighbors = temp_mut_seq[(temp_position - extention):(temp_position + extention + 1)]
        right_vector = np.zeros(extention * len_aa)
        left_vector = np.zeros(extention * len_aa)

        for i in range(len(neighbors[:extention])):
            aa_id = aa_dict[neighbors[i]]
            right_vector[len_aa * i + aa_id] = 1

        for i in range(len(neighbors[(extention + 1):])):
            aa_id = aa_dict[neighbors[i]]
            left_vector[len_aa * i + aa_id] = 1

        if expand:
            bio_score = bio_dic[(original, substitue)]
            SO_vectors.append(np.hstack((right_vector, mutation_info[i], left_vector, [temp, pH, bio_score])).ravel())
        else:
            SO_vectors.append(np.hstack((right_vector, mutation_info[i], left_vector, [temp, pH])).ravel())

    return SO_vectors
