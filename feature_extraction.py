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
def generate_mutation_info(mutations, aa_dict):
    len_mutations = len(mutations)

    mut_info = np.zeros((len_mutations, len(aa_dict)))

    original = mutations['original_residue'].apply(lambda x: aa_dict[x])
    new = mutations['substitute_residue'].apply(lambda x: aa_dict[x])

    mut_info[np.arange(len_mutations), original.tolist()] = -1
    mut_info[np.arange(len_mutations), new.tolist()] = 1
    mutations['mutation_info'] = mut_info.tolist()


def get_SO_vector(row, extension, aa_dict, expand, bio_dic):
    mut_seq = row['sequence']  # sequence of the mutations
    position = row['position'] - 1  # the position of the mutations
    original = row['original_residue']
    substitute = row['substitute_residue']
    sa = row['SA']
    temp = row['temperature']
    pH = row['ph_value']

    len_aa = len(aa_dict)

    neighbors = mut_seq[(position - extension):(position + extension + 1)]
    right_vector = np.zeros(extension * len_aa)
    left_vector = np.zeros(extension * len_aa)

    for i in range(len(neighbors[:extension])):
        aa_id = aa_dict[neighbors[i]]
        right_vector[len_aa * i + aa_id] = 1

    for i in range(len(neighbors[(extension + 1):])):
        aa_id = aa_dict[neighbors[i]]
        left_vector[len_aa * i + aa_id] = 1

    if expand:
        bio_score = bio_dic[(original, substitute)]
        return np.hstack((right_vector, row['mutation_info'], left_vector, [temp, pH, sa, bio_score])).ravel()
    else:
        return np.hstack((right_vector, row['mutation_info'], left_vector, [temp, pH, sa])).ravel()


# %%
# add neighbor vectors 
# add mutation info vector
# add pam250 and blosu62 vector at the end
# create a vector for each mutation     
def generate_SO_vector(mutations, aa_dict, window_size, bio_dic, expand=False):
    extension = int(window_size / 2)

    SO_vectors = mutations.apply(lambda row: get_SO_vector(row, extension, aa_dict, expand, bio_dic), axis=1)
    return SO_vectors.tolist()


def get_TO_vector(row, aa_dict):
    sa = row['SA']
    temp = row['temperature']
    pH = row['ph_value']
    ca_coordinates = row['ca_coordinates']
    position = row['position'] - 1
    sequence = row['sequence']
    coord_of_target = np.array(ca_coordinates[position])
    distances = [np.linalg.norm(coord_of_target - np.array(x)) for x in ca_coordinates]
    TO_vector = np.zeros(len(aa_dict))

    for i in range(len(distances)):
        if distances[i] < 9 and sequence[i] != 'X':
            TO_vector[aa_dict[sequence[i]]] += 1
    return np.hstack((TO_vector, row['mutation_info'], [temp, pH, sa])).ravel()


def generate_TO_vector(mutations, aa_dict):
    to_mutations = mutations.dropna()
    TO_vectors = to_mutations.apply(lambda row: get_TO_vector(row, aa_dict), axis=1)
    return TO_vectors.tolist()
