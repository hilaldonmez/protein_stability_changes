import numpy as np
from Bio.SubsMat import MatrixInfo
from gensim.models import Word2Vec

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


def get_left_right(aa_dict, mut_seq, position, extension):
    len_aa = len(aa_dict)
    len_mut_seq = len(mut_seq)

    neighbors = mut_seq[(position - extension) if (position - extension) > 0 else 0:(position + extension + 1) if (position + extension + 1) <= len_mut_seq else len_mut_seq]
    right_vector = np.zeros(extension * len_aa)
    left_vector = np.zeros(extension * len_aa)

    for i in range(len(neighbors[:extension])):
        aa_id = aa_dict[neighbors[i]]
        right_vector[len_aa * i + aa_id] = 1

    for i in range(len(neighbors[(extension + 1):])):
        aa_id = aa_dict[neighbors[i]]
        left_vector[len_aa * i + aa_id] = 1

    return left_vector, right_vector


def get_SO_vector(row, extension, aa_dict, expand, bio_dic):
    mut_seq = row['sequence']  # sequence of the mutations
    position = row['position'] - 1  # the position of the mutations
    original = row['original_residue']
    substitute = row['substitute_residue']
    temp = row['temperature']
    pH = row['ph_value']

    left_vector, right_vector = get_left_right(aa_dict, mut_seq, position, extension)

    if expand:
        bio_score = bio_dic[(original, substitute)]
        return np.hstack((right_vector, row['mutation_info'], left_vector, [temp, pH, bio_score])).ravel()
    else:
        return np.hstack((right_vector, row['mutation_info'], left_vector, [temp, pH])).ravel()


# %%
# add neighbor vectors 
# add mutation info vector
# add pam250 and blosu62 vector at the end
# create a vector for each mutation     
def generate_SO_vector(mutations, aa_dict, window_size, expand, bio_dic):
    extension = int(window_size / 2)

    SO_vectors = mutations.apply(
        lambda row: get_SO_vector(row, extension, aa_dict, expand, bio_dic), axis=1)
    # print(SO_vectors.tolist()[0])
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


def get_ST_vector(row, extension, aa_dict, expand, bio_dic):
    sa = row['SA']
    temp = row['temperature']
    pH = row['ph_value']
    ca_coordinates = row['ca_coordinates']
    position = row['position'] - 1
    sequence = row['sequence']
    original = row['original_residue']
    substitute = row['substitute_residue']

    coord_of_target = np.array(ca_coordinates[position])
    distances = [np.linalg.norm(coord_of_target - np.array(x)) for x in ca_coordinates]
    TO_vector = np.zeros(len(aa_dict))

    left_vector, right_vector = get_left_right(aa_dict, sequence, position, extension)

    for i in range(len(distances)):
        if distances[i] < 9 and sequence[i] != 'X':
            TO_vector[aa_dict[sequence[i]]] += 1
    if expand:
        bio_score = bio_dic[(original, substitute)]
        return np.hstack((right_vector, row['mutation_info'], left_vector, TO_vector, [temp, pH, sa, bio_score])).ravel()
    else:
        return np.hstack((right_vector, row['mutation_info'], left_vector, TO_vector, [temp, pH, sa])).ravel()


def generate_ST_vector(mutations, aa_dict, window_size, expand, bio_dic):
    extension = int(window_size / 2)

    to_mutations = mutations.dropna()
    TO_vectors = to_mutations.apply(lambda row: get_ST_vector(row, extension, aa_dict, expand, bio_dic), axis=1)
    return TO_vectors.tolist()


def get_node_vector(row, aa_dict, extension, nodeVec):
    sa = row['SA']
    temp = row['temperature']
    pH = row['ph_value']
    position = row['position'] - 1
    mut_seq = row['sequence']  # sequence of the mutations
    substitute = row['substitute_residue']

    len_mut_seq = len(mut_seq)

    neighbors = mut_seq[(position - extension) if (position - extension) > 0 else 0:(position + extension + 1) if (position + extension + 1) <= len_mut_seq else len_mut_seq]
    substitute_id = aa_dict[substitute]
    node_vector = nodeVec[substitute_id].tolist()

    for i in range(7):
        if len(neighbors) > i:
            aa_id = aa_dict[neighbors[i]]
            node_vector = np.hstack((node_vector, nodeVec[aa_id])).ravel()
        else:
            node_vector = np.hstack((node_vector, np.zeros(128))).ravel()

    return np.array(np.hstack((node_vector, row['mutation_info'], [temp, pH, sa])).ravel())


def generate_node_vector(mutations, aa_dict, window_size, nodeVec):
    extension = int(window_size / 2)
    node_vectors = mutations.apply(lambda row: get_node_vector(row, aa_dict, extension, nodeVec), axis=1)
    return node_vectors.tolist()


def get_word2vec_vector(row, extension, model, expand, bio_dic):
    sa = row['SA']
    temp = row['temperature']
    pH = row['ph_value']
    position = row['position'] - 1
    mut_seq = row['sequence']  # sequence of the mutations
    original = row['original_residue']
    substitute = row['substitute_residue']

    left = mut_seq[position - extension:position]
    right = mut_seq[position:position + extension]
    left_vec = np.zeros(200) if left not in model else model[left]
    right_vec = np.zeros(200) if right not in model else model[right]

    if expand:
        bio_score = bio_dic[(original, substitute)]
        return np.array(np.hstack((row['mutation_info'], left_vec, right_vec, [temp, pH, sa, bio_score])).ravel())
    else:
        return np.array(np.hstack((row['mutation_info'], left_vec, right_vec, [temp, pH, sa])).ravel())


def generate_word2vec_vector(mutations, window_size, expand, bio_dic):
    extension = int(window_size / 2)
    model = Word2Vec.load("./data/word2vec_model")
    word2vec_vectors = mutations.apply(lambda row: get_word2vec_vector(row, extension, model, expand, bio_dic), axis=1)
    return word2vec_vectors.tolist()
