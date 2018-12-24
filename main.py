import os
import numpy as np
import preprocessing as pr
import feature_extraction as fe
import classification as clf
import node2vec as nv

data_path = './data/'
aa_dict = {'C': 0, 'Q': 1, 'V': 2, 'K': 3, 'E': 4, 'L': 5, 'Y': 6, 'G': 7, 'S': 8, 'F': 9, 'T': 10, 'I': 11, 'M': 12,
           'W': 13, 'D': 14, 'H': 15, 'N': 16, 'P': 17, 'R': 18, 'A': 19}


def main():
    # %%
    window_size = 7

    for file in os.listdir(data_path):
        if not file == 's1615.txt':
            continue

        dataset_path = data_path + file
        mutations = pr.read_dataset(dataset_path)

        if file == 's1615.txt':
            mutations = pr.generate_dataset(mutations)

        fe.generate_mutation_info(mutations, aa_dict)
        SO_vectors_original = fe.generate_SO_vector(mutations, aa_dict, window_size, False, None)

        pam250 = fe.generate_dic(fe.pam250)
        b62 = fe.generate_dic(fe.b62)

        SO_vectors_pam250 = fe.generate_SO_vector(mutations, aa_dict, window_size, True, pam250)
        SO_vectors_b62 = fe.generate_SO_vector(mutations, aa_dict, window_size, True, b62)

        input_file = "vec_all_blosum.txt"
        node_vectors_blosum = nv.generate_node_vector(input_file)
        SO_vectors_node_blosum = fe.generate_node_vector(mutations, aa_dict, window_size, node_vectors_blosum)
        input_file = "vec_all_pam.txt"
        node_vectors_pam = nv.generate_node_vector(input_file)
        SO_vectors_node_pam = fe.generate_node_vector(mutations, aa_dict, window_size, node_vectors_pam)

        TO_vectors = fe.generate_TO_vector(mutations, aa_dict)
        ST_vectors = fe.generate_ST_vector(mutations, aa_dict, window_size, False, None)
        ST_vectors_pam = fe.generate_ST_vector(mutations, aa_dict, window_size, True, pam250)
        ST_vectors_blosum = fe.generate_ST_vector(mutations, aa_dict, window_size, True, b62)

        word2vec_vectors = fe.generate_word2vec_vector(mutations, window_size, False, None)
        word2vec_vectors_pam = fe.generate_word2vec_vector(mutations, window_size, True, pam250)
        word2vec_vectors_blosum = fe.generate_word2vec_vector(mutations, window_size, False, b62)

        # y = np.array(pr.get_label(mutations))
        # X = np.array(SO_vectors_node_blosum)
        # print('Node Vector Blosum')
        # clf.train_svm(X, y, 20)
        # clf.train_random_forest(X, y, 20)
        #
        # y = np.array(pr.get_label(mutations))
        # X = np.array(SO_vectors_original)
        # print('SO')
        # clf.train_svm(X, y, 20)
        # clf.train_random_forest(X, y, 20)

        y = np.array(pr.get_label(mutations.copy().dropna()))
        X = np.array(TO_vectors)
        print('TO')
        clf.train_svm(X, y, 20)
        clf.train_random_forest(X, y, 20)

        # y = np.array(pr.get_label(mutations.copy().dropna()))
        # X = np.array(ST_vectors)
        # print('ST')
        # clf.train_svm(X, y, 20)
        # clf.train_random_forest(X, y, 20)
        #
        # y = np.array(pr.get_label(mutations.copy().dropna()))
        # X = np.array(ST_vectors_blosum)
        # print('ST Blosum')
        # clf.train_svm(X, y, 20)
        # clf.train_random_forest(X, y, 20)
        #
        # y = np.array(pr.get_label(mutations.copy().dropna()))
        # X = np.array(ST_vectors_pam)
        # print('ST PAM')
        # clf.train_svm(X, y, 20)
        # clf.train_random_forest(X, y, 20)
        #
        # y = np.array(pr.get_label(mutations))
        # X = np.array(word2vec_vectors_pam)
        # print('word2vec PAM')
        # clf.train_svm(X, y, 20)
        # clf.train_random_forest(X, y, 20)
        #
        # y = np.array(pr.get_label(mutations.copy().dropna()))
        # X = np.array(word2vec_vectors_blosum)
        # print('word2vec Blosum')
        # clf.train_svm(X, y, 20)
        # clf.train_random_forest(X, y, 20)


if __name__ == '__main__':
    main()
