import os
import numpy as np
import preprocessing as pr
import feature_extraction as fe
import classification as clf

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
            mutations = pr.generate_dataset(mutations.copy())

        len_aa = len(aa_dict)

        mutation_info = fe.generate_mutation_info(mutations, aa_dict, len_aa)
        SO_vectors_original = fe.generate_SO_vector(mutations, mutation_info, aa_dict, window_size,
                                                    len_aa, bio_dic=None)

        pam250 = fe.generate_dic(fe.pam250)
        b62 = fe.generate_dic(fe.b62)

        SO_vectors_pam250 = fe.generate_SO_vector(mutations, mutation_info, aa_dict, window_size,
                                                  len_aa, pam250, True)
        SO_vectors_b62 = fe.generate_SO_vector(mutations, mutation_info, aa_dict, window_size,
                                               len_aa, b62, True)

        y = np.array(pr.get_label(mutations))
        X = np.array(SO_vectors_original)
        clf.train_svm(X, y, 20)
        clf.train_random_forest(X, y, 20)


if __name__ == '__main__':
    main()
