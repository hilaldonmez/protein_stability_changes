import pandas as pd
import numpy as np


# 1615 mutations
# nested list
# the last element for each mutation is a inner list
# the last element for each mutation has the six attributes which are name, orignal residue, position, substitute
#  residue, SA, ph value, temperature, energy change
# Question : name == PDB code ? , what is mutation in the dataset?
def read_dataset(dataset_path):
    lines = open(dataset_path).read().split('\n')
    lines = [x for x in lines[1:] if not x.startswith('#')]

    all_mutations = []
    for i in range(0, len(lines) - 10, 10):
        mutation = {}
        mutation['sequence'] = lines[i + 3]
        mutation['secondary_structure'] = lines[i + 4]
        mutation['solvent_accessibility'] = [int(x) for x in lines[i + 7].split()]
        mutation['ca_coordinates'] = [[float(y) for y in x.split()] for x in lines[i + 8].split('\t')]
        mutual_info = lines[i + 9].split()
        mutation['name'] = mutual_info[0]
        mutation['original_residue'] = mutual_info[1]
        mutation['position'] = int(mutual_info[2])
        mutation['substitute_residue'] = mutual_info[3]
        mutation['SA'] = float(mutual_info[4])/100
        mutation['ph_value'] = float(mutual_info[5])/10
        mutation['temperature'] = float(mutual_info[6])/100
        mutation['energy_change'] = float(mutual_info[7])
        all_mutations.append(mutation)
    df = pd.DataFrame(data=all_mutations)
    df['label'] = np.where(df['energy_change'] > 0, 1, 0)
    return df


# %%
# generate s1496 dataset after removing duplicate mutations
def generate_dataset(all_mutations):
    all_mutations = all_mutations.drop_duplicates(
        subset=['name', 'original_residue', 'position', 'substitute_residue', 'SA', 'ph_value', 'temperature'])
    return all_mutations


# %%
def get_label(mutations):
    return mutations['label'].tolist()
