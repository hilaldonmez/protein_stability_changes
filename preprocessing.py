

# 1615 mutations
# nested list
# the last element for each mutation is a inner list
# the last element for each mutation has the six attributes which are name, orignal residue, position, substitute
#  residue, SA, ph value, temperature, energy change
# Question : name == PDB code ? , what is mutation in the dataset?
def read_dataset(dataset_path):
    all_mutations = []
    with open(dataset_path, "r") as f:
        count = 0
        inner_array = []
        negatives = 0
        positives = 0
        for line in f:
            if not (line.startswith("#") or len(line.strip()) == 0):
                line = line.replace("\n", '')
                if count == 8:
                    properties = line.split(" ")
                    properties[2] = int(properties[2])
                    properties[4] = float(properties[4]) / 100
                    properties[5] = float(properties[5]) / 10
                    properties[6] = float(properties[6]) / 100
                    properties[7] = float(properties[7])
                    # -1 -> negative examples , 1 -> positive examples
                    if properties[7] < 0:
                        properties.append(0)
                        negatives += 1
                    else:
                        properties.append(1)
                        positives += 1

                    inner_array.append(properties)
                    all_mutations.append(inner_array)
                    inner_array = []
                    count = 0
                elif count != 9:
                    inner_array.append(line)
                    count = count + 1

        print(negatives)
        print(positives)

    return all_mutations


# %%
# generate s1496 dataset after removing duplicate mutations
def generate_dataset(all_mutations):
    remove_index = []
    for i in range(len(all_mutations)):
        for j in range(i + 1, len(all_mutations)):
            if (all_mutations[i][-1][5] == all_mutations[j][-1][5]) and (
                    all_mutations[i][-1][6] == all_mutations[j][-1][6]) and (
                    all_mutations[i][-1][0] == all_mutations[j][-1][0]) and (
                    all_mutations[i][-1][1] == all_mutations[j][-1][1]) and (
                    all_mutations[i][-1][2] == all_mutations[j][-1][2]) and (
                    all_mutations[i][-1][3] == all_mutations[j][-1][3]) and (
                    all_mutations[i][-1][4] == all_mutations[j][-1][4]):
                remove_index.append(j)

    remove_index = list(set(remove_index))
    for index in sorted(remove_index, reverse=True):
        del all_mutations[index]
    return all_mutations


# %%
def get_label(mutations):
    return [i[-1][-1] for i in mutations]
