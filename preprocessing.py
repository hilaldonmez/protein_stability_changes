dataset_path = "./data/s1615.txt"

# 1615 mutations
# nested list
# the last element for each mutation is a inner list
# the last element for each mutation has the six attributes which are name, orignal residue, position, substitute residue, SA, ph value, temperature, energy change
# Question : name == PDB code ? , what is mutation in the dataset? 
def read_dataset(dataset_path):
    all_mutations = []
    
    with open(dataset_path, "r") as f:
        count = 0
        inner_array = []
        for line in f:
            if not ((line.startswith("#") or len(line.strip()) == 0))  :
               if count == 8 :
                   properties = line.split(" ")
                   inner_array.append(properties)
                   all_mutations.append(inner_array)
                   inner_array = []
                   count = 0
                   
               elif count != 9:
                   inner_array.append(line)
                   count = count + 1
    return all_mutations

all_mutations = read_dataset(dataset_path)
 
#%%
