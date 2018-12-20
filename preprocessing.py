import numpy as np
s1615_dataset_path = "./data/s1615.txt"
s388_dataset_path = "./data/s388.txt"

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
               line = line.replace("\n", '')
               if count == 8 :
                   properties = line.split(" ")
                   properties[2] = int(properties[2]) 
                   properties[4] = float(properties[4]) / 100
                   properties[5] = float(properties[5]) / 10
                   properties[6] = float(properties[6]) / 100
                   properties[7] = float(properties[7] )    
                   # 0 -> negative examples , 1 -> positive examples
                   if properties[7] < 0:
                       properties.append(0)
                   else:
                       properties.append(1)
                       
                   inner_array.append(properties)                   
                   all_mutations.append(inner_array)
                   inner_array = []
                   count = 0                   
               elif count != 9:
                   inner_array.append(line)
                   count = count + 1

    return all_mutations

#%%
# generate s1496 dataset after removing duplicate mutations
def generate_s1496_dataset(all_mutations):
    remove_index = []
    for i in range(len(all_mutations)):
        for j in range(i+1,len(all_mutations)):
            if (all_mutations[i][-1][5] == all_mutations[j][-1][5]) and (all_mutations[i][-1][6] == all_mutations[j][-1][6]) and (all_mutations[i][-1][0] == all_mutations[j][-1][0]) and (all_mutations[i][-1][1] == all_mutations[j][-1][1]) and (all_mutations[i][-1][2] == all_mutations[j][-1][2]) and (all_mutations[i][-1][3] == all_mutations[j][-1][3]) and (all_mutations[i][-1][4] == all_mutations[j][-1][4]) :            
                remove_index.append(j)
     
    remove_index = list(set(remove_index))    
    for index in sorted(remove_index, reverse=True):
        del all_mutations[index]
    return all_mutations  
      
#%%
# mutation info vectors
# X aminoacid is unimportant or unknown, remove X  
# create dictionary with aminoacid abbreviations and id given
def generate_aa_dict(s1615_mutations):
    aa_dict = set(''.join([i[2] for i in s1615_mutations]))          
    aa_dict.remove('X')
    d = {value: index for value,index in zip(aa_dict,np.arange(len(aa_dict))) }
    return d

#%%
def get_label(mutations):    
    return [i[-1][-1]  for i in mutations]
#%%
s1615_mutations = read_dataset(s1615_dataset_path)
s388_mutations = read_dataset(s388_dataset_path)
s1496_mutations = generate_s1496_dataset(s1615_mutations.copy())    
aa_dict = generate_aa_dict(s1615_mutations)
len_aa = len(aa_dict)
# for now, only deal with s1496 dataset
# further, deal with all dataset prepared
label = get_label(s1496_mutations)




