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
                   properties[2] = int(properties[2]) 
                   properties[4] = float(properties[4]) / 100
                   properties[5] = float(properties[5]) / 10
                   properties[6] = float(properties[6]) / 100
                   properties[7] = float(properties[7] )    
                   # -1 -> negative examples , 0 -> positive examples
                   if properties[7] < 0:
                       properties.append(-1)
                   else:
                       properties.append(0)
                       
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
temp_mutations = all_mutations.copy()
remove_index = []
for i in range(len(temp_mutations)):
    for j in range(i+1,len(temp_mutations)):
        if (temp_mutations[i][-1][5] == temp_mutations[j][-1][5]) and (temp_mutations[i][-1][6] == temp_mutations[j][-1][6]) and (temp_mutations[i][-1][0] == temp_mutations[j][-1][0]) and (temp_mutations[i][-1][1] == temp_mutations[j][-1][1]) and (temp_mutations[i][-1][2] == temp_mutations[j][-1][2]) and (temp_mutations[i][-1][3] == temp_mutations[j][-1][3]) and (temp_mutations[i][-1][4] == temp_mutations[j][-1][4]) :
        
            remove_index.append(j)
 
remove_index = list(set(remove_index))    
    
#%%
 