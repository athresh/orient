import os
import argparse
def parse_data_file(self, root, data_list_folder, domains, split=None, save=False):
    """Parse file to data list
    Parameters:
        - **data_list_folder** (str): The dir containing the data file
        - **domains** (list of str): List of domains to be read
        - *split* (str) : Name of the split, default: None
        - **return** (list): List of (image path, class_index, domain) tuples
    """
    data_train_list = []
    data_val_list = []
    target_num = {}
    for domain in domains:
        file_name = f"{data_list_folder}/{domain}"
        if split is not None:
            file_name = f"{file_name}_{split}"
        with open(file_name + ".txt", "r") as f:
            for line in f.readlines():
                path, target = line.split()
                if not os.path.isabs(path):
                    path = os.path.join(root, domain + "/images/" + path)
                if target in target_num.keys() and target_num[target]<:
                    data_train_list.append((path, int(target)))
    if save:
        with open('your_file.txt', 'w') as f:
            for item in data_train_list:
                f.write("%s\n" % item)
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str)
    args = parser.parse_args()
    parse_data_file(root="../data/domainnet", data_list_folder="../data/domainnet_list",
                              domains=[args.domain], save=True)
