import random
import torch
import torch.utils.data as data


# building the dataset structure for training 

class small_dataset(data.Dataset):
    """ Sub-dataset containing only the data for a specific cell line and split. 
    Used for easy access and indexing for model training. """

    def __init__(self, data_list):
        """ Builds the dataset. Requires data_list: a list containing tuples of (drug, dose, TF_output)
        for that specific cell line / split. """
        
        self.data = data_list

    def __len__(self):
        """ Returns the length - the number of samples. """
        return len(self.data)

    def __getitem__(self, index):
        """ Gets specific data point associated with index. Returns tuple of 
        drug vector, dose, TF_output vector """
        point = self.data[index]
        return point[0], point[1], point[2]

class TF_Data(data.Dataset):

    """ Given data representing transcription factor (TF) activity for multiple trials of drugs and dosages
    across different cell lines, builds class object containing data in indexable dataset. """

    def __init__(self, trials_to_tf, trials_metadata, drug_fingerprint_dict):
        """
        Builds an instance of class object to store data. 
        
        Requires the following datasets:
            - trials_to_tf: class pd.DataFrame in which the indexes are the trial identifiers and columns are
              the transcription factor activity for specific TFs
            - trials_metadata: class pd.DataFrame in which the indexes are the trial identifiers and columns are
              relevant metadata (drug, dosage, time, cell_line, train/test split)
            - drug_fingerprint_dict: h5py file that returns the ECFP4 fingerprint of drugs as 1d array with SMILEs as keys

        Builds the following things:
            - self.input: list of tuples of torch.Tensor object of the form (drug, dosage, cell_line, split, sample_id) where each tuple
              represents a trial
            - self.output: list of torch.Tensor objects of the TF output of that trial
        """

        # transforming data for easy lookup
        tf_data_dictionary = trials_to_tf.to_dict(orient = 'index')
        metadata_dictionary = trials_metadata.to_dict(orient  = 'index')
        self.cell_line_dictionary = {}
        self.split_dictionary = {split:set() for split in ['train', 'test']}

        self.input = []
        self.output = []

        # iterating over data to build dataset
        idx = 0
        for trial_id , data in tf_data_dictionary.items():

            # getting metadata information
            drug = metadata_dictionary[trial_id]['drug']
            dosage = metadata_dictionary[trial_id]['dosage']
            split = metadata_dictionary[trial_id]['split']
            cell_line = metadata_dictionary[trial_id]['cell_line']

            # building cell-line dataset
            if cell_line not in self.cell_line_dictionary:
                self.cell_line_dictionary[cell_line] = {idx}
            else:
                self.cell_line_dictionary[cell_line].add(idx)

            # building train-test dataset
            self.split_dictionary[split].add(idx)

            # building input of drug, dose, and metadata
            input = (torch.from_numpy(drug_fingerprint_dict[drug][:]), torch.tensor(dosage), cell_line, split, trial_id)

            # building output of TF activity per TF
            tf_list = [activity for tf,activity in data.items()]
            output = torch.Tensor(tf_list)

            self.input.append(input)
            self.output.append(output)

            idx += 1

    def __len__(self):
        """ Returns the length of the dataset - the number of total samples. """

        return len(self.input)

    def __getitem__(self, index):
        """ Returns specific trial as tuple of drug vector, dose, and TF vector. """

        drug = self.input[index][0]
        dose = self.input[index][1]
        tf = self.output[index]

        return drug, dose, tf

    def get_cell_line_indices(self, cell_line):
        """ Returns list of indices of samples associated with a specific cell line """

        return list(self.cell_line_dictionary[cell_line])

    def get_train_test_indices(self, split):
        """ Returns list of indices of samples with train/test split """

        assert split in ['train', 'test'], 'split must be either train or test'

        return list(self.split_dictionary[split])

    def get_cell_line_w_split_indices(self, cell_line, split):
        """ Returns list of indices of samples of specific cell line and train/test split """

        cell_line_idx = self.get_cell_line_indices(cell_line)
        split_idx = self.get_train_test_indices(split)

        #returns the intersection of the two sets
        return(list(set(cell_line_idx) & set(split_idx)))

    def create_sub_dataset(self, indices):
        """ Builds sub-dataset for specific set of indices"""
        
        data = [self[i] for i in indices]
        return small_dataset(data)

    def shuffle(self):
        """ Randomizes dataset in which the y-values / output are shuffled to create mismatched data between drugs and TF output.
        Mutates existing dataset. """
        random.seed(88)
        self.output = random.shuffle(self.output)


