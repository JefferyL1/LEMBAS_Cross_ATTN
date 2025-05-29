import networkx as nx
from itertools import combinations
import pandas as pd
import numpy as np
import scipy
from scipy.sparse.linalg import eigs
import random
import h5py
from typing import Dict, List, Union, Annotated
from annotated_types import Ge
import torch
import torch.nn as nn
import torch.utils.data as data
import os

def get_tanimoto_similarity(fp1, fp2):
    """Calculates tanimoto similarity of two input numpy fingerprints with dtpye = bool."""
    intersection = np.logical_and(fp1, fp2).sum()
    union = np.logical_or(fp1, fp2).sum()
    return intersection / union if union != 0 else 0.0
    
class TDI_Data(data.Dataset):

    """ Given data representing drug target interaction, builds class object containing data in indexable dataset. """

    def __init__(self, data, drug_fingerprint_dict, protein_embedding_dictionary):
        """
        Builds an instance of class object to store data. 
        
        Requires the following datasets:
            - data: class pd.DataFrame in which each row is different trials with columns of 'Drug_ID', 'Protein_ID', 'Drug_SMILES', 'affinity' 
            - drug_fingerprint_dict: h5py file that maps drug smiles to np array of ECFP4 fingerprints
            - protein_embedding_dictionary: h5py file that maps protein ID to protein embeddings

        Builds the following things:
            - self.input: list of tuples of torch.Tensor object of the form (drug, dosage, cell_line, split, sample_id) where each tuple
              represents a trial
            - self.output: list of torch.Tensor objects of the TF output of that trial
        """

        # saving data
        self.data = data
        self.ecfp4 = drug_fingerprint_dict
        self.proteins = protein_embedding_dictionary

        # getting train and test split
        self.train, self.test = self.get_train_test_split()

        # getting training data affinity mean 
        self.mean = self.data.loc[self.train, 'affinity'].mean()
        
        # building data 
        self.input = []
        self.output = []

        for row in self.data.itertuples(index = True):
            self.input.append((row.Drug_SMILES, row.Protein_ID))
            self.output.append(row.affinity)

        print('Successfull built data')

    def __len__(self):
        """ Returns the length of the dataset - the number of total samples. """
        return len(self.input)

    def __getitem__(self, index):
        """ Returns specific sample as tuple of drug ECFP4, protein embedding, and affinity. """

        drug_ECFP4 = torch.from_numpy(self.ecfp4[self.input[index][0]])
        protein_embedding = torch.from_numpy(self.proteins[self.input[index][1]])
        affinity = torch.tensor(self.output[index])

        return drug_ECFP4, protein_embedding, affinity
        
    def get_train_test_split(self, test_split = 0.2):
        """Builds indices of training and testing split in which testing drugs are dissimilar to 
        training drugs. """

        # getting drug SMILES to sample indices 
        SMILES_indices = {k: v.tolist() for k,v in self.data.groupby('Drug_SMILES').groups.items()}

        # getting connected components
        drug_set = set(self.data['Drug_SMILES'])
        connected_components = self.get_connected_components(drug_set, self.ecfp4)
        drugs_w_edges = set.union(*connected_components)
        drugs_wo_edges = drug_set - drugs_w_edges
        connected_components.extend([[drug] for drug in drugs_wo_edges])
        
        # components to number of samples
        component_to_num_sample = dict()
        for i, component in enumerate(connected_components):
            samples = 0
            for drug in component:
                samples += len(SMILES_indices[drug])
            component_to_num_sample[i] = samples
        
        # faux-greedy search for subset sum (testing split)
        comp = sorted(list(component_to_num_sample.keys()), key = lambda x: component_to_num_sample[x])
        
        test = []
        train = []
        for index in comp:
            drugs = connected_components[index]
            if len(test) < 0.2 * len(df):
                for drug in drugs:
                    test.extend(SMILES_indices[drug])
            else:
                for drug in drugs:
                    train.extend(SMILES_indices[drug])

        assert len(test) + len(train) == len(self.data)

        return train, test
        
    def get_connected_components(self, drugs, ecfp4, similarity_cutoff = 0.5):
        """ Given a set of drugs, we represent the similarity problem as an unweighted, undirected graph
        with drugs as nodes and edges between two drugs that are similar (based on our threshold). The 
        connected components of this graph are the drugs that must be in the same training or testing 
        split. """
        G = nx.Graph()
        edges = []
        for drug_pair in combinations(drugs, 2):
            similarity = get_tanimoto_similarity(ecfp4[drug_pair[0]][:], ecfp4[drug_pair[1]][:])
            if similarity >= similarity_cutoff:
                edges.append(drug_pair)
        G.add_edges_from(edges)
        return list(nx.connected_components(G))

    def get_train_test_indices(self, split):
        """ Returns list of indices of samples with train/test split """

        assert split in ['train', 'test'], 'split must be either train or test'

        if split == 'train':
            return self.train
        else:
            return self.test

    def shuffle(self):
        """ Randomizes dataset in which the y-values / output are shuffled to create mismatched data between drugs and TF output.
        Mutates existing dataset. """
        random.seed(88)
        self.output = random.shuffle(self.output)
