"""
Defines the various layers in the SignalingModel RNN.
"""

from typing import Dict, List, Union, Annotated
from annotated_types import Ge
import copy

import pandas as pd
import numpy as np
import scipy
from scipy.sparse.linalg import eigs
import random
import h5py

import torch
import torch.nn as nn

from model_utilities import np_to_torch, format_network
from activation_functions import activation_function_map
from utilities import set_seeds
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


# building the cross-attention based "drug binding" network 
class SingleAttentionHead(nn.Module):

    """ Builds an attention head that uses batch matrix multiplication to get output binding context vector. In essence,
    we are asking the question: Given a specific drug embedding and a designated protein space, which amino acids of each protein 
    should we pay attention to in chemical binding? Returns an output vector that represents our learned interaction
    between drugs and proteins. """
    
    def __init__(self, embedding_dimension, key_query_dim, value_output_dim, device, attn_dropout = 0.0):
        """ Initializes linear layer matrices of specific size respresenting the key, query, and value matrices. """
        
        super().__init__()

        # saving input parameters
        self.device = device
        self.in_dim = embedding_dimension
        self.kq_dim = key_query_dim
        self.out_dim = value_output_dim

        # initializing key, query, and value matrices for attention
        self.W_query = nn.Linear(self.in_dim, self.kq_dim, bias = False)
        self.W_key = nn.Linear(self.in_dim, self.kq_dim, bias = False)
        self.W_value = nn.Linear(self.in_dim, self.out_dim, bias = False)

        ## can normalize the weights of the key, query, and value matrices
        # nn.init.normal_(self.W_query.weight, std=np.sqrt(2 / (self.in_dim + self.kq_dim)))
        # nn.init.normal_(self.W_key.weight, std=np.sqrt(2 / (self.in_dim+ self.kq_dim)))

        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, queries, keys, values, mask = None):

        """ Creates the context matrix using tensor contraction. The inputs of the queries, keys, and values 
        should be the tensors that we want to act as query, key, or value. For example, drug embeddings act as the 
        query to which protein embeddings act as the key/value. In essence, we are asking what parts of the protein 
        to pay attention to based on the presence of different bits in our drug ECFP4 fingerprint embedding. """

        # calculating the query, key, and value tensors
        q = self.W_query(queries) # example dimension: [batch, 64]
        v = self.W_value(values) # example dimension: [num proteins, length (L), 64]
        k = self.W_key(keys) # example_dimension: [num proteins, L, 64]

        ## calculating the attention: batch matrix multiplying the query matrix by every L x 64 matrix for each protein
        ## "simulating" binding for the drug to every protein in our protein space

        # repeating queries into [num proteins, batch, 64]
        q_repeated = q.unsqueeze(dim = 0)
        n = k.size()[0]
        q_repeated = q_repeated.expand(n, -1, -1)

        # bmm calculation (every matrix of q x every matrix of k along the num proteins dimension)
        attn = torch.bmm(q_repeated, k.transpose(1, 2)) # returns example dimension of [num proteins, batch, L]

        # dividing by the square root of key dimension
        attn /= np.sqrt(k.shape[-1])

        # softmaxing over the L dimension - we want to pay attention along the amino acid dimension (which amino acids are important in binding)
        attn = torch.softmax(attn, dim = -1)
        attn = self.attn_dropout(attn)

        ## calculating the context vector: for every drug/batch sample, for every protein, multiply the 1 x L vector by the
        ## corresponding protein matrix to get a context 1 x 64 vector representing the binding of the drug to that specific protein
        num = attn.size()[0]
        batch = attn.size()[1]
        length = attn.size()[2]
                                                                                                                                      
        # building empty tensor to store data
        result = torch.empty(batch, num, 64, device=self.device)
        
        for i in range(batch):

            # getting specific attention matrix for that drug to all proteins 
            drug_context_matrix = attn[:, i, :]

            # batch matrix multiply the attention vector to L x 64 protein value matrix 
            context = torch.bmm(drug_context_matrix.unsqueeze(dim = 1), v)
            
            result[i, :, :] = context.squeeze(dim = 1)

        return result, attn

class TrainableLogistic(nn.Module):
    """ Builds a trainable generalized logistic function that determines the scaling of the dose on the 
    drug pertubation effect before the LEMBAS module. """
    
    def __init__(self):
        """ Initializes instance of Trainable Logistic with trainable parameters """
        
        super(TrainableLogistic, self).__init__()

        self.A = nn.Parameter(torch.tensor(0.0))
        self.K = nn.Parameter(torch.tensor(1.0))
        self.C = nn.Parameter(torch.tensor(1.0))
        self.Q = nn.Parameter(torch.tensor(2.0))
        self.B = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """ Applies TrainableLogistic function """
        
        return self.A + ((self.K - self.A) / (self.C + self.Q * torch.exp(-1 * self.B * x)))

class DrugAttnModule(nn.Module):

    """ Given a single drug as an ECFP4 fingerprint, returns a ligand-like output representing the drugs binding / interaction on each protein of the
    protein space """

    def __init__(self, embedding_dim, key_query_value_dim, layers_to_output, protein_names, protein_file, known_targets_file, ecfp4, dtype = torch.float32, device = 'cuda'):
        """ 
        
        Initializes an instance of this neural network module.

        Requires the following:
        - embedding dimension: size of embeddings (must be the same for the drug and for the protein)
        - key_query_value_dimension: size of intermediate vectors after attention module
        - layers to output: list representing linear layer size when going from context to single binding, must follow form [key_query_value_dimension, ... , 1]
        - protein_names: dataframe containing all of the proteins in our protein space
        - protein file: h5py file containing embeddings via dictionary lookup of UNIPROT ID 
        - known_targets_file: dataframe with drugs as column 1 and other columns as proteins where a 1 represents known drug activity to that protein
        - ecfp4: h5py file containing 1d numpy array of drug fingerprints via dictionary lookup from drug SMILES
        
        """
        super().__init__()

        # sets device, dtype, and cross_attn module 
        self.dtype = dtype
        self.device = torch.device(device)
        self.cross_attn = SingleAttentionHead(embedding_dim, key_query_value_dim, key_query_value_dim, self.device, attn_dropout=0.1)

        # builds linear layers from context vector to binding scalar 
        self.layers = torch.nn.ModuleList()
        for i in range(0, len(layers_to_output) -1):
            self.layers.append(torch.nn.Linear(layers_to_output[i], layers_to_output[i + 1], bias=True))

        # defining aspects of model 
        self.layer_dim = layers_to_output
        self.act_fn = nn.Tanh()
        self.dropout = torch.nn.Dropout(0.20)
        self.trainable_dose = TrainableLogistic()

        # making 3D protein tensor 
        self.n = len(protein_names)
        self.protein, self.protein_ind_dict, self.protein_len_dict = self.create_protein_reference(protein_names, protein_file)
        self.max_L = protein_object.shape[0] # find what dimension the length is across
        self.protein_mask_dict = self.create_protein_mask(self)
        
        # masking for known targets 
        melted_targets = pd.melt(known_targets_file, id_vars = 'drug', var_name='protein', value_name='activity')
        melted_targets = melted_targets[melted_targets['activity'] == 1]
        targets_dictionary = melted_targets.groupby('drug')['protein'].apply(list).to_dict()
        self.known_targets = {tuple(ecfp4[drug]): targets for drug, targets in targets_dictionary.items()}
        self.mask_dict = self.make_target_masks(self.known_targets)

        # masking for attn to non-existent residues
        attn_mask = self.make_attn_mask_tensor()

    def create_protein_reference(self, protein_names, protein_file):
        """ Creates the 3d tensor of proteins to find 'binding' to. Returns this along with dictionaries of protein name to index in 3d tensor
        and protein names to length."""
        
        protein_embeds = []
        protein_ind_dict = {}
        protein_len_dict = {}

        for ind, protein in enumerate(protein_names):
            protein_embeds.append(torch.from_numpy(protein_file[protein][:]))
            protein_ind_dict[protein] = ind 
            protein_len_dict[protein] = protein_file[protein][:].shape[0] # 0 dimension is the length of the protein

        # pads all proteins to maximum length to ensure rectangular tensor 
        protein_object = torch.nn.utils.rnn.pad_sequence(protein_embeds, batch_first = True, padding_value = 0)
        protein_object = protein_object.to(device = self.device, dtype= self.dtype)

        return protein_object, protein_ind_dict, protein_len_dict

    def create_protein_masks(self):
        """ Creates a dictionary mapping protein names to their respective attn mask"""
        protein_mask = {}
        ind_list = [i for i in range(self.max_L)]
        for key, length in protein_len_dict.items:
            mask = torch.zeros(self.n, device = self.device, dtype = self.dtype)
            mask[ind_list[length+1:]] = 1
            protein_mask[self.protein_ind_dict[key]] = mask

        return protein_mask
        
    def forward(self, drug, dose):
        """ Given specific drug and drug, returns the Xin context vector as well as the masked loss. """
        drug, dose = drug.to(dtype = self.dtype), dose.to(dtype = self.dtype)

        context, attn = self.cross_attn(drug, self.protein, self.protein)

        for layer_ind, layer in enumerate(self.layers):
            context = layer(context)
            if layer_ind != len(self.layers) - 1:
                context = context.permute(0,2,1)
                context = torch.nn.BatchNorm1d(num_features=self.layer_dim[layer_ind+1], momentum=0.2).cuda()(context)
                context = context.permute(0,2,1)
                context = self.act_fn(context)
                context = self.dropout(context)
            else:
                context = self.act_fn(context)

        return torch.matmul(torch.diag(self.trainable_dose(dose)), context.squeeze(dim = -1)), attn

    def L2_reg(self, lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the neural network parameters.
        Here, this pushes learned parameters towards `projection_amplitude`

        Parameters
        ----------
        lambda_2 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty)

        Returns
        -------
        projection_L2 : torch.Tensor
            the regularization term
        """
        weight_params = [param for name, param in self.named_parameters() if 'weight' in name]
        l2_reg = torch.sum(torch.square(torch.cat([param.view(-1) for param in weight_params])))
        return lambda_L2 * l2_reg

    def make_target_masks(self, known_targets):
        """ Given a dictionary of known_targets which contains drug ECFP4 tuples to a list/set of their protein targets by Uniprot ID,
        returns a dictionary of tensor masks for every drug. """
        
        output_dict = {}
        for drug, targets in known_targets.items():
            target_ind = [self.protein_ind_dict[target] for target in targets]
            mask = torch.zeros(self.n, device = self.device, dtype = self.dtype)
            mask[target_ind] = 1
            output_dict[drug] = mask

        return output_dict

    def get_mask(self, drug):
        """ Given a drug as numpy 1d array of fingerprint, returns a specific mask for that drug's
        output signaling in which a 1 represents a known drug-target interaction at that protein. """
        return self.mask_dict[tuple(drug.numpy())]

    def calculate_masked_loss(self, drug, Xin, lambda_mask):
        """ Given an output vector after the forward pass, calculate the loss from off-target predictions"""

        mask = self.get_mask(drug.cpu())
        ones_tensor = torch.ones(len(mask), device = self.device, dtype=bool)

        return lambda_mask * torch.sum((ones_tensor - mask) * Xin)

    def get_batched_mask_loss(self, drug_batch_tensor, Xin_batched_tensor, lambda_mask):
        """ Getting masked loss for a batched input"""
        loss = torch.tensor(0, device=self.device, dtype = self.dtype)
        for i in range(drug_batch_tensor.size()[0]):
            loss += self.calculate_masked_loss(drug_batch_tensor[i, :], Xin_batched_tensor[i, :], lambda_mask)
        return loss

    def get_attn_mask(self, protein_index):
        """ Given a protein, returns a specific mask vector for that protein (with dimension of max-length of protein object)
        in which a 1 represents a padded value while 0 represents actual residues) """

        return self.protein_mask_dict[protein_index]

    def make_attn_mask_tensor(self):
        """ Creates a single matrix of dimensions [num_proteins, max_length] for which each row is the attn mask for the protein at that specific index """
        
        mat = torch.empty((self.n, self.max_L), device=self.device)
        
        for i in range(num_proteins):
            mat[i] = get_attn_mask(i)

        return mat 
        
    def calculate_attn_masked_loss(self, attn_batched_input, lambda_attn):
        """ Calculates the loss (the amount of attention paid to non-existent residues) by calculating the Hadamard product between
        broadcasted attention-mask and my actual attention output. """

        batch_size = attn_batched_input.shape[1]

        return lambda_attn * torch.sum(self.mask.unsqueeze(1).repeat(1,batch_size,1) * attn_batched_input)

class ProjectInput(nn.Module):
    """Generate all nodes for the signaling network and linearly scale input ligand values by NN parameters."""
    def __init__(self, node_idx_map: Dict[str, int], input_labels: np.array, projection_amplitude: Union[int, float] = 1, dtype: torch.dtype=torch.float32, device: str = 'cpu'):
        """Initialization method.

        Parameters
        ----------
        node_idx_map : Dict[str, int]
            a dictionary mapping node labels (str) to the node index (float)
            generated by `SignalingModel.parse_network`
        input_labels : np.array
            names of the input nodes (ligands) from net
        projection_amplitude : Union[int, float]
            value with which to initialize learned linear scaling parameters, by default 1. (if turn require_grad = False for this layer, this is still applied simply as a constant linear scalar in each forward pass)
        dtype : torch.dtype, optional
            datatype to store values in torch, by default torch.float32
        device : str
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        """
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.projection_amplitude = projection_amplitude
        self.size_out = len(node_idx_map) # number of nodes total in prior knowledge network
        self.input_node_order = torch.tensor([node_idx_map[x] for x in input_labels], device = self.device) # idx representation of ligand inputs
        weights = self.projection_amplitude * torch.ones(len(input_labels), dtype=self.dtype, device = self.device) # scaled input weights
        self.weights = nn.Parameter(weights)

    def forward(self, X_in: torch.Tensor):
        """Learn the weights for the input ligands to the signaling network (if grad_fn set to False,
        simply scales by projection amplitude).
        Transform from ligand input (samples x ligands) to full signaling network (samples x network nodes).

        Parameters
        ----------
        X_in : torch.Tensor
            the ligand concentration inputs. Shape is (samples x ligands).

        Returns
        -------
        X_full :  torch.Tensor
            the linearly scaled ligand inputs. Shape is (samples x network nodes)
        """
        X_full = torch.zeros([X_in.shape[0],  self.size_out], dtype=self.dtype, device=self.device) # shape of (samples x total nodes in network)
        X_full[:, self.input_node_order] = self.weights * X_in # only modify those nodes that are part of the input (ligands)
        return X_full

    def L2_reg(self, lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the neural network parameters.
        Here, this pushes learned parameters towards `projection_amplitude`

        Parameters
        ----------
        lambda_2 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty)

        Returns
        -------
        projection_L2 : torch.Tensor
            the regularization term
        """
        # if removed the `- self.projection_amplitude` part, would force weights to 0, thus shrinking ligand inputs
        projection_L2 = lambda_L2 * torch.sum(torch.square(self.weights - self.projection_amplitude))
        return projection_L2

class BioNet(nn.Module):
    """Builds the RNN on the signaling network topology."""
    def __init__(self, edge_list: np.array,
                 edge_MOA: np.array,
                 n_network_nodes: int,
                 bionet_params: Dict[str, float],
                 activation_function: str = 'MML',
                 dtype: torch.dtype=torch.float32,
                device: str = 'cpu',
                seed: int = 888):
        """Initialization method.

        Parameters
        ----------
        edge_list : np.array
            a (2, net.shape[0]) array where the first row represents the indices for the target node and the
            second row represents the indices for the source node. net.shape[0] is the total # of interactions
            output from  `SignalingModel.parse_network`
        edge_MOA : np.array
            a (2, net.shape[0]) array where the first row is a boolean of whether the interactions are stimulating and the
            second row is a boolean of whether the interactions are inhibiting
            output from  `SignalingModel.parse_network`
        n_network_nodes : int
            the number of nodes in the network
        bionet_params : Dict[str, float]
            training parameters for the model
            see `SignalingModel.set_training_parameters`
        activation_function : str, optional
            RNN activation function, by default 'MML'
            options include:
                - 'MML': Michaelis-Menten-like
                - 'leaky_relu': Leaky ReLU
                - 'sigmoid': sigmoid
        dtype : torch.dtype, optional
           datatype to store values in torch, by default torch.float32
        device : str
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        seed : int
            random seed for torch and numpy operations, by default 888
        """
        super().__init__()
        self.training_params = bionet_params
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self._ss_seed_counter = 0

        self.n_network_nodes = n_network_nodes
        # TODO: delete these _in _out?
        self.n_network_nodes_in = n_network_nodes
        self.n_network_nodes_out = n_network_nodes

        self.edge_list = (np_to_torch(edge_list[0,:], dtype = torch.int32, device = 'cpu'),
                          np_to_torch(edge_list[1,:], dtype = torch.int32, device = 'cpu'))
        self.edge_MOA = np_to_torch(edge_MOA, dtype=torch.bool, device = self.device)

        # initialize weights and biases
        weights, bias = self.initialize_weights()
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)

        self.weights_MOA, self.mask_MOA = self.make_mask_MOA() # mechanism of action

        # activation function
        self.activation = activation_function_map[activation_function]['activation']
        self.delta = activation_function_map[activation_function]['delta']
        self.onestepdelta_activation_factor = activation_function_map[activation_function]['onestepdelta']

    def initialize_weight_values(self):
        """Initialize the RNN weight_values for all interactions in the signaling network.

        Returns
        -------
        weight_values : torch.Tensor
            a torch.Tensor with randomly initialized values for each signaling network interaction
        bias : torch.Tensor
            a torch.Tensor with randomly initialized values for each signaling network node
        """

        network_targets = self.edge_list[0].numpy() # the target nodes receiving an edge
        n_interactions = len(network_targets)

        set_seeds(self.seed)
        weight_values = 0.1 + 0.1*torch.rand(n_interactions, dtype=self.dtype, device = self.device)
        weight_values[self.edge_MOA[1,:]] = -weight_values[self.edge_MOA[1,:]] # make those that are inhibiting negative

        bias = 1e-3*torch.ones((self.n_network_nodes_in, 1), dtype = self.dtype, device = self.device)

        for nt_idx in np.unique(network_targets):
            if torch.all(weight_values[network_targets == nt_idx]<0):
                bias.data[nt_idx] = 1

        return weight_values, bias

    def make_mask(self):
        """Generates a mask for adjacency matrix for non-interacting nodes.

        Returns
        -------
        weights_mask : torch.Tensor
            a boolean adjacency matrix of all nodes in the signaling network, masking (True) interactions that are not present
        """

        weights_mask = torch.zeros(self.n_network_nodes, self.n_network_nodes, dtype=bool, device = self.device) # adjacency list format (targets (rows)--> sources (columns))
        weights_mask[self.edge_list] = True # if interaction is present, do not mask
        weights_mask = torch.logical_not(weights_mask) # make non-interacting edges False and vice-vesa
        return weights_mask

    def initialize_weights(self):
        """Initializes weights and masks for interacting nodes and mechanism of action.

        Returns
        -------
        weights : torch.Tensor
            a torch.Tensor adjacency matrix with randomly initialized values for each signaling network interaction
        bias : torch.Tensor
            a torch.Tensor with randomly initialized values for each signaling network node
        """

        weight_values, bias = self.initialize_weight_values()
        self.mask = self.make_mask()
        weights = torch.zeros(self.mask.shape, dtype = self.dtype, device = self.device) # adjacency matrix
        weights[self.edge_list] = weight_values

        return weights, bias

    def make_mask_MOA(self):
        """Generates mask (and weights) for adjacency matrix for non-interacting nodes AND nodes were mode of action (stimulating/inhibiting)
        is unknown.

        Returns
        -------
        weights_MOA : torch.Tensor
            an adjacency matrix of all nodes in the signaling network, with activating interactions set to 1, inhibiting interactions set
            to -1, and interactions that do not exist or have an unknown mechanism of action (stimulating/inhibiting) set to 0
        mask_MOA : torch.Tensor
            a boolean adjacency matrix of all nodes in the signaling network, with interactions that do not exist or have an unknown
            mechanism of action masked (True)
        """

        signed_MOA = self.edge_MOA[0, :].type(torch.long) - self.edge_MOA[1, :].type(torch.long) #1=activation -1=inhibition, 0=unknown
        weights_MOA = torch.zeros(self.n_network_nodes_out, self.n_network_nodes_in, dtype=torch.long, device = self.device) # adjacency matrix
        weights_MOA[self.edge_list] = signed_MOA
        mask_MOA = weights_MOA == 0

        return weights_MOA, mask_MOA

    def prescale_weights(self, target_radius: float = 0.8):
        """Scale weights according to spectral radius

        Parameters
        ----------
        target_radius : float, optional
            _description_, by default 0.8
        """

        A = scipy.sparse.csr_matrix(self.weights.detach().cpu().numpy())
        np.random.seed(self.seed)
        eigen_value, _ = eigs(A, k = 1, v0 = np.random.rand(A.shape[0])) # first eigen value
        spectral_radius = np.abs(eigen_value)

        factor = target_radius/spectral_radius.item()
        self.weights.data = self.weights.data * factor

    def forward(self, X_full: torch.Tensor):
        """Learn the edeg weights within the signaling network topology.

        Parameters
        ----------
        X_full : torch.Tensor
            the linearly scaled ligand inputs. Shape is (samples x network nodes). Output of ProjectInput.

        Returns
        -------
        Y_full :  torch.Tensor
            the signaling network scaled by learned interaction weights. Shape is (samples x network nodes).
        """
        self.weights.data.masked_fill_(mask = self.mask, value = 0.0) # fill non-interacting edges with 0

        X_bias = X_full.T + self.bias # this is the bias with the projection_amplitude included
        X_new = torch.zeros_like(X_bias) #initialize all values at 0

        for t in range(self.training_params['max_steps']): # like an RNN, updating from previous time step
            X_old = X_new
            X_new = torch.mm(self.weights, X_new) # scale matrix by edge weights
            X_new = X_new + X_bias  # add original values and bias
            X_new = self.activation(X_new, self.training_params['leak'])

            if (t % 10 == 0) and (t > 20):
                diff = torch.max(torch.abs(X_new - X_old))
                if diff.lt(self.training_params['tolerance']):
                    break

        Y_full = X_new.T
        return Y_full

    def L2_reg(self, lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the neural network parameters.

        Parameters
        ----------
        lambda_2 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty)

        Returns
        -------
        bionet_L2 : torch.Tensor
            the regularization term
        """
        bias_loss = lambda_L2 * torch.sum(torch.square(self.bias))
        weight_loss = lambda_L2 * torch.sum(torch.square(self.weights))

        bionet_L2 = bias_loss + weight_loss
        return bionet_L2

    def get_sign_mistmatch(self):
        """Identifies edge weights in network that have a sign that does not agree
        with the known mode of action.

        Mode of action: stimulating interactions are expected to have positive weights and inhibiting interactions
        are expected to have negative weights.

        Returns
        -------
        sign_mismatch : torch.Tensor
            a binary adjacency matrix of all nodes in the signaling network, where values are 1 if they do not
            match the mode of action and 0 if they match the mode of action or have an unknown mode of action
        """
        sign_mismatch = torch.ne(torch.sign(self.weights), self.weights_MOA).type(self.dtype)
        sign_mismatch = sign_mismatch.masked_fill(self.mask_MOA, 0) # do not penalize sign mismatches of unknown interactions

        return sign_mismatch

    def count_sign_mismatch(self):
        """Counts total sign mismatches identified in `get_sign_mistmatch`

        Returns
        -------
        n_sign_mismatches : float
            the total number of sign mismatches at `iter`
        """
        n_sign_mismatches = torch.sum(self.get_sign_mistmatch()).item()
        return n_sign_mismatches

    def sign_regularization(self, lambda_L1: Annotated[float, Ge(0)] = 0):
        """Get the L1 regularization term for the neural network parameters that
        do not fit the mechanism of action (i.e., negative weights for stimulating interactions or positive weights for inhibiting interactions).
        Only penalizes sign mismatches of known MOA.

        Parameters
        ----------
        lambda_L1 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty)

        Returns
        -------
        loss : torch.Tensor
            the regularization term
        """
        lambda_L1 = torch.tensor(lambda_L1, dtype = self.dtype, device = self.device)
        sign_mismatch = self.get_sign_mistmatch() # will not penalize sign mismatches of unknown interactions

        loss = lambda_L1 * torch.sum(torch.abs(self.weights * sign_mismatch))
        return loss

    # def get_sign_mistmatch_edge_list(self):
    #     """Same as `get_sign_mistmatch`, but converts to coordinates corresponding to `edge_list`

    #     Returns
    #     -------
    #     sign_mismatch : torch.Tensor
    #         a binary vector corresponding to coordinates in `edge_list`, where values are 1 if they do not
    #         match the mode of action and 0 if they match the mode of action or have an unknown mode of action
    #     """
    #     sign_mismatch = self.get_sign_mistmatch()

    #     # violations = sign_mismatch[self.edge_list] # 1 for interactions in edge list that mismatch, 0 otherwise
    #     # activation_mismatch = torch.logical_and(violations, self.edge_MOA[0])
    #     # inhibition_mismatch = torch.logical_and(violations, self.edge_MOA[1])
    #     # all_mismatch = torch.logical_or(activation_mismatch, inhibition_mismatch)

    #     sign_mismatch_edge = sign_mismatch[self.edge_list] # 1 for interactions in edge list that mismatch, 0 otherwise

    #     return sign_mismatch_edge

    def get_SS_loss(self, Y_full: torch.Tensor, spectral_loss_factor: float, subset_n: int = 10, **kwargs):
        """_summary_

        Parameters
        ----------
        Y_full : torch.Tensor
            output of the forward pass
            ensure to run `torch.Tensor.detach` method prior to inputting so that gradient calculations are not effected
        spectral_loss_factor : float
            _description_
        subset_n : int, optional
            _description_, by default 10

        Returns
        -------
        _type_
            _description_
        """
        spectral_loss_factor = torch.tensor(spectral_loss_factor, dtype=Y_full.dtype, device=Y_full.device)
        exp_factor = torch.tensor(self.training_params['exp_factor'], dtype=Y_full.dtype, device=Y_full.device)
    
        if self.seed:
            np.random.seed(self.seed + self._ss_seed_counter)
        selected_values = np.random.permutation(Y_full.shape[0])[:subset_n]
    
        SS_deviation, aprox_spectral_radius = self._get_SS_deviation(Y_full[selected_values,:], **kwargs)        
        spectral_radius_factor = torch.exp(exp_factor*(aprox_spectral_radius-self.training_params['spectral_target']))
        
        loss = spectral_radius_factor * SS_deviation/torch.sum(SS_deviation.detach())
        loss = spectral_loss_factor * torch.sum(loss)
        aprox_spectral_radius = torch.mean(aprox_spectral_radius).item()
    
        self._ss_seed_counter += 1 # new seed each time this (and _get_SS_deviation) is called
    
        return loss, aprox_spectral_radius
    
    def _get_SS_deviation(self, Y_full_sub, n_probes: int = 5, power_steps: int = 5):
        """Quicker version of spectral radius implemented by Olof Nordenstorm."""
        x_prime = self.onestepdelta_activation_factor(Y_full_sub, self.training_params['leak'])     
        x_prime = x_prime.unsqueeze(2)
        
        T = x_prime * self.weights
        if self.seed:
            set_seeds(self.seed + self._ss_seed_counter)
        delta = torch.randn((Y_full_sub.shape[0], Y_full_sub.shape[1], n_probes), dtype=Y_full_sub.dtype, device=Y_full_sub.device)
        for i in range(power_steps):
            new = delta / torch.norm(delta,dim=1).unsqueeze(1)
            delta = torch.matmul(T, new)

        new_delta = torch.matmul(T, delta)
        batch_eigen_not_norm=torch.einsum('ijk,ijk->ik',new_delta,delta)
        normalize=torch.einsum('ijk,ijk->ik',delta,delta)
        batch_SR_values,_=torch.max(torch.abs(batch_eigen_not_norm/normalize),axis=1) # spectral radius approx 

        aprox_spectral_radius = torch.mean(batch_SR_values, axis=0)      
        SS_deviation = batch_SR_values
    
        return SS_deviation, aprox_spectral_radius
    
    def _depr_get_SS_deviation(self, Y_full_sub, n_probes: int = 5, power_steps: int = 50):
        x_prime = self.onestepdelta_activation_factor(Y_full_sub, self.training_params['leak'])     
        x_prime = x_prime.unsqueeze(2)
        
        T = x_prime * self.weights
        if self.seed:
            set_seeds(self.seed + self._ss_seed_counter)
        delta = torch.randn((Y_full_sub.shape[0], Y_full_sub.shape[1], n_probes), dtype=Y_full_sub.dtype, device=Y_full_sub.device)
        for i in range(power_steps):
            new = delta
            delta = torch.matmul(T, new)
    
        SS_deviation = torch.max(torch.abs(delta), axis=1)[0]
        aprox_spectral_radius = torch.mean(torch.exp(torch.log(SS_deviation)/power_steps), axis=1)
        
        SS_deviation = torch.sum(torch.abs(delta), axis=1)
        SS_deviation = torch.mean(torch.exp(torch.log(SS_deviation)/power_steps), axis=1)
    
        return SS_deviation, aprox_spectral_radius

class ProjectOutput(nn.Module):
    """Transforms signaling network to TF activity."""
    def __init__(self, node_idx_map: Dict[str, int], output_labels: np.array,
                    projection_amplitude: Union[int, float] = 1,
                    dtype: torch.dtype=torch.float32, device: str = 'cpu'):
        """Initialization method.

        Parameters
        ----------
        node_idx_map : Dict[str, int]
            a dictionary mapping node labels (str) to the node index (float)
            generated by `SignalingModel.parse_network`
        output_labels : np.array
           names of the out nodes (TFs) from net
        projection_amplitude : Union[int, float], optional
            value with which to initialize learned linear scaling parameters, by default 1.
            (if turn require_grad = False for this layer, this is still applied  simply
            as a constant linear scalar in each forward pass)
        dtype : torch.dtype, optional
            datatype to store values in torch, by default torch.float32
        device : str, optional
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        """
        super().__init__()

        self.size_in = len(node_idx_map)
        self.size_out = len(output_labels)
        self.projection_amplitude = projection_amplitude

        self.output_node_order = torch.tensor([node_idx_map[x] for x in output_labels], device = device) # idx representation of TF outputs

        weights = self.projection_amplitude * torch.ones(len(output_labels), dtype=dtype, device = device)
        self.weights = nn.Parameter(weights)

    def forward(self, Y_full):
        """Learn the weights for the output TFs of the signaling network (if grad_fn set to False,
        simply scales by projection amplitude).
        Transforms full signaling network  (samples x network nodes) to only the space of the TFs.

        Parameters
        ----------
        Y_full : torch.Tensor
            the signaling network scaled by learned interaction weights. Shape is (samples x network nodes).
            Output of BioNet.

        Returns
        -------
        Y_hat :  torch.Tensor
            the linearly scaled TF outputs. Shape is (samples x TFs)
        """
        Y_hat = self.weights * Y_full[:, self.output_node_order]
        return Y_hat

    def L2_reg(self, lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the neural network parameters.
        Here, this pushes learned parameters towards `projection_amplitude`

        Parameters
        ----------
        lambda_2 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty)

        Returns
        -------
        projection_L2 : torch.Tensor
            the regularization term
        """
        projection_L2 = lambda_L2 * torch.sum(torch.square(self.weights - self.projection_amplitude))
        return projection_L2

    # def set_device(self, device: str):
    #     """Sets torch.tensor objects to the device

    #     Parameters
    #     ----------
    #     device : str
    #         set to use gpu ("cuda") or cpu ("cpu")
    #     """
    #     self.output_node_order = self.output_node_order.to(device)

class SignalingModel(torch.nn.Module):
    """Constructs the signaling network based RNN."""
    DEFAULT_TRAINING_PARAMETERS = {'target_steps': 100, 'max_steps': 300, 'exp_factor': 20, 'leak': 0.01, 'tolerance': 1e-5}

    def __init__(self, net: pd.DataFrame, metadata: pd.DataFrame, chem_fingerprints, y_out: pd.DataFrame,
                 drugattn_params, protein_list, protein_embeddings, known_drug_targets,
                 projection_amplitude_in: Union[int, float] = 1, projection_amplitude_out: float = 1,
                 ban_list: List[str] = None, weight_label: str = 'mode_of_action',
                 source_label: str = 'source', target_label: str = 'target',
                 bionet_params: Dict[str, float] = None ,
                 activation_function: str='MML', dtype: torch.dtype=torch.float32, device: str = 'cpu', seed: int = 888):
        """Parse the signaling network and build the model layers.

        Parameters
        ----------
        net: pd.DataFrame
            signaling network adjacency list with the following columns:
                - `weight_label`: whether the interaction is stimulating (1) or inhibiting (-1). Exclude non-interacting (0) nodes.
                - `source_label`: source node column name
                - `target_label`: target node column name
        metadata : pd.DataFrame
            conts data about each sample. Index represents samples and columns represent cell_line, drug, dose, train/test split.
        y_out : pd.DataFrame
            output TF activities. Index represents samples and columns represent TFs. Values represent activity of the TF.
        drugattn_params:
            training parameters for the drug_attn module
            Must include:
                - 'embedding_dim': size of embedding of drugs and proteins
                - 'kqv_dim': size of intermediate key, query, value dimensions of attention
                - 'layers_to_output': list of layers beginning at kqv_dim and ending at one for feedforward layers
        protein_list:
            list of proteins to predict 'binding' to in the drug-attn module (essentially which proteins are directly impacted by drug)
        protein_embeddings:
            h5py File of proteins to retrieve embeddings based on protein Uniprot ID
        known_drug_targets:
            DataFrame in which the first column with the name 'drug' has drug SMILES with the following columns containing the names
            of possible protein targets and cell values representing whether the drug is known to interact with the protein (1) or not (0)
        ban_list : List[str], optional
            a list of signaling network nodes to disregard, by default None
        projection_amplitude_in : Union[int, float]
            value with which to scale ligand inputs by, by default 1 (see `ProjectInput` for details, can also be tuned as a learned parameter in the model)
        projection_amplitude_out : float
             value with which to scale TF activity outputs by, by default 1 (see `ProjectOutput` for details, can also be tuned as a learned parameter in the model)
        bionet_params : Dict[str, float], optional
            training parameters for the model, by default None
            Key values include:
                - 'max_steps': maximum number of time steps of the RNN, by default 300
                - 'tolerance': threshold at which to break RNN; based on magnitude of change of updated edge weight values, by default 1e-5
                - 'leak': parameter to tune extent of leaking, analogous to leaky ReLU, by default 0.01
                - 'spectral_target': _description_, by default np.exp(np.log(params['tolerance'])/params['target_steps'])
                - 'exp_factor': _description_, by default 20
        activation_function : str, optional
            RNN activation function, by default 'MML'
            options include:
                - 'MML': Michaelis-Menten-like
                - 'leaky_relu': Leaky ReLU
                - 'sigmoid': sigmoid
        dtype : torch.dtype, optional
            datatype to store values in torch, by default torch.float32
        device : str
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        seed : int
            random seed for torch and numpy operations, by default 888
        """
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self._gradient_seed_counter = 0
        self.projection_amplitude_out = projection_amplitude_out
        self.output_TF_list = y_out.columns.values

        edge_list, node_labels, edge_MOA = self.parse_network(net, ban_list, weight_label, source_label, target_label)
        if not bionet_params:
            bionet_params = self.DEFAULT_TRAINING_PARAMETERS.copy()
        else:
            bionet_params = self.set_training_parameters(**bionet_params)

        self.proteins_list = protein_list
        self.y_out = y_out

        # making the dataset TF object
        dataset = TF_Data(y_out, metadata, chem_fingerprints)

        # define model layers
        embedding_dim = drugattn_params['embedding_dim']
        kqv_dim = drugattn_params['kqv_dim']
        output_layers = drugattn_params['layers_to_output']
        self.drug_layer = DrugAttnModule(embedding_dim=embedding_dim,
                                        key_query_value_dim=kqv_dim,
                                        layers_to_output=output_layers,
                                        protein_names=protein_list,
                                        protein_file=protein_embeddings,
                                        known_targets_file=known_drug_targets,
                                        ecfp4=chem_fingerprints,
                                        dtype=self.dtype,
                                        device=device)

        self.input_layer = ProjectInput(node_idx_map = self.node_idx_map,
                                        input_labels = self.proteins_list,
                                        projection_amplitude = projection_amplitude_in,
                                        dtype = self.dtype,
                                        device = self.device)

        self.signaling_network = BioNet(edge_list = edge_list,
                                        edge_MOA = edge_MOA,
                                        n_network_nodes = len(node_labels),
                                        bionet_params = bionet_params,
                                        activation_function = activation_function,
                                        dtype = self.dtype, device = self.device, seed = self.seed)

        self.output_layer = ProjectOutput(node_idx_map = self.node_idx_map,
                                          output_labels = self.y_out.columns.values,
                                          projection_amplitude = self.projection_amplitude_out,
                                          dtype = self.dtype, device = device)

    def parse_network(self, net: pd.DataFrame, ban_list: List[str] = None,
                 weight_label: str = 'mode_of_action', source_label: str = 'source', target_label: str = 'target'):
        """Parse adjacency network.

        Parameters
        ----------
        net: pd.DataFrame
            signaling network adjacency list with the following columns:
                - `weight_label`: whether the interaction is stimulating (1) or inhibiting (-1) or unknown (0). Exclude non-interacting (0)
                nodes.
                - `source_label`: source node column name
                - `target_label`: target node column name
        ban_list : List[str], optional
            a list of signaling network nodes to disregard, by default None

        Returns
        -------
        edge_list : np.array
            a (2, net.shape[0]) array where the first row represents the indices for the target node and the
            second row represents the indices for the source node. net.shape[0] is the total # of interactions
        node_labels : list
            a list of the network nodes in the same order as the indices
        edge_MOA : np.array
            a (2, net.shape[0]) array where the first row is a boolean of whether the interactions are stimulating and the
            second row is a boolean of whether the interactions are inhibiting.

            Note: Edge_list includes interactions that are not delineated as activating OR inhibiting, s.t. edge_MOA records this
            as [False, False].
        """
        if not ban_list:
            ban_list = []
        if sorted(net[weight_label].unique()) != [-1, 0.1, 1]:
            raise ValueError(weight_label + ' values must be 1 or -1')

        net = net[~ net[source_label].isin(ban_list)]
        net = net[~ net[target_label].isin(ban_list)]

        # create an edge list with node incides
        node_labels = sorted(pd.concat([net[source_label], net[target_label]]).unique())
        self.node_idx_map = {idx: node_name for node_name, idx in enumerate(node_labels)}

        source_indices = net[source_label].map(self.node_idx_map).values
        target_indices = net[target_label].map(self.node_idx_map).values

        # # get edge list
        # edge_list = np.array((target_indices, source_indices))
        # edge_MOA = net[weight_label].values
        # get edge list *ordered by source-target node index*csr
        n_nodes = len(node_labels)
        A = scipy.sparse.csr_matrix((net[weight_label].values, (source_indices, target_indices)), shape=(n_nodes, n_nodes)) # calculate adjacency matrix
        source_indices, target_indices, edge_MOA = scipy.sparse.find(A) # re-orders adjacency list by index
        edge_list = np.array((target_indices, source_indices))
        edge_MOA = np.array([[edge_MOA==1],[edge_MOA==-1]]).squeeze() # convert to boolean

        return edge_list, node_labels, edge_MOA

    def df_to_tensor(self, df: pd.DataFrame):
        """Converts a pandas dataframe to the appropriate torch.tensor"""
        return torch.tensor(df.values.copy(), dtype=self.dtype, device = self.device)

    def set_training_parameters(self, **attributes):
        """Set the parameters for training the model. Overrides default parameters with attributes if specified.
        Adapted from LEMBAS `trainingParameters`

        Parameters
        ----------
        attributes : dict
            keys are parameter names and values are parameter value
        """
        #set defaults
        default_parameters = self.DEFAULT_TRAINING_PARAMETERS.copy()
        allowed_params = list(default_parameters.keys()) + ['spectral_target']

        params = {**default_parameters, **attributes}
        if 'spectral_target' not in params.keys():
            params['spectral_target'] = np.exp(np.log(params['tolerance'])/params['target_steps'])

        params = {k: v for k,v in params.items() if k in allowed_params}

        return params

    def forward(self, X_in):
        """Linearly scales ligand inputs, learns weights for signaling network interactions, and transforms this to TF activity. See
        `forward` methods of each layer for details."""
        X_full = self.input_layer(X_in) # input ligands to signaling network
        Y_full = self.signaling_network(X_full) # RNN of full signaling network
        Y_hat = self.output_layer(Y_full) # TF outputs of signaling network
        return Y_hat, Y_full

    def L2_reg(self, lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the neural network parameters.

        Parameters
        ----------
        lambda_L2 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty)

        Returns
        -------
         : torch.Tensor
            the regularization term (as the sum of the regularization terms for each layer)
        """
        return self.drugLayer.L2_reg(lambda_L2) + self.input_layer.L2_reg(lambda_L2) + self.signaling_network.L2_reg(lambda_L2) + self.output_layer.L2_reg(lambda_L2)

    def ligand_regularization(self, lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the ligand biases. Intuitively, extracellular ligands should not contribute to
        "baseline (i.e., unstimulated) activity" affecting intrecllular signaling nodes and thus TF outputs.

        Parameters
        ----------
        lambda_L2 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty)

        Returns
        -------
        loss : torch.Tensor
            the regularization term
        """
        loss = lambda_L2 * torch.sum(torch.square(self.signaling_network.bias[self.input_layer.input_node_order]))
        return loss

    def uniform_regularization(self, lambda_L2: float, Y_full: torch.Tensor,
                     target_min: float = 0.0, target_max: float = None):
        """Get the L2 regularization term for deviations of the nodes in Y_full from that of a uniform distribution between
        `target_min` and `target_max`.
        Note, this penalizes both deviations from the uniform distribution AND values that are out of range (like a double penalty).

        Parameters
        ----------
        lambda_L2 : float
            scaling factor for state loss
        Y_full : torch.Tensor
            the signaling network scaled by learned interaction weights. Shape is (samples x network nodes).
            Output of BioNet.
        target_min : float, optional
            minimum values for nodes in Y_full to take on, by default 0.0
        target_max : float, optional
            maximum values for nodes in Y_full to take on, by default 1/`self.projection_amplitude_out`

        Returns
        -------
        loss : torch.Tensor
            the regularization term
        """
        lambda_L2 = torch.tensor(lambda_L2, dtype = Y_full.dtype, device = Y_full.device)
        # loss = lambda_L2 * expected_uniform_distribution(Y_full, target_max = 1/self.projectionAmplitude)
        if not target_max:
            target_max = 1/self.projection_amplitude_out

        sorted_Y_full, _ = torch.sort(Y_full, axis=0) # sorts each column (signaling network node) in ascending order
        target_distribution = torch.linspace(target_min, target_max, Y_full.shape[0], dtype=Y_full.dtype, device=Y_full.device).reshape(-1, 1)

        dist_loss = torch.sum(torch.square(sorted_Y_full - target_distribution)) # difference in distribution
        below_loss = torch.sum(Y_full.lt(target_min) * torch.square(Y_full-target_min)) # those that are below the minimum value
        above_loss = torch.sum(Y_full.gt(target_max) * torch.square(Y_full-target_max)) # those that are above the maximum value
        loss = lambda_L2*(dist_loss + below_loss + above_loss)
        return loss

    def add_gradient_noise(self, noise_level: Union[float, int]):
        """Adds noise to backwards pass gradient calculations. Use during training to make model more robust.

        Parameters
        ----------
        noise_level : Union[float, int]
            scaling factor for amount of noise to add
        """
        all_params = list(self.parameters())
        if self.seed:
            set_seeds(self.seed + self._gradient_seed_counter)
        for i in range(len(all_params)):
            if all_params[i].requires_grad:
                all_noise = torch.randn(all_params[i].grad.shape, dtype=all_params[i].dtype, device=all_params[i].device)
                all_params[i].grad += (noise_level * all_noise)

        self._gradient_seed_counter += 1 # new random noise each time function is called

    def copy(self):
        return copy.deepcopy(self)
