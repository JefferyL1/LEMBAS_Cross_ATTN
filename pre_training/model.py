import torch
import torch.nn as nn
from typing import Dict, List, Union, Annotated
from annotated_types import Ge

class SingleAttentionHead(nn.Module):

    """ Builds an attention head that uses cross-attention to get context binding vector. In essence,
    we are asking the question: Given a specific drug embedding and a designated protein space, which amino acids of each protein 
    should we pay attention to in chemical binding? Returns an output vector that represents our learned interaction
    between drugs and proteins. """
    
    def __init__(self, embedding_dimension, key_query_dim, value_output_dim, device):
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

    def forward(self, queries, keys, values, mask = None):

        """ Creates the context matrix. Runs the attention block"""

        # calculating the query, key, and value tensors
        q = self.W_query(queries) # example dimension: [batch, 64]
        v = self.W_value(values) # example dimension: [batch, length (L), 64]
        k = self.W_key(keys) # example_dimension: [batch, L, 64]

        ## calculating the attention: batch matrix multiplying the query matrix by every L x 64 matrix for each protein
        ## "simulating" binding for the drug to every protein in our protein space

        # attention calculation 
        attn = torch.matmul(q.unsqueeze(dim = 1), k.transpose(1,2)) #[batch, 1, L]
        
        # dividing by the square root of key dimension
        attn /= torch.sqrt(torch.tensor(k.shape[-1]))

        # softmaxing over the L dimension - we want to pay attention along the amino acid dimension (which amino acids are important in binding)
        attn_weights = torch.softmax(attn, dim = -1)

        # getting context vector 
        context = torch.matmul(attn_weights, v) #[batch, 1, 64]

        return context, attn_weights

class DrugAttnModule(nn.Module):

    """ Given a single drug as an ECFP4 fingerprint, returns a ligand-like output representing the drugs binding / interaction on each protein of the
    protein space """

    def __init__(self, embedding_dim, key_query_value_dim, layers_to_output, dtype = torch.float32, device = 'cuda'):
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
        self.cross_attn = SingleAttentionHead(embedding_dim, key_query_value_dim, key_query_value_dim, self.device)

        # builds linear layers from context vector to binding scalar 
        self.layers = torch.nn.ModuleList()
        for i in range(0, len(layers_to_output) -1):
            self.layers.append(torch.nn.Linear(layers_to_output[i], layers_to_output[i + 1], bias=True))

        # defining aspects of model 
        self.layer_dim = layers_to_output
        self.act_fn = nn.Tanh()
        self.dropout = torch.nn.Dropout(0.20)
        
    def forward(self, drug, protein):
        """ Given specific drug and protein, returns the binding affinity and attention. """
        drug, protein = drug.to(dtype = self.dtype), protein.to(dtype = self.dtype)

        context, attn = self.cross_attn(drug, protein, protein) #context is [batch, 1, 64]

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

        # context is [batch, 1, 1]

        return context.squeeze(dim = (-1, -2)), attn

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
