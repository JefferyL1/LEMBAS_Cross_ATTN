# building the cross-attention based "drug binding" network 

class Single_Attn_Head(nn.Module):

    """ Builds an attention head that uses batch matrix multiplication to get output binding context vector. In essence,
    we are asking the question: Given a specific drug embedding and a designated protein space, which amino acids of each protein 
    should we pay attention to in chemical binding? Returns an output vector that represents our learned interaction
    between drugs and proteins. """
    
    def __init__(self, embedding_dim, k_q_v_dim, device, attn_dropout = 0.0):
        """ Initializes linear layer matrices of specific size respresenting the key, query, and value matrices. """
        
        super().__init__()

        # saving input parameters
        self.device = device
        self.in_dim = embedding_dim
        self.kq_dim = k_q_v_dim
        self.out_dim = k_q_v_dim

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

class Multi_Head_Cross_Attn(nn.Module):
  
    """ Builds multiple cross attention heads and concatenates the outputs together to get the final output dimension """

    def __init__(self, embedding_dim, context_dim, n_heads, device):
        super(Multi_Head_Cross_Attn, self).__init__()

        # defining single attention head 
        self.input_embedding_dim = embedding_dim
        self.output_context_dim = context_dim
        self.n_heads = n_heads
        self.single_attn_dim = self.output_context_dim / self.single_attn_dim

        self.heads = nn.ModuleList([
                Single_Attn_Head(embedding_dim, single_attn_dim, device)
                for _ in range(num_heads)])

    def forward(self, queries, keys, values):
        head_outputs = []
        attn_weights = []

      # running attention heads
        for head in self.heads:
          context, attn = head(queries, keys, values)
          head_outputs.append(context)
          attn_weights.append(attn)

      # concatenating across the last dimension
      concatenated = torch.cat(head_outputs, dim = -1)

      return concatenated

class Transformer_DTA_Model(nn.Module):
    """ Builds a cross-attention based transformer model that predicts drug-target affinity binding, directly related to the 
    drug attention module in DT-LEMBAS """

    def __init__(self, embedding_dim, hidden_dim, layers_to_output, protein_names, protein_file, known_targets_file, ecfp4, device = 'cuda'):

      super(Transformer_DTA_Model, self).__init__()

      self.attn_transformer = Multi_Head_Cross_Attn(embbedding_dim, hidden_dim, n_heads = 4, device = device)

      self.layer_norm_hidden = nn.LayerNorm(normalized_shape = hidden_dim)

      self.layer_norm_final = nn.LayerNorm(normalized_shape = layers_to_output[-1])

      # builds linear layers from context vector to binding scalar 
      self.linear_layers = torch.nn.ModuleList()
      for i in range(0, len(layers_to_output) -1):
          self.linear_layers.append(torch.nn.Linear(layers_to_output[i], layers_to_output[i + 1], bias=True))
        
    def forward(self, drug, dose):
      
        context, attn = self.attn_transformer(drug, self.protein, self.protein)
        context = self.layer_norm_hidden.cuda()(context)
      
        for layer_ind, layer in enumerate(self.linear_layers):
            context = layer(context)
            if layer_ind != len(self.layers) - 1:
                context = torch.nn.LayerNorm(normalized_shape=self.layer_dim[layer_ind+1]).cuda()(context)
                context = self.act_fn(context)
                context = self.dropout(context)
            else:
                context = self.layer_norm_final.cuda()(context)
                context = self.act_fn(context)

        return torch.matmul(torch.diag(self.trainable_dose(dose)), context.squeeze(dim = -1)), attn
      
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

    def __init__(self, embedding_dim, key_query_value_dim, layers_to_output, protein_names, protein_file, known_targets_file, ecfp4, batch_size = 8, dtype = torch.float32, device = 'cuda'):
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
        self.max_L = self.protein.shape[1] # find what dimension the length is across
        self.protein_mask_dict = self.create_protein_masks()
        
        # masking for known targets 
        melted_targets = pd.melt(known_targets_file, id_vars = 'drug', var_name='protein', value_name='activity')
        melted_targets = melted_targets[melted_targets['activity'] == 1]
        targets_dictionary = melted_targets.groupby('drug')['protein'].apply(list).to_dict()
        self.known_targets = {tuple(ecfp4[drug]): targets for drug, targets in targets_dictionary.items()}
        self.mask_dict = self.make_target_masks(self.known_targets)

        # masking for attn to non-existent residues
        self.batched_attn_mask, self.attn_mask  = self.make_attn_mask_tensor(batch_size)
        self.batch_size = batch_size
        
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
        for key, length in self.protein_len_dict.items():
            mask = torch.zeros(self.max_L, device = self.device, dtype = self.dtype)
            mask[ind_list[length:]] = 1
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
