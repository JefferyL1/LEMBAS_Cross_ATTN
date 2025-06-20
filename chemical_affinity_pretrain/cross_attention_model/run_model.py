import dataset as dataset
import model as model_architecture
import training as training

import torch
import h5py
import pandas as pd
import pickle

def main():

    df = pd.read_csv('/nobackup/users/jefferyl/LauffenLab/pre_train/data/KIBA.txt', sep = r'\s+', names = ['Drug_ID', 'Protein_ID', 'Drug_SMILES', 'Amino_acid_sequence', 'affinity'])
    ecfp4 = h5py.File('/nobackup/users/jefferyl/LauffenLab/pre_train/data/ecfp4.hdf5', 'r')
    protein_embeddings = h5py.File('/nobackup/users/jefferyl/LauffenLab/pre_train/data/per-residue.h5')
    
    with open('/nobackup/users/jefferyl/LauffenLab/pre_train/data/train_test_indices.pkl', 'rb') as f:
        loaded = pickle.load(f)
        train_indices = loaded[0]
        test_indices = loaded[1]

    tdi_dataset = dataset.TDI_Data(df, ecfp4, protein_embeddings, train_indices, test_indices)
    model_params = {'embedding_dim': 1024, 'kqv_dim': 64, 'layers_to_output': [64,16,4,1], 
            'learning_rate':1e-4, 'max_iter':1000, 'batch_size':128, 'param_lambda_L2': 1e-6}
    output_directory = '/nobackup/users/jefferyl/LauffenLab/pre_train/results/cross_attn_KIBA/'

    model = model_architecture.DrugAttnModule(model_params['embedding_dim'], model_params['kqv_dim'], model_params['layers_to_output'])

    model, cur_loss, cur_eig, cur_corr, stats = training.train_model(model, tdi_dataset, model_params, output_directory)

    torch.save(model.state_dict(), f'{output_directory}/final_model')

    f = open(f'{output_directory}/stats_dict.pkl', 'wb')
    pickle.dump(stats, f)
    f.close()

if __name__=='__main__':
    main()
