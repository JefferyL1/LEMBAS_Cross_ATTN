import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import argparse
import logging
import random
import h5py
import time
import plotnine as p9


import bionetwork as bionetwork
import utilities as utils
import plotting, io

import time
import pickle

def test_model(model, test_dataloader, hyper_params, split = 'test'):
    """ Tests the models on test dataset and stores outputs. """
    
    device = model.device
    dtype = model.dtype

    model.eval()

    with torch.no_grad():
        loss = torch.nn.MSELoss(reduction = 'mean')
        for drug, dose, TF_output in test_dataloader:
            drug, dose, TF_output = drug.to(device), dose.to(device), TF_output.to(device)

             # model pass
            X_in = model.drug_layer(drug, dose) # gets signaling output from attn module
            X_full = model.input_layer(X_in) # project to bionetwork space
            Y_full = model.signaling_network(X_full) # signaling network
            Yhat = model.output_layer(Y_full) # output TF

            # calculating loss
            mse_loss = loss(TF_output, Yhat)
            masked_loss = model.drug_layer.get_batched_mask_loss(drug, X_in, hyper_params['off_target_lambda']) # gets the off-target loss

            # calculating correlation
            tot_correlation = torch.corrcoef(torch.stack([Yhat.view(-1), TF_output.view(-1)]))[0, 1]
            per_TF_corr = torch.zeros(len(model.output_TF_list), device = model.device, dtype = model.dtype)
            for i in range(len(model.output_TF_list)):
                corr = torch.corrcoef(torch.stack([Yhat[:, i].view(-1), TF_output[:, i].view(-1)]))[0,1]
                per_TF_corr[i] = corr

        model.train()
        return {'correlation': tot_correlation, 'per_TF_corr': per_TF_corr, 'loss': mse_loss, 'masked_loss': masked_loss}

def train_model(model, dataset, cell_line, hyper_params, output_directory, verbose = True, reset_epoch = 200, time_limit = 12):
    """ Trains model on training dataset """
    
    device = model.device
    dtype = model.dtype
    start_time = time.time()

    model.to(device)

    stats = utils.initialize_progress(hyper_params['max_iter'])

    # adjusting the model
    model.input_layer.weights.require_grad = False
    model.signaling_network.prescale_weights(target_radius = hyper_params['target_spectral_radius'])

    # defining loss and optimizer
    loss = torch.nn.MSELoss(reduction = 'mean')
    optimizer = torch.optim.Adam(model.parameters(), lr = 1, weight_decay=0)
    reset_state = optimizer.state.copy()

    # getting specific dataset of cell line and train split
    training_indices = dataset.get_cell_line_w_split_indices(cell_line, 'train')
    train_data = dataset.create_sub_dataset(training_indices)
    train_loader = DataLoader(dataset = train_data, batch_size = hyper_params['batch_size'], shuffle = True)

    # datset of testing split
    testing_indices = dataset.get_cell_line_w_split_indices(cell_line, 'test')
    test_data = dataset.create_sub_dataset(testing_indices)
    test_loader = DataLoader(dataset = test_data, batch_size=len(test_data), shuffle=True)

    for e in range(hyper_params['max_iter']):

        # learning rate
        cur_lr = utils.get_lr(e, hyper_params['max_iter'], max_height = hyper_params['learning_rate'],
                            start_height = hyper_params['learning_rate']/10, end_height = 1e-6,
                            peak = 1000)
        optimizer.param_groups[0]['lr'] = cur_lr

        cur_loss = []
        cur_eig = []
        cur_corr = []

        for drug, dose, TF_output in train_loader:
            model.train()
            optimizer.zero_grad()

            drug, dose, TF_output = drug.to(device), dose.to(device), TF_output.to(device)

            # model pass
            X_in = model.drug_layer(drug, dose) # gets signaling output from attn module
            X_full = model.input_layer(X_in) # project to bionetwork space
            network_noise = torch.randn(X_full.shape, device = X_full.device)
            X_full = X_full + (hyper_params['noise_level'] * cur_lr * network_noise)
            Y_full = model.signaling_network(X_full) # signaling network
            Yhat = model.output_layer(Y_full) # output TF

            # getting loss equation

            fit_loss = loss(TF_output, Yhat) # normal loss
            masked_loss = model.drug_layer.get_batched_mask_loss(drug, X_in, hyper_params['off_target_lambda'])
            sign_reg = model.signaling_network.sign_regularization(lambda_L1 = hyper_params['moa_lambda_L1'])
            ligand_reg = model.ligand_regularization(lambda_L2 = hyper_params['ligand_lambda_L2']) # ligand biases
            stability_loss, spectral_radius = model.signaling_network.get_SS_loss(Y_full = Y_full.detach(), spectral_loss_factor = hyper_params['spectral_loss_factor'],
                                                                                subset_n = hyper_params['subset_n_spectral'], n_probes = hyper_params['n_probes_spectral'],
                                                                                power_steps = hyper_params['power_steps_spectral'])
            uniform_reg = model.uniform_regularization(lambda_L2 = hyper_params['uniform_lambda_L2']*cur_lr, Y_full = Y_full,
                                                     target_min = 0, target_max = hyper_params['uniform_max']) # uniform distribution
            param_reg = model.L2_reg(hyper_params['param_lambda_L2']) # all model weights and signaling network biases
            total_loss = fit_loss + masked_loss + sign_reg + ligand_reg + param_reg + stability_loss + uniform_reg

            correlation = torch.corrcoef(torch.stack([Yhat.view(-1), TF_output.view(-1)]))[0, 1] # getting total correlation per TF

            # backpropagation and optimizing
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # store
            cur_eig.append(spectral_radius)
            cur_loss.append(fit_loss.item())
            cur_corr.append(correlation.item())

        stats = utils.update_progress(stats, iter = e, loss = cur_loss, eig = cur_eig, corr = cur_corr, learning_rate = cur_lr,
                                     n_sign_mismatches = model.signaling_network.count_sign_mismatch())
        
        # every 10 iterations, run test model and store results 
        if e % 10 == 0 or e == hyper_params['max_iter'] - 1:
            result_dict = test_model(model, test_loader, hyper_params)
            stats = utils.update_test_progress(stats, iter = e, loss = result_dict['loss'], corr = result_dict['correlation'],
                                              per_TF_corr = result_dict['per_TF_corr'])

        # print stats and save model every 250 iterations
        if verbose and e % 250 == 0:
            utils.print_stats(stats, iter = e)

            if e > 0:
                torch.save(model.state_dict(), f'{output_directory}/model_epoch_{e}')

        # reset optimizer
        if np.logical_and(e % reset_epoch == 0, e > 0):
            optimizer.state = reset_state.copy()

        # check the time and save output if running out of time
        if time.time() - start_time > 60*60*(time_limit - 0.25):
            f = open(f"{output_directory}/stats_dict.pkl",  "wb")
            pickle.dump(stats,f)
            f.close()
            
    return model, cur_loss, cur_eig, cur_corr, stats

def run_model(cell_line, output_directory_path):
    """Given a specific cell_line, runs a model trained on data from that cell_line. Output directory path denotes existing folder to output the results. """

    # loading in necessary data 
    raw_network = pd.read_csv('/nobackup/users/jefferyl/LauffenLab/LEMBAS_w_attn/data/network_data/network_info_from_nikos.tsv', sep ='\t', index_col=False)
    chem_fingerprints = h5py.File('/nobackup/users/jefferyl/LauffenLab/LEMBAS_w_attn/data/ml_data/ecfp4.h5')
    protein_embeddings = h5py.File('/nobackup/users/jefferyl/LauffenLab/ic_50/data/per-residue.h5')
    metadata = pd.read_csv('/nobackup/users/jefferyl/LauffenLab/LEMBAS_w_attn/data/ml_data/metadata.csv', index_col=0)
    TF_output = pd.read_csv('/nobackup/users/jefferyl/LauffenLab/LEMBAS_w_attn/data/raw_data/all_TF_data.tsv', sep = '\t', index_col=0)
    filtTF = TF_output.loc[TF_output.index.isin(metadata.index)]
    known_drug_targets = pd.read_csv('/nobackup/users/jefferyl/LauffenLab/LEMBAS_w_attn/data/raw_data/drug_target_info.tsv', sep = '\t')
    known_drug_targets = known_drug_targets.rename(columns={'Unnamed: 0': 'drug'})
    drug_attn_proteins = pd.read_csv('/nobackup/users/jefferyl/LauffenLab/LEMBAS_w_attn/data/network_data/protein_targets.csv')
    protein_list = drug_attn_proteins['protein']
    device = 'cuda'

    # building the dataset     
    dataset = bionetwork.TF_Data(filtTF, metadata, chem_fingerprints)

    # filtering network
    def create_network_info(raw_network_df):
        network = raw_network_df[['source', 'target', 'stimulation', 'inhibition']]
        mode_of_action = [1.0 if row['stimulation'] == 1 else -1.0 if row['inhibition'] == 1 else 0.1 for _, row in raw_network_df.iterrows()]
        network['mode_of_action'] = mode_of_action
        return network

    network = create_network_info(raw_network)
    
    # linear scaling of inputs/outputs
    projection_amplitude_in = 3
    projection_amplitude_out = 1.2
    
    # bionet parameters
    bionet_params = {'target_steps': 100, 'max_steps': 150, 'exp_factor':50, 'tolerance': 1e-5, 'leak':1e-2} # fed directly to model
    
    # cross_attn parameters
    crossattn_params = {'embedding_dim':1024, 'kqv_dim': 64, 'layers_to_output': [64, 16, 4, 1]} # create drug module for model
    
    # training parameters
    lr_params = {'max_iter': 5000,
                'learning_rate': 2e-3}
    other_params = {'batch_size': 8, 'noise_level': 10, 'gradient_noise_level': 1e-9}
    regularization_params = {'param_lambda_L2': 1e-6, 'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-5, 'uniform_lambda_L2': 1e-4,
                    'uniform_max': 1/projection_amplitude_out, 'spectral_loss_factor': 1e-5, 'off_target_lambda': 0.01}
    spectral_radius_params = {'n_probes_spectral': 5, 'power_steps_spectral': 50, 'subset_n_spectral': 10}
    hyper_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params, 'target_spectral_radius':0.8}

    # loading in the model
    model = bionetwork.SignalingModel(net = network,
                                  metadata = metadata,
                                  chem_fingerprints = chem_fingerprints,
                                  y_out = filtTF,
                                  drugattn_params = crossattn_params,
                                  protein_list = protein_list,
                                  protein_embeddings = protein_embeddings,
                                  known_drug_targets = known_drug_targets,
                                  projection_amplitude_in = projection_amplitude_in,
                                  projection_amplitude_out = projection_amplitude_out,
                                  bionet_params = bionet_params,
                                  device = 'cuda')

    model, cur_loss, cur_eig, cur_corr, stats = train_model(model, dataset, cell_line, hyper_params, output_directory_path)

def main():
    run_model('VCAP', '/nobackup/users/jefferyl/LauffenLab/LEMBAS_w_attn/VCAP_res')

if __name__ = "__main__":
    main()
