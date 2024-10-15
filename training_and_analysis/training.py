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

def train_model(model, dataset, cell_line, hyper_params, verbose = True, reset_epoch = 200, ):
    """ Trains model on training dataset """
    
    device = model.device
    dtype = model.dtype

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

        # print stats every 250 iterations
        if verbose and e % 250 == 0:
            utils.print_stats(stats, iter = e)

        # reset optimizer
        if np.logical_and(e % reset_epoch == 0, e > 0):
            optimizer.state = reset_state.copy()


    return model, cur_loss, cur_eig, cur_corr, stats
