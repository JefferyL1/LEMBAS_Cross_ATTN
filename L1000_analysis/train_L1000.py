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
import training

def run_model(cell_line, output_directory):

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

    # building dataset
    dataset = bionetwork.TF_Data(filtTF, metadata, chem_fingerprints)

    # modifying raw_network_csv into form for model
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
    hyper_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params, 'target_spectral_radius' : 0.8} # fed into training function

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

    # training model
    model, cur_loss, cur_eig, cur_corr, stats = train_model(model, dataset, cell_line, hyper_params)

    # plotting convergence of loss during training
    loss_smooth = utils.get_moving_average(values = stats['loss_mean'], n_steps = 5)
    loss_sigma_smooth = utils.get_moving_average(values = stats['loss_sigma'], n_steps = 10)
    epochs = np.array(range(stats['loss_mean'].shape[0]))

    p1A = plotting.shade_plot(X = epochs, Y = loss_smooth, sigma = loss_sigma_smooth, x_label = 'Epoch', y_label = 'Loss')
    p1A += p9.scale_y_log10()
    p1A.save(filename = 'train_loss', path = output_directory)

    # plotting the learning rate during training
    viz_df = pd.DataFrame(data = {'Epoch': epochs, 'lr': stats['learning_rate']})
    width, height = 5, 3
    p1B = (
        p9.ggplot(viz_df, p9.aes(x='Epoch', y = 'lr')) +
        p9.geom_line(color = '#1E90FF') +
        p9.theme_bw() +
        p9.theme(figure_size=(width, height)) +
        p9.ylab('Learning Rate')
    )
    p1B.save(filename = 'train_learning_rate', path = output_directory)

    # plotting the eigenvalue during training
    eig_smooth = utils.get_moving_average(stats['eig_mean'], 5)
    eig_sigma_smooth = utils.get_moving_average(stats['eig_sigma'], 5)

    p1C = plotting.shade_plot(X = epochs, Y = eig_smooth, sigma = eig_sigma_smooth, x_label = 'Epoch', y_label = 'Spectral Radius')
    p1C.save(filename = 'train_eigenvalue', path = output_directory)

    # plotting the correlation convergence during training
    corr_smooth = utils.get_moving_average(stats['corr_mean'], 5)
    corr_sigma_smooth = utils.get_moving_average(stats['corr_sigma'], 5)

    p1C = plotting.shade_plot(X = epochs, Y = corr_smooth, sigma = corr_sigma_smooth, x_label = 'Epoch', y_label = 'Correlation of Predictions')
    p1C.save(filename = 'train_correlation', path = output_directory)

    # plotting the loss in testing
    p2A = plotting.test_plot(Y_test = stats['test_loss'], x_label = 'Epoch', y_label = 'Loss')
    p2A.save(filename = 'test_loss', path = output_directory)

    # plotting the correlation in testing
    p2B = plotting.test_plot(Y_test = stats['test_corr'], x_label= 'Epoch', y_label= 'Correlation of Prediction')
    p2B.save(filename= 'test_correlation', path = output_directory)

    # plotting the per_TF correlation line plot in testing
    p2C = plotting.ind_TF_corr_plot(corr_TF_data = stats['per_TF_corr'], TF_name_list = model.output_TF_list)
    p2C.save(filename = 'test_per_tf_corr', path = output_directory)
