from typing import Dict, List, Union, Annotated
from annotated_types import Ge
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pickle

def initialize_progress(max_iter: int):
    """Track various stats of the progress of training the model.

    Parameters
    ----------
    max_iter : int
        the maximum number of training iterations

    Returns
    -------
    stats : dict
        a dictionary of progress statistics
    """
    stats = {}

    # training data
    stats['loss_mean'] = np.nan*np.ones(max_iter)
    stats['loss_sigma'] = np.nan*np.ones(max_iter)
    stats['corr_mean'] = np.nan*np.ones(max_iter)
    stats['corr_sigma'] = np.nan*np.ones(max_iter)

    # mean predictor data
    stats['mean_loss_train'] = None
    stats['mean_loss_test'] = None

    # learning rate
    stats['learning_rate'] = np.nan*np.ones(max_iter)

    # testing data
    stats['test_loss'] = np.nan*np.ones(max_iter)
    stats['test_corr'] = np.nan*np.ones(max_iter)
  
    return stats

def update_mean_predictor(stats: dict, mean_loss_train: List[float] = None, mean_loss_test: float = None):

    if mean_loss_train != None:
        stats['mean_loss_train'] = (np.mean(np.array(mean_loss_train)), np.std(np.array(mean_loss_train)))

    if mean_loss_test != None:
        stats['mean_loss_test'] = mean_loss_test
    
def update_train_progress(stats : dict, iter: int,
                  loss: List[float]=None, corr: List[float]=None,
                  learning_rate: float=None):
    """Updates various stats of the progress of training the model.

    Parameters
    ----------
    stats : dict
        a dictionary of progress statistics
    iter : int
        the current training iteration
    loss : List[float], optional
        a list of the loss (excluding regularizations) up to `iter` , by default None
    eig : List[float], optional
        a list of the spectral_radius up to `iter` , by default None
    learning_rate : float, optional
        the model learning rate at `iter`, by default None

    Returns
    -------
    stats : dict
        updated dictionary of progress statistics
    """
    if loss != None:
        stats['loss_mean'][iter] = np.mean(np.array(loss))
        stats['loss_sigma'][iter] = np.std(np.array(loss))
    if corr != None:
        stats['corr_mean'][iter] = np.mean(np.array(corr))
        stats['corr_sigma'][iter] = np.std(np.array(corr))
    if learning_rate != None:
        stats['learning_rate'][iter] = learning_rate

    return stats

def update_test_progress(stats: dict, iter: int, test_loss: int = None, test_corr: int = None):

    if test_loss != None:
        stats['test_loss'][iter] = test_loss
    if test_corr != None:
        stats['test_corr'][iter] = test_corr

    return stats

def lr_time_decay(iter:int, max_iter: int = 1000, beg_height: float = 1e-4, alpha: float = 0.1):
    """Calculates learning rate for given iteration during training using simple time-based decay """
    
    return beg_height / (1 + iter * alpha)

def custom_collate_fn(batch):
    drugs, proteins, affinity = zip(*batch)
    
    drug_batch = torch.stack(drugs)
    padded_proteins = pad_sequence(proteins, batch_first = True)
    affinity = torch.tensor(affinity)

    return drug_batch, padded_proteins, affinity

def test_model(model, test_loader, mean = None):

    device = model.device
    dtype = model.dtype

    model.eval()

    with torch.no_grad():
        loss = torch.nn.MSELoss(reduction = 'mean')
        
        for drug, protein, affinity in test_loader:
            drug, protein, affinity = drug.to(device), protein.to(device), affinity.to(device)
            
            # model pass
            pred_aff, attn = model(drug, dose)

            # calculate mean loss for first iteration
            if iter == 0 and mean != None:
                mean_test_loss = loss(torch.tensor(mean).repeat(*affinity.shape)).item()
                
            # loss + corr
            test_loss = loss(pred_aff, affinity).item()
            test_corr = torch.corrcoef(torch.stack([affinity.view(-1), pred_aff.view(-1)]))[0, 1].item()

    if mean is None:
        mean_test_loss = None
        
    return {'test_loss': test_loss, 'test_corr': test_corr, 'mean_test_loss': mean_test_loss }

def train_model(model, dataset, hyper_params, output_directory, time_limit = 12):
    """ Trains model on training dataset """

    exploding = False
    device = model.device
    dtype = model.dtype
    start_time = time.time()

    model.to(device)

    stats = initialize_progress(hyper_params['max_iter'])

    # defining loss and optimizer
    loss = torch.nn.MSELoss(reduction = 'mean')
    optimizer = torch.optim.Adam(model.parameters(), lr = 1, weight_decay=0, eps = 10e-4)

    # getting specific dataset of cell line and train split
    training_indices = dataset.get_train_test_indices('train')
    train_data = Subset(dataset, training_indices)
    train_loader = DataLoader(dataset = train_data, batch_size = hyper_params['batch_size'], shuffle = True, collate_fn = custom_collate_fn)

    # dataset of testing split
    testing_indices = dataset.get_train_test_indices('test')
    test_data = Subset(dataset, testing_indices)
    test_loader = DataLoader(dataset = test_data, batch_size=len(test_data), shuffle = True, , collate_fn = custom_collate_fn)

    for e in range(hyper_params['max_iter']):
        
        # learning rate
        cur_lr = lr_time_decay(e, hyper_params['max_iter'], beg_height = hyper_params['learning_rate'])
        optimizer.param_groups[0]['lr'] = cur_lr

        cur_loss = []
        cur_corr = []

        if e == 0:
            mean_loss = []
            
        model.train()
        
        # training
        for drug, protein, affinity in train_loader:
            
            optimizer.zero_grad()

            drug, protein, affinity = drug.to(device), protein.to(device), affinity.to(device)

            # model pass
            pred_aff, attn = model(drug, dose) # predicting binding affinity

            # calculating mean predictor loss
            if e == 0:
                mean = dataset.mean
                iter_mean_loss = loss(affinity, torch.tensor(mean).repeat(*affinity.shape))
                mean_loss.append(iter_mean_loss.item())
            
            # getting loss equation
            fit_loss = loss(affinity, pred_aff) # normal loss
            param_reg = model.L2_reg(hyper_params['param_lambda_L2']) # all model weights and signaling network biases
            total_loss = fit_loss + param_reg
            
            # correlation
            correlation = torch.corrcoef(torch.stack([affinity.view(-1), pred_aff.view(-1)]))[0, 1] # getting total correlation per TF

            # backpropagation and optimizing
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # checking for exploding or vanishing loss to stop early
            if math.isnan(fit_loss.item()):
                print(e, fit_loss, cur_lr)
                exploding = True
                break

            # store
            cur_loss.append(fit_loss.item())
            cur_corr.append(correlation.item())

        if e == 0:
            stats = update_mean_predictor(stats, mean_loss_train = mean_loss)
        
        stats = update_train_progress(stats, iter = e, loss = cur_loss, corr = cur_corr, 
                                      learning_rate = cur_lr)

        # test progress every 50 iterations
        if e % 50 == 0 or e == hyper_params['max_iter'] - 1:
            if e == 0:
                test_results = test_model(model, test_loader, dataset.mean)
                stats = update_test_progress(stats, iter = e, test_loss = test_results['test_loss'], 
                                         test_corr = test_results['test_corr'])
                stats = update_mean_predictor(stats, test_results['mean_loss_test'])
            else:
                test_results, mean_loss_test = test_model(model, test_loader)
                stats = update_test_progress(stats, iter = e, test_loss = test_results['test_loss'], 
                                             test_corr = test_results['test_corr'])
            
        
        # save model every 150 iterations
        if e % 150 == 0 and e > 0:
            torch.save(model.state_dict(), f'{output_directory}/model_epoch_{e}')

        # check the time and save output if running out of time
        if time.time() - start_time > 60*60*(time_limit - 0.25):
            f = open(f"{output_directory}/stats_dict.pkl",  "wb")
            pickle.dump(stats,f)
            f.close()
            
    return model, cur_loss, cur_eig, cur_corr, stats
