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
    stats['mean_loss_sigma'] = np.nan*np.ones(max_iter)
    stats['mean_loss_mean'] = np.nan*np.ones(max_iter)

    # learning rate
    stats['learning_rate'] = np.nan*np.ones(max_iter)

    # testing data
    stats['test_loss'] = np.nan*np.ones(max_iter)
    stats['test_corr'] = np.nan*np.ones(max_iter)
  
    return stats

def update_progress(stats : dict, iter: int,
                  loss: List[float]=None, corr: List[float]=None, 
                  test_loss: float=None, test_corr: float=None,
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
    if test_loss != None:
        stats['test_loss'][iter] = test_loss
    if test_corr != None:
        stats['test_corr'][iter] = test_corr

    return stats


def train_model(model, dataset, output_directory, time_limit = 12):
    """ Trains model on training dataset """
    
    device = model.device
    dtype = model.dtype
    start_time = time.time()

    model.to(device)

    stats = utils.initialize_progress(hyper_params['max_iter'])

    # defining loss and optimizer
    loss = torch.nn.MSELoss(reduction = 'mean')
    optimizer = torch.optim.Adam(model.parameters(), lr = 1, weight_decay=0, eps = 10e-4)

    # getting specific dataset of cell line and train split
    training_indices = dataset.get_split_indices(cell_line, 'train')
    train_data = Subset(dataset, training_indices)
    train_loader = DataLoader(dataset = train_data, batch_size = hyper_params['batch_size'], shuffle = True)

    # dataset of testing split
    testing_indices = dataset.get_split_indices(cell_line, 'test')
    test_data = dataset.create_sub_dataset(testing_indices)
    test_loader = DataLoader(dataset = test_data, batch_size=len(test_data), shuffle = True)

    for e in range(hyper_params['max_iter']):

        # learning rate
        cur_lr = get_lr_new(e, hyper_params['max_iter'], max_height = hyper_params['learning_rate'],
                            start_height = hyper_params['learning_rate']/10, end_height = 1e-6,
                            peak = 300)
        optimizer.param_groups[0]['lr'] = cur_lr

        cur_loss = []
        cur_corr = []

        # training
        for drug, protein, affinity in train_loader:
            model.train()
            optimizer.zero_grad()

            drug, protein, affinity = drug.to(device), protein.to(device), affinity.to(device)

            # model pass
            pred_aff, attn = model(drug, dose) # predicting binding affinity

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

            # store
            cur_loss.append(fit_loss.item())
            cur_corr.append(correlation.item())

        #testing
        model.eval()
        with torch.no_grad():
          for drug, protein, affinity in test_loader:
  
            drug, protein, affinity = drug.to(device), protein.to(device), affinity.to(device)
  
            # model pass
            pred_aff, attn = model(drug, dose) # predicting binding affinity
  
            # getting loss
            fit_loss = loss(affinity, pred_aff) # normal loss
  
            # correlation
            correlation = torch.corrcoef(torch.stack([affinity.view(-1), pred_aff.view(-1)]))[0, 1] # getting total correlation per TF
  
        stats = utils.update_progress(stats, iter = e, loss = cur_loss, corr = cur_corr, 
                                      test_loss = fit_loss, test_corr = correlation, 
                                      learning_rate = cur_lr

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
