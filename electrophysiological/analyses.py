import h5py, pickle, os, numpy as np

def split_half_correlation(vvs_data_path, n_iterations, metric): 
    """
    vvs_data_path = path to ventral_neural_data.hdf5, 
    n_iterations = number of independent split halves 
    metric = function to compare splits--e.g np.mean or np.median
    """
    # set seed 
    np.random.seed(0)
    # electrophysiological data from IT and V4 from Majaj et al. 2015 
    data = h5py.File(vvs_data_path, 'r')
    # extract all trials for each time averaged neuron at variation level 3 
    trials = np.array(data['time_averaged']['variation_level_3']) 
    # initialize 
    split_half = {r:[] for r in ['IT_NEURONS', 'V4_NEURONS']}
    # extract indices of IT and V4 neurons
    region_idxs = {r: np.array(data['neural_meta'][r]) for r in split_half}
    # close file
    data.close()
    
    for i_permutation in range(n_iterations): 

        # generate a random split and order it in a way that will be easy to index 
        shuffle = np.random.permutation(trials.shape[0])

        # iterate over regions 
        for i_region in split_half: 

            # extract region -- trial x image x neuron 
            region = trials[:, :, region_idxs[i_region]] 

            # determine split-half correlation for each neuron 
            for i_neuron in range(region.shape[2]): 

                # extract neuron's responses to half of trials -- across images
                split_ = region[shuffle[:len(shuffle)//2],:,i_neuron].mean(0)
                # extract neuron's responses to half of trials -- across images
                _split = region[shuffle[len(shuffle)//2:],:,i_neuron].mean(0)
                # compute the correlation between each half across all images
                neuron_split_reliability = np.corrcoef(split_, _split)
                # store the off diagonal
                split_half[i_region].append(neuron_split_reliability[0, 1]) 
    
    return split_half

def layer_fits(pls_vvs_dir, measure, split_half):

    model_layers = {}
    for i in os.listdir(pls_vvs_dir):
        with open(os.path.join(pls_vvs_dir, i), 'rb') as handle:
            model_layers[ i[:i.find('-')] ] = pickle.load(handle)

    order=['pool1', 'conv2_1', 'conv2_2','pool2',
           'conv3_1','conv3_2','conv3_3','pool3',
           'conv4_1','conv4_2','conv4_3','pool4',
           'conv5_1','conv5_2','conv5_3','pool5',
           'fc6','fc7','fc8']

    it_fit = np.array([measure(model_layers[i]['it_test']) for i in order])
    v4_fit = np.array([measure(model_layers[i]['v4_test']) for i in order])

    it_std = np.array([np.std(model_layers[i]['it_test']) for i in order])
    v4_std = np.array([np.std(model_layers[i]['v4_test']) for i in order])

    it_noise_ceiling = measure(split_half['IT_NEURONS'])
    v4_noise_ceiling = measure(split_half['V4_NEURONS'])

    it_fit_corrected = np.array([measure(model_layers[i]['it_test']) / it_noise_ceiling for i in order])
    v4_fit_corrected = np.array([measure(model_layers[i]['v4_test']) / v4_noise_ceiling for i in order])

    delta = it_fit - v4_fit

    data = {'it':{'mu':it_fit_corrected, 'std':it_std},
            'v4':{'mu':v4_fit_corrected, 'std':v4_std},
            'it_uncorrected':{'mu':it_fit, 'std':it_std},
            'v4_uncorrected':{'mu':v4_fit, 'std':v4_std},
            'delta':delta,
            'layers': order}

    return data

if __name__ == '__main__': 
    
    # base directory 
    electrophysiological_dir = os.getcwd()

    # path to neural data collected in Majaj et al. 2015
    vvs_data_path = os.path.join(electrophysiological_dir, 'ventral_neural_data.hdf5')

    # define n iterations for split half reliability analysis 
    n_iterations = 10
    neural_split_half = split_half_correlation(vvs_data_path, n_iterations, np.mean)
    
    # set path to results from pls fitting procedure 
    pls_vvs_dir = os.path.join(electrophysiological_dir, 'pls_fitting_results')
    
    # extract results from pls fitting procedures and adjust by noise ceiling in neural data
    pls_fits = layer_fits(pls_vvs_dir, np.median, neural_split_half)

    print( pls_fits )
