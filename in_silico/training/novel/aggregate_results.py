import pandas, h5py, os, pickle, numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings; warnings.filterwarnings("ignore")


def across_models(): 

    # base directory for this repository
    base = '/Users/biota/work/mtl_perception/'
    results_directory = os.path.join(base, 'in_silico/training/novel_model_performance/results/')

    models = {} 

    for M in [m for m in os.listdir(results_directory) if 'vgg16_neural' not in m]: 
        with open(os.path.join(results_directory, M), 'rb') as handle: 
            models[M] = pickle.load(handle)

    # initialize data frame
    df_models = pandas.DataFrame()

    for result in models:

        # use '_' to identify information from label
        breaks = np.nonzero([l == '_' for l in result])[0]

        # iterate overa all stimuli--where each unique stimuli has it's own 'marker'
        for i in models[result]:

            # remove silly formating issues
            meta = models[result][i]['meta']

            # extract performance and meta data from stimulus
            i_marker = pandas.Series({'model':result[:breaks[0]],
                                      'training_data': result[breaks[0] + 1: breaks[1]],
                                      'image_type': result[breaks[1] + 1:breaks[2]],
                                      'readout': result[breaks[2] + 1:breaks[3]],
                                      'trial_id':meta['marker'],
                                      'typical_name': meta['typical_name'],
                                      'oddity_name': meta['oddity_name'],
                                      'category':meta['category'],
                                      'accuracy':np.mean(models[result][i]['correct']),
                                     })

            # append to data frame
            df_models = df_models.append(i_marker, ignore_index=True)

    save_name = 'multiple_models_novel_stimuli.csv'
    df_models.to_csv(save_name)
    print(save_name) 

if __name__ == '__main__': 
    across_models()
