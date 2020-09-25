import warnings; warnings.simplefilter('ignore')
import os, pickle, imp, pandas, sys
import numpy as np
from scipy.misc import imread, imresize
from sklearn import cluster
from PIL import Image

def extract_stimuli(stimulus_directory): 
    
    stimuli = {} 
    task_folders = [i for i in os.listdir(stimulus_directory) if ('Faces' in i or 'objects' in i) * ('2' in i)]

    for i_folder in  [i for i in task_folders]: 

        i_type = i_folder[:-6]
        i_set = [1, 2]['2' in i_folder[-6:]]
        if i_type not in stimuli: stimuli[i_type] = {}
            
        for i_file in [i for i in os.listdir(os.path.join(stimulus_directory, i_folder)) if 'zip' not in i] : 
            # load, reshape, and store 
            i_image = Image.open(os.path.join(stimulus_directory, i_folder, i_file))
            i_image = imresize(i_image, (224, 224))
            i_image = np.expand_dims( np.repeat(i_image[ :, : , np.newaxis], 3, axis=2), 0 ) 
            stimuli[i_type][i_file[:-4]] = np.array( i_image ) 
       
    return stimuli

def define_model(path_to_model):
    
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    def get_session():
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        config.gpu_options.allow_growth = True
        return tf.Session(config = config)
    
    print('-- initiate session')
    session = get_session()
    print('-- load model')
    vgg16 = imp.load_source('vgg16', os.path.join(path_to_model, 'vgg16.py'))
    print('-- define input structure')
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    print('-- load model weights')
    vgg = vgg16.vgg16(imgs, os.path.join(path_to_model, 'vgg16_weights.npz'), session)
    
    return vgg, session

def extract_model_responses(vgg, session, stimuli, i_experiment): 
    
    layers = {'conv1_1': vgg.conv1_1, 'conv1_2':vgg.conv1_2, 'pool1': vgg.pool1,
              'conv2_1': vgg.conv2_1, 'conv2_2':vgg.conv2_2, 'pool2': vgg.pool2,
              'conv3_1': vgg.conv3_1, 'conv3_2':vgg.conv3_2, 'conv3_3':vgg.conv3_3, 'pool3': vgg.pool3,
              'conv4_1': vgg.conv4_1, 'conv4_2':vgg.conv4_2, 'conv4_3':vgg.conv4_3, 'pool4': vgg.pool4,
              'conv5_1': vgg.conv5_1, 'conv5_2':vgg.conv5_2, 'conv5_3':vgg.conv5_3, 'pool5': vgg.pool5,
              'fc6': vgg.fc1, 'fc7':vgg.fc2,  'fc8':vgg.fc3l}

    model_responses = {} 
    for image_name in list(stimuli[i_experiment]):

        i_image = stimuli[i_experiment][image_name]
        features = session.run([[layers[i] for i in layers]], feed_dict={vgg.imgs: i_image})[0]
        model_responses[image_name] = {list(layers)[i]: features[i].flatten() for i in range(len(layers))}
        model_responses[image_name]['pixel'] = np.array(i_image).flatten() 
        
    return model_responses

def evaluate_trial(trial_responses, correct_index=0): 
    x = np.array(list(range(len( trial_responses ))))  
    trial_covariance = np.corrcoef(trial_responses)
    trial_decision_space = np.array([trial_covariance[i, x[x!=i]] for i in x])
    trial_decision_space.sort()
    i_choice = trial_decision_space[:,-1].argmin() 
    correct = i_choice == (correct_index)
    return correct 

def evaluate_experiment_one(model_responses, stimuli, i_experiment): 
    
    model_layers = list(model_responses[list(model_responses)[0]])
    layer_responses = {l:[] for l in model_layers}

    if 'Face' in i_experiment: 
        i_len = 6 
    else: 
        i_len = 5 

    n_trials = len(model_responses)//i_len

    for i_object in range(1, n_trials + 1): 

        oddity_group = (i_object%n_trials) + 1
        trial_stimuli = ['%d%d'%(i_object, i_view) for i_view in range(1, i_len+1)]
        oddity_viewpoint = np.random.randint(1, len(trial_stimuli))
        i_oddity = '%d%d'%(oddity_group, oddity_viewpoint+1)
        trial_stimuli.pop(oddity_viewpoint)
        trial_stimuli.insert(0, i_oddity)

        for l in model_layers: 
            trial_responses = np.array( [model_responses[i][l] for i in trial_stimuli] )
            layer_responses[l].append( evaluate_trial(trial_responses) )
        
    return {l: np.mean(layer_responses[l]) for l in model_layers}

def model_performance_experiment_one(model, session, stimulus_directory): 
    
    print('-- extract stimuli for all experiments') 
    stimuli = extract_stimuli(stimulus_directory)    
    print('-- iterate through experiments') 
    model_performance = {} 
    for i_experiment in list(stimuli): 
        
        print('-- extract model responses to stimuli in each experiment: %s'%i_experiment)
        model_responses = extract_model_responses(model, session, stimuli, i_experiment)
        print('-- determine model performance on stimulus set') 
        model_performance[i_experiment] = evaluate_experiment_one(model_responses, stimuli, i_experiment)
        
    return model_performance

def model_experiment_two(features, answer, n_iterations): 
    
    # pseudo subjects
    experimental_n = 3
    # determine all layers 
    layers = list( features[ list(features)[0] ])
    # initialize data type
    performance = {l:[] for l in layers}
    # iterate over pseudo experiments
    for i_iteration in range(n_iterations): 
        
        # true experiment is not known, only protocol used to generate it
        pseudo_experiment = np.random.permutation(list(features))[0:experimental_n]
        
        # estimate performance across all layers
        for l in layers: 
            pseudo_performance = [evaluate_trial(features[t][l], answer[t]) for t in pseudo_experiment]
            performance[l].append( np.mean(pseudo_performance) )
            
    return {l: np.mean(performance[l]) for l in layers}

def experiment_two_model_performance(model, session, stimulus_directory): 
    
    # add lee 2006 to path 
    sys.path.append(os.path.abspath('..'))
    # load 2006 functions 
    import lee_2006.determine_model_performance as lee
    # model performance for all stimuli 
    stimuli, answer_key = lee.extract_stimuli(stimulus_directory)
    
    performance = {} 
    n_iterations = 100

    for exp in list( stimuli ): 
        print('-- extract model responses for experiment %s'%exp)
        model_responses = lee.extract_model_responses(model, session, {exp: stimuli[exp]})
        print('-- determine model performance %s'%exp)
        performance[exp] = model_experiment_two(model_responses[exp], answer_key[exp], n_iterations)
    
    return performance

if __name__ == '__main__': 
     
    # set seed to fix outcome of probablistic protocol
    np.random.seed(0) 
    # set base directory all analyses are contained within
    base_directory = os.path.abspath('..')
    # set path to model
    path_to_model = os.path.join(base_directory, 'model')    
    # define model 
    model, session = define_model(path_to_model)
    # path to stimulus directory
    stim_dir = os.path.join(base_directory, 'experiments/lee_2005/stimuli') 
    # determine model performance for experiment one
    model_performance = {'experiment_one': model_performance_experiment_one(model, session, stim_dir)}
    # stimulus directory for shared 2006 experiment
    shared_directory = os.path.join(base_directory, 'experiments/lee_2006/stimuli')
    # determine model performance for experiment two 
    model_performance['experiment_two'] = experiment_two_model_performance(model, session, shared_directory)
    # save 
    with open('model_performance.pickle', 'wb') as handle: 
        pickle.dump(model_performance, handle)
