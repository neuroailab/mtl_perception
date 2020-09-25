from scipy.misc import imread, imresize
from scipy.stats.mstats import zscore
import tensorflow as tf
import numpy as np
from PIL import Image
import os, sys, imp, itertools, pickle 
import matplotlib.pyplot as plt
import warnings; warnings.simplefilter('ignore')
import tensorflow.compat.v1 as tf
a = np.array

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
    vgg16 = imp.load_source('vgg16', os.path.join(path_to_model,'vgg16.py'))
    print('-- define input structure')
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    print('-- load model weights')
    vgg = vgg16.vgg16(imgs, os.path.join(path_to_model, 'vgg16_weights.npz'), session)

    return vgg, session

def extract_stimuli(path_to_stimuli): 
    
    all_directories = np.sort(os.listdir(path_to_stimuli))

    stimuli = {} 
    meta = {} 
    for i_directory in [i for i in all_directories if i in ['FACES', '3DGREYSC'] ]: 
        file_path = os.path.join(path_to_stimuli, i_directory)
        prefix = np.sort([int(i[0:-4]) for i in os.listdir(file_path)])

        # load and shape image into correct size
        stimuli[i_directory] = [imresize(a(Image.open(os.path.join(file_path, str(i_stim) + '.bmp' ))), (224, 224)) for i_stim in prefix]  

        identifiers = [i[-5:-4] for i in os.listdir(file_path)]
        n_viewpoints = len(np.unique(identifiers))
        n_categories = len(identifiers)/n_viewpoints

        meta[i_directory] = {'identifiers':identifiers, 
                            'n_viewpoints':n_viewpoints, 
                            'n_categories':n_categories}

    # generate snow conditions
    stimuli['3DGREYSC_snow_1'] = [snow(i_image,1) for i_image in stimuli['3DGREYSC']] 
    stimuli['3DGREYSC_snow_2'] = [snow(i_image,2) for i_image in stimuli['3DGREYSC']] 
    stimuli['3DGREYSC_snow_3'] = [snow(i_image,3) for i_image in stimuli['3DGREYSC']]
	   
    return stimuli, meta, n_viewpoints, n_categories
    
def snow(i_image, magnitude): 
    
    img = np.copy(i_image.flatten())
    noise_percent = [0,.3, .5, .7][ magnitude ]  
    if noise_percent: 
        noise_percent = noise_percent + np.random.randint(0,10)/100
        
    mask = np.random.permutation(list(range(len(img)))) > int(len(img) * (1 - noise_percent))
    noise = np.random.randn(len(img)) * np.std(i_image) + np.mean(i_image)    
    img[mask] = noise[mask]
    img.shape = np.shape(i_image)
    
    return img
    
def generate_trial(i_group, noise_magnitude=0, visualize=1, input_type='human'): 
    
    if "3DGREYSC" in i_group: i_group = i_group[:8]
    
    n_viewpoints = meta[i_group]['n_viewpoints']
    n_groups = meta[i_group]['n_categories']
    target_identity = np.random.randint(n_groups)
    
    possible_distractor_identity = [i for i in np.arange(n_groups) if i !=target_identity]
    distractor_identity = possible_distractor_identity[np.random.randint(len(possible_distractor_identity))]
    
    target_viewpoint = np.random.randint(n_viewpoints)
    if i_group == 'FACES': 
        distractor_viewpoints = np.random.permutation([i for i in np.arange(n_viewpoints) if i !=target_viewpoint])
    else: 
        distractor_viewpoints = np.arange(5)
        
    # now convert these to image indices
    ij_target = target_viewpoint+(target_identity*n_viewpoints)
    ij_distractors = [i_distractor + (distractor_identity*n_viewpoints) for i_distractor in distractor_viewpoints]
    ij_distractors = np.random.permutation(ij_distractors)
    trial_stims = [ij_target]
    trial_stims.extend(ij_distractors)
    
    if visualize: 
        
        show_trial_stims(i_group, trial_stims, n_viewpoints, noise_magnitude, input_type=input_type)
    
    # return the indices of these trial stims
    return [int(i) for i in trial_stims]

def model_responses_to_stimuli(vgg, sess, stimuli, i_exp):  
    
    print('extracting model responses for stimulus set:', i_exp)
    layer_map = {'conv1_1': vgg.conv1_1, 'conv1_2':vgg.conv1_2, 'pool1': vgg.pool1, 
                 'conv2_1': vgg.conv2_1, 'conv2_2':vgg.conv2_2, 'pool2': vgg.pool2, 
                 'conv3_1': vgg.conv3_1, 'conv3_2':vgg.conv3_2, 'conv3_3':vgg.conv3_3, 'pool3': vgg.pool3, 
                 'conv4_1': vgg.conv4_1, 'conv4_2':vgg.conv4_2, 'conv4_3':vgg.conv4_3, 'pool4': vgg.pool4, 
                 'conv5_1': vgg.conv5_1, 'conv5_2':vgg.conv5_2, 'conv5_3':vgg.conv5_3, 'pool5': vgg.pool5, 
                 'fc6': vgg.fc1, 
                 'fc7':vgg.fc2, 
                 'fc8':vgg.fc3l, 
                }
    
    labels, activations, errors, removed_pairs = [], [], [], [] 
    model_responses = {l:[] for l in list(layer_map)} ; 
    model_responses['pixel'] = [] 
    
    for i_image in range(len(stimuli[i_exp])):
    
        # format images
        image_i = np.expand_dims(np.repeat(stimuli[i_exp][i_image][ :, : , np.newaxis], 3, axis=2), axis=0)
        
        # extract model representations 
        i_responses = sess.run([[layer_map[i] for i in layer_map]], feed_dict={vgg.imgs: image_i})[0]
        
        for i in range(len(list(layer_map))): 
            
            model_responses['pixel'].append( stimuli[i_exp][i_image].flatten() ) 
            model_responses[list(layer_map)[i]].append(i_responses[i].flatten() ) 
            
    return model_responses

def run_single_experiment(model_data, i_category, n_subjects, n_trials):
    
    print('estimating model performance for %d pseudosubjects completing %d trials'%(n_subjects, n_trials))
         
    x = np.array(list(range(6)))
    layers = list(model_data)
    layers.insert(0, 'pixel')
    trial_decision = {l:[] for l in layers}
    condition_accuracy = {l:[] for l in layers}
    
    for i_subject in range(n_subjects):  
        
        for i_iteration in range(n_trials): 
            
            trial_stims = generate_trial(i_category, visualize=0)
            
            for i_layer in layers: 
                
                responses = [model_data[i_layer][i] for i in trial_stims]
                trial_covariance = np.corrcoef(responses)
                trial_decision_space = np.array([trial_covariance[i, x[x!=i]] for i in x])
                trial_decision_space.sort()
                i_choice = trial_decision_space[:,-1].argmin() 
                correct = i_choice == 0 
                trial_decision[i_layer].append(correct)
    
    condition_accuracy = {l: np.mean(trial_decision[l]) for l in layers}
    
    return condition_accuracy


if __name__ == '__main__': 
    
    # experiment is probablistic -- generating noised stimuli & distribution of trials 
    np.random.seed(0) 
    # location for all stimuli and model folders 
    base_directory = os.path.abspath('..') 
    # set path to experimental stimuli
    path_to_stimuli = os.path.join(base_directory, 'experiments/stark_2000/stimuli')
    # extract all stimuli 
    stimuli, meta, n_viewpoints, n_categories = extract_stimuli(path_to_stimuli)
    # path 
    path_to_model = os.path.join(base_directory, 'model')
    # load model 
    vgg, sess = define_model(path_to_model)
    # set experimental parameters 
    n_trials = 100
    n_subjects = 3
    # perform experiment in sequence
    experiments = {} 
    for i_experiment in [s for s in stimuli if s != '3DGREYSC']: 
	# extract model responses for each stimulus set
        model_data = model_responses_to_stimuli(vgg, sess, stimuli, i_experiment)
	# determine model performance on stimulus set
        experiments[i_experiment] = run_single_experiment(model_data, i_experiment, n_subjects, n_trials)

    # store results
    with open('model_performance.pickle', 'wb') as handle: 
        pickle.dump(experiments, handle)
