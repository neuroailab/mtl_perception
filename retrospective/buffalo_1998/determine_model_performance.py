import warnings; warnings.simplefilter('ignore')
from scipy.misc import imread, imresize
import tensorflow as tf
import numpy as np
from PIL import Image
import os, sys, imp, pickle

# for convenience 
a = np.array

def extract_stimuli(path_to_stimuli): 
    """return all stimuli"""
    # extract all stimulus files 
    stim_filenames = [i for i in np.sort(os.listdir(path_to_stimuli)) if 'jpg' in i]
    
    stimuli = []
    for i_file in stim_filenames:
        # identify full path to file 
        file_path = os.path.join(path_to_stimuli, i_file)
        # load and resize image 
        img = imresize(a(Image.open(file_path)), (224, 224))
        # make sure we're not repeating stimuli! 
        if not sum( [np.sum(img==i) // len(img.flatten()) for i in stimuli]):
            # append stimulus
            stimuli.append(img)
   
   return stimuli

def generate_trial(stimuli): 
    """returns indices for randomly selected trial"""
    
    # set number of stimuli in each trial
    n_observations = 4
    # get a random order of all stimulus labels
    shuffle = np.random.permutation(list(range(1, len(stimuli))))
    # select the first four stimuli in the random shuffle
    match_screen = shuffle[:n_observations]
    # set the sample that should be used in this trial
    sample_screen = match_screen[np.random.randint(n_observations)]
    # start to create the trial with the sample image
    combine = [ sample_screen ] 
    # append the other stimuli to generate a list
    combine.extend( match_screen ) 
    
    return match_screen, sample_screen, combine

def model_responses_to_stimuli(stimuli, model, session):
    
    # initialize all data structures
    labels, activations, errors, removed_pairs = [], [], [], []
    
    for i_image in range(len(stimuli)):
    
        # repeat in z and then expand the first index
        image_i = np.expand_dims(stimuli[i_image], axis=0)
        # extract model representations
        activation = session.run([model.conv5_1], feed_dict={model.imgs: image_i})[0]
        # append flattened model responses 
        activations.append(activation.flatten())

    return np.array( activations )


def define_model(path_to_model):
    """return model and session"""
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
    def get_session():
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        config.gpu_options.allow_growth = True
        return tf.Session(config = config)

    print('-- initiate session')
    session = get_session()
    print('-- load model')
    vgg16 = imp.load_source('vgg16', path_to_model + 'vgg16.py')
    print('-- define input structure')
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    print('-- load model weights')
    vgg = vgg16.vgg16(imgs, path_to_model + 'vgg16_weights.npz', session)

    return vgg, session

def determine_model_performance(model_responses):
    """returns model performance on experiment"""

    # set parameters for generating pseudo experiments 
    n_trials, n_blocks, n_subjects = 100, 3, 3
    # initialize data structure
    accuracy = []
    
    # begin experiment for first "subject" 
    for i_subject in range(n_subjects): 

        # generate all "trials" 
        for i_trial in range(n_trials*n_blocks):
            
            # generate this pseudo trial 
            match, sample, trial_indices = generate_trial(stimuli)
            # determine covariance matrix of model responses
            trial_covariance = np.corrcoef(model_responses[trial_indices, :])
            # determine correct response location
            true_index = np.nonzero(np.array(sample)==match)[0][0] 
            # determine model-selected oddity
            model_index = np.argmax( trial_covariance[0,1:] ) 
            # append model accuracy
            accuracy.append( true_index == model_index )
    
    # convert to average accuracy and save 
    with open('model_performance.pickle', 'wb') as f: 
        pickle.dump( {'zero_delay': np.mean(accuracy) }, f) 
    
    return {'zero_delay': np.mean( accuracy )}

if __name__ == '__main__': 
    
    # set base directory all analyses are contained within
    base_directory = os.path.abspath('..')
    # stimulus directory=
    stimuli_directory = os.path.join(base_directory, 'experiments/buffalo_1998/stimuli')
    # extract stimuli 
    stimuli = extract_stimuli(stimuli_directory)    
    # model directory
    path_to_model = '/Users/biota/work/perirhinal_cortex/analysis/models/'
    # define model
    model, session = define_model(path_to_model)
    # extract model responses to all stimuli
    model_responses = model_responses_to_stimuli(stimuli, model, session)
    # determine model performance
    model_performance = determine_model_performance(model_responses)
