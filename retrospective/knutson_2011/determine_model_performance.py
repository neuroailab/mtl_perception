import warnings; warnings.simplefilter('ignore')
from scipy.misc import imread, imresize
import numpy as np
from PIL import Image
import os, sys, imp, pickle, pandas
import tensorflow.compat.v1 as tf
a = np.array

def load_image_data(base_directory): 
    
    # locate directory with all images
    image_path = os.path.join(base_directory, 'stimuli')
    # get all image file names 
    files = np.array([f[0:-4] for f in np.sort(os.listdir(image_path)) if 'png' in f])
    # arrange image file names and ignore broken trial 
    slides = np.arange(1,65)[np.arange(1,65)!=48]
    # locate metadata file  
    behavior_file = os.path.join(base_directory, 'human_behavior/difficulty_key.xlsx')
    # load metadata 
    difficulty_key = pandas.read_excel(behavior_file)
    # clean up metadata 
    difficulty = {'level':difficulty_key['difficulty level'].values, 
                  'slide_number':difficulty_key['Powerpoint slide number'].values}
    
    stimuli = {} 

    for i_slide in slides:
        
        # initialize data structure 
        stimuli[i_slide] = {} 
        # define human readable name 
        names = [f + '.png' for f in files if '_' + str(i_slide)+'_' in f]
        # extract and resize all images in trial
        imgs = [imresize(a(Image.open(os.path.join(image_path,i))),(224,224))[:,:,0:3] for i in names]
        # store images
        stimuli[i_slide]['images'] = imgs
        # store name 
        stimuli[i_slide]['labels'] = names
        # store number of groups 
        stimuli[i_slide]['n_groups'] = int((len(names)-1)/2)
        # store difficulty
        stimuli[i_slide]['difficulty'] = difficulty['level'][int(i_slide)-1]
        
    return stimuli, difficulty

def define_model(path_to_model):

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

def model_responses_to_stimuli(vgg, sess, stimuli):

    # initialize data types
    data = {'conv5_1':{},'labels':{}, 'difficulty':{}} 

    # iterate over slides
    for i_slide in list(stimuli.keys()):
        
        # extract slide images
        slide_images = a(stimuli[i_slide]['images'])

        # extract model responses to images
        features = sess.run([ vgg.conv5_1 ], feed_dict={vgg.imgs: slide_images})[0]
        # store model representations and labels 
        data['conv5_1'][i_slide] = np.reshape(features, (len(features), -1))
        # store labels for this strial 
        data['labels'][i_slide] = stimuli[i_slide]['labels']
        # store difficulty for this trial 
        data['difficulty'][i_slide] = stimuli[i_slide]['difficulty'] 

    return data

def evaluate_trial(trial_responses, correct_index=0):
    """returns model performance on trial"""
    
    # set structure to extract off diagonal 
    x = np.array(list(range(len( trial_responses ))))
    # determine item-by-item covariance matrix from model responses
    trial_covariance = np.corrcoef(trial_responses)
    # extract off diagonal 
    trial_decision_space = np.array([trial_covariance[i, x[x!=i]] for i in x])
    # sort off diagonal 
    trial_decision_space.sort()
    # determine model-selected oddity
    i_choice = trial_decision_space[:,-1].argmin()
    # determine whether model selected oddity is correct
    correct = i_choice == (correct_index)

    return correct

def determine_model_performance(model_responses): 
    
    # initialize data type
    model_performance = {i:[] for i in np.unique( difficulty['level'] )}

    for i_trial in list(model_responses['labels']): 
        # determine model accuracy on trial
        trial_correct = evaluate_trial( model_responses['conv5_1'][i_trial] ) 
        # determine difficulty of trial
        trial_difficulty = model_responses['difficulty'][i_trial]
        # save model performance by trial difficulty
        model_performance[trial_difficulty].append( trial_correct )
    
    # compute average
    model_performance = {i: np.mean(model_performance[i]) for i in model_performance}
    
    # save model performance
    with open('model_performance.pickle', 'wb') as f: 
        pickle.dump(model_performance, f)
    
    return model_performance

if __name__ == '__main__': 
   
    # set base directory all analyses are contained within
    base_directory = os.path.abspath('..')
    # location of experimental data
    stimulus_directory  = os.path.join(base_directory, 'experiments/knutson_2011/stimuli')
    # load stimuli information
    stimuli, difficulty = load_image_data(stimulus_directory)
    # set path to model
    path_to_model = os.path.join(base_directory, 'models')
    # load model
    model, session = define_model(path_to_model)
    # extract model responses to stimuli
    model_responses = model_responses_to_stimuli(model, session, stimuli)
    # determine model performance 
    model_performance = determine_model_performance(model_responses)
