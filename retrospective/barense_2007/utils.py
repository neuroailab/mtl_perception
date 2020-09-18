import os, pickle
import numpy as np
import os, imp, pickle
import numpy as np
from PIL import Image
from scipy.misc import imread, imresize
import tensorflow as tf

stimulus_directory = '../../prc_stimuli/Barense_2007_NYS/'

def find(a_list):
    return [i for i in range(len(a_list)) if a_list[i] == 1 ]

def import_rotation_keys(): 
    
    with open('rotation_keys.pkl', 'rb') as f:
        keys = pickle.load(f)

    return keys

def load_images(source):

    if source=='extract':

        stimuli, originals = extract_all_stimuli()

        obj = {'stimuli':stimuli, 'originals':originals}

        with open('experimental_stimuli.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    if source=='preload':

        with open('experimental_stimuli.pkl', 'rb') as f:
            obj = pickle.load(f)

        stimuli = obj['stimuli']
        originals = obj['originals']

    return stimuli, originals


def extract_answer_key(stimulus_directory):
    answer_key = {}
    answer_directory = os.path.join(stimulus_directory, 'anwser_keys')
    answer_files = [i for i in os.listdir(answer_directory) if '.edat' not in i]
    model_names = {'famobjectHIGH':'familiar','famobjectLOW':'familiar','greeblesHIGH':'novel', 'greeblesLOW':'novel', 'size':'size', 'colour':'color', 'fribbles_block2':'oddity'}

    for stimulus_type in answer_files:


        answer_file = os.path.join(answer_directory, stimulus_type)
        answers = open(answer_file).readlines()

        # convert to name in the models
        tag = stimulus_type[0:-9]
        name = model_names[stimulus_type[0:-9]]
        if name not in answer_key.keys(): answer_key[name] = {}

        if 'HIGH' in tag: answer_key[name]['high'] = {}
        elif 'LOW' in tag: answer_key[name]['low'] = {}
        else: answer_key[name] = {}

        for i_line in range(len(answers)):

            if 'ImageShow' in answers[i_line]:
                q = answers[i_line][str.find(answers[i_line],'_')+1:str.find(answers[i_line],'.')]
                if 'HIGH' in tag: answer_key[name]['high'][q] = answers[i_line-5][-2:-1]
                elif 'LOW' in tag : answer_key[name]['low'][q] = answers[i_line-5][-2:-1]
                else: answer_key[name][q] = answers[i_line-5][-2:-1]
        
    return answer_key


def extract_all_stimuli():

    """ 
    this function is rediculous... extracts all stimulus types and segments objects in preparation for modeling. 
    """

    def stabalize_and_segment_image(img): 
        
        # get half width and hight
        l = int(img.shape[0]/2)
        h = int(img.shape[1]/2)
        # set area around object
        padding = 10
        # extract objects from image -- IN THE CORRECT ORDER XD 
        shapes = {0:img[0:l,0:h], 1:img[0:l,h:], 2:img[l:,0:h], 3:img[l:,h:]}
        # determine what the width and height of the largest object in this image is
        l_max = max([sum(np.mean(shapes[i],0)<250) for i in shapes.keys()]) + padding*2
        h_max = max([sum(np.mean(shapes[i],1)<250) for i in shapes.keys()]) + padding*2
        generic_size = 1 
        if generic_size:
            if l_max < 224: l_max = 224
            if h_max < 224: h_max = 224

        # determine where each object starts  
        h_start = {i:np.nonzero(np.mean(shapes[i],0)<222)[0][0]-padding for i in shapes.keys()}
        l_start = {i:np.nonzero(np.mean(shapes[i],1)<222)[0][0]-padding for i in shapes.keys()}
        # extract stereotypical box around each object 
        imgs = [shapes[i][l_start[i]:l_start[i]+l_max, h_start[i]:h_start[i]+h_max] for i in range(4)]
        # create image of the right size for modeling (224x224x3) 
        
        return [imresize([imgs[j] for i in range(3)], (224, 224)) for j in range(4)]  

    stimuli = {}
    original = {}
    print('working on ', end='')
    stim_folder = stimulus_directory
    task_folders = [i for i in os.listdir(stim_folder) if 'pdf' not in i]
    rotation = import_rotation_keys()
    types = ['control', 'familiar', 'novel', 'oddity']

    for i_type in types:

        print('\n...', i_type, end=' ')

        # initialize stim-type dictionaries
        if i_type == 'control':
            stimuli[i_type] = {'color': {}, 'size':{}}
            original[i_type] = {'color': {}, 'size':{}}
        else:
            stimuli[i_type] = {'low':{}, 'high':{}}
            stimuli['rotate_%s'%i_type] = {'low':{}, 'high':{}}
            original[i_type] = {'low':{}, 'high':{}}

        # stimulus information for loading
        i_folder = [i for i in task_folders if i_type in i][0]
        files = os.listdir(os.path.join(stim_folder, i_folder))
        high = [i for i in files if 'HIGH' in i]
        low = [i for i in files if 'LOW' in i]

        for i_file in files:
            
            # main variable of interest: will hold segmented objects
            slide_images = []

            i_number  = i_file[str.find(i_file, '_')+1:str.find(i_file, '.')]

            # set variability type and unique label for this slide
            tag = i_file[find([i == '_' for i in i_file])[0]+1:-4]
            level = ['high', 'low']['LOW' in i_file]

            # load images
            i_image = np.array(Image.open(os.path.join(stim_folder, i_folder, i_file)))

            if i_type != 'oddity':

                half0 = int(i_image.shape[0]/2)
                half1 = int(i_image.shape[1]/2)

                # the shape extraction here should preserve everything, expand
                if (i_type == 'control') and (len(np.shape(i_image)) == 2):
                    
                    imgs = stabalize_and_segment_image(i_image)
                    im0, im1, im2, im3 = imgs[0], imgs[1], imgs[2], imgs[3]

                # the color extraction here should be ONLY the color
                elif (i_type == 'control') and (len(np.shape(i_image)) > 2):

                    im0 = i_image[:half0,:half1][50:-50,50:-50]
                    im1 = i_image[:half0:,half1:][50:-50,50:-50]
                    im2 = i_image[half0:,:half1:][50:-50,50:-50]
                    im3 = i_image[half0:,half1:][50:-50,50:-50]

                # otherwise just split everything into quarters
                else:

                    im0 = i_image[:half0,:half1]
                    im1 = i_image[:half0:,half1:]
                    im2 = i_image[half0:,:half1:]
                    im3 = i_image[half0:,half1:]

                slide_images.append(imresize(im0, (224, 224)))
                slide_images.append(imresize(im1, (224, 224)))
                slide_images.append(imresize(im2, (224, 224)))
                slide_images.append(imresize(im3, (224, 224)))

            else: # oddities need to be split into sevens

                height = np.shape(i_image)[0]
                length = np.shape(i_image)[1]
                height_seg = int(height/4)
                length_seg = int(length/4)
                height_start = int(height_seg - (length_seg/2))

                # top row
                for i in [0,1,2,3]:
                    i_slide = i_image[height_start:height_start+length_seg,length_seg*i:length_seg*(i+1)]
                    slide_images.append(imresize(i_slide, (224, 224)))

                # bottom row
                length_start = int(length*.5 - (3/2)*length_seg)
                height_start = int(height*(3/4) - (.5 * length_seg))
                for i in [0,1,2]:

                    i_slide = i_image[height_start:height_start+length_seg,
                                      length_start+length_seg*i:length_start+length_seg*(i+1)]
                    slide_images.append(imresize(i_slide, (224, 224)))


            if i_type == 'control':

                if len(np.shape(i_image)) == 3:

                    stimuli[i_type]['color'][i_number] = slide_images
                    original[i_type]['color'][i_number] = i_image
                
                else:

                    stimuli[i_type]['size'][i_number] = slide_images
                    original[i_type]['size'][i_number] = i_image

            else:

                
                stimuli[i_type][level][i_number] = slide_images
                original[i_type][level][i_number] = i_image

                if i_type in ['familiar', 'novel']:
                    
                    flipped = []

                    try:
                        rotations = rotation[i_type][level][i_file]

                        for i in range(len(slide_images)):

                            flipped.append(np.rot90(slide_images[i], k=rotations[i]))

                        stimuli['rotate_%s'%i_type][level][i_number] = flipped

                    except:
                        print('\n\tLOADING ERROR:', i_file, end=' ')


    stimuli['color'] = stimuli['control']['color']
    stimuli['size'] = stimuli['control']['size']
    original['color'] = original['control']['color']
    original['size'] = original['control']['size']
    del(stimuli['control']) ; del(original['control'])
    return stimuli, original

def load_experimental_data(load_type='from_original'): 

    answer_key = extract_answer_key()
    #    stimuli, originals =load_images(load_type)
    if load_type=='from_original':

        stimuli, originals = extract_all_stimuli()

        obj = {'stimuli':stimuli, 'originals':originals}

        with open('experimental_stimuli.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:

        with open('experimental_stimuli.pkl', 'rb') as f:
            obj = pickle.load(f)

        stimuli = obj['stimuli']
        originals = obj['originals']
    
    print('Data loaded and preprocessed')  
    return stimuli, originals, answer_key


def load_model(): 

    print('Loading model ... ', end='')  
    # if we're running the model
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    # only necessary on ccncluster
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # set parameters for memory allocation
    #def get_session():

        #config = tf.ConfigProto(allow_soft_placement = True)
        #config.gpu_options.per_process_gpu_memory_fraction = 0.4
        #config.gpu_options.allow_growth = True
        #return {'sess' : tf.Session(config = config)}

    # initialize session
    tf_session = tf.Session()

    # load and initialize model
    path_to_model = '../models/'
    vgg16 = imp.load_source('vgg16', path_to_model + 'vgg16.py')
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16.vgg16(imgs, path_to_model + 'vgg16_weights.npz', tf_session)
    print('done')
    return vgg, tf_session

def model_stimuli(stimuli, model, session, select_types):
    
    if select_types: types = select_types
    else: types = list(stimuli.keys())

    print('Modeling stimuli from groups', types)

    # initialize data types
    response = {i:{} for i in types}
    
    for i_type in types: 
        
        print(i_type, end=' ')
        
        if i_type in ['color', 'size']: 

            response[i_type] = {}
    
            # iterate over slides
            for i_slide in stimuli[i_type].keys():
                    
                print('.', end='')
                    
                # extract slide images
                slide_images = np.array(stimuli[i_type][i_slide])
    
                # extract model responses to images
                fc6 = session.run([ model.fc1 ], feed_dict={model.imgs: slide_images})
    
                # store model representations and labels 
                response[i_type][i_slide] = fc6[0]
    
            print(' ')

        else: 
            
            for i_level in stimuli[i_type].keys(): 
                
                response[i_type][i_level] = {}
    
                # iterate over slides
                for i_slide in list(stimuli[i_type][i_level].keys()):
                    
                    print('.', end='')
                    
                    # extract slide images
                    slide_images = np.array(stimuli[i_type][i_level][i_slide])
    
                    # extract model responses to images
                    fc6 = session.run([ model.fc1 ], feed_dict={model.imgs: slide_images})
    
                    # store model representations and labels 
                    response[i_type][i_level][i_slide] = fc6[0]
    
                print(' ')
    
    return response

def extract_representations(select_types):

    # preprocess images and extract answers from txt files
    stimuli, originals, answer_key = load_experimental_data('preload')

    if how=='preload':

        # load representations we've already extracted
        model_responses = np.load(file_name + '.npy').item()

    else:

        # extract model responses to stimuli
        model_responses = model_stimuli(stimuli, select_types)
        # save representations
        np.save(file_name, model_data)

    return stimuli, originals, model_responses



def extract_average_model_performance(model_responses, answer_key, stimuli): 
    type_results = {'average':{}, 
                    'image':{}, 
                    'missing':{}, 
                    'difference':{}, 
                    'average_difference':{}}

    errors = []

    for i_type in ['rotate_familiar','rotate_novel', 'color', 'size']: 

        if i_type in ['color', 'size']: 

            type_tag = i_type

            correct, image_response, missed_trial, item_similarity = [], {}, [],[]

            for i in list(stimuli[i_type].keys()): 

                try: 
                    trial_response = model_responses[i_type][i]
                    trial_cov = (np.corrcoef(trial_response))
                    r = np.array(list(range(len(trial_cov))))
                    trial_decision_space = np.array([trial_cov[i, r[r!=i]] for i in r])


                    if 'rotate' in i_type: TYPE = i_type[7:]
                    else: TYPE = i_type

                    image_label = stimuli[TYPE][i][0:-4]    
                    image_number = i #image_label[str.find(image_label,'_')+1:]

                    try: 

                        TRUE = answer_key[TYPE][image_number]
                        model_decision_ = trial_decision_space.mean(1).argmin()
                        correct.append(int(TRUE) == (model_decision_+1))
                        item_covariance = trial_decision_space.mean(1)
                        pattern_similarity = np.sort(item_covariance)[1] - np.sort(item_covariance)[0]
                        item_similarity.append(pattern_similarity)
                        #print(i, i_type, np.sort(item_covariance))                    
                    except: 

                        errors.append(image_number)
                        TRUE = np.nan

                    image_response[image_number] = TRUE
                except: 
                    errors.append([i_type + '_' + i])
            type_results['image'][type_tag] = image_response
            type_results['average'][type_tag] = np.mean(correct)
            type_results['missing'][type_tag] = missed_trial
            type_results['difference'][type_tag] = item_similarity
            type_results['average_difference'][type_tag] = np.mean(item_similarity)
        else: 

            for i_var in ['high', 'low']: 

                type_tag = i_type + '_' + i_var

                correct, image_response, missed_trial, item_similarity = [], {}, [], []

                for i in list(stimuli[i_type][i_var].keys()): 

                    try: 

                        trial_response = model_responses[i_type][i_var][i]
                        trial_cov = (np.corrcoef(trial_response))

                        r = np.array(list(range(len(trial_cov))))
                        trial_decision_space = np.array([trial_cov[i, r[r!=i]] for i in r])


                        if 'rotate' in i_type: TYPE = i_type[7:]
                        else: TYPE = i_type
                        #image_label = stimuli[TYPE]['%s_files'%i_var][i][0:-4]
                        image_number = i #image_label[str.find(image_label,'_')+1:]
                        try: 

                            if i_type == 'oddity': 
                                TRUE = answer_key[TYPE][image_number]
                            else: 
                                TRUE = answer_key[TYPE][i_var][image_number]

                            model_decision_ = trial_decision_space.mean(1).argmin()
                            correct.append(int(TRUE) == (model_decision_+1))
                            item_covariance = trial_decision_space.mean(1)
                            pattern_similarity = np.sort(item_covariance)[1] - np.sort(item_covariance)[0]

                            item_similarity.append(pattern_similarity)

                        except: 

                            errors.append(image_number)
                            TRUE = np.nan

                        image_response[image_number] = TRUE
                    except: 
                        errors.append([i_type + '_' + i_var + '_' + i])
                type_results['image'][type_tag] = image_response
                type_results['average'][type_tag] = np.mean(correct)
                type_results['missing'][type_tag] = missed_trial
                type_results['difference'][type_tag] = item_similarity
                type_results['average_difference'][type_tag] = np.mean(item_similarity)

    return type_results, errors
