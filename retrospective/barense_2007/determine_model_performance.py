import warnings; warnings.simplefilter('ignore')
import os, imp, pickle
import numpy as np
from PIL import Image 
from scipy.misc import imread, imresize
from sklearn import cluster
# custom functions in this directory
from utils import extract_answer_key, load_model
import pandas
import tensorflow.compat.v1 as tf

def load_model(path_to_model):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    tf.disable_v2_behavior()
    session = tf.Session()
    vgg16 = imp.load_source('vgg16', path_to_model + 'vgg16.py')
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16.vgg16(imgs, path_to_model + 'vgg16_weights.npz', session)
    return vgg, session

def import_rotation_keys():
    rotation_location = 'rotation_keys.pkl'
    with open(rotation_location, 'rb') as f:
        return pickle.load(f)

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

def extract_stimuli(stimulus_directory): 
    
    stimuli = {}
    original = {}
    task_folders = [i for i in os.listdir(stimulus_directory) if 'pdf' not in i]
    rotation = import_rotation_keys()
    types = ['familiar']

    conditions = {'familiar':['high', 'low']}

    for i_type in types:

        # initialize stim-type dictionaries

        stimuli[i_type] = {i: {} for i in conditions[i_type]}
        original[i_type] = {i: {} for i in conditions[i_type]}

        # stimulus information for loading
        i_folder = [i for i in task_folders if i_type in i][0]
        files = os.listdir(os.path.join(stimulus_directory, i_folder))

        for i_file in files:

            # main variable of interest: will hold segmented objects
            slide_images = []

            i_number  = i_file[str.find(i_file, '_')+1:str.find(i_file, '.')]

            # set variability type and unique label for this slide
            #print(i_file)
            #print( i_file.find('_') )
            #zzztag = i_file[find([i == '_' for i in i_file])[0]+1:-4]
            tag = i_file[i_file.find('_')+1:-4]
            
            level = ['high', 'low']['LOW' in i_file]

            # load images
            i_image = np.array(Image.open(os.path.join(stimulus_directory, i_folder, i_file)))

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

                        stimuli[i_type][level][i_number] = flipped

                    except:
                        pass # print('\n\tLOADING ERROR:', i_file, end=' ')
                    
    
    return stimuli

def determine_order_of_clusters(clusters, i_image): 
    """determine how to map cluster order onto image order """
    
    # split into halves 
    ys = (clusters[:,0] > (i_image.shape[0]/2) ) 
    xs = (clusters[:,1] > (i_image.shape[1]/2) ) 
    
    # manually determine how to map cluster onto image order 
    order = []
    order.append( np.nonzero((xs == 0) * (ys == 0))[0][0]) # first 
    order.append( np.nonzero((xs == 1) * (ys == 0))[0][0]) # second 
    order.append( np.nonzero((xs == 0) * (ys == 1))[0][0]) # third 
    order.append( np.nonzero((xs == 1) * (ys == 1))[0][0]) # fourth
    
    return np.array(order)   

def extract_and_rotate_novel_images(stimulus_directory, to_save=False):

    full_width = 350
    d = int(full_width/2)
    i_type = 'novel'
    answer_key = extract_answer_key(stimulus_directory)

    #stimulus_directory = '../../prc_stimuli/Barense_2007_NYS/'

    task_folders = [i for i in os.listdir(stimulus_directory) if 'pdf' not in i]
    rotation = import_rotation_keys()

    i_type = 'novel'
    # stimulus information for loading
    i_folder = [i for i in task_folders if i_type in i][0]
    #print(i_folder, stimulus_directory)
    files = os.listdir(os.path.join(stimulus_directory, i_folder))

    k_means = cluster.KMeans(n_clusters=4, n_init=4)

    novel_images = {'high':{}, 'low':{}}

    for i_file in files:

        if 'LOW' in i_file: ambiguity = 'low'
        if 'HIG' in i_file: ambiguity = 'high'

        i_number = i_file[i_file.find('_')+1:len(i_file)-4]

        if (i_file in rotation['novel'][ambiguity].keys()) * (i_number in answer_key['novel'][ambiguity].keys()):

            # load rotations necessary for image
            image_rotations = rotation['novel'][ambiguity][ i_file ]
            # load image
            i_image = np.array(Image.open(os.path.join(stimulus_directory, i_folder, i_file)))

            ## kmeans segmentation protocol ##
            # binarize layer of image
            binary = i_image[:,:,1] != 255
            # convert binaryized image into a vector for each non-blank location
            points = [[i,j] for i in  range(binary.shape[0]) for j in range(binary.shape[1]) if binary[i,j]]
            # cluster non-blank locations to determine object center of masses
            k_means.fit(points)
            # determine mapping between cluster order and image order
            order = determine_order_of_clusters(k_means.cluster_centers_, i_image)
            ## segmentation complete        ##

            # iterate over all objects in image
            i_slide = []
            for i_segmented_object in range(len(order)):

                x, y = k_means.cluster_centers_[order[i_segmented_object],:]
                # select an area around the center of mass defined above
                i_object =  i_image[int(x-d):int(x+d), int(y-d):int(y+d)]
                # rotate the selected area into its connonical orientation
                i_object_rotated = np.rot90(i_object, k=image_rotations[i_segmented_object])

                i_slide.append(i_object_rotated)

            # resize and append to ambiguity type
            if to_save:
                novel_images[ambiguity][i_number] = i_slide
            else:
                novel_images[ambiguity][i_number] = [ imresize(i_slide[i], (224, 224)) for i in range(len(i_slide)) ]

    return novel_images

def extract_fribbles(stimulus_directory, answer_key): 
    
    extract_list = list( answer_key['oddity'])
    
    full_width = 200
    d = int(full_width/2)

    #stimulus_directory = '../../prc_stimuli/Barense_2007_NYS/'
    task_folders = [i for i in os.listdir(stimulus_directory) if 'pdf' not in i]
    rotation = import_rotation_keys()

    i_type = 'oddity'
    # stimulus information for loading
    i_folder = [i for i in task_folders if i_type in i][0]
    files = os.listdir(os.path.join(stimulus_directory, i_folder))

    k_means = cluster.KMeans(n_clusters=7, n_init=4)

    oddity_images = {}

    for i_file in files: 

        i_number = i_file[i_file.find('_')+1:len(i_file)-4]
        if i_number in extract_list: 
            # load image 
            i_image = np.array(Image.open(os.path.join(stimulus_directory, i_folder, i_file)))

            ## kmeans segmentation protocol ##
            # binarize layer of image 
            binary = i_image[:,:,1] != 255
            # convert binaryized image into a vector for each non-blank location
            points = [[i,j] for i in  range(binary.shape[0]) for j in range(binary.shape[1]) if binary[i,j]]
            # cluster non-blank locations to determine object center of masses
            k_means.fit(points)
            # determine mapping between cluster order and image order
            order = np.array([0 for i in range(7)])
            top_row = k_means.cluster_centers_[:,0] < 400
            order[top_row] = k_means.cluster_centers_[:,1][top_row].argsort().argsort() 
            order[top_row==0] = k_means.cluster_centers_[:,1][top_row==0].argsort().argsort() + 4
            order = order.argsort() 
            ## segmentation complete        ##

            # iterate over all objects in image
            i_slide = [] 
            for i_segmented_object in range(len(order)): 

                x, y = k_means.cluster_centers_[order[i_segmented_object],:]

                # select an area around the center of mass defined above
                i_object =  i_image[int(x-d):int(x+d), int(y-d):int(y+d)] 

                i_slide.append(i_object)

            # resize and append to ambiguity type
            oddity_images[i_number] = [ imresize(i_slide[i], (224, 224)) for i in range(len(i_slide)) ]
    
    return oddity_images

def extract_stimuli_and_answer_key(stimulus_directory): 
    
    stimuli = extract_stimuli(stimulus_directory)
    answer_key = extract_answer_key(stimulus_directory) 
    
    stimuli['novel'] = extract_and_rotate_novel_images(stimulus_directory) 
    
    stimuli['familiar_high'] = stimuli['familiar']['high']
    stimuli['familiar_low'] = stimuli['familiar']['low']
    stimuli['novel_high'] = stimuli['novel']['high']
    stimuli['novel_low'] = stimuli['novel']['low']
    answer_key['familiar_high'] = answer_key['familiar']['high']
    answer_key['familiar_low'] = answer_key['familiar']['low']
    answer_key['novel_high'] = answer_key['novel']['high']
    answer_key['novel_low'] = answer_key['novel']['low']

    # mis labeled from the pretraining data -- an error in my automatic selection
    del(answer_key['color']['2'])
    del(answer_key['familiar'])
    del(answer_key['novel'])    
    del(stimuli['novel']);
    del(stimuli['familiar']);    
    
    return stimuli, answer_key

def determine_model_performance(model, session, stimuli, answer_key):
    
    model_responses = {} 
    performance = pandas.DataFrame({'experiment':[], 'layer':[], 'accuracy':[]})
    
    model_layers = ['conv1_1', 'conv1_2', 'pool1',
                    'conv2_1', 'conv2_2', 'pool2', 
                    'conv3_1', 'conv3_2', 'conv3_3', 'pool3', 
                    'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 
                    'conv5_1', 'conv5_2', 'conv5_3', 'pool5', 
                    'fc6', 'fc7', 'fc8']
    
    for i_experiment in list(stimuli): 
        
        print('extracting features for', i_experiment)
        
        model_responses = {i:{} for i in stimuli[i_experiment]} 
        
        # iterate over slides
        for i_slide in list(stimuli[i_experiment]):
            
            #model_responses[i_slide] = {} 
            # extract slide images
            slide_images = np.array(stimuli[i_experiment][i_slide])
            n_images = len(slide_images)

            # extract model responses to images
            outs = session.run([vgg.conv1_1, vgg.conv1_2, vgg.pool1,
                                vgg.conv2_1, vgg.conv2_2, vgg.pool2, 
                                vgg.conv3_1, vgg.conv3_2, vgg.conv3_3, vgg.pool3, 
                                vgg.conv4_1, vgg.conv4_2, vgg.conv4_3, vgg.pool4, 
                                vgg.conv5_1, vgg.conv5_2, vgg.conv5_3, vgg.pool5, 
                                vgg.fc1, vgg.fc2, vgg.fc3l], 
                               feed_dict={model.imgs: slide_images})

            # pixel representations
            model_responses[i_slide]['pixel'] =np.reshape( slide_images, (n_images, -1))

            # store model representations and labels
            trial_ = {model_layers[i]: np.reshape( outs[i], (n_images, -1)) for i in range(len(outs))}

            for i_layer in trial_.keys(): 
                model_responses[i_slide][i_layer] = trial_[i_layer]
            
        print('estimating model performance...')
        
        performance = evaluate_experiment(model_responses, answer_key[i_experiment], performance, i_experiment)
        
    return performance

def evaluate_experiment(experiment, answer_key, df, i_experiment): 

    trials = [i for i in list(experiment) if i in list(answer_key)]
    layers = list(experiment[trials[0]])
    n_stimuli = len(experiment[trials[0]][layers[0]])
    x = np.array(list(range(n_stimuli)))

    for i_layer in layers: 
    
        accuracy = [] 

        for i_trial in trials:     
            trial_covariance = np.corrcoef(experiment[i_trial][i_layer])
            trial_decision_space = np.array([trial_covariance[i, x[x!=i]] for i in x])
            trial_decision_space.sort()
            i_choice = trial_decision_space[:,-1].argmin() + 1
            i_correct = i_choice == int(answer_key[i_trial])
            accuracy.append( i_correct )

        i_result = {'accuracy': np.mean( accuracy ), 'experiment':i_experiment, 'layer':i_layer}
        df = df.append(i_result,  ignore_index=True)

    return df

if __name__ == '__main__': 
    
    base_dir = '/Users/biota/work/perirhinal_cortex/analysis/barense_2007/'

    stimulus_dir = base_dir + 'stimuli/'

    stimuli, answer_key = extract_stimuli_and_answer_key(stimulus_dir) 

    path_to_model = '/Users/biota/work/perirhinal_cortex/analysis/models/'
    
    vgg, session = load_model(path_to_model)

    model_performance = determine_model_performance(vgg, session, stimuli, answer_key)

    model_performance.to_csv('barense_model_performance.csv')
