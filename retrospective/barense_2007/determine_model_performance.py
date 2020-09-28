import warnings; warnings.simplefilter('ignore')
import os, imp, pickle, pandas, numpy as np
from PIL import Image
from scipy.misc import imread, imresize
from sklearn import cluster
import tensorflow.compat.v1 as tf

def load_model(path_to_model):
    """define computational proxy for the vvs from pretained model"""
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    tf.disable_v2_behavior()
    session = tf.Session()
    vgg16 = imp.load_source('vgg16', os.path.join(path_to_model, 'vgg16.py'))
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16.vgg16(imgs, os.path.join(path_to_model, 'vgg16_weights.npz'), session)
    return vgg, session

def import_rotation_keys(stimulus_directory):
    """returns keys to rotate images into their canonical orientation"""
    rotation_location = os.path.join(stimulus_directory, 'rotation_keys.pkl')
    with open(rotation_location, 'rb') as f:
        return pickle.load(f)

def extract_and_rotate_familiar_images(stimulus_directory):
    """returns stimuli in canonical orientation"""
    
    # folders of interest 
    task_folders = [i for i in os.listdir(stimulus_directory) if 'pdf' not in i]
    # extract rotation keys 
    rotation = import_rotation_keys(stimulus_directory)

    # initialize stim-type dictionaries
    stimuli = {'high':{}, 'low':{}}

    # stimulus information for loading
    i_folder = [i for i in task_folders if 'familiar' in i][0]

    # all files in stimulus directory
    files = os.listdir(os.path.join(stimulus_directory, i_folder))
    
    print( rotation ) 

    for i_file in files:

        # main variable of interest: will hold segmented objects
        slide_images = []
        # define a more legible name for each file
        i_number  = i_file[str.find(i_file, '_')+1:str.find(i_file, '.')]
        # set variability type and unique label for this slide
        tag = i_file[i_file.find('_')+1:-4]
        # 
        level = ['high', 'low']['LOW' in i_file]
        
        # load images
        i_image = np.array(Image.open(os.path.join(stimulus_directory, i_folder, i_file)))
        
        # define width and height midscreen positino 
        half0 = int(i_image.shape[0]/2)
        half1 = int(i_image.shape[1]/2)
        
        # segment all four images
        im0 = i_image[:half0,:half1]
        im1 = i_image[:half0:,half1:]
        im2 = i_image[half0:,:half1:]
        im3 = i_image[half0:,half1:]
        
        # resize and append all images
        slide_images.append(imresize(im0, (224, 224)))
        slide_images.append(imresize(im1, (224, 224)))
        slide_images.append(imresize(im2, (224, 224)))
        slide_images.append(imresize(im3, (224, 224)))

        flipped = []
        # only load images that have saved rotations 
        if i_file in list( rotation['familiar'][level] ):
            
            # load rotations for this trial 
            rotations = rotation['familiar'][level][i_file]
            # rotate each image 
            for i in range(len(slide_images)):
                flipped.append(np.rot90(slide_images[i], k=rotations[i]))
            # store rotated images 
            stimuli[level][i_number] = flipped

    return stimuli

def extract_and_rotate_novel_images(stimulus_directory):
    """return novel images in their canonical orientation"""
    
    # load answer key 
    answer_key = extract_answer_key(stimulus_directory)
    # identify task folders 
    task_folders = [i for i in os.listdir(stimulus_directory) if 'pdf' not in i]
    # import rotation keys 
    rotation = import_rotation_keys(stimulus_directory)
    # stimulus information for loading
    i_folder = [i for i in task_folders if 'novel' in i][0]
    # all files 
    files = os.listdir(os.path.join(stimulus_directory, i_folder))
    # initialize image segmentation protocol--four objects per stimulus screen
    k_means = cluster.KMeans(n_clusters=4, n_init=4)
    # initialize data storage 
    novel_images = {'high':{}, 'low':{}}
    # set diameter of images (determined manually) 
    d = int(350/2)

    for i_file in files:
        
        # determine ambiguity level of this trial
        if 'LOW' in i_file: amb = 'low'
        if 'HIG' in i_file: amb = 'high'
        # define human readable filename
        i_number = i_file[i_file.find('_')+1:len(i_file)-4]
        
        # only select those images that we have rotation keys for 
        if (i_file in rotation['novel'][amb].keys()) * (i_number in answer_key['novel_%s'%amb].keys()):

            # load rotations necessary for image
            image_rotations = rotation['novel'][amb][ i_file ]
            # load image
            i_image = np.array(Image.open(os.path.join(stimulus_directory, i_folder, i_file)))

            ##### begin kmeans segmentation protocol #####
            # binarize layer of image
            io = i_image[:,:,1] != 255
            # convert binaryized image into a vector for each non-blank location
            points = [[i,j] for i in  range(io.shape[0]) for j in range(io.shape[1]) if io[i,j]]
            # cluster non-blank locations to determine object center of masses
            k_means.fit(points)
            # determine mapping between cluster order and image order
            order = determine_order_of_clusters(k_means.cluster_centers_, i_image)
            ##### end segmentation protocol #######

            # iterate over all objects in image
            i_slide = []
            for i_segmented_object in range(len(order)):
                
                # identify centroids of clusters 
                x, y = k_means.cluster_centers_[order[i_segmented_object],:]
                # select an area around the center of mass defined above
                i_object =  i_image[int(x-d):int(x+d), int(y-d):int(y+d)]
                # rotate the selected area into its connonical orientation
                i_object_rotated = np.rot90(i_object, k=image_rotations[i_segmented_object])
                # add to list
                i_slide.append(i_object_rotated)

            # resize images
            i_slide = [imresize(i_slide[i],(224, 224)) for i in range(len(i_slide))]
            # append to ambiguity type
            novel_images[amb][i_number] = i_slide

    return novel_images

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

def extract_fribbles(stimulus_directory):
    """returns fribbles""" 
    
    # load answer key 
    answer_key = extract_answer_key(stimulus_directory)
    # return list of trial names 
    extract_list = list( answer_key['oddity'])
    # set diameter of images (determined manually)  
    d = int(200/2)
    # identify experiment folder 
    task_folders = [i for i in os.listdir(stimulus_directory) if 'pdf' not in i]
    # import rotation keys 
    rotation = import_rotation_keys(stimulus_directory)
    # stimulus information for loading
    i_folder = [i for i in task_folders if 'oddity' in i][0]
    # identify all files in directory
    files = os.listdir(os.path.join(stimulus_directory, i_folder))
    # initialize kmeans segmentation protocol -- 7 images per screen 
    k_means = cluster.KMeans(n_clusters=7, n_init=4)

    oddity_images = {}
    for i_file in files:
        
        # create human readable filename 
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
            # determine whether image is in the top row 
            top_row = k_means.cluster_centers_[:,0] < 400
            # extract all top row images             
            order[top_row] = k_means.cluster_centers_[:,1][top_row].argsort().argsort()
            # extract all bottom row images
            order[top_row==0] = k_means.cluster_centers_[:,1][top_row==0].argsort().argsort() + 4
            # sort 
            order = order.argsort()
            ## segmentation complete        ##

            # iterate over all objects in image
            i_slide = []
            for i_segmented_object in range(len(order)):
                
                # identify center of mass for each object 
                x, y = k_means.cluster_centers_[order[i_segmented_object],:]
                # select an area around the center of mass defined above
                i_object =  i_image[int(x-d):int(x+d), int(y-d):int(y+d)]
                # add segmented image 
                i_slide.append(i_object)

            # resize and append to ambiguity type
            oddity_images[i_number] = [ imresize(i_slide[i], (224, 224)) for i in range(len(i_slide)) ]

    return oddity_images

def extract_stimuli_and_answer_key(stim_dir): 
    
    # answer key for all experiments 
    answer_key = extract_answer_key(stim_dir)
    # novel high and low ambiguity experiments
    novel = extract_and_rotate_novel_images(stim_dir)
    # familiar high and low ambiguity experiments
    familiar = extract_and_rotate_familiar_images(stim_dir)
    # fribbles 
    fribbles = extract_fribbles(stim_dir)
    
    # combine into single 
    stimuli = {'novel_high': novel['high'], 
               'novel_low' : novel['low'], 
               'familiar_high': familiar['high'], 
               'familiar_low': familiar['low'], 
               'oddity': fribbles}
    
    return stimuli, answer_key

def extract_answer_key(stimulus_directory):
    
    # initialize data structure 
    answer_key = {}
    # identify directory with answer keys
    answer_directory = os.path.join(stimulus_directory, 'anwser_keys')
    # extract all files in directory
    answer_files = [i for i in os.listdir(answer_directory) if '.edat' not in i]
    # determine mapping between given and desired names 
    model_names = {'famobjectHIGH':'familiar_high',
                   'famobjectLOW':'familiar_low',
                   'greeblesHIGH':'novel_high',
                   'greeblesLOW':'novel_low',
                   'size':'size',
                   'colour':'color',
                   'fribbles_block2':'oddity'}
    
    # iterate of files 
    for stimulus_type in answer_files:
        # load answer key from file 
        answer_file = os.path.join(answer_directory, stimulus_type)
        # load raw file 
        answers = open(answer_file).readlines()
        # determine type 
        tag = stimulus_type[0:-9]
        # convert to eventual name 
        name = model_names[stimulus_type[0:-9]]
        # initialize data structure 
        answer_key[name] = {}
        
        # very idensyncratic code given the available file structure 
        for i_line in range(len(answers)):
            if 'ImageShow' in answers[i_line]:
                q = answers[i_line][str.find(answers[i_line],'_')+1:str.find(answers[i_line],'.')]
                answer_key[name][q] = answers[i_line-5][-2:-1]
                answer_key[name][q] = answers[i_line-5][-2:-1]
                answer_key[name][q] = answers[i_line-5][-2:-1]

    return answer_key

def determine_model_performance(vgg, session, stimuli, answer_key):
    
    # initialize data structure for model representations
    model_responses = {}
    # initalize data structure for model performance
    performance = pandas.DataFrame({'experiment':[], 'layer':[], 'accuracy':[]})
    
    # names for all model layers
    model_layers = ['conv1_1', 'conv1_2', 'pool1',
                    'conv2_1', 'conv2_2', 'pool2',
                    'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
                    'conv4_1', 'conv4_2', 'conv4_3', 'pool4',
                    'conv5_1', 'conv5_2', 'conv5_3', 'pool5',
                    'fc6', 'fc7', 'fc8']

    for i_experiment in stimuli:

        print('extracting features for', i_experiment)

        # iterate over slides
        for i_slide in list(stimuli[i_experiment]):
            
            # initialize data structure for slide 
            model_responses[i_slide] = {}
            # extract slide images
            slide_images = np.array(stimuli[i_experiment][i_slide])
            # determine number of images 
            n_images = len(slide_images)

            # extract model responses to images
            outs = session.run([vgg.conv1_1, vgg.conv1_2, vgg.pool1,
                                vgg.conv2_1, vgg.conv2_2, vgg.pool2,
                                vgg.conv3_1, vgg.conv3_2, vgg.conv3_3, vgg.pool3,
                                vgg.conv4_1, vgg.conv4_2, vgg.conv4_3, vgg.pool4,
                                vgg.conv5_1, vgg.conv5_2, vgg.conv5_3, vgg.pool5,
                                vgg.fc1, vgg.fc2, vgg.fc3l],
                               feed_dict={vgg.imgs: slide_images})

            # pixel representations
            model_responses[i_slide]['pixel'] = np.reshape( slide_images, (n_images, -1))

            # store model representations and labels
            trial_ = {model_layers[i]: np.reshape( outs[i], (n_images, -1)) for i in range(len(outs))}
            
            # transfer model responses into data structure for all images 
            for i_layer in trial_.keys():
                model_responses[i_slide][i_layer] = trial_[i_layer]
        
        # determine model performance for this experiment
        print('estimating model performance...')
        performance = evaluate_experiment(model_responses, answer_key, performance, i_experiment)

    return performance

def evaluate_experiment(model_responses, answer_keys, df, i_experiment):
    """returns model performance""" 
    # load answer keys for experiment
    answer_key = answer_keys[i_experiment]
    # get trial names 
    trials = [i for i in list(model_responses) if i in list(answer_key)]
    # get all layers used 
    layers = list(model_responses[trials[0]])
    # determine number of stimuli 
    n_stimuli = len(model_responses[trials[0]][layers[0]])
    # convenience array used to extract model responses
    x = np.array(list(range(n_stimuli)))
    
    for i_layer in layers:
        
        # initialize data structure
        accuracy = []

        for i_trial in trials:
            # determine covariance matrix for trial
            trial_covariance = np.corrcoef(model_responses[i_trial][i_layer])
            # only use off-diagonal values
            trial_decision_space = np.array([trial_covariance[i, x[x!=i]] for i in x])
            # sort 
            trial_decision_space.sort()
            # determine model-estimated oddity
            i_choice = trial_decision_space[:,-1].argmin() + 1
            # determine whether model choice is correct
            accuracy.append(i_choice == int(answer_key[i_trial]))
        # compute average across trials, for this layer
        i_result = {'accuracy': np.mean( accuracy ), 'experiment':i_experiment, 'layer':i_layer}
        # append results for this result # 
        df = df.append(i_result,  ignore_index=True)
    
    return df

if __name__ == '__main__': 

    # set base directory all analyses are contained within
    base_directory = os.path.abspath('..')
    # stimulus directory
    stimulus_dir = os.path.join(base_directory,'experiments/barense_2007/stimuli')
    # extract stimuli
    stimuli, answer_key = extract_stimuli_and_answer_key(stimulus_dir)
    # path to model
    path_to_model =  os.path.join(base_directory, 'model')
    # define modela
    vgg, session = load_model(path_to_model)
    # determine model performance
    model_performance = determine_model_performance(vgg, session, stimuli, answer_key) 
    # save
    model_performance.to_csv('model_performance.csv')
