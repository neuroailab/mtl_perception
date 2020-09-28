import warnings; warnings.simplefilter('ignore')
from PIL import Image
import imp, os, pickle, pandas, numpy as np
from scipy.misc import imresize
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def load_model(path_to_model):
    """return model and session"""
    session = tf.Session()
    vgg16 = imp.load_source('vgg16', os.path.join(path_to_model, 'vgg16.py'))
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16.vgg16(imgs, os.path.join(path_to_model, 'vgg16_weights.npz'), session)
    return vgg, session

def model_performance(model, session, image_path):
    """returns model performance on experiment"""
    
    # all images in stimulus directory
    images = np.sort( [i for i in os.listdir(image_path) if 'BMP' in i] )
    # all unique trials
    trial_names = np.unique([i[0:-8] for i in images])
    # trial/condition markers
    markers = ['030', '053', '082', '097']
    # initialize data types
    model_accuracy = {'082':[], '097':[], 'name':trial_names, 'markers':markers}

    for i_trial in trial_names:

        # all images in trial
        i_files = [i for i in images if i_trial in i and i[-7:-4] in markers]

        # only
        if len(i_files) == len( markers ):

            for i_face in range(len(i_files)):
                imgs = np.array([np.array(Image.open(os.path.join(image_path, i))) for i in i_files])
                imgs = [imresize(imgs[i], (224, 224, 3)) for i in range(len(imgs))]
                i_response = session.run(model.conv5_1, feed_dict={model.imgs: imgs})

            # flatten model responses to trial images
            i_response = np.reshape(i_response, (len(imgs), -1))

            # determine model accuracy for 'near' and 'far' conditions
            for condition in [2, 3]:
                # generate covariance
                i_covariance = np.corrcoef(i_response[np.array([0,1,condition]), :])
                # sort covariance matrix
                i_covariance.sort()
                # extract nearest neighbor
                i_nearest = i_covariance[:,-2]
                # determine if farthest is the oddity
                correct = i_nearest.argmin() == len(i_nearest)-1
                # store
                model_accuracy[markers[condition]].append(correct)
    
    # determine average accuracy
    model_performance = {i: np.mean(model_accuracy[i]) for i in [ '082', '097']}
    # save
    pandas.DataFrame({'accuracy' : list(model_performance.values()), 
                      'condition': list(model_performance)}).to_csv('model_performance.csv') 
    
    return model_performance

if __name__ == '__main__':

    # set base directory all analyses are contained within
    base_directory = os.path.abspath('..')
    # set path to model
    path_to_model = os.path.join(base_directory, 'model')
    # define model
    model, session = load_model(path_to_model)
    # path to stimuli
    image_path = os.path.join(base_directory, 'experiments/imhoff_2018/stimuli/oddity')
    # determine model performance
    model_accuracy = model_performance(model, session, image_path)
