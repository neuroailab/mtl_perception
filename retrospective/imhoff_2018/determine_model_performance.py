from PIL import Image
import warnings; warnings.simplefilter('ignore')
import imp, os, pickle, numpy as np
from scipy.misc import imresize
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def load_model(path_to_model):
    session = tf.Session()
    vgg16 = imp.load_source('vgg16', path_to_model + 'vgg16.py')
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16.vgg16(imgs, path_to_model + 'vgg16_weights.npz', session)
    return vgg, session

def model_performance(model, image_path): 
    
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
                imgs = np.array([np.array(Image.open(image_path + i)) for i in i_files])
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
    
    return model_accuracy

if __name__ == '__main__': 
    
    # 
    path_to_model = '/Users/biota/work/perirhinal_cortex/analysis/models/'
    
    model, session = load_model(path_to_model)
    
    image_path = '/Users/biota/work/perirhinal_cortex/analysis/imhoff_2018/stimuli/oddity/'

    model_accuracy = model_performance(model, image_path)

    with open('model_performance.pickle', 'wb') as handle: 
        pickle.dump(model_accuracy, handle)
