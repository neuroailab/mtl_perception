import sys, h5py, imp, os, pickle, numpy as np 
from sklearn.cross_decomposition import PLSRegression
from scipy.misc import imresize
#from skimage.transform import resize as imresize 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def extract_neural_data(path): 
    # load our dataset
    data = h5py.File(path, 'r')
    # extract neural data
    neural_data = np.array(data['time_averaged_trial_averaged'])
    # extract variation labels
    variation_data = np.array(data['image_meta']['variation_level'])
    # extract IT data
    it_data = neural_data[:, data['neural_meta']['IT_NEURONS'] ]
    # extract V4 data 
    v4_data = neural_data[:, data['neural_meta']['V4_NEURONS'] ]
    # extract images 
    images = np.array(data['images'])
    # close h5py file 
    data.close() 
    return v4_data, it_data, variation_data, images

def generate_shuffle(n, ratio=3/4):
    sh = np.random.permutation(range( n ))
    return sh[:int(len(sh)*(ratio))], sh[int(len(sh)*(ratio)):]

def fit_response(model_, neural_, train_, test_, pls): 
    # find mapping between model responses and population
    pls.fit(model_[train_], neural_[train_] )
    # extract best fit to training data
    pred_train = pls.predict(model_[train_]).flatten()
    # extract predictions to testing data
    pred_test = pls.predict(model_[test_]).flatten()
    # compute correlation between model and neural responses in training data 
    train_r =  np.corrcoef(pred_train, neural_[train_].flatten())
    # compute correlation between model and neural responses in testing data
    test_r = np.corrcoef(pred_test, neural_[test_].flatten())
    return train_r[0, 1], test_r[0, 1]

def model_neural_map(layer_, neural_responses,  n_components=25, per_neuron=True): 
    # define pls model 
    pls = PLSRegression(n_components=n_components, scale=False)    
    # we'll use the same split across regions 
    train_, test_ = generate_shuffle(layer_.shape[0])
    # initialize data types 
    fits_ = {'it': {'train':[], 'test':[]}, 'v4': {'train':[], 'test':[]}}    
    print('---MODELING %s DATA'%['POPULATION', 'SINGLE UNIT'][per_neuron*1]) 
    for region in neural_responses: 
        print('---- %s'%region)
        # define region of interest 
        neural_ = neural_responses[region]
        if per_neuron: 
            for i_neuron in range( neural_.shape[1]): 
                print('------ %s NEURON %d'%(region.upper(), i_neuron))
                # single neuron's response 
                neuron_ = neural_[:, i_neuron]
                # find mapping between model responses and single neuron
                r_train, r_test = fit_response( layer_, neuron_, train_, test_, pls )
                # store 
                fits_[region]['train'].append( r_train )
                fits_[region]['test'].append( r_test )
                print('----NEURON %d CORRELATION: %.02f'%(i_neuron, r_test))
        else: 
            # fit population 
            train_r, test_r = fit_response(layer_, neural_, train_, test_, pls)
            # store correlations
            fits_[region]['train'].append( train_r )
            fits_[region]['test'].append( test_r )            
    return fits_

def rgb_from_grey(grey):
    wide, high = grey.shape
    rgb = np.empty((wide, high, 3), dtype=np.uint8)
    rgb[:, :, 2] =  rgb[:, :, 1] =  rgb[:, :, 0] =  grey
    return rgb

def load_model(path_to_model): 
    session = tf.Session()
    vgg16 = imp.load_source('vgg16', path_to_model + 'vgg16.py')
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16.vgg16(imgs, path_to_model + 'vgg16_weights.npz', session)
    return vgg, session

def extract_model_responses(images, model, session, layer): 
    
    layer_map = {'conv1_1': vgg.conv1_1, 'conv1_2':vgg.conv1_2, 'pool1': vgg.pool1, 
                 'conv2_1': vgg.conv2_1, 'conv2_2':vgg.conv2_2, 'pool2': vgg.pool2, 
                 'conv3_1': vgg.conv3_1, 'conv3_2':vgg.conv3_2, 'conv3_3':vgg.conv3_3, 'pool3': vgg.pool3, 
                 'conv4_1': vgg.conv4_1, 'conv4_2':vgg.conv4_2, 'conv4_3':vgg.conv4_3, 'pool4': vgg.pool4, 
                 'conv5_1': vgg.conv5_1, 'conv5_2':vgg.conv5_2, 'conv5_3':vgg.conv5_3, 'pool5': vgg.pool5, 
                 'fc6': vgg.fc1, 'fc7':vgg.fc2, 'fc8':vgg.fc3l}

    layer_responses = [] 
    for i_image in range(len(images)): 
        tmp_img = rgb_from_grey(images[i_image])  
        rbg_image = [imresize(tmp_img, (224, 224))]
        ##print('RGB_IMAGE', np.array(rbg_image).shape )
        i_responses = session.run(layer_map[layer], feed_dict={model.imgs: rbg_image})[0]
        ##print(i_responses.shape)
        layer_responses.append(i_responses.flatten())
        if not i_image % 500: print('EXTRACTING MODEL RESPONSES: %.02f %% DONE'%(i_image/len(images)))
    return np.array(layer_responses)

if __name__ == '__main__': 
 
    i_layer = sys.argv[1]     
    print('-LOADING MODEL RESPONSES FOR %s'%i_layer) 
    
    n_components = int(sys.argv[2])
    print('-USING %d COMPONENTS'%n_components) 

    n_iterations = int(sys.argv[3])
    print('-RUNNING FOR %d ITERATIONS'%n_iterations) 

    variation_type = sys.argv[4]
    print('-RUNNING ANALYSIS ON %s VARIATION DATA'%variation_type.upper())

    save_location = '/home/users/bonnen/perirhinal_cortex/analysis/yamins_2014/model_neural_pls_fits'
    save_file_name = '%s/%s-fit_%d-iterations_%d-components_%s.pickle'%(save_location, i_layer, n_iterations, n_components, variation_type)
    print('-SAVING RESULTS TO :', save_file_name) 

    print('-LOADING AND SUBSETTING NEURAL DATA')
    path_ = '/home/users/bonnen/perirhinal_cortex/analysis/yamins_2014/ventral_neural_data.hdf5'
    v4_data, it_data, variation_, images = extract_neural_data( path_ ) 
    
    print('original images shape', images.shape)
    if variation_type == 'all': 
      select_indices = variation_ !=  b'V0'
    else: 
      select_indices = variation_ == variation_type.encode() 

    neural_ = {'v4': v4_data[select_indices], 'it': it_data[select_indices]}
    images = images[select_indices]
    print( '--SUBSET IMAGES SHAPE:', images.shape)  
    
    model_path = '/home/users/bonnen/perirhinal_cortex/analysis/models/'
    vgg, session = load_model(model_path) 
    model_ = extract_model_responses(images, vgg, session, i_layer) 
    print('MODEL RESPONSES SHAPE:', model_.shape)  
    
    print('-BEGINNING LAYER-NEURAL FIT') 
    np.random.seed(3)
    it_test, it_train, v4_test, v4_train = [] , [] , [] , [] 
   
    del(vgg)
    del(session) 
    del(images) 
    
   
    for i_iteration in range(n_iterations): 
        fits_ = model_neural_map(model_, neural_, n_components, per_neuron=True)
        it_test.append(fits_['it']['test']  ) 
        it_train.append(fits_['it']['train']) 
        v4_test.append(fits_['v4']['test']  ) 
        v4_train.append(fits_['v4']['train'])

    save_data = {'it_test':it_test, 'it_train':it_train, 'v4_test':v4_test, 'v4_train':v4_train}
    
    print('SAVING...') 
    with open(save_file_name, 'wb') as handle: 
        pickle.dump(save_data, handle) 
    print(':D') 
