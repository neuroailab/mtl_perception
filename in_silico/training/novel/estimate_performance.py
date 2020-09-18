import sys, h5py, os, pickle, numpy as np
import torch, torchvision, torch.nn as nn
from PIL import Image 
import foveate # custom  

def generate_paths(params): 
    
    if 'oak' not in os.listdir('/'): 
        print('- LOCAL ENVIRONMENT') 
        base = '/Users/biota/work/perirhinal_cortex'
        response_location = base
    else: 
        print('- SHERLOCK ENVIRONMENT') 
        base = '/home/users/bonnen/perirhinal_cortex'
        response_location = '/scratch/users/bonnen/model_responses/'

    # directory for analysis scripts and directories
    experiment_directory = os.path.join(base, 'analysis/yamins_2014/foveating_stimuli')
    # location to save model responses to stimuli
    response_results = 'vgg16_%s_%s_responses.npy'%(params['training_data'], params['stimuli'])
    # generic template to save results
    results = 'vgg16_%s_%s_%s_%diterations.pickle'
    # substitute model params into template
    results = results%(params['training_data'], params['stimuli'], params['readout_method'], params['n_iterations']) 

    paths = {
     'experiment_directory': experiment_directory,
     'hvm_data': os.path.join(base, 'analysis/yamins_2014/ventral_neural_data.hdf5'), 
     'oddity_metadata': os.path.join(experiment_directory, 'ACFP5_oddity_metadata.pickle'), 
     'vggface_model':os.path.join(base, 'analysis/synthesis/face/models/'), 
     'vggface_weights': os.path.join(base, 'analysis/synthesis/face/pretrained/VGG_FACE.t7'), 
     'saved_responses': os.path.join(response_location, response_results), 
     'oddity_results': os.path.join(experiment_directory, 'results', results)}

    return paths 

def model_readout_method(params):
  """learned readout determines model-based performance on oddity task"""
  print('- USING %s READOUT'%params['readout_method'].upper())
  if params['readout_method']=='linear_svm': 
    from sklearn.svm import LinearSVC
    model = LinearSVC(random_state=0)
  elif params['readout_method']=='rbf_svm': 
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', gamma='auto', random_state=0)
  elif params['readout_method']=='logistic_regression': 
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=0)
  elif params['readout_method']=='mlp':     
    from sklearn.neural_network import MLPClassifier
    # should be better optimized, to be fair
    if params['training_data'] == 'neural': 
      model = MLPClassifier(hidden_layer_sizes=(200,200,200), learning_rate_init=.001, max_iter=1000, random_state=0, tol=0.00001)
    else: 
      model = MLPClassifier(hidden_layer_sizes=(5000,5000,5000), learning_rate_init=.00001, max_iter=1000, random_state=0, tol=0.00001, verbose=True)
  return model

## STIMULI EXTRACTION
def prep_image(image):
    image = torch.Tensor(image).permute(2, 0, 1).view(1, 3, 224, 224).double()
    image -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1)
    return image

def model_image(model, image):
    return model(image).detach().numpy().flatten()

def rgb_from_grey(grey):
    wide, high = grey.shape
    rgb = np.empty((wide, high, 3), dtype=np.uint8)
    rgb[:, :, 2] =  rgb[:, :, 1] =  rgb[:, :, 0] =  grey
    return rgb

def resize_all_images(imgs, xy=224): 
  return [np.array(Image.fromarray(i).resize((xy, xy))) for i in imgs]

def extraction_function(layer, inp, output):
    output = output.cpu().detach().numpy()
    responses.append(output.flatten())

def define_face_model(params):
    """pretrained vgg16 face model"""
    # add location of face model to path
    print(sys.path)
    print(params['paths']['vggface_model'])
    sys.path.insert(len(sys.path), params['paths']['vggface_model'])
    print(sys.path)
    # import from custom library
    import vgg_face as face
    # define model
    face_model = face.VGG_16().double()
    # load weights
    face_model.load_weights(params['paths']['vggface_weights'])
    return face_model.eval()

def feature_extraction_protocol(model, function, params):
    """passive feature extraction during forward pass""" 
    # identify model layer to extract features from
    if params['training_data'] == 'vggface': 
      model_layer = getattr(model, 'fc6')
    if params['training_data'] == 'imagenet': 
      model_layer = getattr(getattr(model, 'classifier'), '1') # relu or no? 
    if params['training_data'] == 'untrained': 
      model_layer = getattr(getattr(model, 'classifier'), '1') # relu or no? 
    # register forward hook
    model_layer.register_forward_hook(function)

def extract_experiment_info(params, responses_loaded=0): 
    """extract metadata and preprocess stimuli"""
    # hvm contains stimuli, metadata, neural data
    HVM = h5py.File(params['paths']['hvm_data'], 'r')
    # extract relevant metadata 
    metadata = {'categories': np.array([i.decode() for i in HVM['image_meta']['category']]),
                'objects': np.array([i.decode() for i in HVM['image_meta']['object_name']])}
    # don't need stimuli if model responses have been loaded
    if responses_loaded:
      # close h5py 
      HVM.close() 
      # exit.
      return metadata
    elif params['training_data'] == 'neural':  
      # extract neural data from v4 and it
      neural_data = np.array( HVM['time_averaged_trial_averaged'] )
      it_neurons  = np.array( HVM['neural_meta']['IT_NEURONS'] ) 
      v4_neurons  = np.array( HVM['neural_meta']['V4_NEURONS'] ) 
      objects = {'it': neural_data[:, it_neurons], 
                 'v4': neural_data[:, v4_neurons]} 
    else: 
      # foveate or retain original images
      if params['stimuli'] == 'foveated':
          print('- FOVEATED IMAGES') 
          objects = foveate.all_stimuli(HVM) 
      else: 
          print('- ORIGINAL IMAGES') 
          objects = np.array(HVM['images']) 
      # resize    
      objects = resize_all_images(objects, xy=params['size']) 
      # convert images to rgb 
      objects = [rgb_from_grey(objects[i]) for i in range(len(objects))]
      # preprocess image for model 
      objects = [prep_image(objects[i]) for i in range(len(objects))]
      # close h5py 
    
    HVM.close() 
    return objects, metadata

def extract_model_responses(params, responses): 
    
    
    # check if model responses have already been saved 
    if os.path.isfile(params['paths']['saved_responses']):
        print('- LOADING MODEL RESPONSES FROM FILE') 
        # extract saved model responses
        responses = np.load(params['paths']['saved_responses'])
        # extract metadata for experiment generation
        metadata = extract_experiment_info(params, responses_loaded=1) 
    elif params['training_data'] == 'neural': 
        # load it and v4 data 
        print('- LOADING NEURAL RESPONSES FROM V4 & IT')
        responses, metadata = extract_experiment_info(params) 
    else: 
        # extract metadata for experiment generation
        stimuli, metadata = extract_experiment_info(params) 
        # define model to extract features from 
        if params['training_data'] == 'vggface':
            print('- VGGFACE TRAINED VGG16') 
            # define model from pretrained weights
            model = define_face_model(params) 
        elif params['training_data'] == 'imagenet': 
            print('- IMAGENET TRAINED VGG16') 
            # load imagenet trained vgg
            model = torchvision.models.vgg16(pretrained=True).double() 
        elif params['training_data'] == 'untrained': 
            print('- UNTRAINED VGG16') 
            model = torchvision.models.vgg16(pretrained=False).double() 
        
        # set up feature extraction protocol  
        feature_extraction_protocol(model, extraction_function, params)
        print('- EXTRACTING FEATURES') 
        # pass all stimuli through model 
        with torch.no_grad():
          # stimli are extracted with forward hook 
          [model(stimuli[i]) for i in range(len(stimuli))];

        print(responses) 
        # reshape variable scoped locally and globally 
        responses = np.array(responses).squeeze()
        print('responses shape:', responses.shape ) 
        np.save(params['paths']['saved_responses'], responses)
        print('- MODEL RESPONSES SAVED') 
    
    return responses, metadata

## ODDITY EXPERIMENT
def nonzero(x): 
    return np.nonzero(x)[0]

def permutation(x): 
    return np.random.permutation(x)

def generate_training_indices(metadata, i_marker, n_typicals=2): 
    """"""
    trial_indices = list( i_marker['indices'] )
    typical_name = i_marker['typical_name']
    oddity_name = i_marker['oddity_name']
    oddity_index = trial_indices.pop(i_marker['oddity_index'])
    
    typical_indices = permutation([i for i in nonzero(metadata['objects'] == typical_name ) if i not in trial_indices])
    oddity_indices = permutation([i for i in nonzero(metadata['objects'] == oddity_name ) if i != oddity_index] )
    
    training_indices, training_labels = [] , [] 
    
    for i_trial in range(0, len(typical_indices), n_typicals): 
        i_trial_indices = [i for i in typical_indices[i_trial:i_trial+2]]
        oddity_index = np.random.randint(3)
        i_trial_indices.insert(oddity_index, oddity_indices[i_trial]) 
        training_indices.append( i_trial_indices )
        training_labels.append( oddity_index )
        
    return training_indices, training_labels

def model_oddity_behavior(params, responses): 
    
    def perform_trial(responses, training_indices, training_labels, marker_info, model): 
      train_data_ = np.array([responses[trial].flatten() for trial in training_indices]) 
      test_data_  = np.expand_dims( responses[marker_info['indices']].flatten(), 0) 
      model.fit(train_data_, training_labels)
      i_prediction = model.predict(test_data_)
      i_correct = (i_prediction==marker_info['oddity_index'])[0]
      return i_correct 

    # extract model responses -- NEEDS TO RETURN NEURAL DATA IF SPECIFIED 
    responses, meta = extract_model_responses(params, responses)
   
    with open(params['paths']['oddity_metadata'], 'rb') as handle: 
        markers = pickle.load(handle)
 
    # define model
    model = model_readout_method(params)
    # initialize data structure
    results = {}
    print('- ESTIMATING ODDITY PERFORMANCE OVER %d ITERATIONS'%params['n_iterations'])

    for i_marker in markers: 
        
        # 
        marker_info = markers[i_marker]
        # initialize data structure
        results[i_marker] = {'meta': markers[i_marker], 'iterations':params['n_iterations'], 'correct':[]}
        # modify for neural data if necessary
        if params['training_data'] == 'neural': results[i_marker]['correct'] = {'it':[], 'v4':[]}

        for i_iteration in range(params['n_iterations']): 
            
            # generate a random configuration of oddity experiments to train a readout with 
            training_indices, training_labels = generate_training_indices(meta, marker_info)
            
            if params['training_data'] != 'neural': 
              i_correct = perform_trial(responses, training_indices, training_labels, marker_info, model) 
              results[i_marker]['correct'].append( i_correct )
            else: 
              # fit it and v4 
              for i_region in ['v4', 'it']: 
                i_correct = perform_trial(responses[i_region], training_indices, training_labels, marker_info, model) 
                results[i_marker]['correct'][i_region].append( i_correct )
            
              pass 
        
        if params['training_data'] == 'neural': 
          print('index: %d - typical %s - accuracy it %.02f | v4 %.02f'%(i_marker, 
                                                                         marker_info['typical_name'],
                                                                         np.mean(results[i_marker]['correct']['it']), 
                                                                         np.mean(results[i_marker]['correct']['v4'])
                                                                        ))
        else: 
          # create results down here
          print('index: %d - typical %s - accuracy %.02f'%(i_marker, marker_info['typical_name'],
                                                          np.mean(results[i_marker]['correct'])))

    # save to pickle 
    with open(params['paths']['oddity_results'], 'wb') as handle: 
        pickle.dump(results, handle)

if __name__ == '__main__': 
    
    # sets variable globally, passes locally--for scoping issues with registering forward hooks
    responses = []  
    
    if len(sys.argv) != 5: 
        print('\nexample usage:') 
        print('           $ python estimate_performance.py <vggface|imagenet|untrained> '
              '<foveated|original> <linear_svm|logistic_regression|...> <n_iterations>\n')
        sys.exit()
    
    # parameters for analysis
    params = {'training_data': sys.argv[1], 
                    'stimuli': sys.argv[2], 
             'readout_method': sys.argv[3], 
               'n_iterations': int(sys.argv[4]), 
                       'size': 224, }
    
    # path and result locations  
    params['paths'] = generate_paths(params)  
    # those selected markers we will focus on specifically
    model_oddity_behavior(params, responses) 
    # printout for inventory
    print('- PARAMS', params)
