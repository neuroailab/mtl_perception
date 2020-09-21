import scipy, os, h5py, pickle 
import numpy as np
from sklearn.linear_model import LogisticRegression

permutation = np.random.permutation

def make_list(l):
    return np.array([i.decode() for i in l])

def generate_trial(typicals, oddity, variation_level, n_distractors=2): 
   
    i_variation = np.array([i ==variation_level for i in variation_labels])
    i_categories = np.random.randint(0,7,2)+1
    typicals = np.nonzero(np.array([i == typicals for i in object_labels]) * i_variation)[0]
    typicals = np.random.permutation(typicals)
    oddity = np.nonzero(np.array([i == oddity for i in object_labels]) * i_variation)[0]
    i_oddity = oddity[np.random.randint(0,len(oddity), 1)[0]]
    image_numbers = [i for i in typicals[0:n_distractors]]
    image_numbers.append(i_oddity)
    image_numbers = np.random.permutation(image_numbers)
    correct = np.nonzero(image_numbers == i_oddity)[0][0]   

    return image_numbers, correct, {'typical': object_labels[typicals[0]], 'oddity': object_labels[i_oddity]}

    
def extract_diagonal(X): 
     return np.array( [[X[i, j] for i in range(len(X)) if i != j] for j in range(len(X)) ] )

def format_trial(trial_indices, typicals, oddity, n_distractors): 
    
    typical_indices = permutation([i for i in np.nonzero( object_labels == typicals )[0] if i not in trial_indices ])
    oddity_indices = permutation([i for i in np.nonzero( object_labels == oddity )[0] if i not in trial_indices ])

    trial_stimuli = []
    trial_answers = [] 

    for i_trial in range(0, len(typical_indices), n_distractors): 
        i_oddity_index = np.random.randint(n_distractors+1)
        i_trial_stimuli = typical_indices[i_trial:i_trial+n_distractors]
        i_trial_stimuli = np.insert(i_trial_stimuli, i_oddity_index, oddity_indices[i_trial]) 
        trial_stimuli.append( model_data[i_trial_stimuli].flatten() ) 
        trial_answers.append( i_oddity_index)

    trial_stimuli = np.array( trial_stimuli )
    
    return trial_stimuli, trial_answer

def generate_stimulus_set(load_saved_data=1): 

    n_distractors = 2
    n_iterations = 100

    summaries = {} 

    for i_category in category_names: 

        print('\n', i_category)
        summaries[i_category] = {} 
        category_objects = np.unique(object_labels[category_labels == i_category]) 

        for i_object in category_objects: 

            summaries[i_category][i_object] = {}

            for j_object in [j for j in category_objects if j != i_object]: 

                print(i_object, j_object)
                linear_accuracy = []
                distance_accuracy = [] 
                distances = [] 

                for _ in range(n_iterations): 

                    trial_indices, oddity_index, object_names = generate_trial(i_object, 
                                                                               j_object, 'V3', 
                                                                               n_distractors=n_distractors)

                    typical_name, oddity_name =  list(object_names.values()) 

                    trial_stimuli, trial_answers = format_trial(trial_indices, 
                                                                typical_name, oddity_name, 
                                                                n_distractors=n_distractors)

                    clf = LogisticRegression(random_state=0, solver='lbfgs').fit(trial_stimuli, trial_answers)
                    linear_correct = clf.predict(  model_data[trial_indices].flatten()[np.newaxis] )[0] == oddity_index  
                    linear_accuracy.append(linear_correct)

                    trial_covariance = np.corrcoef( model_data[trial_indices] )    
                    object_distances = np.mean(extract_diagonal(trial_covariance), 1)
                    distance_correct = object_distances.argmin() == oddity_index

                    distance_accuracy.append( distance_correct )

                    typical_distances = [i for i in object_distances if i != object_distances[oddity_index]]
                    mean_closetypical_distance = typical_distances[np.array(typical_distances).argmin()]
                    mean_oddity_distance = object_distances[oddity_index]
                    trial_distance = mean_oddity_distance - mean_closetypical_distance
                    distances.append(trial_distance) 

                summaries[i_category][i_object][j_object] = {'linear_accuracy': linear_accuracy, 
                                                             'distance_accuracy': distance_accuracy, 
                                                             'distances': distances}
        f = open("summaries.pkl","wb")
        pickle.dump(summaries,f)
        f.close()
                
if __name__ == '__main__':

    # load representations extracted from vgg16
    model_data = np.load('fc6_all_images.npy')
    # load datasets
    data = h5py.File('ventral_neural_data.hdf5', 'r')
    # # extract some meta data that we'll be using a lot 
    category_names = np.unique(make_list(data['image_meta']['category']))
    category_labels = make_list(data['image_meta']['category'])
    object_labels = make_list(data['image_meta']['object_name'])
    variation_labels = make_list(data['image_meta']['variation_level'])
    variation_levels = np.unique(variation_labels)
    n_stimuli = len(data['image_meta']['object_name'])
    neural_data = np.array(data['time_averaged_trial_averaged'])
    data.close()

    generate_stimulus_set() 
