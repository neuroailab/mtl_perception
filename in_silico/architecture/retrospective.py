import pickle, pandas 
import os, numpy as np 
import torch 
import torchvision.models as models
import torch, torch.nn as nn
import colour 
from torchvision import transforms

def get_experiments():
    base = '/Users/biota/work/perirhinal_cortex/analysis'
    stimuli = {}
    #### barense
    with open(os.path.join(base, 'barense_2007/barense_diagnostic_experimental_info.pickle'), 'rb') as handle:
        stimuli['barense'] = pickle.load(handle)

    #### stark
    with open(os.path.join(base, 'stark_2000/stark_stimuli.pickle'), 'rb') as handle:
        stimuli['stark'] = pickle.load(handle)

    #### lee 2006
    with open(os.path.join(base, 'lee_2006/lee_2006_stimuli.pickle'), 'rb') as handle:
        #lee_2006_stim, lee_2006_ans  = pickle.load(handle)
        #stimuli['lee_2006'] = {'stimuli': lee_2006_stim, 'answer_key': lee_2006_ans}
        stimuli['lee_2006'] = pickle.load(handle)

    #### lee 2005
    with open(os.path.join(base, 'lee_2005/lee_2005_experiment_one_stimuli.pickle'), 'rb') as handle:
        stimuli['lee_2005'] = pickle.load(handle)

    return stimuli

def model_barense_experiments(model, expmt):

    stimuli = expmt['stimuli']
    accuracy = {}

    for i_type in expmt['stimuli']:

        accuracy[i_type] = []

        stimuli_with_answers = [i for i in list(expmt['stimuli'][i_type]) if i in expmt['answer_key'][i_type]]

        for i_image in stimuli_with_answers:

            images = [to_tensor(image) for image in  stimuli[i_type][i_image] ]

            with torch.no_grad():
                outputs = [model(i_image).numpy().flatten() for i_image in images]

            correct_index = int(expmt['answer_key'][i_type][i_image])-1
            trial_correct = evaluate_trial(outputs, correct_index)
            accuracy[i_type].append( trial_correct )

    #print(i_type, np.mean(accuracy[i_type]).round(3))

    return {i:np.mean(accuracy[i]).round(3) for i in accuracy}

def model_responses_to_stimuli(model, stimuli):

    model_responses = {e:[] for e in list(stimuli)}

    for i_set in list(stimuli.keys()):

        #print(i_set, end=' ')

        labels, activations, errors, removed_pairs = [], [], [], []

        for i_image in range(len(stimuli[i_set])):

            # repeat in z and then expand the first index
            image_i = np.repeat(stimuli[i_set][i_image][ :, : , np.newaxis], 3, axis=2)

            # extract model representations
            with torch.no_grad():
                i_responses = model(to_tensor(image_i)).numpy().flatten()

            model_responses[i_set].append( i_responses )
        model_responses[i_set] = np.array(  model_responses[i_set] )

    return model_responses

def generate_trial(i_group, meta, noise_magnitude=0, visualize=1, input_type='human'):

    if "3DGREYSC" in i_group: i_group = i_group[:8]

    n_viewpoints = meta[i_group]['n_viewpoints']
    n_groups = meta[i_group]['n_categories']
    target_identity = np.random.randint(n_groups)

    possible_distractor_identity = [i for i in np.arange(n_groups) if i !=target_identity]
    distractor_identity = possible_distractor_identity[np.random.randint(len(possible_distractor_identity))]

    target_viewpoint = np.random.randint(n_viewpoints)
    if i_group == 'FACES':
        distractor_viewpoints = np.random.permutation([i for i in np.arange(n_viewpoints) if i !=target_viewpoint])
    else:
        distractor_viewpoints = np.arange(5)

    # now convert these to image indices
    ij_target = target_viewpoint+(target_identity*n_viewpoints)
    ij_distractors = [i_distractor + (distractor_identity*n_viewpoints) for i_distractor in distractor_viewpoints]
    ij_distractors = np.random.permutation(ij_distractors)
    trial_stims = [ij_target]
    trial_stims.extend(ij_distractors)

    if visualize:

        show_trial_stims(i_group, trial_stims, n_viewpoints, noise_magnitude, input_type=input_type)

    # only returns the indices of these trial stims
    return [int(i) for i in trial_stims]

def run_single_experiment(model_data, meta, i_category, n_subjects, n_trials):

    x = np.array(list(range(6)))
    trial_decision = []
    condition_accuracy = []

    for i_subject in range(n_subjects):

        for i_iteration in range(n_trials):
            trial_stims = generate_trial(i_category, meta, visualize=0)
            responses = [model_data[i_category][i] for i in trial_stims]
            trial_covariance = np.corrcoef(responses)
            trial_decision_space = np.array([trial_covariance[i, x[x!=i]] for i in x])
            trial_decision_space.sort()
            i_choice = trial_decision_space[:,-1].argmin()
            correct = i_choice == 0
            trial_decision.append(correct)

    condition_accuracy = np.mean(trial_decision).round(3)

    #print( i_category, condition_accuracy )
    return condition_accuracy

def model_stark_experiments(model, experiment):

    responses = model_responses_to_stimuli(model, experiment['stimuli'])

    n_trials = 100
    n_subjects = 100
    np.random.seed(0)
    s_results = {}

    for e in responses:
        s_results[e] = run_single_experiment(responses, experiment['meta'], e, n_subjects, n_trials)

    return s_results

def model_lee2006_experiments(model, experiments):

    model_responses = {}
    experimental_accuracy = {e:[] for e in list(experiments['stimuli'])}
    n_images = 4

    for i_experiment in experiments['stimuli']:

        accuracy = []

        for i_trial in list(experiments['stimuli'][i_experiment]):

            i_images = experiments['stimuli'][i_experiment][i_trial]

            with torch.no_grad():
                i_responses = np.array([model(to_tensor(i)).numpy().flatten() for i in i_images])

            i_correct = evaluate_trial(i_responses, int(experiments['answer_key'][i_experiment][i_trial]))
            accuracy.append( i_correct )

            if i_experiment == 'faces':
                model_responses[i_trial] = i_responses

        #print(  i_experiment, np.mean( accuracy ) )
        experimental_accuracy[i_experiment] = np.mean( accuracy ).round(3)

    return experimental_accuracy, model_responses

def model_lee2005_experiments(model, experiment, lee2006_responses):

    def answer(s):
        return int(s[-5])-1

    ### EXPERIMENT ONE
    model_responses = {i:{} for i in experiment}
    for i_experiment in experiment:

        model_responses[i_experiment] = {}
        for i_trial in list(experiment[i_experiment]):

            images = experiment[i_experiment][i_trial]
            with torch.no_grad():
                i_responses = [model(to_tensor(i_image)).numpy().flatten() for i_image in images]
            model_responses[i_experiment][i_trial] = i_responses

    experimental_accuracy = {}
    np.random.seed(0)
    n_iterations = 1000
    for i_experiment in list( model_responses ):

        if 'Face' in i_experiment: i_len = 6
        else: i_len = 5

        n_trials = len([i for i in model_responses[i_experiment] ])//i_len
        accuracy = []

        for i_object in range(1, n_trials + 1):

            oddity_group = (i_object%n_trials) + 1
            for i_iteration in range(n_iterations):

                trial_stimuli = ['%d%d'%(i_object, i_view) for i_view in range(1, i_len+1)]
                oddity_viewpoint = np.random.randint(1, len(trial_stimuli))
                i_oddity = '%d%d'%(oddity_group, oddity_viewpoint+1)
                trial_stimuli.pop(oddity_viewpoint)
                trial_stimuli.insert(0, i_oddity)
                trial_responses = np.array([model_responses[i_experiment][i] for i in trial_stimuli]).squeeze()
                accuracy.append( evaluate_trial(trial_responses) )

        experimental_accuracy[i_experiment] = np.mean( accuracy ).round(3)
        #print( i_experiment, experimental_accuracy[i_experiment] )

    ##### EXPERIMENT TWO
    experimental_n = 31
    experiment_two_performance = []
    np.random.seed(0)
    for i_iteration in range(n_iterations):

        random_experiment = np.random.permutation(list(lee2006_responses))[0:experimental_n]
        random_performance = [evaluate_trial(lee2006_responses[t], answer(t)) for t in lee2006_responses]
        experiment_two_performance.append(np.mean(random_performance))

    experimental_accuracy['E2_faces'] = np.mean(experiment_two_performance).round(3)
    #print( i_experiment, experimental_accuracy['E2_faces'] )

    return experimental_accuracy

def model_all_experiments(model, experiments):

    results = {}

    results['barense'] = model_barense_experiments(model, experiments['barense'])
    results['stark'] = model_stark_experiments(model, experiments['stark'])
    results['lee_2006'], lee2006_representations = model_lee2006_experiments(model, experiments['lee_2006'])
    results['lee_2005'] = model_lee2005_experiments(model, experiments['lee_2005'], lee2006_representations)

    renaming = { 'FACES': 'faces',
                 '3DGREYSC_snow_1': 'snow3',
                 '3DGREYSC_snow_2':'snow4',
                 '3DGREYSC_snow_3':'snow5',
                 'faces':'dfaces',
                 'scenes':'dscenes',
                 'Faces':'faces_E1',
                 'Novel objects':'novel_objects_E1',
                 'Familiar objects':'familiar_objects_E1',
                 'E2_faces':'faces_E2'}

    resnet_results = {}
    for ex in results:
        for study in results[ex]:
            if study in renaming:
                resnet_results[renaming[study]] = results[ex][study]
            else:
                 resnet_results[study] = results[ex][study]

    return resnet_results


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def to_tensor(x):
    return preprocess(x).unsqueeze(0)

def evaluate_trial(trial_responses, correct_index=0):
    x = np.array(list(range(len( trial_responses ))))
    trial_covariance = np.corrcoef(trial_responses)
    trial_decision_space = np.array([trial_covariance[i, x[x!=i]] for i in x])
    trial_decision_space.sort()
    i_choice = trial_decision_space[:,-1].argmin()
    correct = i_choice == (correct_index)
    return correct

def define_model(model, pretrained=True):
    model = model(pretrained=pretrained).eval()
    #print(model)
    modules=list(model.children())[:-2]
    model=nn.Sequential(*modules)
    for p in model.parameters(): p.requires_grad = False
    #print(model)
    return model


if __name__ == '__main__': 

    experiments = get_experiments()
    modeling_results = {} 


    models = {'18': models.resnet18, 
              '34': models.resnet34, 
              '50': models.resnet50, 
              '101': models.resnet101, 
              '152': models.resnet152}
    
    retrospective = {} 
    for i_model in models: 
        
        print('\t resnet %s...'%i_model) 
        model = define_model(models[i_model])
        retrospective[i_model] = model_all_experiments(model, experiments)

    with open('retrospective.pickle', 'wb') as handle:
        pickle.dump(retrospective, handle)
