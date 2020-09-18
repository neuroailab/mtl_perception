from __future__ import print_function
from boto.mturk.connection import MTurkConnection
from boto.mturk.question import ExternalQuestion
import boto.mturk.qualification as mtqu
from dateutil.parser import *
import numpy as np
import sys, os, datetime 
import json, math

param_location = '../params.js'
port_number = '8881'
experiment_script = 'index.html'
n_hours_to_complete_hit = 1
n_hours_to_accept_hit = 2
credentials= '../../credentials/aws_keys.json'

def set_path_to_experiment(experiment_script): 
    
    path = os.path.realpath('..')
    node_directory = 'tasks/' 
    experiment_directory = path[path.find(node_directory) + len(node_directory):]
    node_path_to_experiment = os.path.join(experiment_directory, experiment_script) 
    return node_path_to_experiment

def extract_values_from_params( param_file ):
    """ generic script to extract information from js formatted params file""" 
    contents = open( param_file, 'r' ).readlines()
    params = {} 
    for i_line in contents: 
        line = i_line.replace(' ', '') 
        i_key = line[:line.find(':')]
        i_value = line[ line.find(':') +1 : line.find(',')]        
        #[ i for i in params if variable in i][0].replace(' ', '')
        # value = line[ line.find(':') +1 : line.find(',')]

        try: params[i_key] = float(i_value) 
        except: params[i_key] = i_value

    return params

def determine_bonus_and_time(params): 
    """ ideosyncratic script: calculates information to show in HIT description """ 
 
    max_bonus = params['max_experiment_bonus']
    max_time = np.ceil(  (params['n_trials'] * params['estimate_seconds_per_trial'])  /60)

    return max_bonus, max_time

experiment_path = set_path_to_experiment(experiment_script)
params = extract_values_from_params( os.path.join( os.path.realpath('.'), param_location)) 
params['max_bonus'], params['max_time'] = determine_bonus_and_time(params) 

print( "https://stanfordmemorylab.com:%s"%(os.path.join(port_number, experiment_path)) ) 

# send info about inputs if they're aren't enough
if len(sys.argv) < 3:

    sys.exit("""
    usage for either sandbox or live:\n
        $ python submit_hit.py sandbox <n_hits_per_task> <amount> 
        $ python submit_hit.py live <n_hits_per_task> <amount>\n""")

else: # make sure data are formated correctely, and we're in the right context

    # format user input
    context =  sys.argv[1]
    # task_identifier = sys.argv[2] 
    n_hits = int(sys.argv[2])
    compensation_amount = sys.argv[3]
        
    if (context == 'live') or (context == 'sandbox'):

        print('\nCreate %s %s HITs for $%s each?\n'%(str(n_hits), str(context), str(compensation_amount)))
        #print(task_names)
        print('\n(yes/no)\n')
        try: user_response = raw_input()
        except: user_response = raw_input()
        if user_response[0].lower() != 'y':
            sys.exit('\ncareful :)\n')
    else:
        sys.exit("\ncontext needs to be either 'live' or 'sandbox'\n")

# set mturk dependencies
if context == 'sandbox':

    host = 'mechanicalturk.sandbox.amazonaws.com'
    base_url = 'https://workersandbox.mturk.com/mturk/preview?groupId='
    external_submit = 'https://workersandbox.mturk.com/mturk/externalSubmit'

elif context == 'live':

    print('okay... this one is for real!\n')
    host = 'mechanicalturk.amazonaws.com'
    base_url = 'https://www.mturk.com/mturk/preview?groupId='
    external_submit = "https://www.mturk.com/mturk/externalSubmit"

# load acces key information
# key_info = np.load('snail_rootkey_may6.npy').item() 
with open(credentials, 'rb') as aws_keys:
    key_info = json.load(aws_keys)

access_id = str(key_info['access_key_id'])
secret_key = str(key_info['secret_access_key'])

def generate_hit_info(context, compensation_amount, i_hits_per_subsubmission): 
    
    hit_info = {} 
    # mongo info 
    hit_info['database'] = params['database'] 
    hit_info['collection'] = params['collection'] 
    hit_info['iteration'] = params['iteration']
    hit_info['time_of_submission'] = datetime.datetime.now().strftime("%H:%M_%m_%d_%Y")
    hit_info['platform'] = context
    # mturk description info 
    hit_info['external_url'] = "https://stanfordmemorylab.com:%s"%(os.path.join(port_number, experiment_path)) 
    hit_info['keywords'] = ['perception', 'neuroscience', 'game', 'fun', 'experiment', 'research']
    hit_info['description'] = 'An experiment on the relationship between perception and memory'  
    hit_info['experiment_name'] = 'TEXTTEXTEXT'
    hit_info['title'] = 'A perceptual experiment where you can earn up to $%.02f bonus in less than %d minutes!'%( params['max_bonus'], params['max_time'] + 1)  
    # payment and bonus info
    hit_info['payment_for_experiment'] = compensation_amount 
    # mturk interface and worker details 
    hit_info['max_assignments'] = i_hits_per_subsubmission
    hit_info['frame_height'] = 700
    hit_info['approval_rating_cutoff'] = 90
    # experimental timing details -- time is in seconds, e.g: 60 * 60 = 1 hour
    hit_info['lifetime_of_experiment'] = 60 * 60 * n_hours_to_accept_hit
    hit_info['duration_of_experiment'] = 60 * 60 * n_hours_to_complete_hit
    hit_info['approval_delay'] = 1 * 30

    return hit_info 

def post_hits(hit_info, n_sub_hits):

    mtc = MTurkConnection(aws_access_key_id=access_id, aws_secret_access_key=secret_key, host=host)
    q = ExternalQuestion(external_url = hit_info['external_url'],  frame_height=hit_info['frame_height'])
    qualifications = mtqu.Qualifications()
    qualifications.add(mtqu.PercentAssignmentsApprovedRequirement('GreaterThanOrEqualTo', hit_info['approval_rating_cutoff']))
    qualifications.add(mtqu.LocaleRequirement("EqualTo", "US"))
    
    #print('url:', hit_info['external_url'], n_sub_hits)

    the_HIT = mtc.create_hit(question=q,
                          lifetime = hit_info['lifetime_of_experiment'], 
                          max_assignments = n_sub_hits, 
                          title = hit_info['title'],
                          description = hit_info['description'],
                          keywords = hit_info['keywords'],
                          qualifications = qualifications,
                          reward = hit_info['payment_for_experiment'], 
                          duration = hit_info['duration_of_experiment'], 
                          approval_delay = hit_info['approval_delay'],  
                          annotation = hit_info['experiment_name'])

    assert(the_HIT.status == True)

    hit_info['hit_id'] = the_HIT[0].HITId
    hit_url = "{}{}".format(base_url, the_HIT[0].HITTypeId)
    hit_info['hit_url'] = hit_url


    record_name = '%s_submission_records.npy'%(context)

    if record_name not in os.listdir(os.getcwd()):
      turk_info = {}
    else: 
      turk_info = np.load(record_name).item()

    key_name = 'submission_%d'%len(turk_info.keys())
    turk_info[key_name] = hit_info
    np.save(record_name, turk_info)

    print('HIT_ID:', the_HIT[0].HITId, "\nwhich you can see here:", hit_url)
    
### make the magic happen
max_hits = 9
full_cycles = int(n_hits/max_hits)
partial_cycle = n_hits%max_hits
submissions = list(np.repeat(max_hits, full_cycles))
if partial_cycle: submissions.append(partial_cycle)

for n_hits_per_submission in submissions: 
    
    hit_info = generate_hit_info(context, compensation_amount, n_hits_per_submission)
    #print(hit_info)
    post_hits(hit_info, n_hits_per_submission)

max_hits = 9
full_cycles = int(n_hits/max_hits)
partial_cycle = n_hits%max_hits
submissions = list(np.repeat(max_hits, full_cycles))
if partial_cycle: submissions.append(partial_cycle)
