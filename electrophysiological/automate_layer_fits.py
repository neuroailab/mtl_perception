#!/usr/bin/env python
import os

python_location = '/home/users/bonnen/env/bin/python'
script_location = '/home/users/bonnen/perirhinal_cortex/analysis/yamins_2014/model_neural_pls_fits/layer_neural_fit.py'

def mkdir_p(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)

working_directory = os.getcwd() 
sbatch_directory = '.jobs'
output_directory = '.outs'

job_directory = os.path.join( working_directory, sbatch_directory )
output_directory = os.path.join( working_directory, output_directory ) 

# make sbatch and output directories if necessary
mkdir_p(job_directory)
mkdir_p(output_directory)

layers = ['conv1_1', 'conv1_2', 'pool1', 
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
          'conv4_1', 'conv4_2', 'conv4_3', 'pool4',
          'conv5_1', 'conv5_2', 'conv5_3', 'pool5',
          'fc6', 
          'fc7', 
          'fc8']

##layers = ['fc6', 'fc7', 'fc8'] 
n_components = 5 
variation_level = 'all'
n_iterations = 1 
# 512000
memory_requested = 256000

for i_layer in layers:

    i_submission_name = 'neural-%s_fits_%d-components_%d_%s'%(i_layer, n_components, memory_requested, variation_level)
    i_job = os.path.join(job_directory, "%s.sbatch"%i_submission_name)

    with open(i_job, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s\n"%i_submission_name) 
        fh.writelines("#SBATCH --output=%s/%s.out\n"% (output_directory, i_submission_name))
        fh.writelines("#SBATCH --error=%s/%s.err\n" % (output_directory, i_submission_name))
        fh.writelines("#SBATCH --time=30:00:00\n")
        fh.writelines("#SBATCH --mem=%d\n"%memory_requested)
        fh.writelines('#SBATCH -p yamins,owners,normal\n')
        fh.writelines('module load python/3.6.1\n')
        fh.writelines("srun %s %s %s %d %d %s\n"%(python_location, script_location, i_layer, n_components, n_iterations, variation_level))

    print('--Submitting job %s'%i_job) 
    os.system("sbatch %s" %i_job)
