#!/usr/bin/env python
import os

# function to make directory if necessary
def make_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

# information for python and scripts to run
python_location = '/home/users/bonnen/env/bin/python'
working_directory  = '/home/users/bonnen/perirhinal_cortex/analysis/yamins_2014/foveating_stimuli/'
script_location = os.path.join(working_directory, 'estimate_performance.py')

# information about jobs to print out--eg progess, errors, etc.
job_directory = os.path.join( working_directory, '.jobs')
output_directory = os.path.join( working_directory, '.outs') 

# make sbatch and output directories if necessary
make_directory(job_directory)
make_directory(output_directory)

# analysis parameters to iterate over
train_data  = ['vggface', 'imagenet', 'untrained', 'neural']
view_types = ['foveated', 'original']
readout_types = ['mlp', 'logistic_regression', 'linear_svm'] 

# iterations per stimuli 
n_iterations = 100

for i_train in train_data: 
  for i_view in view_types: 
    for i_readout in readout_types: 
      
      # the mlp is a heavier lift 
      if (i_readout == 'mlp') * (i_train != 'neural') : 
        memory_requested = 256000
      else: 
        memory_requested = 128000

      # generate submission name from parameters within this iteration of the loop 
      i_submission = 'oddity-%s_%s_%s_%diterations'%(i_train, i_view, i_readout, n_iterations)
      # generate file to execute with parameters given from loop  
      i_job = os.path.join(job_directory, "%s.sbatch"%i_submission)
  
      # generate file that we can submit to slurm -- white spaces are critical 
      with open(i_job, 'w') as fh:
          fh.writelines("#!/bin/bash\n")
          fh.writelines("#SBATCH --job-name=%s\n"%i_submission) 
          fh.writelines("#SBATCH --output=%s/%s.out\n"% (output_directory, i_submission))
          fh.writelines("#SBATCH --error=%s/%s.err\n" % (output_directory, i_submission))
          fh.writelines("#SBATCH --time=30:00:00\n")
          fh.writelines("#SBATCH --mem=%d\n"%memory_requested)
          fh.writelines('#SBATCH -p yamins,owners,normal\n')
          fh.writelines('module load python/3.6.1\n')
          fh.writelines('source /home/users/bonnen/env/bin/activate\n')
          fh.writelines("python %s %s %s %s %s\n"%(script_location, i_train, i_view, i_readout, n_iterations))
      
      # submit
      os.system("sbatch %s" %i_job)
      print('submitted job %s'%i_job)
