params = { 
  // total number of trials in experiment
  n_trials: 60, 
  // categories to draw the typicals from
  reference_categories: ['cars', 'chairs', 'animals', 'planes', 'tables', 'fruits', 'faces', 'boats'], 
  feedback: false, 
  // between-category frequency is every between_frequency -1 trials
  between_frequency : undefined, 
  // control frequency
  control_frequency : 10, 
  // variation level for trial types
  experimental_variation_level : 'V3', 
  control_variation_level : 'V0', 

  // number of trials correct in a row before practice is over
  practice_criterion : 5, 

  // bonus per each correct response
  trial_bonus : .05, 
  // penalty for each incorrect response
  trial_penalty : .15, 
  // max time per trial till marked as incorrect
  max_decision_time : 10000, // ten seconds
  // 
  max_experiment_bonus: 2.00, 
  // server info
  database : 'oddity', 
  collection : 'objectome',
  iteration : 'within_category_all0', 
  stimulus_path  : '../stimuli/objectome/',
  array_type : '2x2',
  // tell subjects max time experiment should takee
  estimate_seconds_per_trial : 4,
  key_ : {52:1, 57:2, 82:3, 73:4},
}

params.completion_time =  Math.ceil((params.n_trials * params.estimate_seconds_per_trial)/60)
params.cutoff_percent = (100 * ( ( params.trial_penalty ) / ( params.trial_bonus + params.trial_penalty) )).toFixed(0)
