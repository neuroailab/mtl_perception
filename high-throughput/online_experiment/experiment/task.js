experiment_id = jsPsych.randomization.randomID(20)

//params.group_type = ['even', 'odd'][ 1 * (Math.random() > .6 )]
//params.iteration = params.iteration + params.group_type 
//console.log(params.iteration)

// instructions up to practice trials: includes visualization of trials and responses
var instructions = {
  type: "instructions",
  pages: [
    // welcome page
    "<p style= 'font-size:200%' ><b>Welcome to our experiment!</b></br></p>" + 
    "<p>This is a challenging visual task where your bonus depends on your performance.</p>" + 
    "<p><b>You'll be able to earn up to $" + (params.max_experiment_bonus).toFixed(2) + 
    " in less than " + params.completion_time + " minutes, but only if you do well.</b></p>" + 
    "<p>We want you to perform well, so we'll start off with some instructions.</p>", 
    // example trial 
    "<p style='font-size: 150%'><b>Experiment Layout</b></p>" +
    "<p>In each trial, you'll be looking at three black and white images at the center of the screen, like this:</p>" + 
    '<p><img  style="width:40%" src="utils/example_trial.png"></img></p>'+    
    "<p><b>You can always ignore the backgrounds; the backgrounds are irrelevant.</b><br>" + 
    "Two of the images (superimposed on the backgrounds) will be of the same object shown from different viewpoints." + 
    "<br>One of these objects will be an image of a different object. We'll call the different object <b>the oddity</b>. ", 
    // explicit about oddity in example trial
    "<p style='font-size:150%'><b>Your goal</b><p>" + 
    "<p>In every trial, your task is to look at all the images presented to you and select the oddity. </p>" + 
    '<p><img  style="width:40%" src="utils/example_trial_explicit.png"></img></p>' +  
    "<p><br>In the example above, there are two images of the same Elephant, shown from different viewpoints." + 
    "<br>One of the images is of a Lion. The lion is the oddity</p>",  
    // bonus info 
    "<p style='font-size: 150%'><b>Bonus information</b></p>" +
    "<p>You'll earn $" + params.trial_bonus.toFixed(2) + ' towards your bonus for every trial you get right; ' +  
    "you'll loose $" + params.trial_penalty.toFixed(2) +" when you get it wrong.</p>" + 
    "<p>That means you'll have to get at least " + params.cutoff_percent + "% correct to earn any bonus!<br>", 
    // difficulty info 
    "<p style='font-size: 150%'><b>Difficulty</b></p>" + 
    "<p>You'll be finding the oddity among very similar objects--like selecting one face from another face.</p>" +
    "<p>This should be fun and challenging, so make sure you take your time in each trial!" ,  
     // timing info
    "<p style='font-size: 150%'><b>Timing information</b></p>" +
    "<p>You have ten seconds to complete each trial; if you don't respond in this time the trial will be marked as incorrect</p>",  
    // keyboard responses
    "<p style='font-size: 150%'><b>Response keys</b></p>" +
    "Once you identify the oddity, you'll have to use the following keys on your keyboard to select it: &#x2190;, &#x2192; or &#x2193; </p>" + 
    "<p>" + 
      "<b>&#x2190;</b> (Left Arrow) Object on the left" + 
      '<br><b>&#x2192;</b> (Right Arrow) Object on the right' + 
      "<br><b>&#x2193;</b> (Down Arrow) Object on bottom</p>" + 
    '<p><img  style="width:30%" src="utils/key_map.png"></img></p>'+ 
    "<p>We hope that these keyboard options are intuitive once you start using them! If not, let us know :)</p>",
    // practice trial description 
    "<p style='font-size: 150%'><b>Practice trials</b></p>" + 
    "<p>We'll give you a few practice trials now, to get familiar with the response keys and see what the experiment is like.<br>" + 
    "You can also use these practice trails to decide if you want to participate in the study.</p>" + 
    "<p>Once you get " + params.practice_criterion + " right you'll move on to the consent form, then the experiment itself.</p>", 
  ], 
  choices: ['space'],
  show_clickable_nav: true,
  show_page_number: false,
  post_trial_gap: 500,
  on_finish: function(){
    document.body.style.backgroundColor = "#808080"
  }
};

// practivce trial choice array 
var practice_trial  = {
  type: 'choice-array',
  stimulus: '',   
  choices: [37, 39, 40], 
  prompt: '',  // 'Use either the <b>Left</b>, <b>Right</b>, or <b>Down</b> Arrow to choose the oddity',
  response_ends_trial: true,
  post_trial_gap: 200,
  on_start: function(data) {
    practice_index = Math.round(Math.random() * (params.reference_categories.length-1) )
    test_trial = generate_stimuli(params.reference_categories[practice_index], params.reference_categories[practice_index], 'V3', params.stimulus_path)
    data.stimulus_info = test_trial
    data.stimulus =  test_trial.stimuli
    data.correct_response = test_trial.correct_response
    data.trial_type = 'practice'
  }, 
  on_finish: function(data) {
    data.choice = params.key_[data.key_press]
    if (data.correct_response==params.key_[data.key_press]){ data.correct = true
    } else { data.correct = false
    }
    
    data.trial_type = 'practice'

    practice_trials = jsPsych.data.get().filter({trial_type: 'practice'})
    n_practice_trials = practice_trials.count()
    
    if (n_practice_trials >= params.practice_criterion) { 
      correct_count = practice_trials.filter({correct:true}).count() 
    } else { correct_count = 0} 
    
    if ( ( correct_count >= params.practice_criterion)  ) {
      data.practice_threshold_met = true
    }
 }
}

// practice trial feedback
var practice_inter_trial_screen  = {
  type: 'image-keyboard-response',
  stimulus: '',
  // prompt is conditional -- changes once they meet criterion
  prompt: function() {
    
    // feedback about whether they got the answer right
    emoji = ['incorrect D:', 'correct :D'][jsPsych.data.get().last(1).filter({correct:true}).count()]
    // change display if they've met the practice threshhold
    practice_met = jsPsych.data.get().filter({practice_threshold_met:1}).count() 
    if (practice_met) { 
      exit = '<p>Press the space bar to end the practice trials </p>' 
    } else { 
      exit = '<p>Press the space bar to begin another practice trial</p>' 
    } 
    // set complete feedback string to present to subjects
    display = '<p style="font-size:200%"><b>'+emoji+'</b></p>'+exit
    return display
  }, 
  choices: ['space', 'enter'], 
}

// block with conditional loop based on criterion performance
var  practice_block= {
  timeline: [ practice_trial, practice_inter_trial_screen],
  loop_function: function(data){
    exit_key = 'enter'
    
    practice_trials = jsPsych.data.get().filter({trial_type: 'practice'})
    n_practice_trials = practice_trials.count()
    
    if (n_practice_trials > 4) { 
      correct_count = practice_trials.filter({correct:true}).count() 
    } else { correct_count = 0} 
    
    last_key_press = jsPsych.data.getLastTrialData().values()[0].key_press
    exit_key_convert = jsPsych.pluginAPI.convertKeyCharacterToKeyCode(exit_key)
    practice_met = jsPsych.data.get().filter({practice_threshold_met:1}).count() 
    if ( ( practice_met) ) { // * (exit_key_convert == last_key_press ) ){
      return false; // break loop 
    } else {
      return true; // continue loop 
    }
  }, 

}

var consent_form = { 
  type: 'html-keyboard-response', 
  stimulus: '' + 
    '<p style="font-size:140%"><b>Nice work!</b></p>' + 
    '<p>Before we get started, feel free to take a look at our consent form, and download a copy for your records if you like:<p>' + 
    '<div style="padding:1%" >'  + 
      "<embed src='utils/memory_lab_online_consent.pdf' width='100%' height='400px' style='border: 2px solid lightgrey';/>" + 
    '</div>' + 
    "<p>Press 'y' if you agree to participate in this study</p>" ,  
  choices: ['y'],
  on_start: function(){
    document.body.style.backgroundColor = "#ffffff"
  }, 
}

// experimental screens between trials
var inter_trial_screen  = {
  type: 'image-keyboard-response',
  stimulus: '',
  prompt: function() { 
    if (params.feedback) { 
      emoji = ['D:', ':D'][jsPsych.data.get().last(1).filter({correct:true}).count()]
    } else { emoji = '' 
    } 
    feedback = '<p><b>' + emoji + '</b></p><p>Press the space bar to begin the next trial</p>' 
    return feedback}, 
  choices: ['space'], 
}

var all_stimuli = [] 
stimulus_path = params.stimulus_pathh
// generate stimuli for experiment with the right distributionss
for (i_trial=0; i_trial < params.n_trials; i_trial++) {
  
  // select our reference categories in equal proportion 
  var typicals_ = params.reference_categories [ Math.floor(i_trial/(params.n_trials/params.reference_categories.length)) ] 
  // select distractor category 
  if (i_trial%params.between_frequency==0) {
    // only select from non-typical categories  
    var remaining_ = Object.keys(meta).filter(function(value){ return value != typicals_;});
    oddity_ = remaining_[ get_random_index(remaining_) ] 
    var oddity_type = 'between'
  } else {
    //console.log('WITHIN!')
    // within category discrimination
    var oddity_ = typicals_
    var oddity_type = 'within' 
  }
 
  // select distractor category 
  if (i_trial%params.control_frequency==0) { 
    // within category discrimination
    i_variation = params.control_variation_level
  } else {
    i_variation = params.experimental_variation_level
  }
  
  // generate stimuli from typical and oddity categories
  stim_info = generate_stimuli(typicals_, oddity_, i_variation, params.stimulus_path)
  stim_info.oddity_type = oddity_type 
  var trial_info  = {
    type: 'choice-array',
    stimulus_info: stim_info, 
    stimulus: stim_info.stimuli,
    correct_response: stim_info.correct_response, 
    choices: [37, 39, 40],
    prompt: '', // '<br>Use either <b>4</b>, <b>9</b>, <b>r</b>, or <b>i</b> to choose the oddity',  
    response_ends_trial: true,
    trial_duration: params.max_decision_time,
    post_trial_gap: 200,
    on_finish: function(data) {
      console.log( 'ON FINISH:', data.correct_response, params.key_, data.key_press ) 
      data.correct = data.correct_response == params.key_[data.key_press]
      data.array_type = params.array_type 
      data.trial_type = 'experiment'
      data.oddity_type = [ 'between', 'within' ] [ ( data.typical_category == data.oddity_category ) * 1  ] 
      data.choice_object = [ data.typical_name, data.oddity_name ] [ data.correct * 1 ]  
      data.choice_category = [ data.typical_category, data.oddity_category ] [ data.correct * 1 ] 
      data.trial_number = jsPsych.data.get().filter({trial_type: 'experiment'}).count()  
      data.choice = params.key_[data.key_press]
      data = format_data_for_server(data, params)
      console.log(data)  
      data.experiment_id = experiment_id
      save_trial_to_database(data, 'experiment')
   }
  }
  all_stimuli.push(trial_info)
}

var full_screen_start = {
 type: 'fullscreen',
 message: "" +  
  '<p style="font-size:150%"><b>Now you\'re ready to start the experiment. </b></p>' + 
  "<p>You'll have ten seconds to find the oddity in each trial, and remember:</p>" + 
  "<p>There's a bonus (+$" + params.trial_bonus.toFixed(2) + ") for each correct choice you make " + 
  "and a big penalty (-$" + params.trial_penalty.toFixed(2) + ") for each mistake!<p>" + 
  "<p>In the experiment, you'll recieve feedback at the end of the experiment, not after every trial</p>", 
  button_label: '<p style="color:"><b>Click to enter full screen and begin experiment</b></p>', 
 fullscreen_mode: true, 
 on_finish: function(){ 
  document.body.style.backgroundColor = "#808080"
 }
};

var experiment_debrief = {
  type: 'survey-text',
  preamble: function() {
 
    var trials = jsPsych.data.get().filter({trial_type: 'experiment'});
    var correct_trials = trials.filter({correct: true});
    var incorrect_trials = trials.filter({correct: false});
    var total_bonus = correct_trials.count() * params.trial_bonus - incorrect_trials.count() * params.trial_penalty
    n_trials_total = trials.count()
    accuracy = Math.round(correct_trials.count() / trials.count() * 100);
    rt = (Math.round(trials.select('rt').mean())/1000).toFixed(2)
    time_elapsed = ( jsPsych.data.getLastTrialData().values()[0].time_elapsed / 1000 / 60).toFixed(2)
    total_bonus = Math.max(0, total_bonus) 
    total_bonus = Math.min(total_bonus, params.max_experiment_bonus)
    pretty_bonus = Math.max(total_bonus, 0).toFixed(2)
    var short_debrief = "<p>You responded correctly on " + accuracy + "% of the trials"+
      " with an average response time was " + rt + " seconds.</p>"+
      "<p><b>You earned a total $" + pretty_bonus + ' in ' + time_elapsed  + ' minutes!</b></p>' 
    return short_debrief
  },
  questions: [
    { prompt: "We'd love to hear your thoughts about the experiment! " + 
              "Was it hard, easy, boring--is the bonus fair?" + 
              "<br>Any and all feedback is helpful :)" ,
      rows: 10, 
      columns: 100
    }
  ],
  button_label: "Click to continue on to MTurk's 'Submit' button",
  on_finish:  function() {
    data = {}
    data.params = params
    data.n_trials = n_trials_total
    data.trial_type = 'summary'
    data.accuracy = accuracy
    data.average_rt = rt
    data.experimental_duration = time_elapsed
    data.total_bonus = pretty_bonus
    data.worker_feedback = jsPsych.data.get().last(1).values()[0]['responses']
    data = format_data_for_server(data, params)
    data.experiment_id = experiment_id
    save_trial_to_database(data)
    console.log('final data to save:', data)
  }
};

// POPULATE TIMELINE 
var timeline = [] 
// instructions
timeline.push(instructions);
// practice block
timeline.push(practice_block)
// consent form
timeline.push(consent_form)
// final reminder 
timeline.push(full_screen_start)
// shuffle stimuli
shuffled_stims = shuffle(all_stimuli) 
// add each trial to timeline + iti
for (i=0; i < all_stimuli.length; i++) { 
  timeline.push(shuffled_stims[i], inter_trial_screen)
}
// add debrief block
timeline.push(experiment_debrief)
// initialize jsPsych experiment  
jsPsych.init({
  timeline: timeline,
  on_finish: function() { 
    // define mturk exit protocol
    show_mturk_submit_button() 
  },
});
