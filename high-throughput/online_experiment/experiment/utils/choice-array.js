/**
 * choice-array
 *  tyler
 *
 * plugin for displaying a 2x2 choice array and getting a keyboard response
 *
 * documentation: docs.jspsych.org
 *
 **/


jsPsych.plugins["choice-array"] = (function() {

  var plugin = {};

  jsPsych.pluginAPI.registerPreload('choice-array', 'stimulus', 'image');

  plugin.info = {
    name: 'choice-array',
    description: '',
    parameters: {
      stimulus: {
        type: jsPsych.plugins.parameterType.IMAGE,
        pretty_name: 'Stimulus',
        default: undefined,
        description: 'The image to be displayed'
      },
      choices: {
        type: jsPsych.plugins.parameterType.KEYCODE,
        array: true,
        pretty_name: 'Choices',
        default: jsPsych.ALL_KEYS,
        description: 'The keys the subject is allowed to press to respond to the stimulus.'
      },
      prompt: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Prompt',
        default: null,
        description: 'Any content here will be displayed below the stimulus.'
      },
      stimulus_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Stimulus duration',
        default: null,
        description: 'How long to hide the stimulus.'
      },
      trial_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Trial duration',
        default: null,
        description: 'How long to show trial before it ends.'
      },
      response_ends_trial: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Response ends trial',
        default: true,
        description: 'If true, trial will end when subject makes a response.'
      },
      correct_resonse: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'The correct response XD',
        default: null,
        description: 'Data to pass along.'
      },
      stim_info: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: '',
        default: null,
        description: 'Data to save.'
      },

    }
  }

  plugin.trial = function(display_element, trial) {
    
    var new_html = trial.prompt
    new_html += '<br><div>' + 
      '<div style="text-align:center; display: flex" >' + 
        '<img style="flex: 1"  src="'+trial.stimulus[0]+'" id="choice-array-stimulus"></img>'+
        '<img style="flex: 2" src="'+trial.stimulus[1]+'" id="choice-array-stimulus"></img>'+
      '</div>' + 
      '<div style="text-align:center; display: flex" >' +
        '<img style="flex: 3" src="'+trial.stimulus[2]+'" id="choice-array-stimulus"></img>'+
        '<img style="flex: 4" src="'+trial.stimulus[3]+'" id="choice-array-stimulus"></img>'+
      '</div>'+ 
      '</div>'

    // add prompt
   // if (trial.prompt !== null){
   //   new_html += trial.prompt;
   // }

    // draw
    display_element.innerHTML = new_html;

    // store response
    var response = {
      rt: null,
      key: null,
      correct_response: trial.correct_response,
    };

    // function to end trial when it is time
    var end_trial = function() {

      // kill any remaining setTimeout handlers
      jsPsych.pluginAPI.clearAllTimeouts();

      // kill keyboard listeners
      if (typeof keyboardListener !== 'undefined') {
        jsPsych.pluginAPI.cancelKeyboardResponse(keyboardListener);
      }

      // gather the data to store for the trial
      stim_info = trial.stimulus_info
      stim_inds = stim_info.typical_indices
      stimulus_response_map = {} 
      for (i=0;i<trial.stimulus.length;i++){
	      stimulus_response_map[i+1] = Number(trial.stimulus[i].slice('stimuli/'.length, -5))
      }
      
      var trial_data = {
        "rt": Math.round(response.rt),
        "timed_out": response.rt == undefined, 
        "stimulus": trial.stimulus,
        "key_press": response.key,
        'correct_response': trial.correct_response, 
        'typical_category': stim_info.typical_category, 
        'oddity_category': stim_info.oddity_category,
        'typical_indices': stim_info.typical_indices, 
        'oddity_index': stim_info.oddity_index,
        'stimulus_response_map': stimulus_response_map,
        'variation_level': stim_info.variation_level, 
        'typical_rotation': stim_info.typical_rotation, 
        'oddity_rotation': stim_info.oddity_rotation,
        'typical_name': stim_info.typical_identity, 
        'oddity_name': stim_info.oddity_identity, 
      };
      
    // clear the display
    display_element.innerHTML = '';

      // move on to the next trial
      jsPsych.finishTrial(trial_data);
    };

    // function to handle responses by the subject
    var after_response = function(info) {

      // after a valid response, the stimulus will have the CSS class 'responded'
      // which can be used to provide visual feedback that a response was recorded
      display_element.querySelector('#choice-array-stimulus').className += ' responded';

      // only record the first response
      if (response.key == null) {
        response = info;
      }

      if (trial.response_ends_trial) {
        end_trial();
      }
    };

    // start the response listener
    if (trial.choices != jsPsych.NO_KEYS) {
      var keyboardListener = jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: after_response,
        valid_responses: trial.choices,
        rt_method: 'performance',
        persist: false,
        allow_held_key: false
      });
    }

    // hide stimulus if stimulus_duration is set
    if (trial.stimulus_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function() {
        display_element.querySelector('#choice-array-stimulus').style.visibility = 'hidden';
      }, trial.stimulus_duration);
    }

    // end trial if trial_duration is set
    if (trial.trial_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function() {
        trial.timed_out = true
        end_trial();
      }, trial.trial_duration);
    }

  };

  return plugin;
})();
