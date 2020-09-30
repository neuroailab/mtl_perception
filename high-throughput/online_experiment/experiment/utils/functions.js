/// mongo details

var supported_browsers = ['Chrome']

////////////////////////////////////////////// CUSTOM EXPERIMENT FUNCTIONS /////////////////////////////////////

function get_preset_trial_stimuli(i_cat, i_var, i_object, n_to_return){
  
  // this function is grotesque :::laughs:::   
  
  var keep_inds = []
  var rotation_info = {'xy':[], 'xz':[], 'yz':[]}
  var variation_info = []   
  for (i_item=0; i_item < meta[i_cat]['names'].length; i_item ++) {
    
    var object_match = meta[i_cat]['names'][i_item] == i_object
    var variation_match = meta[i_cat]['variation_level'][i_item] == i_var

    if ( object_match & variation_match ) {
      keep_inds.push(meta[i_cat]['indices'][i_item])
      rotation_info['xy'].push(meta[i_cat].rotation_xy[i_item])
      rotation_info['xz'].push(meta[i_cat].rotation_xz[i_item])
      rotation_info['yz'].push(meta[i_cat].rotation_yz[i_item]) 
      variation_info.push(meta[i_cat].variation_level[i_item])
    }
  } 
  
  // get the random indices we'll use for both data types 
  var random_indices = [...Array(keep_inds.length).keys()]
  random_indices = shuffle(random_indices).slice(0,n_to_return)
  // extract the preset trials here

  var return_index = [] 
  var rotation_return = {'xy':[], 'xz':[], 'yz':[]}
  var variation_return = []   
  for (i=0; i < random_indices.length; i++) {
    
    return_index.push( keep_inds[random_indices[i]] )
    rotation_return['xy'].push( rotation_info['xy'][random_indices[i]] )
    rotation_return['xz'].push( rotation_info['xz'][random_indices[i]] )
    rotation_return['yz'].push( rotation_info['xz'][random_indices[i]] )
    variation_return.push( variation_info[random_indices[i]] )
  }
  // separate oddity and typical info 
  // some version of arr.splice(oddity_index, 1) 
  return {index: return_index, rotation: rotation_return, variation: variation_return}
  // change to return both oddity and typical info at once 
}


function get_image_info(i_cat, i_var, i_object, j_object, n_to_return){
  
  // this function is grotesque :::laughs:::   
   /////// LEGACY CODE WE CAN USE TO GENERATE CONTROL TRIALS FOR THE MTURK VALIDATION
  
  if ( i_var == 'V0' ){ 
       
    ///////////// GENERATE CONTROL TRIALS ////////////////////
    
    var n_to_return = params.n_objects_per_trial  

    var control_indices = []
    var _control_indices = [] 
    for (i_item=0; i_item < meta[i_cat]['names'].length; i_item ++) {
      
      var object_match = meta[i_cat]['names'][i_item] == i_object
      var variation_match = meta[i_cat]['variation_level'][i_item] == i_var
      var _object_match = meta[i_cat]['names'][i_item] == j_object

      if ( object_match & variation_match ) {
        control_indices.push(meta[i_cat]['indices'][i_item]) }
      if ( _object_match & variation_match ) { 
        _control_indices.push(meta[i_cat]['indices'][i_item])
      }
    }

    control_indices = shuffle(control_indices).slice(0,n_to_return)
    oddity_info = {index: shuffle(_control_indices)[0], variation:'V0'}
    typical_info = {index: control_indices.slice(0, n_to_return-1), variation:'V0'}

  } else { 
     
    ////////// EXTRACT STIMULI FROM PRESET INDICIES ///////////// 
    console.log( params.group_type )
    var _ind_array = [0, 1, 2, 3, 4] 
    if (params.group_type=='even'){
      _ind_select = _ind_array.filter(n => n%2==0)  
    } else{ 
      _ind_select = _ind_array.filter(n => n%2)    
    } 
    
    _ind_choice = shuffle(_ind_select)[0]
    console.log(i_cat, i_object, j_object, _ind_choice)
  
    var i_trial = lesion_stimuli[i_cat][i_object][j_object][_ind_choice]
    
    /// generic method that isn't segmenting things into even and odds 
    /// var i_trial = shuffle(lesion_stimuli[i_cat][i_object][j_object])[0]
    
    oddity_info = {index: i_trial['oddity'], variation:'V3'}
    typical_info = {index: i_trial['typicals'], variation:'V3'}
    
    new_indices = i_trial['stimuli']
  }

  return {typical:typical_info, oddity: oddity_info}

}

function generate_stimuli(typical_category, oddity_category, i_variation, stimulus_path){
  
  // get a random object from list of all objects
  //console.log('typical_category:', typical_category)
  var typical_object_index = get_random_index(meta[typical_category]['template_names'])
  var oddity_object_index = get_random_index(meta[oddity_category]['template_names'])
  
  // set category type
  if (typical_category==oddity_category) {
    // if it's a within category, make sure they aren't the same object :) 
    while (typical_object_index == oddity_object_index ) {
      oddity_object_index = get_random_index( meta[oddity_category]['template_names'])
    }
    var category_type = 'within_category'
  } else {
    var category_type = 'between_category'
  }
  
  // get the name of each object
  var typical_identity = meta[typical_category]['template_names'][typical_object_index]
  var oddity_identity =  meta[oddity_category]['template_names'][oddity_object_index]

  // extract indices and meta in
  //var typical_info = get_image_info(typical_category, i_variation, typical_identity, 3)
  //var oddity_info = get_image_info(oddity_category, i_variation, oddity_identity, 1)
  /////////// THIS IS WHAT WE'RE WORKING ON CHANGING, THE TWO LINES ABOVE ////////////////
  var stimulus_info = get_image_info(typical_category, i_variation, typical_identity, oddity_identity) 

  // finalize typical stimulus information (includes stimulus location) 
  var stimuli = []
  for (i_stim=0; i_stim<stimulus_info.typical.index.length; i_stim++){
    stimuli.push(stimulus_path + stimulus_info.typical.index[i_stim] + '.jpeg')
  }
  
  // finalize oddity stimulus infor (includes stimulus location)
  var oddity_stimulus_file = stimulus_path + stimulus_info.oddity.index + '.jpeg'
   
  // insert oddity at random location in stimuli to finalize choice array 
  var random_location_within_array = Math.round(Math.random()*stimuli.length)
  stimuli.splice(random_location_within_array, 0, oddity_stimulus_file);
  
  // data to return  
  trial_structure_info = {
                          stimuli:stimuli, 
                          comparison: category_type, 
                          correct_response: random_location_within_array+1,
                          typical_category:typical_category, 
                          oddity_category: oddity_category,
                          typical_indices: stimulus_info.typical.index, 
                          oddity_index: stimulus_info.oddity.index,
                          category_type: category_type, 
                          typical_identity:  typical_identity, 
                          oddity_identity: oddity_identity, 
                          variation_level: i_variation, 
                          //typical_rotation: stimulus_info.typical.rotation, 
                          //oddity_rotation: stimulus_info.oddity.rotation, 
                         }
  //console.log( 'TRIAL STRUCTURE', trial_structure_info )  
  return trial_structure_info
}

////////////////////////////////////////////// GENERIC HELPER FUNCTIONS /////////////////////////////////////

// load node process 'io' as socket
socket = io.connect();

// define javascript facing node functions
save_trial_to_database = function(trial_data){
  socket.emit('insert', trial_data)
}

function format_data_for_server(trial_data, params) { 
  trial_data.worker_id= get_turk_param('workerId')
  trial_data.assignment_id= get_turk_param('assignmentId')
  trial_data.hit_id= get_turk_param('hitId')
  trial_data.browser = get_browser_type()
  trial_data.collection = params.collection
  trial_data.database = params.database
  trial_data.iteration = params.iteration
  return trial_data
}
function shuffle(a) {
    var j, x, i;
    for (i = a.length - 1; i > 0; i--) {
        j = Math.floor(Math.random() * (i + 1));
        x = a[i];
        a[i] = a[j];
        a[j] = x;
    }
    return a;
}

function get_random_index(list) {
  return Math.floor(Math.random()*list.length)
}


////////////////////////////////////////////// GENERIC MTURK FUNCTIONS /////////////////////////////////////

function show_mturk_submit_button(){

  submit_button = document.createElement('div');
  submit_button.innerHTML = "" + 
  "<div id='hidden_button' style='position: absolute; top:50%; left: 50%; '>" + 
    "<form name='hitForm' id='hitForm' method='post' action=''>" + 
      "<input type='hidden' name='assignmentId' id='assignmentId' value=''>" + 
      "<input type='submit' name='submitButton' id='submitButton' value='Submit' class='submit_button'>" + 
    "</form>" + 
  "</div>"

  document.body.appendChild(submit_button);
  document.getElementById('hitForm').setAttribute('action', get_submission_url())
  document.getElementById('assignmentId').setAttribute('value', get_turk_param('assignmentId')) 

}

function get_submission_url(){
  if (window.location.href.indexOf('sandbox')>0) {
      console.log('SANDBOX!')
      submission_url = 'https://workersandbox.mturk.com/mturk/externalSubmit'
  } else {
      console.log('REAL LYFE!')
      submission_url = "https://www.mturk.com/mturk/externalSubmit"
    }
  return submission_url
}
   
  function get_turk_param( param ) {
    // worker id : 'workerId'
    // assignmen ID : 'assignmentId'
    // hit ID : 'hitId'
    var search_term = "[\?&]"+param+"=([^&#]*)";
    var reg_exp = new RegExp( search_term );
    var search_url = window.location.href;
    results = reg_exp.exec( search_url );
    if( results == null ) {
        return 'NONE'
    } else {
      return results[1];
    }
  }
  

////////////////////////////////////////////// MANAGE ZOOM SETTINGS /////////////////////////////////////

function get_browser_type(){
  var N= navigator.appName, ua= navigator.userAgent, tem;
  var M= ua.match(/(opera|chrome|safari|firefox|msie)\/?\s*(\.?\d+(\.\d+)*)/i);
  if(M && (tem= ua.match(/version\/([\.\d]+)/i))!= null) M[2]= tem[1];
  M= M? [M[1], M[2]]: [N, navigator.appVersion,'-?'];
  ////// includes version: ////////  return M.join(' '),
  return  M[0]
 };

