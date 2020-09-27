### Server- and client-side code to perform online data collection

For a detailed tutorial on implementing this browswer-based experiment, see `https://github.com/tzler/server_side_psych`. In short, after setting up a server for online data collection, set up this folder with: 

- Create a `credentials/` folder in this directory which contains mongo, ssl, and mturk info 
- Create a `jsPsych/` folder by cloning the jsPsych repository with `$ git clone https://github.com/jspsych/jsPsych.git`
- Create a `node_modules/` folder that contains server-side dependencies using npm: 

  ```  
  $ npm init --yes # initialize and accept all defaults
  ```
  
  and install the dependencies we'll need
  
  ```
  $ npm install express mongodb https socket.io 
  ```
- Create a `stimuli/` folder with all experimental stimuli (available upon request) 

With those steps complete, to open a port that makes your server available for data collection, you can run 

```
$ node app.js
```

The stimuli used in this experiment are available upon request. 
