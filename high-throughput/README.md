# High-throughput human psychophysics experiments

This directory contains all code used to collect human psychophysics experiments online, download and then preprocess these trial-level data. A randomized instance of this experiment can be viewed [here](https://stanfordmemorylab.com:8881/high-throughput_data_collection/index.html).  

- `human_performance.csv`: trial-by-trial human behavior on novel stimulus sets
- `summary_dataframe.csv`: summary of human, model, and IT-supported performance on novel stimulus set
- `online_experiment/`: server- and client-side code use to collect data online 
- `download_data/`: scripts used to download data from server and preprocess them into `human_performance.csv`
