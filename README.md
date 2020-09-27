## A deep learning account of medial temporal lobe involvement in perception

Our recent [findings](summary/manuscript/preprint.pdf) are reflected in the structure of this repository: 

- `electrophysiological/`: model fits to electrophysiological recordings from IT and V4 cortex 
- `retrospective/`: generate model performance on all experiments in the  retrospective dataset
- `high-throughput/`: collect human behavior on novel dataset and preprocess the results 
- `in_silico/`: examine effects of changing model architecture and trained data on PRC-relevant behavior 

Results across each of these studies are synthesized in 

- `summary/`: reporting statistical effects, generating figures, and final manuscript 

To generate our main findings and figures, install conda (v4.8.5, tested on an osx-64 platform) and import our python environment with

```
$ conda env create -f conda_environment.yml
```

and open the jupyter notebook [summary/reporting_statistics.ipynb](summary/reporting_statistics.ipynb) using the `mtl_perception` kernel.
