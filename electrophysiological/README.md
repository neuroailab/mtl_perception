# Model fits to electrophysiological responses throughout the VVS 

Code to evaluate the correspondence between model and electrophysiological responses through the VVS, then identify the model `IT-like` layer for subsequent analyses. 
    
- `layer_neural_fit.py`: protocol for fitting model responses to neural responses. 
- `automate_layer_fits.py`: script to parallelize `layer_neural_fit.py` on cluster. Results in `pls_fitting_results/`
- `analyses.py` includes functions to estiamte split-half neural reliability and extract pls fitting results into main analysis. 

Neural data from: 

> Majaj, N. J., Hong, H., Solomon, E. A., & DiCarlo, J. J. (2015). Simple learned weighted sums of inferior temporal neuronal firing rates accurately predict human core object recognition performance. Journal of Neuroscience, 35(39), 13402-13418.

See [Brain Score](http://www.brain-score.org/#about) for a more information about fitting this and related models to neural data.     
