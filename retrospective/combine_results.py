import os, pandas, pickle, numpy as np
import statsmodels.formula.api as smf
import warnings ; warnings.filterwarnings('ignore')

def across_studies(it_layer, v4_layer, base_directory): 

    ############### STARK

    # patient data
    stark_location = os.path.join(base_directory, 'stark_2000') 
    stark_human_data = pandas.read_excel(os.path.join( stark_location, 'stimuli/collapse_runs_1.xlsx'))

    conditions = ['faces', 'snow3', 'snow4', 'snow5']
    stark_ = {}
    stark_['prc_lesion'] = [stark_human_data.iloc[5:8][c].mean()/100 for c in conditions]
    stark_['hpc_lesion'] = [stark_human_data.iloc[11:14][c].mean()/100 for c in conditions]
    stark_['prc_intact'] = [stark_human_data.iloc[0:5][c].mean()/100 for c in conditions]
    stark_['hpc_intact'] = [stark_human_data.iloc[0:5][c].mean()/100 for c in conditions]
    stark_['experiment'] = [c for c in conditions]
    stark_['study'] = ['stark' for c in conditions]
    stark_['claim'] = ['against' for c in conditions]
    stark_['type'] = ['diagnostic' for c in conditions]
    stark_ = pandas.DataFrame(stark_)

    # model data
    stark_model_data = os.path.join(stark_location, 'stark_2000_modelperformance.pickle')
    name_map = {'FACES':'faces',
                '3DGREYSC_snow_1':'snow3',
                '3DGREYSC_snow_2': 'snow4',
                '3DGREYSC_snow_3': 'snow5', }

    with open(stark_model_data, 'rb') as handle:
        _data = pickle.load(handle)

    for i in _data[list(name_map)[0]]: stark_[i] = ''

    for exp in _data:
        indices = np.sum(stark_[['experiment']] == name_map[exp], axis=1).values
        for i in _data[exp]:
            stark_[i].iloc[np.nonzero(indices)[0][0]] = _data[exp][i]


    ############### BARENSE 

    # patient data
    # GET THESE NUMBERS
    barense_ = {}
    barense_['prc_lesion'] = [.50,  1., .18, .95]
    barense_['hpc_lesion'] = [.78, .97, .73, .99]
    barense_['prc_intact'] = [.85, .98, .82, .98]
    barense_['hpc_intact'] = [.85, .98, .82, .98]
    barense_['experiment'] = ['familiar_high', 'familiar_low', 'novel_high', 'novel_low']
    barense_['study'] = ['barense' for i in range(len(barense_['experiment']))]
    barense_['claim'] = ['for' for c in barense_['experiment']]
    barense_['type'] = ['diagnostic', 'control', 'diagnostic', 'control']
    barense_ = pandas.DataFrame(barense_)


    # model data
    #barense_model_results = 'barense_2007/barense_diagnostic_VGG16_behavior.pickle'

    _bdata = pandas.read_csv(os.path.join(base_directory, 'barense_2007/barense_model_performance.csv'))

    for i in _bdata.layer.unique(): barense_[i] = ''
        
    for exp in _bdata.experiment.unique() :
        indices = np.sum(barense_[['experiment']] == exp, axis=1).values
        for i in _bdata.layer.unique(): 
            i_value =_bdata[(_bdata.experiment==exp) * (_bdata.layer==i) ].accuracy.values[0]
            barense_[i].iloc[np.nonzero(indices)[0][0]] = i_value


    ################ LEE 2006 

    # patient data
    lee_human_data_location = os.path.join(base_directory, 'lee_2006/stimuli/Lee_2006_JNeurosci.xlsx')
    lee_human_data = pandas.read_excel(lee_human_data_location,  header=1)
    conditions = ['dscenes', 'dfaces']
    lee_ = {}
    lee_['prc_lesion'] = [lee_human_data.iloc[2:10][c].mean() for c in conditions]
    lee_['hpc_lesion'] = [lee_human_data.iloc[16:23][c].mean() for c in conditions]
    lee_['prc_intact'] = [lee_human_data.iloc[29:39][c].mean() for c in conditions]
    lee_['hpc_intact'] = [lee_human_data.iloc[45:54][c].mean() for c in conditions]
    lee_['experiment'] = [c for c in conditions]
    lee_['study'] = ['lee2006' for i in lee_['experiment']]
    lee_['claim'] = ['for' for c in lee_['experiment']]
    lee_['type'] = ['control', 'diagnostic']
    lee2006 = pandas.DataFrame(lee_)


    # model data
    lee_2006_model_results =  os.path.join(base_directory, 'lee_2006/lee_diagnostic_VGG16_behavior.pickle')

    name_map = {'faces':'dfaces', 
                'scenes':'dscenes'}

    with open(lee_2006_model_results, 'rb') as handle: 
        _data = pickle.load(handle)

    for i in _data[list(_data)[0]]: lee2006[i] = ''
    lee2006['pixel'] = ''
    for exp in _data:
        indices = np.sum(lee2006[['experiment']] == name_map[exp], axis=1).values
        for i in _data[exp]: 
            lee2006[i].iloc[np.nonzero(indices)[0][0]] = _data[exp][i]



    ################ LEE 2005 

    # patient data
    ## experiment 1
    lee05_human_data_location = os.path.join(base_directory, 'lee_2005/Lee_2005_Hippocampus_Expt1.xlsx')
    E1 = pandas.read_excel(lee05_human_data_location,  header=0)
    E1 = E1.rename(columns={
        'Unnamed: 0':'group',
        'Object nov %':'novel_objects_E1',
        'Object fam %':'familiar_objects_E1',
        'Face %':'faces_E1'})

    CON_ = 'CON'
    lee_2005 = {}
    lee_2005['prc_lesion']  = E1[['mtl' in i if i == i  else False for i in E1['group'].values]]
    lee_2005['hpc_lesion']  = E1[['hc'  in i if i == i  else False for i in E1['group'].values]]
    lee_2005['prc_control'] = E1[[CON_ in i if i == i else False for i in E1['group'].values]]
    lee_2005['hpc_control'] = E1[[CON_ in i if i == i else False for i in E1['group'].values]]

    lee05_conditionsE1 = [ 'faces_E1', 'novel_objects_E1', 'familiar_objects_E1']
    lee_ = {}
    lee_['prc_lesion'] = [lee_2005['prc_lesion'][c].mean() for c in lee05_conditionsE1]
    lee_['hpc_lesion'] = [lee_2005['hpc_lesion'][c].mean() for c in lee05_conditionsE1]
    lee_['prc_intact'] = [lee_2005['prc_control'][c].mean() for c in lee05_conditionsE1]
    lee_['hpc_intact'] = [lee_2005['hpc_control'][c].mean() for c in lee05_conditionsE1]
    lee_['experiment'] = [c for c in lee05_conditionsE1]
    lee_['study'] = ['lee2005' for i in lee_['experiment']]
    lee_['claim'] = ['for' for c in lee_['experiment']]
    lee_['type'] = ['diagnostic', 'diagnostic', 'control']

    ## experiment 2
    lee_human_data_location = os.path.join(base_directory, 'lee_2005/Lee_2005_Hippocampus_Expt2.xlsx')
    experiment_two= pandas.read_excel(lee_human_data_location,  header=1)

    experiment_two = experiment_two.rename(columns={'Unnamed: 0': 'group',
                                                    'SCENES':'scenes_same', 'Unnamed: 4':'scenes',
                                                    'FACES':'faces_same', 'Unnamed: 6':'faces'})

    lee_subset2  = {}
    lee_subset2['prc_lesion']  = experiment_two[['MTL' in i if i == i else False for i in experiment_two['group'].values]]
    lee_subset2['hpc_lesion']  = experiment_two[['HC'  in i if i == i else False for i in experiment_two['group'].values]]
    lee_subset2['prc_control'] = experiment_two[[CON_ in i if i == i else False for i in experiment_two['group'].values]]
    lee_subset2['hpc_control'] = experiment_two[[CON_ in i if i == i else False for i in experiment_two['group'].values]]

    lee2_ = {}
    lee_['prc_lesion'].append( lee_subset2['prc_lesion']['faces'].mean() )
    lee_['hpc_lesion'].append( lee_subset2['hpc_lesion']['faces'].mean() )
    lee_['prc_intact'].append( lee_subset2['prc_control']['faces'].mean() )
    lee_['hpc_intact'].append( lee_subset2['hpc_control']['faces'].mean() )
    lee_['experiment'].append( 'faces_E2' )
    lee_['study'].append( 'lee2005' )
    lee_['claim'].append( 'for')
    lee_['type'].append( 'diagnostic' )
    lee2005 = pandas.DataFrame( lee_ )


    # model data
    studies = {'lee_2005E1': os.path.join(base_directory, 'lee_2005/lee_2005_diagnostic_VGG16_behavior_E1_1-iterations.pickle'), 
               'lee_2005E2': os.path.join(base_directory, 'lee_2005/lee_2005_diagnostic_VGG16_behavior_E2_1-iterations.pickle')
              }

    name_map = { 
        # not using the scenes because of lesion type
        'Faces':'faces_E1', 
        'Novel objects':'novel_objects_E1', 
        'Familiar objects': 'familiar_objects_E1',
        'faces': 'faces_E2'}

    for i_set in studies: 
        with open(studies[i_set], 'rb') as handle: 
            _data = pickle.load(handle)
            
            if i_set == list(studies)[0]:
                lee2005['pixel'] = ''
                for i in _data[list(_data)[0]]: lee2005[i] = ''
                    

            for exp in [i for i in _data if i != 'scenes']:
                indices = np.sum(lee2005[['experiment']] == name_map[exp], axis=1).values

                for i in _data[exp]: 
                    lee2005[i].iloc[np.nonzero(indices)[0][0]] = _data[exp][i]


    meta_df = pandas.concat([stark_, barense_,lee2005, lee2006], ignore_index=True)


    meta_df['prc_delta'] = meta_df['prc_intact']-meta_df['prc_lesion']
    meta_df['hpc_delta'] = meta_df['hpc_intact']-meta_df['hpc_lesion']
    meta_df['prclesion_model_delta'] = meta_df['prc_lesion']-meta_df[it_layer]
    meta_df['hpclesion_model_delta'] = meta_df['hpc_lesion']-meta_df[it_layer]
    meta_df['prcintact_model_delta'] = meta_df['prc_intact']-meta_df[it_layer]
    meta_df['hpcintact_model_delta'] = meta_df['hpc_intact']-meta_df[it_layer]


    for i_layer in meta_df.columns[9:30]: 
        meta_df[i_layer] = np.array(meta_df[i_layer]).astype('float64')


    layers = [ 'conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 
               'conv3_1', 'conv3_2', 'conv3_3','pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4',
               'conv5_1', 'conv5_2', 'conv5_3', 'pool5', 'fc6', 'fc7', 'fc8']


    interactions = {'prc_con':[], 'hpc_con':[]}

    prc_lesion = meta_df['prc_lesion'].values.astype('float64')
    hpc_lesion = meta_df['hpc_lesion'].values.astype('float64')
    prc_intact = meta_df['prc_intact'].values.astype('float64')
    hpc_intact = meta_df['hpc_intact'].values.astype('float64') 

    n_ = len(prc_lesion)
    prc_lesion_rmse, hpc_lesion_rmse, prc_intact_rmse , hpc_intact_rmse = [] , [] , [] , [] 
    r_squared = {'prc_lesion':[], 'prc_intact':[], 'hpc_lesion':[], 'hpc_intact':[]}

    for i_layer in layers: 
        
        model_responses = np.array(meta_df[i_layer].values).astype('float64')
        
        # stats 
        prc_data = pandas.DataFrame(
            {'model': np.concatenate((model_responses, model_responses)), 
             'human': np.concatenate((prc_lesion, prc_intact)),
             'group': np.concatenate([np.repeat(0,n_), np.repeat(1, n_)])})

        prc_lesion_rmse_ = np.sqrt( np.mean( (np.array(prc_lesion) - np.array(model_responses) )**2 ) )
        hpc_lesion_rmse_ = np.sqrt( np.mean( (np.array(hpc_lesion) - np.array(model_responses) )**2 ) )
        prc_intact_rmse_ = np.sqrt( np.mean( (np.array(prc_intact) - np.array(model_responses) )**2 ) )
        hpc_intact_rmse_ = np.sqrt( np.mean( (np.array(hpc_intact) - np.array(model_responses) )**2 ) )

        prc_interaction = smf.ols("human ~ model * group", prc_data).fit()
        interactions['prc_con'].append( prc_interaction.pvalues[-1] )
        # stats 
        hpc_data = pandas.DataFrame(
            {'model': np.concatenate((model_responses, model_responses)), 
             'human': np.concatenate((hpc_lesion, hpc_intact)),
             'group': np.concatenate([np.repeat(0,n_), np.repeat(1, n_)])})


        hpc_interaction = smf.ols("human ~ model * group", hpc_data).fit().pvalues[-1]
        interactions['hpc_con'].append( hpc_interaction )    
        
        prc_lesion_rmse.append(prc_lesion_rmse_)
        hpc_lesion_rmse.append(hpc_lesion_rmse_)
        prc_intact_rmse.append(prc_intact_rmse_)
        hpc_intact_rmse.append(hpc_intact_rmse_)
        
    prc_lesion_rmse = np.array(prc_lesion_rmse) 
    prc_intact_rmse = np.array(prc_intact_rmse)
    hpc_lesion_rmse = np.array(hpc_lesion_rmse) 
    hpc_intact_rmse = np.array(hpc_intact_rmse) 

    layer_data = pandas.DataFrame({'prc_interaction': interactions['prc_con'], 
                                     'hpc_interaction': interactions['hpc_con'], 
                                     'prc_lesion_rmse': prc_lesion_rmse, 
                                     'hpc_lesion_rmse': hpc_lesion_rmse, 
                                     'prc_intact_rmse': prc_intact_rmse, 
                                     'hpc_intact_rmse': hpc_intact_rmse, 
                                     'prc_delta_rmse': prc_intact_rmse-prc_lesion_rmse, 
                                     'hpc_delta_rmse': hpc_intact_rmse-hpc_lesion_rmse, 
                                     'layer': layers})




    return meta_df, layer_data

if __name__ == '__main__': 

    base_directory = '/Users/biota/work/mtl_perception/retrospective/' 
    print( across_studies( 'conv5_1', 'pool3', base_directory ) )  