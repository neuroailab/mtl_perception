from sklearn.linear_model import LinearRegression
import colour, pickle, pandas, os, sys
from scipy.stats.mstats import zscore
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from IPython.display import Markdown as md
from scipy import stats
import warnings ; warnings.filterwarnings('ignore')

def show_model_parameters(m_, idx_ =-1):
    
    # fit model 
    m_ = m_.fit() 
    # extract significant figures from float 
    def sigfigs(x):
        d = int(str('%.2e'%x)[('%.2e'%x).find('-')+1:])
        n = np.round(float(str('%.02e'%x)[0:3]))
        return n, d
    
    # extract model parameters 
    beta, pval, df_model = m_.params[idx_], m_.pvalues[idx_], m_.df_model
    rsqrd, df_resid, tvalues = m_.rsquared, m_.df_resid, m_.tvalues[idx_]
    
    # show exact p values up to three significant figures 
    if sigfigs(pval)[1] < 4:
        stat_str = "$\\beta = %.2f$, $F(%d, %d)$ = $%.02f, P = %.03f $"
        report = stat_str%(beta, df_model, df_resid, tvalues, pval, )
    else:
        stat_str = "$\\beta = %.2f$, $F(%d, %d)$ = $%.02f, P = %.0f $ x $ 10 ^{-%d} $"
        report = stat_str%(beta, df_model, df_resid, tvalues, sigfigs(pval)[0], sigfigs(pval)[1])
    
    # return markdown visualization 
    return md(report) #, report

def retrospective_interaction(i_layer, meta_df, misclassified, _location):

    # helper function
    def plot_nice_line(model_, params={}, title=''):
        m, b = model_.coef_[0], model_.intercept_[0]
        xs = np.array([0, min(1 , ( ( 1 - b) / m ))])
        return plt.plot(xs, xs * m + b, **params,
                        solid_capstyle = ['butt','round','projecting'][1])

    # helper function
    def plot_diagonal(label=''):
        return plt.plot([0, 1], [0, 1], color='grey',
                        linestyle='--', zorder=-1, label=label)

    # data
    mod_ = meta_df[i_layer].values
    prc_ = meta_df['prc_lesion'].values
    hpc_ = meta_df['hpc_lesion'].values
    con_prc = meta_df['prc_intact'].values
    con_hpc = meta_df['hpc_intact'].values
    i = misclassified['hpc_lesion']#nondiagnostic['flat']['intact']
    l = misclassified['prc_lesion']#nondiagnostic['flat']['lesion']
    c = misclassified['control']#nondiagnostic['flat']['control']

    fig = plt.figure(constrained_layout=True, figsize=[7.5, 4])
    gs = fig.add_gridspec(2, 4)

    y_lim = (-.05, 1.05)
    prc_c = '#9a2487'
    con_c = '#a3a3a3'
    hpc_c = '#037397'
    x_model_name = 'Model Performance'
    xy_lsize = 12
    p_width = 1
    p_size = 30
    l_width = 5
    t_size = 8
    xlabelpad = 10
    ylabelpad = 4
    _alpha = .9
    nd_lwidth=.7

    ################
    ax = fig.add_subplot(gs[0:2, 0:2]);

    con_c = '#dadada'#'#d1d1d1'
    point_params = {'s':p_size,'linewidth':p_width, 'zorder':-2, 'facecolor':con_c, 'edgecolor':'white',}
    prccon_model_ = LinearRegression().fit(np.reshape(mod_, (-1,1)), np.reshape(con_prc, (-1, 1)))
    points = plt.scatter(x = mod_, y = con_prc, **point_params)
    prediction_line = plot_diagonal()
    line_params = {'linewidth':l_width,'color':con_c,'label':'PRC-intact', 'alpha':_alpha, 'zorder':-3}
    con_line = plot_nice_line(prccon_model_, line_params)
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', zorder=-2)
    # prc-lesion data
    point_params = {'s':p_size,'linewidth':p_width, 'zorder':2, 'facecolor':prc_c, 'edgecolor':'white'}
    prc_model_ = LinearRegression().fit(np.reshape(mod_, (-1,1)), np.reshape(prc_, (-1, 1)))
    _params = {'linewidth':l_width, 'zorder':-1, 'color':prc_c, 'alpha':_alpha, 'label':'PRC-lesion'}
    prc_line = plot_nice_line(prc_model_, _params)
    plt.scatter(x = mod_, y = prc_, **point_params)
    # aesthetics
    plt.xlabel(x_model_name, fontsize=xy_lsize, labelpad=xlabelpad)
    plt.ylabel('Human Performance', fontsize=xy_lsize, labelpad=ylabelpad)
    #plt.title('PRC-Lesioned Behavior\n'  , fontsize=t_size, y=1.05)
    plt.ylim(y_lim)

    plt.xticks(size=t_size); plt.yticks(size=t_size)
    ax.text(-0.05, 1.06, ['a','b','c', 'd'][0], transform=ax.transAxes,fontsize=12, va='top', ha='right')

    plt.scatter(misclassified['model_performance'], i, facecolor='white', edgecolor=con_c, s=p_size-10,linewidth=nd_lwidth)
    plt.scatter(misclassified['model_performance'], l, facecolor='white', edgecolor=prc_c,s=p_size-10,linewidth=nd_lwidth)

    plt.legend(fontsize=10, title_fontsize=8, framealpha=0, loc=4)

    ################
    ax = fig.add_subplot(gs[0:2, 2:4]);

    # mtl-intact data
    con_c = '#d1d1d1'
    point_params = {'s':p_size,'linewidth':p_width, 'zorder':-5, 'facecolor':con_c, 'edgecolor':'white'}
    A = plt.scatter(x = mod_, y = con_hpc, **point_params )
    con_model_ = LinearRegression().fit(np.reshape(mod_, (-1,1)), np.reshape(con_hpc, (-1, 1)))
    line_params = {'linewidth':l_width,'color':con_c,'label':'HPC-intact', 'alpha':_alpha, 'zorder':0}
    plot_nice_line(con_model_, line_params)
    # hpc-lesion data
    point_params = {'s':p_size,'linewidth':p_width, 'zorder':2, 'facecolor':hpc_c, 'edgecolor':'white'}
    plt.scatter(x = mod_, y = hpc_, **point_params);
    hpc_model_ = LinearRegression().fit(np.reshape(mod_, (-1,1)), np.reshape(hpc_, (-1, 1)))
    line_params = {'linewidth':l_width, 'zorder':1, 'color':hpc_c, 'label':'HPC-lesion', 'alpha':_alpha}
    plot_nice_line(hpc_model_, line_params)
    # aesthetics
    plot_diagonal()
    plt.ylim(y_lim)

    plt.xlabel(x_model_name, fontsize=xy_lsize, labelpad=xlabelpad)
    L1 = plt.legend(title='', fontsize=10, title_fontsize=10, framealpha=0, loc=4)
    plt.xticks(size=t_size); plt.yticks(size=t_size)
    ax.text(-0.05, 1.06, ['a','b','c', 'd'][1], transform=ax.transAxes,fontsize=12, va='top', ha='right')

    plt.scatter(misclassified['model_performance'], i, facecolor='white', edgecolor=con_c, s=p_size-10, linewidth=nd_lwidth)
    plt.scatter(misclassified['model_performance'], c, facecolor='white', edgecolor=hpc_c,s=p_size-10, linewidth=nd_lwidth)

    filename = 'figure_two.pdf'
    save_name = os.path.join(_location, filename)
    plt.savefig(save_name, format='pdf', bbox_inches = "tight")

def annotate_fit_to_layer(region, pls_fits):
    _xy = pls_fits[region]['mu']
    plt.annotate("'%s-like'"%region.upper(), xy=(np.argmax(_xy)-.8, max(_xy)+.08), size=7, rotation=0, color='grey')
    plt.scatter(x=np.argmax(_xy), y=max(_xy), marker='|', color='black', s=200, linewidth=.5)

def model_electrophysiological_fit(layer_fit, ax):

    ################# AESTHETICS
    v4_color='black'
    it_color='black'
    n_layers = len(layer_fit['layers'])
    v4 = layer_fit['v4']
    it = layer_fit['it']
    delta = layer_fit['delta']
    alpha=.1

    ################## V4
    plt.plot(v4['mu'], color=v4_color, linestyle='--', linewidth=1, alpha=1, label='V4')
    v4_min = v4['mu']-(v4['std']/np.sqrt(n_layers))
    v4_max = v4['mu']+(v4['std']/np.sqrt(n_layers))
    plt.fill_between(x=range(n_layers),y1=v4_min, y2=v4_max, color=v4_color, alpha=alpha, edgecolor='')
    annotate_fit_to_layer('v4', layer_fit)

    ################## IT
    plt.plot(it['mu'], color=it_color, linestyle='-', linewidth=1, alpha=1, label='IT')
    it_min = it['mu']-(it['std']/np.sqrt(n_layers))
    it_max = it['mu']+(it['std']/np.sqrt(n_layers))
    plt.fill_between(x=range(n_layers),y1=it_min, y2=it_max, color=it_color, alpha=alpha, edgecolor='')
    annotate_fit_to_layer('it', layer_fit)

    ################ DELTA
    import matplotlib.patheffects as pe
    params = {'solid_capstyle':'round', 'linewidth':5, 'zorder':1,}
    plt.plot(range(n_layers), delta, color='white', label='$\Delta_{IT - V4}$',
             path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], )
    plt.ylim(min(delta)-.1, max(v4['mu'])+.15)

    ################ LABELS
    plt.yticks(size=6)
    plt.ylabel('Cross-validated fit to Neural Data', labelpad=0, fontsize=11)
    plt.xlabel("Model Layer", labelpad=5, fontsize=10)
    plt.xticks(range(n_layers), layer_fit['layers'], rotation=90, fontsize=7);
    plt.legend(framealpha=0, fontsize=9, title_fontsize=9,loc=4)

def show_layer_fit(retro, color_df, i_layer, group, show_x=0, show_y=0, show_tag=0):

    xleg = {'pool1':'(Pre \'V4-like\' Layer)',
            'pool3':'(\'V4-like\' Layer)',
            'conv5_1': '(\'IT-like\' Layer)' ,
            'fc7':'(Latter \'IT-like\' Layer)'}

    y_tag = 'Human Performance'#%group.upper()
    l_tag = '%s-lesioned'%group.upper()
    x_layer = retro[i_layer].values
    y_intact = retro['%s_intact'%group].values
    y_lesion = retro['%s_lesion'%group].values

    i_color = color_df['%s_colors'%group].values[np.nonzero( color_df.layer.values == i_layer )[0][0]]

    l_line = LinearRegression().fit(np.reshape(x_layer, (-1,1)), np.reshape(y_lesion, (-1, 1)))
    i_line = LinearRegression().fit(np.reshape(x_layer, (-1,1)), np.reshape(y_intact, (-1, 1)))
    params = {'facecolor':i_color, 'edgecolor':'white', 's':5, 'linewidth':.2, 'zorder':10}

    plt.scatter(x = x_layer, y = y_lesion, **params)
    plt.plot((0, 1), (0, 1), color='grey', linestyle='--', linewidth=.6, alpha=.5)

    line_params = {'solid_capstyle':'round', 'linewidth':3}
    xs = np.array([0, min(1 , ( ( 1 - l_line.intercept_[0] ) / l_line.coef_[0] ))] )
    plt.plot(xs, xs * l_line.coef_[0] + l_line.intercept_[0], color=i_color, **line_params, label='patient')
    intact_params = {'color': 'lightgrey', 'alpha':1, 'zorder':-5}

    xs = np.array([0, min(1 , ( ( 1 - i_line.intercept_[0] ) / i_line.coef_[0] ))] )
    plt.plot(xs, xs * i_line.coef_[0] + i_line.intercept_[0], **line_params, **intact_params)
    plt.plot([], [] , **line_params, **intact_params,label='control')
    plt.scatter(x = x_layer, y = y_intact, facecolor='lightgrey',s=5, linewidth=.5, edgecolor='', zorder=-2)

    if show_y:
        plt.yticks([0.0, .5, 1.0], [0.0, .5, 1.0], size=7)
        plt.ylabel('%s'%y_tag, labelpad=5, fontsize=9, x=10)
        plt.legend( loc=4, framealpha=0, fontsize=7, title_fontsize=7, title=l_tag)
    else:
        plt.yticks([])

    plt.ylim([-.1, 1.1]); plt.xlim([-.1, 1.1])
    if group=='hpc':
        plt.xticks(fontsize=7)
        plt.xlabel('Model$_{%s}$ Performance\n %s' %(i_layer, xleg[i_layer]), labelpad=7, fontsize=9)
    else:
        plt.xticks([])

def gradient_show():

    y_start = .063
    prc_offset = .007
    y_center=.0015
    x_start = -.30
    x_shift = .09
    l_fontsize=6

    colors = [i.rgb for i in colour.Color('#4B0082').range_to(colour.Color('#FF1493'), 20)]
    [plt.plot(np.array([i, i+1])/300+x_start,(y_start+prc_offset, y_start+prc_offset), color=colors[i], linewidth=4) for i in range(len(colors))];
    plt.annotate('PRC', xy=(x_start+x_shift, y_start+prc_offset-y_center), fontsize=l_fontsize)
    cs = [i.rgb for i in colour.Color('darkblue').range_to(colour.Color('#00FFFF'), 20)]
    [plt.plot(np.array([i, i+1])/300+x_start,(y_start, y_start), color=cs[i], linewidth=4) for i in range(len(cs))];
    plt.annotate('HPC', xy=(x_start+x_shift,y_start-y_center), fontsize=l_fontsize)
    plt.annotate('Lesion Group', xy=(x_start-.035, y_start+prc_offset*2), fontsize=l_fontsize)


def nice_legends(meta_statistics, pls_fits):
    l_fontsize=6

    plt.scatter([], [], s=20,  linewidth=.7, facecolor='white', edgecolor='black', label='p$_{corrected}$ < 0.05')
    #plt.scatter([], [], **plot_params, **color_params, facecolor='white', edgecolor='lightgrey', label='non-significant')
    plt.legend(title='Interaction', framealpha=0, title_fontsize=l_fontsize, fontsize=l_fontsize-1,
               loc=3, bbox_to_anchor=[.45, .335]) # 'Interaction ('r'$\alpha$''=.05)'

    for region in ['it', 'v4']:

        i_layer = {'it':'conv5_1', 'v4':'pool3'}[region]
        i_index = np.nonzero(np.array(pls_fits['layers']) == i_layer)[0][0]

        __y = meta_statistics[meta_statistics.layer==i_layer].prc_delta_rmse.values[0]
        __x = pls_fits['delta'][i_index]

        y_offset = __y+[-.02, .02][region=='v4']
        plt.annotate("'%s-like' Layer"%region.upper(),
                     xy=(__x-.06, y_offset), fontsize=l_fontsize, color='grey',
                     bbox={'alpha':1, 'color':'white'})
        plt.plot([__x, __x], [__y, y_offset], linewidth=.5, alpha=.3, linestyle=":", zorder=-3, color='black')

def focal_neuroanatomical_dependencies(pls_fits, retrospective, meta_statistics, _location):

    fig = plt.figure(constrained_layout=True, figsize=[13.5, 4])
    gs = fig.add_gridspec(2, 8)

    # MODEL FITS TO ELECTROPHYSIOLOGICAL DATA
    ax = fig.add_subplot(gs[0:2, 0:2]);
    order = pls_fits['layers']
    model_electrophysiological_fit(pls_fits, ax)
    ax.text(-0.08, 1.05, 'a', transform=ax.transAxes, fontsize=13, va='top', ha='right')

    # LAYER FITS TO PRC-LESIONED PERFORMENCE
    f_ax = fig.add_subplot(gs[0:1, 2:3])
    show_layer_fit(retrospective, meta_statistics, 'pool1', 'prc', show_y=1)
    f_ax.text(-0.08, 1.10, 'b', transform=f_ax.transAxes,fontsize=13, va='top', ha='right')
    f_ax = fig.add_subplot(gs[0:1, 3:4])
    show_layer_fit(retrospective, meta_statistics, 'pool3', 'prc', show_tag=1)
    f_ax = fig.add_subplot(gs[0:1, 4:5])
    show_layer_fit(retrospective, meta_statistics, 'conv5_1', 'prc')
    f_ax = fig.add_subplot(gs[0:1, 5:6])
    show_layer_fit(retrospective, meta_statistics, 'fc7', 'prc')
    plt.annotate('    significant interaction \n        between slopes',
                 xy=(.17, -.05), fontsize=6, color='black')

    # LAYER FITS TO HPC-LESIONED PERFORMENCE
    f_ax = fig.add_subplot(gs[1:2, 2:3])
    show_layer_fit(retrospective, meta_statistics, 'pool1', 'hpc', show_x=1, show_y=1)
    f_ax = fig.add_subplot(gs[1:2, 3:4])
    show_layer_fit(retrospective, meta_statistics, 'pool3', 'hpc', show_x=1, show_tag=1)
    f_ax = fig.add_subplot(gs[1:2, 4:5])
    show_layer_fit(retrospective, meta_statistics, 'conv5_1','hpc', show_x=1, show_tag=1)
    f_ax = fig.add_subplot(gs[1:2, 5:6])
    show_layer_fit(retrospective, meta_statistics, 'fc7', 'hpc')
    plt.annotate(' non-significant interaction \n          between slopes ',
                 xy=(.17, -.05), fontsize=6, color='grey')

    # RELATING ELECTROPHYSIOLOGICAL DATA TO HUMAN BEHAVIOR
    f_ax = fig.add_subplot(gs[0:2, 7:8])
    f_ax.text(-0.08, 1.05, 'c', transform=f_ax.transAxes,fontsize=13, va='top', ha='right')

    # only extract human fits to model layers that have fits to electrophysiological data
    _meta = meta_statistics[[i in pls_fits['layers'] for i in meta_statistics.layer]]

    # params for both plots
    i_size = 30
    l_fontsize=6
    plot_params = {'marker':'o', 's':i_size}
    color_params = {'linewidth':.5}
    _plsfits = pls_fits['delta']
    x_label_size = 12

    #### PRC DATA
    alphas = (.05/len(pls_fits['it']['mu']))
    p_corrected = np.array(_meta['prc_interaction']) < alphas
    _prc = _meta['prc_intact_rmse'] - _meta['prc_lesion_rmse']
    sig_line = [['white', 'black'][i==True] for i in p_corrected]
    lwidth = [[.5,1][i==True] for i in p_corrected]
    plt.scatter(_plsfits, _prc, **plot_params, linewidth=lwidth,
                facecolor=_meta['prc_colors'], edgecolor=sig_line)

    #### HPC DATA
    alphas = (.05/len(pls_fits['it']['mu']))
    p_corrected = np.array(_meta['hpc_interaction']) < alphas
    sig_line = [['white', 'black'][i==True] for i in p_corrected]
    _hpc = _meta['hpc_intact_rmse'] - _meta['hpc_lesion_rmse']
    plt.scatter(_plsfits, _hpc, **plot_params, **color_params,
                facecolor=_meta['hpc_colors'], edgecolor=sig_line)

    ### AESTHETICS
    plt.xticks(size=6)
    #plt.xlim(-.45, .35)
    plt.yticks(fontsize=5)
    x_label = 'Differential Neural Fit\n$\Delta_{IT - V4}$'
    plt.xlabel(x_label, fontsize=10, labelpad=5)
    plt.ylabel('Differential Fit to Human Behavior\n$\Delta_{intact-lesion}$',
               fontsize=10.5, labelpad=5)
    plt.ylim(-.005, .191)

    gradient_show()
    nice_legends(_meta, pls_fits)
    save_location = os.path.join(_location, 'figure_three.pdf')
    plt.savefig(save_location, format='pdf', bbox_inches = "tight")

def high_throughput_results(novel_summary, _location):

    def plot_nice_line(model_, params={}, title=''):
        m, b = model_.coef_[0], model_.intercept_[0]
        xs = np.array([.2, min(1 , ( ( 1 - b) / m ))])
        plt.plot(xs, xs * m + b, **params)

    color = { 'v4':'#c9d1d3', 'fc6': '#8b2f97', 'prc': '#00909e'}

    plt.figure(figsize=[17, 4])
    for plot, region in enumerate(list(color), 1):

        ax = plt.subplot(1, 5, plot)
        x_ = novel_summary['it'].values
        y_ = novel_summary[region].values
        plt.plot([.2, 1], [.2, 1], color='lightgrey', linestyle=':', zorder=-1)
        _dotparams = {'edgecolor':'black', 'linewidth': .4, 's':35, 'color':color[region]}
        plt.scatter(x=x_, y=y_, **_dotparams)
        plt.xlabel('IT-Supported performance', fontsize=15, labelpad=10)
        if region == 'fc6':
            plt.ylabel('Model Performance', fontsize=13, labelpad=1)
        elif region == 'prc':
            plt.ylabel('PRC-Intact Performance', fontsize=13, labelpad=2)
        else:
            plt.ylabel('%s-Supported Performance'%region.upper(), fontsize=12)

        ax.text(-0.11, 1.08, ['','a','b','c'][plot], transform=ax.transAxes,fontsize=16, va='top', ha='right')

        ax = plt.subplot(1, 5, 4)
        _lineparams = {'linewidth':7, 'zorder':-1, 'color':color[region], 'label':region.upper(),
                       'solid_capstyle': 'round'}

        x_ = novel_summary['fc6'].values
        if region == 'fc6':
            y_ = novel_summary['it'].values
            _lineparams['label']='IT'
        else:
            y_ = novel_summary[region].values

        _model = LinearRegression().fit(np.reshape(x_, (-1,1)), np.reshape(y_, (-1, 1)))
        plot_nice_line(_model, _lineparams)

    plt.ylabel('VVS-Supported Performance', fontsize=12, labelpad=0)
    plt.xlabel('Model Performance', fontsize=15, labelpad=10)
    plt.plot([.2, 1], [.2, 1], color='lightgrey', linestyle=':', zorder=-3)
    plt.legend(loc=4, title='READOUT', framealpha=0, fontsize=9, title_fontsize=9)
    ax.text(-0.11, 1.08, 'd', transform=ax.transAxes,fontsize=16, va='top', ha='right')
    ax = plt.subplot(155)
    _dotparams = {'edgecolor':'black', 'linewidth': .4, 's':35}
    _x = novel_summary['delta_prc_it']
    _y = novel_summary['rt']
    plt.scatter(x=novel_summary['delta_prc_v4'], y=_y, facecolor=color['v4'], edgecolor='', label='V4', zorder=-2)
    plt.scatter(x=_x, y=_y, color=color['fc6'], **_dotparams, label='IT')
    plt.ylim(min(_y)-200,  max(_y)+200)
    plt.xlim(min(_x)-.08,  max(_x)+.08)
    plt.legend(framealpha=0, title='READOUT')
    #plt.xlabel('PRC-intact — VVS-supported Accuracy', fontsize=12)
    plt.xlabel('$\Delta_{\mathregular{PRC-VVS}}$'' Performance', fontsize=15, labelpad=10)
    plt.ylabel('Reaction Time (ms)', fontsize=12, labelpad=-2)
    plt.yticks(size=7)
    plt.subplots_adjust(right=1.2)
    ax.text(-0.11, 1.08, 'e', transform=ax.transAxes,fontsize=16, va='top', ha='right')

    save_location = os.path.join(_location, 'figure_five.pdf')
    plt.savefig(save_location, format='pdf', bbox_inches = "tight")

def show_resnets(df, resnets, _location):

    def plot_nice_line(model_, params={}, title=''):
        m, b = model_.coef_[0], model_.intercept_[0]
        xs = np.array([0, min(1 , ( ( 1 - b) / m ))])
        plt.plot(xs, xs * m + b, **params)

    prc_color = ('#a6229a', '#72249a')
    con_color = ('#d8d8d8', '#b2b2b2')
    hpc_color = ('#032997', '#037397')

    lw = 3
    _a = 1
    _cap = ['butt', 'round', 'projecting'][0]
    _jstyle = ['miter', 'round', 'bevel'][2]
    #resnets = [i for i in df.columns if 'resnet' in i]
    prc_c = [i.rgb for i in colour.Color(prc_color[0]).range_to(colour.Color(prc_color[1]), len(resnets))]
    hpc_c = [i.rgb for i in colour.Color(hpc_color[0]).range_to(colour.Color(hpc_color[1]), len(resnets))]
    con_c = [i.rgb for i in colour.Color(con_color[0]).range_to(colour.Color(con_color[1]), len(resnets))]

    prc_ = df['prc_lesion'].values
    hpc_ = df['hpc_lesion'].values
    con_ = df['prc_intact'].values

    for i_resnet in range(len(resnets)):

        mod_ = df[resnets[i_resnet]].values

        con_model_ = LinearRegression().fit(np.reshape(mod_, (-1,1)), np.reshape(con_, (-1, 1)))
        params = {'color':con_c[i_resnet],'linewidth':lw,'zorder':-10,
                 'solid_capstyle':_cap, 'alpha':_a, 'solid_joinstyle':_jstyle}
        plot_nice_line(con_model_, params)

        hpc_model_ = LinearRegression().fit(np.reshape(mod_, (-1,1)), np.reshape(hpc_, (-1, 1)))
        params = {'color':hpc_c[i_resnet], 'linewidth':lw,'zorder':-3,
                 'solid_capstyle':_cap, 'alpha':_a}
        plot_nice_line(hpc_model_, params)

        prc_model_ = LinearRegression().fit(np.reshape(mod_, (-1,1)), np.reshape(prc_, (-1, 1)))
        params = {'color':prc_c[i_resnet], 'linewidth':lw, 'zorder':-5, 'label':'%3d'%int(resnets[i_resnet]), 
                 'solid_capstyle':_cap, 'alpha':_a}
        plot_nice_line(prc_model_, params)


    y_i = .0085
    y_x = .0595
    ly = .29
    _params = {'solid_capstyle':_cap,'linewidth':lw}
    # PRC ANNOTATION
    plt.annotate('$PRC$', xy=(.825, ly), fontsize=9)
    # HPC ANNOTATION
    [plt.plot([.71, .79],[y_i + y_x*i, y_i+y_x*i], color=hpc_c[i], **_params) for i in range(len(resnets))]
    plt.annotate('$HPC$', xy=(.715, ly), fontsize=9)
    # PRC INTACT
    [plt.plot([.60, .68],[y_i + y_x*i, y_i+y_x*i], color=con_c[i], **_params) for i in range(len(resnets))]
    plt.annotate('$NON$', xy=(.59, ly), fontsize=9)
    plt.annotate('$n$', xy=(.96, ly), fontsize=9)
    plt.annotate('Lesion Group', xy=(.62, .37), fontsize=10)
    plt.plot([0, 1], [0, 1], color='grey', linestyle=':', zorder=-15)
    plt.legend(title='', framealpha=0, fontsize=8, loc=4)
    plt.xlabel('Model$_{layers=n} $'' ''Performance', fontsize=12, labelpad=10)
    plt.ylabel('Human Performance', fontsize=12)
    plt.xticks(size=8)
    plt.yticks(size=8)
    plt.savefig(os.path.join(_location, 'figure_six.pdf'), format='pdf', bbox_inches = "tight")

def face_retrospective_models(meta_df, PARAMS):
    np.random.seed(1234)
    con_color =  ('#2a2a2a', '#cccccc')
    n_objects = sum(['face' not in i for i in meta_df.experiment])
    c_objects = [i.rgb for i in colour.Color(con_color[1]).range_to(colour.Color(con_color[0]), n_objects)]
    c_faces = PARAMS['FACECOLOR']
    s_ = PARAMS['s'] + 5
    x0 = .1
    x1 = .9
    #lw = PARAMS['pointlinewidth']+.25
    _params = {'edgecolor':PARAMS['edgecolor'], 's':PARAMS['s'], 'linewidth':PARAMS['pointlinewidth']}

    i_df = meta_df[ ['face' in i for i in meta_df.experiment] ]
    plt.scatter(x=[x1+np.random.randn()/100 for i in range(len(i_df))],y=i_df['vggface'],
                facecolor=c_faces, **_params)
    plt.scatter(x=[x0+np.random.randn()/100 for i in range(len(i_df))],y=i_df['fc6'],
                facecolor=c_faces, **_params)
    faces_ = [i_df['vggface'].mean(), i_df['fc6'].mean()]

    _params = {'edgecolor':'white', 's':s_, 'linewidth':PARAMS['pointlinewidth'], 'zorder':-2}
    i_df = meta_df[ ['face' not in i for i in meta_df.experiment] ]
    plt.scatter(x=[x1+np.random.randn()/100 for i in range(len(i_df))],y=i_df['vggface'],
                facecolor=c_objects, **_params)
    plt.scatter(x=[x0+np.random.randn()/100 for i in range(len(i_df))],y=i_df['fc6'],
                facecolor=c_objects, **_params)
    objects_ = [i_df['vggface'].mean(), i_df['fc6'].mean()]

    _params = {'linewidth':2, 'zorder':-5, 'linestyle':'-', 'alpha':1, 'solid_capstyle':'round'}
    plt.plot([x0, x1], [faces_[0], objects_[0]], color='grey', **_params, label='object')
    plt.plot([x0, x1], [faces_[1], objects_[1]], color=c_faces, **_params, label='faces')
    plt.xlim(-.2, 1.2)
    #plt.ylim(.01, 1.15)
    plt.ylim([.05, 1.05])
    plt.xticks([x0, x1], ['Objects', 'Faces'], fontsize=PARAMS['xtick_size']+2, y=0,)
    plt.ylabel('Model Performance on Retrospective Experiments', labelpad=5,
               fontsize=PARAMS['ylabel_fontsize']+2);
    plt.yticks(fontsize=PARAMS['xtick_size'])
    plt.xlabel('Training Data', labelpad=5, fontsize=PARAMS['xlabel_fontsize']+2)
    plt.legend(title='category mean', framealpha=0, fontsize=PARAMS['legend_fontsize'],
               labelspacing=.2, bbox_to_anchor=[.25, .2], title_fontsize=PARAMS['legend_title_fontsize'])

def face_retrospective_human(meta_df, ax, PARAMS, group, model, show_legend=1): 
    #prc_color = ('#72249a', '#9a2487') 
    con_color =  ('#2a2a2a', '#cccccc')
    #hpc_color = ('#032997', '#037397') 

    n_objects = sum(['face' not in i for i in meta_df.experiment])
    c_objects = [i.rgb for i in colour.Color(con_color[1]).range_to(colour.Color(con_color[0]), n_objects)]
    #c_objects = c_objects[-1::-1]
    x0 = .1
    x1 = .9
    l_fsize= PARAMS['legend_fontsize']
    i_alpha= .3
    tsize = PARAMS['xtick_size']
    lw = .3
    label_adjustsize=0
    _params = {'edgecolor':PARAMS['edgecolor'], 's':PARAMS['s'], 'linewidth':PARAMS['pointlinewidth'], 
               'facecolor':PARAMS['FACECOLOR']}
    
    face_df = meta_df[ ['face' in i for i in meta_df.experiment] ] 
    ax.scatter(x=face_df[model]    , y=face_df[group], **_params, zorder=-2, label='faces')
    #ax.plot([0, 1], [0, 1], color='grey', linestyle=':', zorder=-15)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=.5, alpha=.5, zorder=-2)
            
    _params = {'facecolor':c_objects, 'edgecolor':PARAMS['edgecolor'],
               's':PARAMS['s'], 'linewidth':PARAMS['pointlinewidth']}
    obj_df = meta_df[ ['face' not in i for i in meta_df.experiment] ] 
    close_black = ax.scatter(x=obj_df[model]    , y=obj_df[group], **_params, zorder=-2,label='objects')    
    if show_legend:
        ax.legend(loc=4, title='Stimulus Type', framealpha=0,fontsize=l_fsize, title_fontsize=l_fsize) 
        
    else: 
        a, = plt.plot([], [], linestyle='--', color='grey',  linewidth=.8, alpha=1)
        plt.legend([a], ['Model Prediction'], loc=4,  fontsize=PARAMS['legend_fontsize'],
                   labelspacing=.2, framealpha=0) 

    plt.yticks([0.0, .25, .5, .75, 1], ['0.00', .25, '0.50', .75, '1.00'], fontsize=tsize)
    plt.xticks([0.0, .25, .5, .75, 1], ['0.00', .25, '0.50', .75, '1.00'], fontsize=tsize)
    plt.ylabel('Human Performance\nPRC-Intact ', labelpad=4, fontsize=PARAMS['ylabel_fontsize']+label_adjustsize); 
    plt.xlabel('Model Performance\n%s-Trained '%['Face', 'Object'][model=='fc6'], 
               labelpad=4, fontsize=PARAMS['xlabel_fontsize']+label_adjustsize)

def face_novel_model(df_select, PARAMS, view_type='foveated'): 
    
    category_sets = {c: df_select[df_select.category==c].oddity_name.unique() for c in df_select.category.unique()}
    np.random.seed(765)

    x_face, x_object = [.25, 0]
    C = {'animals':'#000000','chairs':'#696969', 'planes':'#a8a8a8',  'faces':PARAMS['FACECOLOR']}

    object_color = 'white'
    _params = {'edgecolor':PARAMS['edgecolor'], 'linewidth':PARAMS['pointlinewidth'], 's':PARAMS['s']-5}
    _lparams = {'linestyle':'-', 'linewidth':2, 'zorder':-1}

    resolution = 'trial_id'
    object_alpha = 1 
    
    for i_category in ['animals', 'chairs', 'planes', 'faces']: 

        category_set = category_sets[i_category]

        i_df = df_select[(df_select.category==i_category) * (df_select.image_type==view_type)]
        face_values = i_df[i_df.training_data=='vggface'].groupby([resolution]).mean().accuracy.values
        object_values = i_df[i_df.training_data=='imagenet'].groupby([resolution]).mean().accuracy.values

        face_jitter = x_face + np.random.randn(len(face_values))/100
        object_jitter = x_object + np.random.randn(len(object_values))/100
        i_alpha=[object_alpha, 1][i_category=='faces']
        plt.scatter(y = face_values, x = face_jitter, facecolor=C[i_category], **_params, alpha=i_alpha) 
        plt.scatter(y = object_values,x = object_jitter, facecolor=C[i_category], **_params, alpha=i_alpha) 

        plt.plot([x_face, x_object], [np.mean(face_values), np.mean(object_values)], 
                 color=C[i_category], **_lparams, label='%s'%i_category, solid_capstyle='round')

    plt.legend(framealpha=1, fontsize=PARAMS['legend_fontsize'], labelspacing=.5,
               title_fontsize=PARAMS['legend_title_fontsize'], bbox_to_anchor=[.51, .25], 
               edgecolor='white', title='category mean', frameon=0)    
    plt.xlim(x_object-.1, x_face+.1); 
    #plt.xticks([x_object, x_face, x_object], [True, False], size=PARAMS['xtick_size']); 
    plt.yticks(fontsize=PARAMS['xtick_size'])
    plt.xlabel('Training Data', labelpad=5, fontsize=PARAMS['xlabel_fontsize']+2)
    plt.ylim([.05, 1.05])
    plt.ylabel('Model Performance on Novel Experiment',  labelpad=5, 
               fontsize=PARAMS['ylabel_fontsize']+2)
    plt.xticks([x_face, x_object], ['Faces', 'Objects'], fontsize=7); 
    plt.yticks(size=PARAMS['xtick_size'])

def face_novel_human(novel, i_data, y_data, PARAMS, view_type='foveated', legend=0):

    C = {'animals': '#000000','chairs': '#696969','planes': '#a8a8a8','faces': PARAMS['FACECOLOR']}
    resolution = 'typical_name'


    for i_category in novel.category.unique():

        i_df = novel[novel.category==i_category]
        it_values = i_df.groupby([resolution]).mean().it.values
        v4_values = i_df.groupby([resolution]).mean().v4.values # i_df[i_df.model=='v4'].groupby([resolution]).mean().accuracy.values

        life_values = i_df.groupby([resolution]).mean().human.values
        time_values = i_df.groupby([resolution]).mean().rt.values


        face_values = i_df.groupby('typical_name').mean()['vggface%s'%['','_orig']['original'==view_type]].values
        object_values = i_df.groupby('typical_name').mean()['imagenet%s'%['','_orig']['original'==view_type]].values

        ts = PARAMS['xtick_size']
        s_ = PARAMS['s']

        #####
        i_set = {'it':it_values, 'imagenet':object_values, 'vggface':face_values}
        dot_params = {'label':'%s'%i_category, 'facecolor':C[i_category],
                      's':s_, 'edgecolor':'white', 'linewidth':.25}

        def x_label(i_data):
            plt.xlabel('Model Performance\n%s-Trained'%['Object', 'Face'][i_data=='vggface'],
                       fontsize=PARAMS['xlabel_fontsize'])

        plt.xlim(0.15, 1.03)
        plt.xticks([.25, .5, .75, 1], [.25, .50, .75, 1.0])

        # IT
        if y_data == 'it':

            plt.scatter(x=i_set[i_data], y=it_values, **dot_params)
            plt.plot([.18, 1], [.18, 1], color='grey', linestyle='--', linewidth=.5, alpha=.5, zorder=-2)
            x_label(i_data)
            plt.ylabel('IT-Supported Performance', fontsize=PARAMS['ylabel_fontsize'])
            plt.xticks(fontsize=ts)
            plt.yticks(fontsize=ts)

        # model x human accuracy
        elif y_data == 'human':
            plt.scatter(x=i_set[i_data], y=life_values, **dot_params)
            x_label(i_data)
            plt.plot([.18, 1], [.18, 1], color='grey', linestyle='--', linewidth=.5, alpha=.5, zorder=-2)
            plt.ylabel('Human Performance\nPRC-Intact', fontsize=PARAMS['ylabel_fontsize'], labelpad=4)
            plt.xticks(fontsize=ts) ; plt.yticks(fontsize=ts)

        elif y_data =='rt':

            x_ = i_set[i_data]
            plt.scatter(x=x_, y=time_values, **dot_params)
            plt.yticks(size=8)
            plt.xticks(fontsize=ts) ; plt.yticks(fontsize=ts)
            x_label(i_data)
            plt.ylabel('Reaction Time (ms)', fontsize=PARAMS['ylabel_fontsize']+1, labelpad=4)
            plt.xticks(fontsize=ts) ;
            plt.yticks([2000, 2500, 3000, 3500], [2000, 2500, 3000, 3500], fontsize=ts)
            plt.ylim(1600, 3800)
            plt.xlim(0.19, 1.03)

    if view_type == 'original':
        plt.xlim(0.1, 1.03)
    if legend:
        plt.legend(framealpha=0, title='category', loc=4,
                   fontsize=PARAMS['legend_fontsize'], title_fontsize=PARAMS['legend_title_fontsize'],
                   labelspacing=.2,)
    if (not legend) * (y_data == 'human') :
        a, = plt.plot([], [], linestyle='--', color='grey',  linewidth=.8, alpha=1)
        plt.legend([a], ['Model Prediction'], loc=4,  fontsize=PARAMS['legend_fontsize'],
                   title_fontsize=PARAMS['legend_title_fontsize']+1, labelspacing=.2, framealpha=0)

def changing_distribution_of_training_data(retrospective, mm_select, novel, _location): 
    
    face_color = '#c53a73'
    PARAMS = {'FACECOLOR':face_color, 'edgecolor':'white', 'pointlinewidth':.25, 'legend_title_fontsize':6, 
              'legend_fontsize':6, 's':15, 'xtick_size':6, 'ylabel_fontsize':7, 'xlabel_fontsize':7, 'labelpad':2}

    label_size = 13
    label_height = 1.05
    label_dist = -.30

    fig = plt.figure(constrained_layout=True, figsize=[8, 8])
    gs = fig.add_gridspec(4, 4, wspace=0, hspace=1) 

    ###
    ax = fig.add_subplot(gs[0:2, 0:1]); 
    face_retrospective_models(retrospective, PARAMS)
    plt.text(label_dist, label_height-.02, 'a', fontsize=label_size, transform=ax.transAxes,)
    ###
    ax = fig.add_subplot(gs[0:1, 1:2]); 
    face_retrospective_human(retrospective, ax, PARAMS, 'prc_intact', 'fc6')
    #plt.text(label_dist, label_height, 'b', fontsize=label_size, transform=ax.transAxes,)
    ax = fig.add_subplot(gs[1:2, 1:2]); 
    face_retrospective_human(retrospective, ax, PARAMS, 'prc_intact', 'vggface', show_legend=0)
    #plt.text(label_dist, label_height, 'c', fontsize=label_size, transform=ax.transAxes,)

    ###
    ax = fig.add_subplot(gs[2:4, 0:1]);
    face_novel_model(mm_select, PARAMS)
    plt.text(label_dist, label_height, 'b', fontsize=label_size, transform=ax.transAxes,)
    ###
    ax = fig.add_subplot(gs[2:3, 1:2]); 
    face_novel_human(novel, 'imagenet', 'human', PARAMS, legend=1)
    #plt.text(label_dist, label_height, 'e', fontsize=label_size, transform=ax.transAxes,)
    ax = fig.add_subplot(gs[3:4, 1:2]); 
    face_novel_human(novel, 'vggface', 'human', PARAMS, legend=0)
    #plt.text(label_dist, label_height, 'f', fontsize=label_size, transform=ax.transAxes,)

    # ###
    ax = fig.add_subplot(gs[2:3, 2:3]); 
    face_novel_human(novel, 'imagenet', 'it', PARAMS)
    plt.text(label_dist, label_height+.05, 'c', fontsize=label_size, transform=ax.transAxes,)
    ax = fig.add_subplot(gs[3:4, 2:3]); 
    face_novel_human(novel, 'vggface', 'it', PARAMS)
    #plt.text(label_dist, label_height, 'h', fontsize=label_size, transform=ax.transAxes,)

    ax = fig.add_subplot(gs[2:3, 3:4]); 
    face_novel_human(novel, 'imagenet', 'rt', PARAMS)
    #plt.text(label_dist, label_height, 'i', fontsize=label_size, transform=ax.transAxes,)
    ax = fig.add_subplot(gs[3:4, 3:4]); 
    face_novel_human(novel, 'vggface', 'rt', PARAMS)
    
    plt.text(label_dist, label_height, 'j', fontsize=label_size, transform=ax.transAxes,)
    plt.savefig(os.path.join(_location, 'figure_seven.pdf'), format='pdf', bbox_inches = "tight")

def unfoveated_model_behavior(retrospective, mm_select, novel, _location):
    
    face_color = '#c53a73'
    PARAMS = {'FACECOLOR':face_color, 'edgecolor':'white', 'pointlinewidth':.25, 'legend_title_fontsize':6,
              'legend_fontsize':6, 's':15, 'xtick_size':6, 'ylabel_fontsize':7, 'xlabel_fontsize':7, 'labelpad':2}

    label_size = 13
    label_height = 1.05
    label_dist = -.30

    fig = plt.figure(constrained_layout=True, figsize=[8, 4])
    gs = fig.add_gridspec(2, 4, wspace=0, hspace=1)

    ax = fig.add_subplot(gs[0:2, 0:1]);
    face_novel_model(mm_select, PARAMS, view_type='original')
    plt.text(label_dist, label_height, 'a', fontsize=label_size, transform=ax.transAxes,)
    ###
    ax = fig.add_subplot(gs[0:1, 1:2]);
    face_novel_human(novel, 'imagenet', 'human', PARAMS, legend=1, view_type='original')
    plt.text(label_dist, label_height+.05, 'b', fontsize=label_size, transform=ax.transAxes,)
    ax = fig.add_subplot(gs[1:2, 1:2]);
    face_novel_human(novel, 'vggface', 'human', PARAMS, legend=0, view_type='original')

    # ###
    ax = fig.add_subplot(gs[0:1, 2:3]);
    face_novel_human(novel, 'imagenet', 'it', PARAMS, view_type='original')
    ax = fig.add_subplot(gs[1:2, 2:3]);
    face_novel_human(novel, 'vggface', 'it', PARAMS, view_type='original')

    ax = fig.add_subplot(gs[0:1, 3:4]);
    face_novel_human(novel, 'imagenet', 'rt', PARAMS, view_type='original')
    ax = fig.add_subplot(gs[1:2, 3:4]);
    face_novel_human(novel, 'vggface', 'rt', PARAMS, view_type='original')
    plt.savefig(os.path.join(_location, 'supplemental_figure_four.pdf'), format='pdf', bbox_inches = "tight")



def show_misclassified_studies(misclassified, _location):
    study_order = ['buffalo', 'knutson', 'barense', 'inhoff']

    titles = {'buffalo': 'Buffalo et al. 1999 (Fractals)\nModel Accuracy: 100%',
             'knutson': 'Knutson et al. 2011 (Object Pairs)\nModel Accuracy: 100%',
             'barense': 'Barense et al. 2007 (Fribbles)\nModel Accuracy: 100%',
             'inhoff': 'Inhoff et al. 2018 (Face Morphs)\n Model Accuracy: 100%'}

    params = {'m':'p', 's':70, 'C0': '#130619', 'C1': '#ff3edf','x_label':15, 'title': 13}
    
    def gradient_display(x_, y_):
        colors = [i.hex for i in colour.Color(params['C0']).range_to(colour.Color(params['C1']), (100))]
        [plt.plot(np.array([i, i+1])/190 + x_,(y_, y_), color=colors[i], linewidth=8) for i in range(len(colors))];
        plt.annotate('"STIMULUS COMPLEXITY"', xy=[x_-.08, y_+.05])
        plt.annotate('LOW', xy=[x_-.03, y_ - .08 ], fontsize=8)
        plt.annotate('HIGH', xy=[x_+.43, y_ - .08 ], fontsize=8)
    
    def get_gradient(n):
        g = [i.hex for i in colour.Color(params['C0']).range_to(colour.Color(params['C1']), n)]
        return [params['C1'], g][len(g)>1]
    
    i = 1
    plt.figure(figsize=[18, 4])
    for i_study in study_order:
        ax = plt.subplot(1, 4, i)
        plt.plot([0,1], [0, 1], linestyle='--', color='grey', zorder=-1)

        i_data = misclassified[[ i_study in i.lower() for i in misclassified.study ]]

        ax.text(-0.05, 1.1,  ['a','b','c','d'][i-1],
                transform=ax.transAxes,fontsize=14, va='top', ha='right')
        colors = get_gradient(len(i_data))

        if i_study == 'knutson': gradient_display(.05, .95)
        if i_study == 'buffalo':
            p = {'marker':params['m'], 's':params['s']}

            plt.scatter([], [], **p, edgecolor='black', facecolor='',  label='PRC-LESION')
            plt.scatter([], [], **p, edgecolor='black', facecolor='black', label='PRC-INTACT')
            plt.legend(framealpha=0, title='PATIENT GROUP')

        plt.scatter(y=i_data['control'], x=i_data['model_performance'],
                    marker=params['m'], s=params['s'], edgecolor=colors, facecolor=colors)

        plt.scatter(y=i_data['prc_lesion'], x=i_data['model_performance'],
                    marker=params['m'], s=params['s'], edgecolor=colors, facecolor='white')


        plt.title(titles[i_study], y=1.05)
        plt.xlim(-.1, 1.1) ; plt.ylim(-.1, 1.1)
        i+=1
    
    plt.savefig(os.path.join(_location, 'S2.pdf'), format='pdf', bbox_inches = "tight")

def transformed_model_performance(retrospective, summary_location, _location):
    # weighted and uniform model performance on novel experiment
    with open(summary_location, 'rb') as f:
        _s = pickle.load(f)
        _s = {i.lower():_s[i] for i in _s}

    # generate dataframe
    df = pandas.DataFrame({})
    for c in _s:
        for t in _s[c]:
            for o in _s[c][t]:
                trial = _s[c][t][o]
                i_comparison = {'distance':np.mean(trial['distance_accuracy']),
                                'linear':np.mean(trial['linear_accuracy'])}
                df = df.append(i_comparison, ignore_index=True)

    # for convenience
    def plot_nice_line(model_, params={}, lim=0, title=''):
        m, b = model_.coef_[0], model_.intercept_[0]
        xs = np.array([lim, min(1 , ( ( 1 - b) / m ))])
        plt.plot(xs, xs * m + b, **params)

    # fit between weighted and unweighted/distance performance
    transpose = LinearRegression().fit(np.reshape(df['distance'].values, (-1,1)),
                                       np.reshape(df['linear'].values, (-1, 1)))

    ###
    fig = plt.figure(figsize=[9,4])

    # line width for all figures
    lw = 5
    # set style for all figures
    linestyle = '--'

    # first figure
    ax = fig.add_subplot(121)
    # label figure
    ax.text(-0.05, 1.08,'a',transform=ax.transAxes,fontsize=12, va='top', ha='right')

    # plot distance readout and weighted readout
    plt.scatter(x=df['distance'], y=df['linear'],
                # aesthetics
                facecolor='black', edgecolor='', linewidth=.2, s=15)

    # plot
    plot_nice_line(transpose,
     {'linewidth':lw, 'color':'#229aa6', 'zorder':-5,
      'solid_capstyle': 'round', 'label':'transform'}, lim=.2)

    # plot diagonal
    plt.plot([.3, 1], [.3, 1],
             linestyle=':', color='grey', zorder=-3, label='prediction',alpha=.8)

    # axis labels and legends
    plt.xlabel(r'Model Performance$_{unweighted}$')
    plt.ylabel(r'Model Performance$_{weighted}$')
    plt.xticks(fontsize=7); plt.yticks(fontsize=7);
    plt.legend(framealpha=0, fontsize=8, title='Readout', title_fontsize=9)

    # second plot
    ax = fig.add_subplot(122)
    # label figure
    ax.text(-0.05, 1.08, ['a','b','c'][1], transform=ax.transAxes,fontsize=12, va='top', ha='right')
    # set colors for groups
    prc_color, hpc_color = '#a6229a', '#a3a3a3'

    # model performance from an it-like layer
    _x = retrospective['conv5_1'].values
    # prc lesioned performance
    _y = retrospective['prc_lesion'].values

    # find line of best fit between model and prc-lesioned performance
    _prcmodel = LinearRegression().fit(np.reshape(_x, (-1,1)), np.reshape(_y, (-1, 1)))

    # plot line of best fit between model and prc-lesioned performance
    plot_nice_line(_prcmodel, { 'linestyle':linestyle, 'color':prc_color, 'linewidth':lw-3,
                   'label':'unweighted', 'zorder':1, 'alpha':.8, 'solid_capstyle': 'round'})

    # transform model performance according to what we know from the novel dataste (fig 1)
    transform_model_performance = transpose.predict(np.expand_dims(retrospective['conv5_1'], 1))
    # for aestietics: make sure our plot fits within the visualization box
    x_fit = [min(i[0], 1) for i in transform_model_performance]
    # find line of best fit between transformed model performance and prc-lesioned beahvior
    _prcmodelT = LinearRegression().fit(np.reshape(x_fit, (-1,1)), np.reshape(_y, (-1, 1)))
    # plot transformed model performance
    plot_nice_line(_prcmodelT, {'color':prc_color, 'alpha':1, 'linewidth':lw, 'alpha':.8,
                                'label':'transformed', 'solid_capstyle': 'round', 'zorder':3})
    # label
    transformed_legend = plt.legend(framealpha=0, loc=4, title='Fit to Behavior',
                                    title_fontsize=8, fontsize=7)


    # fit line between model performance and prc-intact participants
    _model = LinearRegression().fit(np.reshape(_x, (-1,1)),
                                    np.reshape(retrospective['prc_intact'].values, (-1, 1)))
    # visualize model fit to prc-intact subjects
    plot_nice_line(_model, {'color':hpc_color, 'alpha':1, 'linewidth':lw-3, 'linestyle':linestyle,
                            'zorder':-3, 'solid_capstyle': 'round'})
    # find line of best fit between transformed model performance and prc-intact beahvior
    _model = LinearRegression().fit(np.reshape(x_fit, (-1,1)), np.reshape(_y, (-1, 1)))
    # visualize transormed model fit to prc-intact behavior
    plot_nice_line(_model, {'color':hpc_color, 'alpha':.8,  'linewidth':lw,
                            'alpha':1, 'zorder':2, 'solid_capstyle': 'round', })

    # plot diagonal
    plt.plot([0, 1], [0, 1], linestyle=':', color='grey', zorder=-3, label='prediction',alpha=.8)

    plt.xlabel('Model Performance')
    plt.ylabel('Human Performance')
    plt.xticks(fontsize=7);
    plt.yticks(fontsize=7);

    a, = plt.plot([],[],
        **{'color':prc_color, 'alpha':.8, 'linewidth':lw, 'solid_capstyle': 'round',
           'label':'transformed', 'zorder':-5})

    b, = plt.plot([],[],
       **{'color':hpc_color, 'alpha':1, 'linewidth':lw, 'solid_capstyle': 'round'})

    leg2 = ax.legend([b, a],['prc-intact', 'prc-lesion'], bbox_to_anchor=[1, .40],
                     title='Group', framealpha=0, title_fontsize=9, fontsize=8)

    ax.add_artist( transformed_legend )
    plt.savefig(os.path.join(_location, 'S3.pdf'), format='pdf', bbox_inches = "tight")
