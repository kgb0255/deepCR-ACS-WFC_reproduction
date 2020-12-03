import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import sys
import shutil
import tarfile
import urllib
%matplotlib inline

def download():
    #Download training data
    print('------------------------------------------------------------')
    print('Downloading training data')
    print('------------------------------------------------------------')
    urllib.request.urlretrieve('https://zenodo.org/api/files/8998d801-f877-4301-8b9d-efd3a3adaed7/deepCR.ACS-WFC.train.tar?versionId=9157d939-07da-4ce2-89a3-f59b298b51fd','deepCR.ACS-WFC.train.tar')
    
    #Donwload test data
    print('------------------------------------------------------------')
    print('Downloading test data')
    print('------------------------------------------------------------')
    urllib.request.urlretrieve('https://zenodo.org/api/files/8998d801-f877-4301-8b9d-efd3a3adaed7/deepCR.ACS-WFC.test.tar?versionId=571378b4-9e36-478e-9f8d-c6cc04ce7b8d','deepCR.ACS-WFC.test.tar')
    
    print('Datasets downloaded')
    print('Sorting...')
    shutil.move('deepCR.ACS-WFC.train.tar',data_base)
    shutil.move('deepCR.ACS-WFC.test.tar',data_base)
    print('Complete')
    
    print('Extracting tar files...')
    train_tar = tarfile.open(os.path.join(data_base,'deepCR.ACS-WFC.train.tar'))
    test_tar = tarfile.open(os.path.join(data_base,'deepCR.ACS-WFC.test.tar'))
    
    train_tar.extractall(data_base)
    test_tar.extractall(data_base)
    print('Complete')
    
    return None


#Directories
def get_dirs():
    train_dirs = []
    test_dirs = []

    test_base = os.path.join(data_base,'npy_test')
    train_base = os.path.join(data_base,'npy_train')

    print('------------------------------------------------------------')
    print('Fetching directories for the test set')
    print('------------------------------------------------------------')
    for _filter in os.listdir(test_base):
        filter_dir = os.path.join(test_base,_filter)
        if os.path.isdir(filter_dir):
            for prop_id in os.listdir(filter_dir):
                prop_id_dir = os.path.join(filter_dir,prop_id)
                if os.path.isdir(prop_id_dir):
                    for vis_num in os.listdir(prop_id_dir):
                        vis_num_dir = os.path.join(prop_id_dir,vis_num)
                        if os.path.isdir(vis_num_dir):
                            for f in os.listdir(vis_num_dir):
                                if '.npy' in f and f != 'sky.npy':
                                    test_dirs.append(os.path.join(vis_num_dir,f))

    print('------------------------------------------------------------')
    print('Fetching directories for the training set')
    print('------------------------------------------------------------')
    for _filter in os.listdir(train_base):
        filter_dir = os.path.join(train_base,_filter)
        if os.path.isdir(filter_dir):
            for prop_id in os.listdir(filter_dir):
                prop_id_dir = os.path.join(filter_dir,prop_id)
                if os.path.isdir(prop_id_dir):
                    for vis_num in os.listdir(prop_id_dir):
                        vis_num_dir = os.path.join(prop_id_dir,vis_num)
                        if os.path.isdir(vis_num_dir):
                            for f in os.listdir(vis_num_dir):
                                if '.npy' in f and f != 'sky.npy':
                                    train_dirs.append(os.path.join(vis_num_dir,f))
#     print(train_dirs)
    np.save(os.path.join(base_dir,'test_dirs.npy'), test_dirs)
    np.save(os.path.join(base_dir,'train_dirs.npy'), train_dirs)

    return None

#Train
def model_train():
    from deepCR import train

    #deepCR scalable HST branch
    train_dirs = np.load(os.path.join(base_dir,'train_dirs.npy'),allow_pickle = True)

    f435_train_dirs = []
    f814_train_dirs = []
    f606_train_dirs = []

    num_trains = len(train_dirs)
    for i,train_dir in enumerate(train_dirs):
        arr = train_dir.split('/')
        _filter = arr[-4]
        if _filter == 'f435w':
            f435_train_dirs.append(train_dir)
        elif _filter == 'f606w':
            f606_train_dirs.append(train_dir)
        elif _filter == 'f814w':
            f814_train_dirs.append(train_dir)
        else:
            raise ValueError('Check filter')
    aug_sky = (-0.9,3)

    '''
    *****************************************************************************************************************************
    Training ACS/WFC F435W model
    *****************************************************************************************************************************
    '''
    print('------------------------------------------------------------')
    print('Training ACS/WFC F435W model')
    print('------------------------------------------------------------')

    if not os.path.exists(os.path.join(base_dir,'deepCR.ACS-WFC.F435W')):
        os.makedirs(os.path.join(base_dir,'deepCR.ACS-WFC.F435W'))

    trainer = train(f435_train_dirs[::], mode = 'pair', aug_sky = aug_sky, name = 'F435W', epoch=50)
    trainer.train()
    mdl_f435 = trainer.save()
    shutil.move(f'{mdl_f435}.pth',os.path.join(base_dir,'deepCR.ACS-WFC.F435W',f'{mdl_f435}.pth'))
    np.save(os.path.join(base_dir,'deepCR.ACS-WFC.F435W',f'loss_{mdl_f435}.npy'), np.array(trainer.validation_loss))
    del trainer

    '''
    *****************************************************************************************************************************
    Training ACS/WFC F606W model
    *****************************************************************************************************************************
    '''
    print('------------------------------------------------------------')
    print('Training ACS/WFC F606W model')
    print('------------------------------------------------------------')

    if not os.path.exists(os.path.join(base_dir,'deepCR.ACS-WFC.F606W')):
        os.makedirs(os.path.join(base_dir,'deepCR.ACS-WFC.F606W'))	

    trainer = train(f606_train_dirs[::], mode = 'pair', aug_sky = aug_sky, name = 'F606W', epoch=50)
    trainer.train()
    mdl_f606 = trainer.save()
    shutil.move(f'{mdl_f606}.pth',os.path.join(base_dir,'deepCR.ACS-WFC.F606W',f'{mdl_f606}.pth'))
    np.save(os.path.join(base_dir,'deepCR.ACS-WFC.F606W',f'loss_{mdl_f606}.npy'), np.array(trainer.validation_loss))
    del trainer

    '''
    *****************************************************************************************************************************
    Training ACS/WFC F814W model
    *****************************************************************************************************************************
    '''
    print('------------------------------------------------------------')
    print('Training ACS/WFC F814W model')
    print('------------------------------------------------------------')

    if not os.path.exists(os.path.join(base_dir,'deepCR.ACS-WFC.F814W')):
        os.makedirs(os.path.join(base_dir,'deepCR.ACS-WFC.F814W'))	

    trainer = train(f814_train_dirs[::], mode = 'pair', aug_sky = aug_sky, name = 'F814W', epoch = 50)
    trainer.train()
    mdl_f814 = trainer.save()
    shutil.move(f'{mdl_f814}.pth',os.path.join(base_dir,'deepCR.ACS-WFC.F814W',f'{mdl_f814}.pth'))
    np.save(os.path.join(base_dir,'deepCR.ACS-WFC.F814W',f'loss_{mdl_f814}.npy'), np.array(trainer.validation_loss))
    del trainer

    '''
    *****************************************************************************************************************************
    Training ACS/WFC Global model
    *****************************************************************************************************************************
    '''
    print('------------------------------------------------------------')
    print('Training ACS/WFC Global model')
    print('------------------------------------------------------------')

    if not os.path.exists(os.path.join(base_dir,'deepCR.ACS-WFC')):
        os.makedirs(os.path.join(base_dir,'deepCR.ACS-WFC'))

    trainer = train(train_dirs[::], mode = 'pair', aug_sky = aug_sky, name = 'ACS-WFC', epoch = 50)
    trainer.train()
    mdl_3xF= trainer.save()
    shutil.move(f'{mdl_3xF}.pth',os.path.join(base_dir,'deepCR.ACS-WFC',f'{mdl_3xF}.pth'))
    np.save(os.path.join(base_dir,'deepCR.ACS-WFC',f'loss_{mdl_3xF}.npy'), np.array(trainer.validation_loss))
    del trainer

    gc.collect()

    return mdl_f435,mdl_f606,mdl_f814,mdl_3xF

#Test
def model_test(mdl_names):
    from deepCR import evaluate
    from deepCR import deepCR
    from deepCR.evaluate import roc_lacosmic
    _mdl_f435, _mdl_f606, _mdl_f814, _mdl_3xF=mdl_names
    test_dirs = np.load(os.path.join(base_dir,'test_dirs.npy'),allow_pickle = True)

    field_type = {'10595_2': 'GC',
              '10595_7': 'GC',
              '9442_1': 'GC',
              '9442_3': 'GC',
              '9442_5': 'GC',
              '10760_2': 'GAL',
              '10760_4': 'GAL',
              '10631_3': 'EX',
              '10631_1': 'EX',
              '10631_4': 'EX',
              '12103_a3': 'EX',
              '13498_b1': 'EX',
              '13737_2': 'GAL',
              '13737_3': 'GAL',
              '9490_a3': 'GAL',
              '10349_30': 'GC', 
              '10005_10': 'GC', 
              '10120_3': 'GC',
              '12513_2': 'GAL', 
              '12513_3': 'GAL', 
              '14164_9': 'EX', 
              '13718_6': 'EX', 
              '10524_7': 'GC', 
              '10182_pb': 'GAL',
              '10182_pd': 'GAL',
              '9425_2': 'EX',
              '9425_4': 'EX',
              '9583_99': 'EX',
              '10584_13': 'GAL',
              '9978_5e': 'EX',
              '15100_2': 'EX',
              '15647_13': 'EX',
              '11340_11': 'GC',
              '13389_10': 'EX',
              '9694_6': 'EX',
              '10342_3': 'GAL',
              
              '14343_1': 'GAL', 
              '10536_13': 'EX',
              '13057_1': 'GAL', 
              '10260_7': 'GAL',
              '10260_5': 'GAL',
              '10407_3': 'GAL',
              '13375_4': 'EX',
              '13375_7': 'EX',
              '13364_95': 'GAL',
              '10190_28': 'GAL',
              '10190_13': 'GAL',
              '10146_4': 'GC',
              '10146_3': 'GC',
              '10775_ab': 'GC',
              '11586_5':'GC',
              '12438_1': 'EX', 
              '13671_35': 'EX',
              '14164_1': 'GC', 
              
              '9490_a2': 'GAL',
              '9405_6d': 'EX',
              '9405_4b': 'EX',
              '9450_14': 'EX',
              '10092_1': 'EX',
              '13691_11': 'GAL',
              '12058_12': 'GAL',
              '12058_16': 'GAL',
              '12058_1': 'GAL',
              '9450_16': 'EX',
              '10775_52': 'GC',
              '12602_1': 'GC',
              '12602_2': 'GC',
              '10775_29': 'GC',
              '10775_ad': 'GC',
              '12058_6': 'GAL', #NEW
              '14704_1': 'GAL', #NEW
              '13804_6': 'GAL' #NEW
             }

    f814_test_field_dirs = {'GC': [], 'EX': [], 'GAL': []}
    f606_test_field_dirs = {'GC': [], 'EX': [], 'GAL': []}
    f435_test_field_dirs = {'GC': [], 'EX': [], 'GAL': []}

    for _dir in test_dirs:
        arr = _dir.split('/')
        _filter = arr[-4]
        key = f'{arr[-3]}_{arr[-2]}'
        f_type = field_type[key]
        if _filter == 'f435w':
            f435_test_field_dirs[f_type].append(_dir)
        elif _filter == 'f606w':
            f606_test_field_dirs[f_type].append(_dir)
        elif _filter == 'f814w':
            f814_test_field_dirs[f_type].append(_dir)

    '''
    *****************************************************************************************************************************
    Testing F435W model
    *****************************************************************************************************************************
    '''
    print('------------------------------------------------------------')
    print('Testing F435W model on F435W testset')
    print('------------------------------------------------------------')
    f435w_mdl=deepCR(mask=os.path.join(base_dir,'deepCR.ACS-WFC.F435W',_mdl_f435)+'.pth')
    for f_type in list(f435_test_field_dirs.keys()):
        filter_field_test_dirs = f435_test_field_dirs[f_type]
        tpr_fpr, tpr_fpr_dilate = evaluate.roc(f435w_mdl,filter_field_test_dirs[::], mode = 'pair', dilate = True)

        tpr,fpr = tpr_fpr

        tpr_d,fpr_d = tpr_fpr_dilate
        fpr_d = fpr

        np.save(os.path.join(base_dir,'deepCR.ACS-WFC.F435W',f'F435W_{f_type}_deepCR.ACS-WFC.F435W.npy'),[[tpr,fpr],[tpr_d,fpr_d]])
    del f435w_mdl

    '''
    *****************************************************************************************************************************
    Testing F606W model
    *****************************************************************************************************************************
    '''
    print('------------------------------------------------------------')
    print('Testing F606W model on F606W testset')
    print('------------------------------------------------------------')
    f606w_mdl=deepCR(mask=os.path.join(base_dir,'deepCR.ACS-WFC.F606W',_mdl_f606)+'.pth')
    for f_type in list(f606_test_field_dirs.keys()):
        filter_field_test_dirs = f606_test_field_dirs[f_type]
        tpr_fpr, tpr_fpr_dilate = evaluate.roc(f606w_mdl,filter_field_test_dirs[::], mode = 'pair', dilate = True)

        tpr,fpr = tpr_fpr

        tpr_d,fpr_d = tpr_fpr_dilate
        fpr_d = fpr

        np.save(os.path.join(base_dir,'deepCR.ACS-WFC.F606W',f'F606W_{f_type}_deepCR.ACS-WFC.F606W.npy'),[[tpr,fpr],[tpr_d,fpr_d]])
    del f606w_mdl

    print('------------------------------------------------------------')
    print('Testing F606W model on F435W testset')
    print('------------------------------------------------------------')
    f606w_mdl=deepCR(mask=os.path.join(base_dir,'deepCR.ACS-WFC.F606W',_mdl_f606)+'.pth')
    for f_type in list(f435_test_field_dirs.keys()):
        filter_field_test_dirs = f435_test_field_dirs[f_type]
        tpr_fpr, tpr_fpr_dilate = evaluate.roc(f606w_mdl,filter_field_test_dirs[::], mode = 'pair', dilate = True)

        tpr,fpr = tpr_fpr

        tpr_d,fpr_d = tpr_fpr_dilate
        fpr_d = fpr

        np.save(os.path.join(base_dir,'deepCR.ACS-WFC.F606W',f'F435W_{f_type}_deepCR.ACS-WFC.F606W.npy'),[[tpr,fpr],[tpr_d,fpr_d]])
    del f606w_mdl

    '''
    *****************************************************************************************************************************
    Testing F814W model
    *****************************************************************************************************************************
    '''
    print('------------------------------------------------------------')
    print('Testing F814W model on F814W testset')
    print('------------------------------------------------------------')
    f814w_mdl=deepCR(mask=os.path.join(base_dir,'deepCR.ACS-WFC.F814W',_mdl_f814)+'.pth')
    for f_type in list(f814_test_field_dirs.keys()):
        filter_field_test_dirs = f814_test_field_dirs[f_type]
        tpr_fpr, tpr_fpr_dilate = evaluate.roc(f814w_mdl,filter_field_test_dirs[::], mode = 'pair', dilate = True)

        tpr,fpr = tpr_fpr

        tpr_d,fpr_d = tpr_fpr_dilate
        fpr_d = fpr

        np.save(os.path.join(base_dir,'deepCR.ACS-WFC.F814W',f'F814W_{f_type}_deepCR.ACS-WFC.F814W.npy'),[[tpr,fpr],[tpr_d,fpr_d]])
    del f814w_mdl

    '''
    *****************************************************************************************************************************
    Testing ACS/WFC model
    *****************************************************************************************************************************
    '''
    print('------------------------------------------------------------')
    print('Testing ACS/WFC model on F435W testset')
    print('------------------------------------------------------------')
    global_mdl=deepCR(mask=os.path.join(base_dir,'deepCR.ACS-WFC',_mdl_3xF)+'.pth')
    for f_type in list(f435_test_field_dirs.keys()):
        filter_field_test_dirs = f435_test_field_dirs[f_type]
        tpr_fpr, tpr_fpr_dilate = evaluate.roc(global_mdl,filter_field_test_dirs[::], mode = 'pair', dilate = True)

        tpr,fpr = tpr_fpr

        tpr_d,fpr_d = tpr_fpr_dilate
        fpr_d = fpr

        np.save(os.path.join(base_dir,'deepCR.ACS-WFC',f'F435W_{f_type}_deepCR.ACS-WFC.npy'),[[tpr,fpr],[tpr_d,fpr_d]])

    print('------------------------------------------------------------')
    print('Testing ACS/WFC model on F606W testset')
    print('------------------------------------------------------------')
    for f_type in list(f606_test_field_dirs.keys()):
        filter_field_test_dirs = f606_test_field_dirs[f_type]
        tpr_fpr, tpr_fpr_dilate = evaluate.roc(global_mdl,filter_field_test_dirs[::], mode = 'pair', dilate = True)

        tpr,fpr = tpr_fpr

        tpr_d,fpr_d = tpr_fpr_dilate
        fpr_d = fpr

        np.save(os.path.join(base_dir,'deepCR.ACS-WFC',f'F606W_{f_type}_deepCR.ACS-WFC.npy'),[[tpr,fpr],[tpr_d,fpr_d]])

    print('------------------------------------------------------------')
    print('Testing ACS/WFC model on F814W testset')
    print('------------------------------------------------------------')
    for f_type in list(f814_test_field_dirs.keys()):
        filter_field_test_dirs = f814_test_field_dirs[f_type]
        tpr_fpr, tpr_fpr_dilate = evaluate.roc(global_mdl,filter_field_test_dirs[::], mode = 'pair', dilate = True)

        tpr,fpr = tpr_fpr

        tpr_d,fpr_d = tpr_fpr_dilate
        fpr_d = fpr

        np.save(os.path.join(base_dir,'deepCR.ACS-WFC',f'F814W_{f_type}_deepCR.ACS-WFC.npy'),[[tpr,fpr],[tpr_d,fpr_d]]) #DELETE gal

    del global_mdl

#     *****************************************************************************************************************************
#     Testing LACosmic
#     *****************************************************************************************************************************

#     '''

#     if not os.path.exists(os.path.join(base_dir,'LACosmic')):
#         os.makedirs(os.path.join(base_dir,'LACosmic'))
#     print('------------------------------------------------------------')
#     print('Testing LACosmic on F435W testset')
#     print('------------------------------------------------------------')
#     for f_type in list(f435_test_field_dirs.keys()):
#         filter_field_test_dirs = f435_test_field_dirs[f_type]
#         tpr_fpr, tpr_fpr_dilate = roc_lacosmic(np.linspace(1,1000,200),filter_field_test_dirs[::20], mode = 'pair', dilate = True)

#         tpr,fpr = tpr_fpr

#         tpr_d,fpr_d = tpr_fpr_dilate
#         fpr_d = fpr

#         np.save(os.path.join(base_dir,'LACosmic',f'F435W_{f_type}_LACosmic.npy'),[[tpr,fpr],[tpr_d,fpr_d]])

#     print('------------------------------------------------------------')
#     print('Testing LACosmic on F606W testset')
#     print('------------------------------------------------------------')
#     for f_type in list(f606_test_field_dirs.keys()):
#         filter_field_test_dirs = f606_test_field_dirs[f_type]
#         tpr_fpr, tpr_fpr_dilate = roc_lacosmic(np.linspace(1,1000,200),filter_field_test_dirs[::20], mode = 'pair', dilate = True)

#         tpr,fpr = tpr_fpr

#         tpr_d,fpr_d = tpr_fpr_dilate
#         fpr_d = fpr

#         np.save(os.path.join(base_dir,'LACosmic',f'F606W_{f_type}_LACosmic.npy'),[[tpr,fpr],[tpr_d,fpr_d]])

#     print('------------------------------------------------------------')
#     print('Testing LACosmic on F814W testset')
#     print('------------------------------------------------------------')
#     for f_type in list(f814_test_field_dirs.keys()):
#         filter_field_test_dirs = f814_test_field_dirs[f_type]
#         tpr_fpr, tpr_fpr_dilate = roc_lacosmic(np.linspace(1,1000,200),filter_field_test_dirs[::20], mode = 'pair', dilate = True)

#         tpr,fpr = tpr_fpr

#         tpr_d,fpr_d = tpr_fpr_dilate
#         fpr_d = fpr

#         np.save(os.path.join(base_dir,'LACosmic',f'F814W_{f_type}_LACosmic.npy'),[[tpr,fpr],[tpr_d,fpr_d]])

    return None

def plot_result(mdl_names):
    import seaborn as sns
    sns.set_theme(style="ticks", context='talk')
    if not os.path.exists(os.path.join(base_dir,'plots')):
        os.makedirs(os.path.join(base_dir,'plots'))
    if dilate:
        _plot = _plotter
    else:
        _plot = _plotter_no_dilate

    print('------------------------------------------------------------')
    print('Plotting F435W testing result')
    print('------------------------------------------------------------')
    _plot('F435W',mdl_names)

    print('------------------------------------------------------------')
    print('Plotting F606W testing result')
    print('------------------------------------------------------------')
    _plot('F606W',mdl_names)

    print('------------------------------------------------------------')
    print('Plotting F814W testing result')
    print('------------------------------------------------------------')
    _plot('F814W',mdl_names)

    return None

def _plotter(filter_name,mdl_names):
    _mdl_f435, _mdl_f606, _mdl_f814, _mdl_3xF=mdl_names
    fig, axs = plt.subplots(2,3, figsize = (18,12), sharey = True, sharex= True)
    fig.suptitle(f'{filter_name} Test set', y = 0.97, fontsize = 35)
    
    tf_global_gc,tf_global_d_gc = np.load(os.path.join(base_dir,'deepCR.ACS-WFC',f'{filter_name}_GC_deepCR.ACS-WFC.npy'), allow_pickle = True)
    tf_global_ex,tf_global_d_ex = np.load(os.path.join(base_dir,'deepCR.ACS-WFC',f'{filter_name}_EX_deepCR.ACS-WFC.npy'), allow_pickle = True)
    tf_global_gal,tf_global_d_gal = np.load(os.path.join(base_dir,'deepCR.ACS-WFC',f'{filter_name}_GAL_deepCR.ACS-WFC.npy'), allow_pickle = True)
    
    tf_global = [tf_global_gc,tf_global_ex,tf_global_gal]
    tf_global_d = [tf_global_d_gc,tf_global_d_ex,tf_global_d_gal]
    
    tf_single_gc,tf_single_d_gc = np.load(os.path.join(base_dir,f'deepCR.ACS-WFC.{filter_name}',f'{filter_name}_GC_deepCR.ACS-WFC.{filter_name}.npy'), allow_pickle = True)
    tf_single_ex,tf_single_d_ex = np.load(os.path.join(base_dir,f'deepCR.ACS-WFC.{filter_name}',f'{filter_name}_EX_deepCR.ACS-WFC.{filter_name}.npy'), allow_pickle = True)
    tf_single_gal,tf_single_d_gal = np.load(os.path.join(base_dir,f'deepCR.ACS-WFC.{filter_name}',f'{filter_name}_GAL_deepCR.ACS-WFC.{filter_name}.npy'), allow_pickle = True)
    
    tf_single = [tf_single_gc,tf_single_ex,tf_single_gal]
    tf_single_d = [tf_single_d_gc,tf_single_d_ex,tf_single_d_gal]
    
#     tf_lac_gc, tf_lac_d_gc = np.load(os.path.join(base_dir,'LACosmic',f'{filter_name}_GC_LACosmic.npy'), allow_pickle = True)
#     tf_lac_ex, tf_lac_d_ex = np.load(os.path.join(base_dir,'LACosmic',f'{filter_name}_EX_LACosmic.npy'), allow_pickle = True)
#     tf_lac_gal, tf_lac_d_gal = np.load(os.path.join(base_dir,'LACosmic',f'{filter_name}_GAL_LACosmic.npy'), allow_pickle = True)
    
#     tf_lac = [tf_lac_gc,tf_lac_ex,tf_lac_gal]
#     tf_lac_d = [tf_lac_d_gc,tf_lac_d_ex,tf_lac_d_gal]


    axs[0,0].set_title('Globular Clusters')
    for i in range(3):
        
        if i ==0:
            axs[1,i].set_ylabel('TPR [%]')
            axs[0,i].set_ylabel('TPR [%]')
        if i == 1:
            axs[1,i].set_xlabel('FPR [%]')

        axs[0,i].set_ylim(20,100)
        axs[1,i].set_ylim(20,100)
        axs[1,i].set_xlim(0,1)
        axs[0,i].grid()
        axs[1,i].grid()

        axs[0,i].set_xlim(0,1)
    axs[0,1].set_title('Extragalactic Fields')
    axs[0,2].set_title('Resolved Galaxies')
    
    for i in range(3):
        t_global, f_global =tf_global[i]
        axs[0,i].plot(f_global,t_global, label = 'deepCR-ACS/WFC w/o dilate',  ls = '-', c = 'r')
        
#         t_lac, f_lac =tf_lac[i]
#         axs[0,i].plot(f_lac,t_lac, label = 'LACosmic w/o dilate',  ls = ':',c = 'C0')
        
        t_single, f_single =tf_single[i]
        axs[0,i].plot(f_single,t_single, label = f'deepCR-ACS/WFC {filter_name} w/o dilate',  ls = '-.',zorder= 1,c = 'k')
        if i == 2:
            axs[0,i].legend(frameon=False,bbox_to_anchor=(1.05, 1), loc='upper left')
    for i in range(3):
        t_global, f_global =tf_global[i]
        t_global_d, f_global_d = tf_global_d[i]
        axs[1,i].plot(f_global_d,t_global_d, label = 'deepCR-ACS/WFC with dilate', c = 'r', ls = '-')
        
        # t_lac, f_lac =tf_lac[i]
        # t_lac_d, f_lac_d = tf_lac_d[i]
        # axs[1,i].plot(f_lac_d,t_lac_d, label = 'LACosmic with dilate', c = 'C0', ls = ':')
        
        t_fsingle, f_fsingle =tf_single[i]
        t_single_d, f_single_d = tf_single_d[i]
        axs[1,i].plot(f_single_d,t_single_d, label = f'deepCR-ACS/WFC {filter_name} with dilate', c = 'k', ls = '-.', zorder = 1)
        if i == 2:
            axs[1,i].legend(frameon=False,bbox_to_anchor=(1.05, 1), loc='upper left')
    ax = fig.add_axes([1.2, 0.5, 0, 0])
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels([' ',' '])
    ax.set_yticklabels([' ',' '])
    fig.savefig(os.path.join(base_dir,'plots',f'{filter_name}.pdf'), format = 'pdf', bbox_inches='tight')

    return None

if __name__=='__main__':
    os.makedirs('deepCR.ACS-WFC')
    base_dir = os.path.join('deepCR.ACS-WFC')
    os.makedirs(os.path.join(base_dir,'data'))
    data_base = os.path.join(base_dir,'data')

    download()
    get_dirs()
    mdl_f435,mdl_f606,mdl_f814,mdl_3xF = model_train()
    model_test([mdl_f435,mdl_f606,mdl_f814,mdl_3xF])
    plot_result([mdl_f435,mdl_f606,mdl_f814,mdl_3xF])
    