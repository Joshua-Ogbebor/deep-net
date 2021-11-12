from typing import Union, Optional, Dict
from model.residual_net import Resnet_Classifier as res_net
from model.alex_net import AlexNet as alex_net
from model.inception_net import Googlenet_Classifier as google_net
import pickle, os,torch

def save_model(model:Union[res_net,alex_net,google_net],  arch:str, unique_id:str='', save_state_dict:bool=True,save_entire_model:bool=True, save_serialized=True):
    """ Saves model 
    """
    model_dict_path, pickled_model_path, entire_model_path=model_path_er(arch,unique_id=unique_id )
    if save_entire_model: torch.save(model, entire_model_path)
    if save_state_dict: torch.save(model.state_dict(),model_dict_path)
    if save_serialized: pickle.dump(model, open(pickled_model_path, 'wb'))

    return

def load_model(arch:str,unique_id:str=None, load_entire_model:bool=False, load_serialized=False)->Union[res_net,alex_net,google_net,None]:
    """Loads a saved final model using saved state dict by default. specify arch, and other model load types if needed
    
    """
    model_dict_path, pickled_model_path, entire_model_path=model_path_er(arch=arch, unique_id=unique_id)
    

    if load_serialized and os.path.isfile(pickled_model_path):
        return pickle.load(open(pickled_model_path, 'rb'))
    elif load_entire_model and os.path.isfile(entire_model_path):
        return torch.load(entire_model_path)
    elif os.path.isfile(model_dict_path):# load_state_dict:
        return arch_to_model(arch).load_state_dict(torch.load(model_dict_path)) 
    else:
        print('Path error or enter model load format or directory name')
        return None

def model_path_er(arch:str, unique_id:str='', model_folder:Optional[str]='trained-models' )->str:
    """ Returns the path for saving models
    """
    entire_model_path = os.path.join(model_folder, unique_id + arch+'_entire.pth')
    pickled_model_path= os.path.join(model_folder, unique_id+arch+'_pickled.sav')
    model_dict_path=os.path.join(model_folder, unique_id+arch+'_dict.pth')
    return model_dict_path, pickled_model_path, entire_model_path

def arch_to_model_ref (model_arch:str):
   return {'inc':google_net,
    'res':res_net,
    'alex': alex_net ,
    #'vgg': VGG
   }[model_arch]

def arch_to_model(arch:str, **kwargs)->Union[res_net,alex_net,google_net]:
    return arch_to_model_ref(arch)(**config_s(arch), **kwargs)
def ckpt_dir(arch:str)->str:
    return os.path.join('ckpt',arch+'_model')
def config_s(arch:str)->Dict:
    """Returns tuned hyper parameters for models
    """
    return{'res':{
        "lr":0.0152923,
        "mm":0.6,
        "dp":0,
        "wD":0.00001,
        "bloc_1":128,
        "bloc_2":512,
        "bloc_3":256,
        "bloc_4":512,
        "depth_1":2,
        "depth_2":3,
        "depth_3":0,
        "depth_4":0,
        "actvn":'relu',
        "batch_size": 64,
        "opt":'adadelta',
        "b1":0.9,
        "b2":0.999,
        "eps":2.33231e-08,
        "rho":0.9,
},'inc':{
        "lr":0.000216735,
        "mm":0.6,
        "dp":0,
        "wD":0.00003,
        "depth":4,
        "actvn":'leaky_relu',
        "batch_size": 48,
        "opt":'adam',
        "b1":0.9,
        "b2":0.999,
        "eps":1.7739e-07,
        "rho":0.9
}, 'alex':{
        "lr": 0.0163072,
        "mm":1.2,
        "dp":0,
        "wD":8e-06,
        "actvn":'relu',
        "batch_size": 64,
        "opt":'adadelta',
        "b1":0.9,
        "b2":0.999,
        "eps": 2.04677e-06,
        "rho":0.9
}, 'vgg':{
        "lr":0.000117906,
        "mm":0.6,
        "dp":0,
        "wD":8e-06,
        "vgg_config":'A',
        "actvn":'leaky_relu',
        "batch_size": 64,
        "opt":'adam',
        "b1":0.9,
        "b2":0.999,
        "eps":2.82947e-05,
        "rho":0.9
}}[arch]


