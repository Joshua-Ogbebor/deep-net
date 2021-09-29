import os, copy, comet_ml
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from train import fit
import pytorch_lightning as pl
from ray import tune

def main (num_epochs=50, folder="Dataset", arch='inc', experiment=Experiment(api_key='dummy_key', disabled=True)):
    data_dir = os.path.join(os.getcwd(), folder)
    pl.seed_everything(42, workers=True)    

    #________________________________train_____________________________________
    fit.train_fn(
        config=hparams,
      model_arch=arch,
      num_epochs=num_epochs,
      data_dir = data_dir
        )

    experiment.end()


def config_dict (arch):
     #_______________ tuned Config for architectures _____________________
    config_inc = {
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
    }
    config_vgg = {
        "lr":0.000117906,
        "mm":0.6,
        "dp":0,
        "wD":8e-06,
        "vgg_config":',
        "actvn":'leaky_relu',
        "batch_size": 64, 
        "opt":'adam',
        "b1":0.9,
        "b2":0.999,
        "eps":2.82947e-05,
        "rho":0.9       
    }
    config_alex = {
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
    }


    config_res = {
        "lr":0.0152923,
        "mm":0.6,
        "dp":0,
        "wD":0.00001,
        "bloc_1":256,
        "bloc_2":512,
        "bloc_3":64,
        "bloc_4":64,
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
    }


    if arch =='inc':
       config= config_inc
    elif arch =='res':
       config =config_res
    elif arch == 'alex':
       config =config_alex
    elif arch =='vgg':
       config = config_vgg
    return config #config_

if __name__ == "__main__":
    main(num_epochs=50, folder="Dataset", arch='res')
    #main(num_epochs=50, folder="Dataset", arch='vgg')
    #main(num_epochs=50, folder="Dataset", arch='asha')
    #main(num_epochs=50, folder="Dataset", arch='inc')
    

    
