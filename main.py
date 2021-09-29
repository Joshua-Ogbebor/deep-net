import os, copy, comet_ml
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from train import fit
import pytorch_lightning as pl
from ray import tune

#comet_logger=CometLogger(api_key = "rI52i2RdTWgAjRD8MlZrdAhoS", experiment_name = "train",project_name = "deep-net")

def main (num_epochs=50, folder="Dataset", arch='inc', experiment=Experiment(api_key='dummy_key', disabled=True)):
    data_dir = os.path.join(os.getcwd(), folder)
    experiment_name = arch
    experiment = Experiment(api_key = "rI52i2RdTWgAjRD8MlZrdAhoS", project_name = "deep-net", parse_args=False)
    experiment.set_name(experiment_name)

    hparams=config_dict(arch)
    experiment.log_parameters(hparams)
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
     ######## Config for architectures ############
    config_inc = {

        "lr": tune.loguniform(1e-4, 1e-1),
        "mm":tune.choice([0.6,0.9,1.2]),
        "dp":tune.choice([0,0.9,0.995]),
        "wD":tune.choice([0,0.000008,0.00001,0.00003 ]),
        "depth":tune.choice([1,2,3,4,5]),
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size":tune.choice([48,64,96]),
        "opt": tune.choice(['adam','sgd', 'adadelta']),
        "b1": 0.9,
        "b2":0.999 ,
        "eps":tune.loguniform(1e-08 ,1e-04),
        "rho":0.9
    }
    config_vgg = {

        "lr": tune.loguniform(1e-4, 1e-1),
        "mm":tune.choice([0.6,0.9,1.2]),
        "dp":tune.choice([0,0.9,0.995]),
        "wD":tune.choice([0,0.000008,0.00001,0.00003 ]),
        "vgg_config":tune.choice(['A','B','D','E']),
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size":tune.choice([48,64,96]),
        "opt": tune.choice(['adam','sgd', 'adadelta']),
        "b1":0.9,
        "b2":0.999 ,
        "eps":tune.loguniform(1e-08 ,1e-04),
        "batch_norm": tune.choice([True,False]),
        "rho":0.9
    }
    config_alex = {

        "lr": tune.loguniform(1e-4, 1e-1),
        "mm":tune.choice([0.6,0.9,1.2]),
        "dp":tune.choice([0,0.9,0.995]),
        "wD":tune.choice([0,0.000008,0.00001,0.00003 ]),
        #"depth":tune.choice([1,2,3,4,5]),
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
         "batch_size":tune.choice([48,64,96]),
        "opt": tune.choice(['adam','sgd', 'adadelta']),
        "b1": 0.9,
        "b2":0.999 ,
        "eps":tune.loguniform(1e-08 ,1e-04),
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
        "opt":'adam',
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
    elif arch =='def':
       pass
    return config #config_

if __name__ == "__main__":
    main(num_epochs=40, folder="Dataset", arch='res')
    #main(num_samples=100, num_epochs=35, folder="Dataset_new", arch='alex', optim='asha')
    #main(num_samples=40, num_epochs=35, folder="Dataset", arch='alex', opt='pbt')
    #main(num_samples=40, num_epochs=35, folder="Dataset", arch='vgg', opt='pbt')

    
