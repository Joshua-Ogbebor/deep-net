import os, copy, comet_ml
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from train import fit


experiment_name = "train"
experiment = Experiment(api_key = "rI52i2RdTWgAjRD8MlZrdAhoS", project_name = "deep-net", parse_args=False)
experiment.set_name(experiment_name)

#comet_logger=CometLogger(api_key = "rI52i2RdTWgAjRD8MlZrdAhoS", experiment_name = "train",project_name = "deep-net")

def main (num_samples=40, num_epochs=50, folder="Dataset", arch='inc', experiment=Experiment(api_key='dummy_key', disabled=True)):
    data_dir = os.path.join(os.getcwd(), folder)
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

        "lr":tune.loguniform(1e-4, 1e-1),
        "mm":tune.choice([0.6,0.9,1.2]),
        "dp":tune.choice([0,0.9,0.995]),
        "wD":tune.choice([0,0.000008,0.00001,0.00003 ]),
        "bloc_1":tune.choice([64,128,256,512]),
        "bloc_2":tune.choice([64,128,256,512]),
        "bloc_3":tune.choice([64,128,256,512]),
        "bloc_4":tune.choice([64,128,256,512]),
        #"bloc_2":0,
        #"bloc_3":0,
        #"bloc_4":0,
        "depth_1":tune.choice([1,2,3]),
        "depth_2":tune.choice([1,2,3]),
        #"depth_3":tune.choice([1,2,3]),
        #"depth_4":tune.choice([1,2,3]),
        #"depth_2":0,
        "depth_3":0,
        "depth_4":0,
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size":tune.choice([96, 64, 128]),
        "opt":tune.choice(['adam','sgd', 'adadelta']),
        "b1":tune.choice([0.9]),
        "b2":tune.choice([0.999]),
        "eps":tune.loguniform(1e-08,1e-04),
        "rho":tune.choice([0.9])
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
    main(num_samples=100, num_epochs=35, folder="Dataset_new", arch='alex')
    #main(num_samples=100, num_epochs=35, folder="Dataset_new", arch='alex', optim='asha')
    #main(num_samples=40, num_epochs=35, folder="Dataset", arch='alex', opt='pbt')
    #main(num_samples=40, num_epochs=35, folder="Dataset", arch='vgg', opt='pbt')

    
