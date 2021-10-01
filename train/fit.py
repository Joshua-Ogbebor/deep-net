import pytorch_lightning as pl
import sys
sys.path.append("..")
from model import residual_net, inception_net, vgg_net, alex_net
from data import datamodule
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch, os


def model_name (model_arch):
   return {'inc':inception_net.Googlenet_Classifier,
    'res':residual_net.Resnet_Classifier,
    'alex': alex_net.AlexNet ,
    'vgg': vgg_net.VGG
   }[model_arch]

def train_fn(config, model_arch, data_dir=os.path.join(os.getcwd(), "Dataset") , num_epochs=60, num_gpus=0, checkpoint_dir=None):
   dm = datamodule.ImgData(num_workers=8, batch_size=config["batch_size"],data_dir=data_dir)
   model = model_name(model_arch)(config,  dm.num_classes)
   
   comet_logger=CometLogger(api_key = "", experiment_name =model_arch, project_name = "deep-net")
   comet_logger.log_hyperparams(config)
   comet_logger.log_graph(model) 
   checkpoint_callback = ModelCheckpoint(
     monitor='test_loss',
     dirpath='model',
     filename='arch-{epoch:02d}-{test_loss:.2f}'
     )

   trainer = pl.Trainer(
      max_epochs=num_epochs,
      gpus=num_gpus,
      callbacks=[checkpoint_callback],
      logger = comet_logger,
      log_every_n_steps=5000,
      progress_bar_refresh_rate=0,
      accelerator='ddp',
      plugins=DDPPlugin(find_unused_parameters=False),
      deterministic=True)

   trainer.fit(model, dm)
