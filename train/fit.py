import pytorch_lightning as pl
import sys
sys.path.append("..")
from data import datamodule
from pytorch_lightning.plugins import DDPPlugin
#from pytorch_lightning.loggers import CometLogger
import torch, os
from pytorch_lightning.callbacks import ModelCheckpoint
from funcs import save_model, config_s,ckpt_dir, arch_to_model, arch_to_model_ref


def train_fn(model_arch, data_dir=os.path.join(os.getcwd(), "Dataset"),logger=None, num_epochs=60, num_gpus=1):
   dm = datamodule.ImgData(num_workers=4, batch_size=config_s(arch=model_arch)["batch_size"],data_dir=data_dir)
   model = arch_to_model(arch=model_arch, num_classes=dm.num_classes)
   
   #comet_logger=CometLogger(api_key = key(), experiment_name =model_arch, project_name = "deep-net")
   
   #comet_logger.log_graph(model) 
   checkpoint_callback = ModelCheckpoint(
     monitor='val_loss',
     mode='min',
     save_top_k=1,
     dirpath=ckpt_dir(model_arch),
     filename=model_arch+'-{epoch:02d}-{val_loss:.2f}'
     )

   trainer = pl.Trainer(
      max_epochs=num_epochs,
      gpus=1,
      callbacks=[checkpoint_callback],
      logger = logger,
      progress_bar_refresh_rate=0,
      accelerator='ddp',
      plugins=DDPPlugin(find_unused_parameters=False),
      deterministic=True)


   trainer.fit(model, dm)
   save_model(model=model,arch=model_arch,unique_id='full_train')
   model=arch_to_model_ref(model_arch)(**config_s(model_arch),num_classes=dm.num_classes).load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)
   trainer.test(ckpt_path="best", dataloaders=test_dataloaders)
   #save_model(model=model,arch=model_arch,unique_id='full_train')
   
   return  model#arch_to_model_ref(model_arch)(**config_s(model_arch),num_classes=dm.num_classes).load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)#,config,  dm.num_classes)
