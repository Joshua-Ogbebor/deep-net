import os, comet_ml
from pytorch_lightning.loggers import CometLogger
from train import fit
import pytorch_lightning as pl
from funcs import save_model, load_model
from predict import predict
#from comet_ml import Experiment
from private.comet_key import key

def main (num_epochs=50, folder="Dataset", arch='inc'):
    data_dir = os.path.join(os.getcwd(), folder)
    pl.seed_everything(42, workers=True)    
    comet_logger=CometLogger(api_key = key(), experiment_name =arch, project_name = "deep-net")
    #________________________________train_____________________________________
    model=fit.train_fn(
        model_arch=arch,
        num_epochs=num_epochs,
        logger=comet_logger,
        data_dir = data_dir)
    save_model(model=model, arch=arch)
    comet_logger.experiment.log_graph(model)

    #______________________________test________________________________________

    image_folder=os.path.join(os.getcwd(), 'test-imgs')
    for image_path in os.listdir(image_folder):
        comet_logger.experiment.log_image(cv2.imread(os.path.join(image_folder,image_path)))
        comet_logger.experiment.log_image(predict.main(input_img_path =os.path.join(image_folder,image_path),base_model=model))
        comet_logger.experiment.log_image(predict.main(input_img_path =os.path.join(image_folder,image_path),base_model=load_model(arch=arch, unique_id='full_train')))

if __name__ == "__main__":
    main(num_epochs=41, folder="Dataset", arch='alex')
    #main(num_epochs=50, folder="Dataset", arch='vgg')
    #main(num_epochs=50, folder="Dataset", arch='inc')
    main(num_epochs=41, folder="Dataset", arch='res')
    

    
