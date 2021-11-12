import os, cv2, comet_ml
from predict.predict import main
from comet_ml import Experiment
from private.comet_key import key
#from pytorch_lightning.loggers import CometLogger
image_folder=os.path.join(os.getcwd(), 'test-imgs')
comet_logger=Experiment(api_key = key(),project_name = "deep-predict")

if __name__ == '__main__':    
    for image_path in os.listdir(image_folder):
        comet_logger.log_image(cv2.imread(os.path.join(image_folder,image_path)))
        comet_logger.log_image(main(input_img_path =os.path.join(image_folder,image_path),base_model_path="trained-models/entire_alex_model.pth"))

    comet_logger.end()
