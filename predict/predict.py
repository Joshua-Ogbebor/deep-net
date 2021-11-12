from typing import Union, Callable,Tuple, Optional,List,Dict,Any
import os, cv2, torch
import numpy as np
import matplotlib as plt
from torchvision import transforms
from PIL import Image
from model.residual_net import Resnet_Classifier as res_net
from model.alex_net import AlexNet as alex_net
from model.inception_net import Googlenet_Classifier as google_net
from private.comet_key import key


class err_out(Exception):
    def __init__(*args,**kwargs:Any)->None:
        """
            Throws out error and prints all arguments to screen
        """
        [print(key,':',value) for key, value in kwargs.items()]
        #print(msg)#parse kw arg

def resize_full_img(input_img:List[List[List[int]]], 
            reverting:bool=False, 
            initial_h:int=None, 
            initial_w:int=None, 
            win_height:int=256, 
            win_width:int=256
         )->Optional[List[List[List[int]]]]:
    """
        Called before sliding window and after sliding window + prediction + masking
        Resizes image to multiple of win_height, and win_width for sliding window or reverts to initial size after sliding window + prediction + masking
        Parameters: input_img : read image from cv2
                    reverting: resizing to new size based on  or previous size. If True, reverting to previous size is the case. initial_h and initial_w must not be None
                    initial_h: initial image height if reverting to previous size
                    initial_w: initial image weight if reverting to previous size
                    win_height: height of sliding window if preparing for sliding window
                    win_width: width of sliding window if preparing for sliding window

        returns resized image

    """
    if reverting:
        if not (initial_h and initial_w):
            raise err_out(initial_h=initial_h, initial_w=initial_w, msg='not specified')
        else:
            input_img=cv2.resize(input_img, (initial_w,initial_h), interpolation = cv2.INTER_AREA)

    else:
        imgheight, imgwidth, _ = input_img.shape
        imH=win_height*round(imgheight/win_height)               #resize image
        imW=win_width*round(imgwidth/win_width)                   
        input_img=cv2.resize(input_img, (imW,imH), interpolation = cv2.INTER_AREA)
    return input_img


def save_img(input_img:List[List[List[int]]], folder_name:str, k:int=None)->None:
    """
        Saves image input_img in specified folder_name using a unique identifier k.
        will overwrite!
        Parameters:
                    input_img: image to be saved. output from im.read()
                    folder-name: relative folder name. will be created if non existent
                    k: unique identifier to save image
        returns None
    """
    if not os.path.exists(os.path.join('saved_patches', folder_name)):
        os.makedirs(os.path.join('saved_patches', folder_name))
    filename = os.path.join('saved_patches', folder_name,'img_{}.png'.format(k))
    if os.path.exists(filename):
        print("{}: Image patch with the same filename already exists. Overwriting...".format('img_{}.png'.format(k)))
    cv2.imwrite(filename, input_img)




def predict(model:Union[alex_net,res_net,google_net], 
                idx_to_class:Callable[[],Dict[int,str]], 
                image_tensor:torch.Tensor, 
                channels:int, height:int, width:int, 
                batch:int=1)->str: #Callable for idx2clas
    """
    Predicts on image patches, test_image_tensor of size:(height, width) using model and idx_to_class dictionary for class identification
    parameters:
            model: inference model
            idx_to_class: Dictionary mapping value to class
            test_image_tensor: image tensor
            channels: image property
            height: image property
            width: image property  
    returns True if cracked
    """
     # it uses the model to predict on test_image...
    
    if torch.cuda.is_available():                       # checks if gpu available
        image_tensor = image_tensor.view(batch, channels, height, width)#.cuda()   #hard_code
    else:
        image_tensor = image_tensor.view(batch, channels, height, width)      #hard_code
    
    with torch.no_grad():
        model.eval()
        out = model(image_tensor)                  # Model outputs log probabilities... computes the output of the model
        pred, ind = torch.max(out,1)                    #  computes the probability of each classes. #... choose the top class. That is, the class with highest probability
        class_name = idx_to_class[ind.item()]
    return class_name, class_name [0]== 'C'             # HARD CODE HERE ASSUMES THAT POSITIVE STARTS WITH 'C'


def get_folder_name(input_img_label:str)->str:
    """
        returns a folder name from an image name.
    """
    file, ext = os.path.splitext(input_img_label)
    image_name = file.split('/')[-1]
    return 'saved_' + image_name


def load_model(path:str='./cifar__mODEL_net.pth')-> Union[alex_net,res_net,google_net]:
    model = torch.load(path)
    return model.eval()


def show_image(image:List[List[List[int]]])->None:
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def global_transforms():
    return {'train': transforms.Compose([
        #torchvision.transforms.Grayscale
        transforms.Resize(size=256),
        #transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=<InterpolationMode.BILINEAR: 'bilinear'>, fill=0)
        #transforms.ColorJitter(brightness=0.1, contrast=0.2),
        #transforms.Normalize(mean_nums, std_nums)
        transforms.ToTensor()
        
]), 'val': transforms.Compose([
        #torchvision.transforms.Grayscale
        #transforms.Resize(256),
        #transforms.CenterCrop(227),
        #transforms.Normalize(mean_nums, std_nums) 
        transforms.ToTensor()
       ]),
    }

def transform_img(image:List[List[List[int]]], mode:str='val')->List[List[List[int]]]:
    transformation=global_transforms()[mode]
    return transformation(image)


def global_idx2class()->Dict[int,str]:
    return {0:'Cracked_asp', 1:'Cracked_con', 2:'Uncracked_asp', 3:'Uncracked_con'}


def img_from_array(image:List[List[List[int]]])->Image:
    """
    array to image
            image:
    
    """
    return Image.fromarray(image)


def mask_image(image:List[List[List[int]]])->List[List[List[int]]]:
    """
    Parameters: 
        image:  cracked image
    """
    color = (0,0, 255)                            #Blue
    b = np.zeros_like(image, dtype=np.uint8)
    #print(b,image,color)
    b[:] = color
    #print(b)
    return cv2.addWeighted(image, 0.9, b, 0.1, 0) 


def sub_image_pred(input_img:List[List[List[int]]], 
        base_model:Union[alex_net,res_net,google_net],     
        save_path:str=None, 
        win_height:int=256, 
        win_width:int=256, 
        save_crops:bool = False,
        **kwargs) -> List[List[List[int]]]:
    """
    Parameters: 
        input_img: Fully cracked Image
        basemodel: prediction model
        input_img_label:
        win_height: sliding window size
        win_width: sliding window size
        save_crops: False by default
    """
    init_height, init_width, channels = input_img.shape             # Get initial image properties
    resized_img=resize_full_img(input_img=input_img)                # Process
    imgheight,imgwidth, _=resized_img.shape                  # Get post processed shape
    folder_name=get_folder_name(input_img_label=save_path) if save_crops and save_path else None
    
    #for looop
    #img_=patch_pred_assemble(resized_img=resized_img)
    
    output_image=np.zeros_like(resized_img)        
    k=0
    
    for i in range(0,imgheight,win_height):
        for j in range(0,imgwidth,win_width):
            img =resized_img[i:i+win_height, j:j+win_width]
            img_t = transform_img(img_from_array(image=img))

            is_cracked = predict(model=base_model,idx_to_class=global_idx2class(), image_tensor=img_t, channels=channels,height=win_height,width=win_width)
            if is_cracked:
                img=mask_image(image=img) ## Put predicted class on the image
            if save_crops:
                save_img(input_img=img, folder_name=folder_name,k=k)
            k+=1
            output_image[i:i+win_height, j:j+win_width,:] = img
 
    return resize_full_img(input_img= output_image, reverting=True,initial_h=init_height,initial_w=init_width)

def arch_to_model (model_arch:str):
   return {'inc':google_net,
    'res':res_net,
    'alex': alex_net ,
    #'vgg': VGG
   }[model_arch]

def main(input_img:Optional[Any]=None, 
            input_img_path:Optional[str]=None, 
            base_model_path:Optional[str]=None, 
            base_model_dict_path:Optional[str]=None,
            arch:Optional[str]='generic',
            base_model:Optional[Union[alex_net, res_net, google_net]]=None, 
            save_crops:bool=False, save_path:str=None, height:int=256, width:int=256,
            )-> List[List[List[int]]]:
    """
    Predict on new image. Required arguments:
                                base_model_path:PATH to model
                                                 or base_model: model
                                                 or base_model_dict_path: PATH to model dict and arch: res, inc or alex for 
                                input_img_path: PATH to input image
                                                or input_img: image
                        Optionally specify height: sliding window height
                                           width: sliding window width

    """

    if (base_model_path or base_model or (base_model_dict_path and arch)) and (input_img_path or input_img) is None:
        raise err_out(base_model_path=base_model_path if base_model_path else None,
                            base_model= "specified" if base_model else None,
                            base_model_dict_path=base_model_path if base_model_dict_path else None,
                            arch = arch if arch else None,
                            input_img_path =input_img_path if input_img_path else None,
                            input_img ="passed" if input_img else None,
                            msg="Required arguments:    base_model_path:PATH to model      or base_model: model        or base_model_dict_path: PATH to model dict and arch: res, inc or alex for       input_img_path: PATH to input image   or input_img: image Optionally specify height: sliding window height     width: sliding window width")
    if not base_model:
        base_model=torch.load(base_model_path) if base_model_path else arch_to_model(arch)().load_state_dict(torch.load(base_model_path))

    input_img=input_img if input_img is not None else cv2.imread(input_img_path) 
    
    return sub_image_pred(input_img=input_img,base_model=base_model,save_crops=save_crops, save_path=save_path,win_height=height, win_width=width)



