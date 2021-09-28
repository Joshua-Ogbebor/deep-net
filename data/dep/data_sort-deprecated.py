import os
import random
import shutil


class data_sort:
    
    def __init__(self, directory = "Dataset"):
        
        '''
        param : The directory name that contains the classes of files
        
        '''

        cwd = os.getcwd()
        self.dataset_dir = os.path.join(cwd, directory)  
        #self.Num_classes=0
        #self.mean_nums=0
        #self.std_nums=0

        
    def create_training_data(self):
        """ This function creates the Training and Validation dataset for data saved in a folder in the 
        current working directory. The Dataset folder is expected to contain labelled images in subfolders
        The number of subfolders indicates the number of classes while the name of the subfolders are the classnames """
        #base_dir = cwd


        #Check if already sorted
        list_dirs_init= [f.path for f in os.scandir(self.dataset_dir) if f.is_dir()]
        i=0
        for dirs in list_dirs_init:
            list_dirs_init[i] = dirs.split('/')[-1]
            i=1+i

        if not list_dirs_init== ['train','val']:  ## train only nko?
            print ("..... ")
            list_dirs=[f.path for f in os.scandir(self.dataset_dir) if f.is_dir()]    
            Num_classes = len(list_dirs)
        else:
            print ("Previously created")
            list_dirs=[f.path for f in os.scandir(f"{self.dataset_dir}/{list_dirs_init[0]}") if f.is_dir()]
            Num_classes = len(list_dirs)

        for dirs in list_dirs: 
            class_name = dirs.split('/')[-1]
            train_dir = f"{self.dataset_dir}/train/{class_name}"                    # Create folder names
            val_dir = f"{self.dataset_dir}/val/{class_name}" 

            if not list_dirs_init == ['train','val']:  ## if element of listdirs_init==train_dir    OOOORRR   list_dirs_init [0]== 'train':,,,,,,else:print(error?)
                # Training set
                shutil.move(dirs, train_dir)
                # Validation set
                os.makedirs(val_dir)    ## if element of listdirs_init==val_dir
                for f in os.listdir(train_dir):
                    if random.random() > 0.80: #generates a random float uniformly in the semi-open range [0.0, 1.0)
                        shutil.move(f'{train_dir}/{f}', val_dir)
            num_T=len(os.listdir(train_dir))
            num_V=len(os.listdir(val_dir)) 

            # Print number of classes                

            print("{:s}: {:.0f} training images and {:.0f} validation images ({:.1f}%) ".format(class_name, num_T, num_V, 100*num_V/(num_T+num_V)))   
        print("There are {:.0f} Classes.".format(Num_classes))
        return Num_classes
    
        ## Create the data loader
    
#class data_pre_load:

    def do_prep(self,sum=False):
        Num_classes=self.create_training_data()

        #data_loader, Num_images, classes =self.load_dataset()
        #if not self.std_nums,self.mean_nums == 0:  this should be some external file...
        #    print("mean and std dev previously calculated")
        #else:
        #if sum:
            #self.mean_nums, self.std_nums=self.compute_mean_n_deviation(data_loader)
        return Num_classes,self.dataset_dir#,self.mean_nums,self.std_nums
