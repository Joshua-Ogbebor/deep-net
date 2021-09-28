import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import cv2
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from ..data import data_load


# Convert model to be used on GPU do this before optimizer



class trainer():
    def __init__(self,model,device,criterion= nn.BCEWithLogitsLoss(),lr=0.01, momentum=0.9, dampening=0, weight_decay=0.0001,step_size=10, gamma=0.5):
        self.device = device
        self.model = model.to(self.device)
        self.dataloaders, self.dataset_sizes, self.class_names=data_load.data_load().do_prep()
        self.idx_to_class=self.idx2class()
        
        
        # Define Optimizer and Loss Function
        # learning rate decay.  change the learning rate dynamically- the schedular 
        #@ IN THE FIRST epoch, the learning rate was 0.01, then the learning rate in epoch 10 would be 0.05. 
        # torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
        # Decay LR by a factor of 0.5 every 10 epochs
        #self.exp_lr_scheduler = scheduler
        self.optimizer=optim.SGD(self.model.parameters(), lr, momentum, dampening, weight_decay)
        self.exp_lr_scheduler=lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        self.criterion = criterion.cuda() if torch.cuda.is_available() else criterion            

    
    def idx2class(self):
        idx=0
        idx_to_class=dict()
        for classes in self.class_names:
            idx_to_class[idx] = classes
            idx=idx+1
        print (idx_to_class)
        return idx_to_class

    def get_num_correct(self, pred, labels):
        pred_c, ind_pred = torch.max(pred,1)
        labels_c, ind_label = torch.max(labels,1)

        return ind_pred.eq(ind_label).sum().item()



    def train_model(self,num_epochs=60):# model=self.model, criterion=self.criterion, optimizer=self.optimizer, scheduler=self.scheduler, ):
        writer=SummaryWriter()
        since = time.time()

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        highest_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:                                          ## where is phase from
                if phase == 'train':
                    #scheduler.step()                                                             #######
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                current_loss = 0.0
                current_corrects = 0

                #training starts...
                print('Iterating through data...')
                # 
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # Zero the accumulated gradients for each step,
                    self.optimizer.zero_grad()

                    # Time to carry out the forward training poss
                    with torch.set_grad_enabled(phase == 'train'): #This is false for validation
                        outputs = self.model(inputs)
                        #outputs=outputs.squeeze(3)
                        #outputs=outputs.squeeze(2)
                        #print(outputs.size())

                        #the loss...
                        loss = self.criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward() # this is the back-propragation step. 
                            #writer.add_scalar('Loss/train', loss.item(), epoch)
                            self.optimizer.step()
                            # optimizer updates the model paramters...

                    # We want variables to hold the loss statistics
                    #print(inputs.size(0),dataset_sizes[phase])
                    # combine our model losses and compute the wrongly classified examples
                    current_loss += loss.item() * inputs.size(0)
                    current_corrects += self.get_num_correct(outputs,labels)
                # compute the loss and accuracy for each of our epoch
                if phase == 'train':
                    self.exp_lr_scheduler.step() 
                epoch_loss = (1.0*current_loss) / self.dataset_sizes[phase]
                epoch_acc = (1.0* current_corrects)/ self.dataset_sizes[phase]

                if phase == 'val':
                    #writer.add_scalar("Val Loss", current_loss, epoch)
                    Val_epoch_loss=epoch_loss
                    Val_epoch_acc=epoch_acc
                    writer.add_scalar("Validation Loss", epoch_loss, epoch)
                    writer.add_scalar("Validation Accuracy", epoch_acc,epoch)#total_correct/ len(train_set), epoch)
                if phase == 'train':
                    #writer.add_scalar("Train Loss", current_loss, epoch)
                    Trn_epoch_loss=epoch_loss
                    Trn_epoch_acc=epoch_acc
                    writer.add_scalar("Training Loss", epoch_loss, epoch)
                    writer.add_scalar("Training Accuracy", epoch_acc, epoch)#total_correct/ len(train_set), epoch)


                #writer.add_scalars('Compact_Acc', {"Training_Accuracy": Trn_epoch_acc,
                #                        "Validation_Accuracy": val_epoch_acc}, epoch)
                #writer.add_scalars('Compact_Loss', {"Training_Loss": Trn_epoch_loss,
                #                        "Validation_Loss": val_epoch_loss}, epoch)
                #tb.add_histogram("conv1.bias", model.conv1.bias, epoch)
                #tb.add_histogram("conv1.weight", model.conv1.weight, epoch)
                #tb.add_histogram("conv2.bias", model.conv2.bias, epoch)
                #tb.add_histogram("conv2.weight", model.conv2.weight, epoch)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # Make a copy of the model if the accuracy on the validation set has improved
                if phase == 'val' and epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    #SAVE IT
                #if ......early stopping

            print()

        time_since = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_since // 60, time_since % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        #layout = {"Accuracy":{"Acc":["Margin",["Training Accuracy","Validation Accuracy"]]}, "Loss":{"L":["Margin",["Training Loss","Validation Loss"]]}}
        #writer.add_scalars
        writer.close()
        # Now we'll load in the best model weights and return it
        self.model.load_state_dict(self.best_model_wts)
        PATH_dICT = './cifar_dICT_net.pth'######################################33
        PATH_mODEL = './cifar__mODEL_net.pth'##########################################333333333333

        torch.save(self.model.state_dict(), PATH_dICT)
        torch.save(self.model, PATH_mODEL)

        return self.model



    def visualize_model(self, num_images=20):
        was_training = model.training
        self.model.eval()
        images_handeled = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                val, ind =torch.max(output,1)
                #print(val,ind)



                for j in range(inputs.size()[0]):
                    images_handeled += 1
                    class_name = self.idx_to_class[ind.item()]
                    #class_name = idx_to_class[ind.squeeze(1)[j].item()]
                    ax = plt.subplot(num_images//2, 2, images_handeled)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(class_name))
                    imshow(inputs.cpu().data[j])

                    if images_handeled == num_images:
                        self.model.train(mode=was_training)
                        return
        model.train(mode=was_training)