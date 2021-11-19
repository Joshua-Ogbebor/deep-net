import  torch
from typing import List, Any
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
#from comet_ml import ConfusionMatrix


class deep_net(pl.LightningModule):
    """
        define self.losss
                self.accuracy
                self.forward

    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x.float())
        y=y.long()
        loss=self.losss(logits,y)
        acc = self.accuracy(logits, y)
        return {'loss': loss, 'train_accuracy_per_step': acc}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x.float())
        y=y.long()
        loss=self.losss(logits,y)
        acc = self.accuracy(logits, y)
        mat=confusion_matrix_t(label=y, l=logits, num_classes=self.num_classes)
        #CM=self.logger.experiment.create_confusion_matrix()
        #mat2=self.logger.experiment.create_confusion_matrix(y_true=y.cpu(), y_predicted=logits.cpu())
        return {'val_loss_per_step': loss, 'val_accuracy_per_step': acc, 'mat_t':mat}

    def test_step(self,test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x.float())
        y=y.long()
        loss=self.losss(logits,y)
        acc = self.accuracy(logits, y)
        mat=confusion_matrix_t(label=y, l=logits, num_classes=self.num_classes)
        #CM=self.logger.experiment.create_confusion_matrix()
        #mat2=self.logger.experiment.create_confusion_matrix(y_true=y.cpu(), y_predicted=logits.cpu())
        # 
        return {'logits':logits, 'labels':y, 'loss_per_step': loss, 'accuracy_per_step': acc, 'mat_t':mat}

    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack(
            [x['train_accuracy_per_step'] for x in outputs]).mean()
        with self.logger.experiment.train():
            self.logger.experiment.log_metric('loss', avg_loss, step=self.current_epoch)
            self.logger.experiment.log_metric('accuracy', avg_acc, step=self.current_epoch)
            #self.log("val_loss", avg_loss,rank_zero_only=True)

    def validation_epoch_end(self, outputs, **kwargs):
        avg_loss = torch.stack(
            [x['val_loss_per_step'] for x in outputs]).mean()
        avg_acc = torch.stack(
            [x['val_accuracy_per_step'] for x in outputs]).mean()
        mat_t = torch.stack(
            [x['mat_t'] for x in outputs])#
        mat_t=torch.sum(mat_t,0)        
        #print(mat_t)
        self.logger.experiment.log_metrics(classification_metrics(mat=mat_t, num_classes=self.num_classes, **kwargs),step=self.current_epoch)
        with self.logger.experiment.validate():
            self.logger.experiment.log_metric('loss', avg_loss, step=self.current_epoch)
            self.logger.experiment.log_metric('accuracy', avg_acc, step=self.current_epoch)
            self.log("val_loss", avg_loss,rank_zero_only=True,logger=False)
        
        
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x['loss_per_step'] for x in outputs]).mean()
        avg_acc = torch.stack(
            [x['accuracy_per_step'] for x in outputs]).mean()
        mat_t = torch.stack(
            [x['mat_t'] for x in outputs])#
        mat_t=torch.sum(mat_t,0)
        
        y = torch.cat(
            [x['labels'] for x in outputs])
        l = torch.cat(
            [x['logits'] for x in outputs])
        print(y,l)
        mat_t_C, metrics=classification_metrics(mat=mat_t, num_classes=self.num_classes, test=True)
        mat2=self.logger.experiment.create_confusion_matrix(y_true=y.cpu(), y_predicted=l.cpu())
        self.logger.experiment.log_confusion_matrix(mat_t_C, title='me_stacked')
        self.logger.experiment.log_metrics(metrics)
        with self.logger.experiment.test():
            self.logger.experiment.log_confusion_matrix(mat2, title='comet_stacked') 
            self.logger.experiment.log_confusion_matrix(mat_t_C, title='me_stacked')
            self.logger.experiment.log_other('me_mat', mat_t_C)
            self.logger.experiment.log_metrics(metrics)
            self.logger.experiment.log_metric('loss', avg_loss)#, step=self.current_epoch)
            self.logger.experiment.log_metric('accuracy', avg_acc)#, step=self.current_epoch)
            #self.log("test_loss", avg_loss,rank_zero_only=True,logger=False)
    
    def configure_optimizers(self):
        optim={
            'sgd':torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, dampening=self.damp, weight_decay=self.wghtDcay),
            'adam':torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.wghtDcay),
            'adadelta':torch.optim.Adadelta(self.parameters(), lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.wghtDcay)
        }
        return optim[self.optim_name]

def decoder (l,num_classes,onehotcoded:bool=False):
    if not onehotcoded:
        return torch.max(l,1) 
    else:
        max_,max_ind=torch.max(l,1)
        #print(max_,max_ind,torch.eye(num_classes)[max_ind].tolist())
        return max_, torch.eye(num_classes)[max_ind]

def confusion_matrix_t (label, l, num_classes)->List[List[int]]:
    #print(label,l)
    mat=torch.zeros([num_classes,num_classes], dtype=torch.float)
    max_, max_indices = decoder(l=l,num_classes=num_classes,onehotcoded=True)
    for index, elements in enumerate (label):
        #print( )
        mat[elements]+=max_indices[index]
    return mat 

def act_fn_by_name(act_fn_name):
    return{
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "linear": nn.Identity
}[act_fn_name]


def classification_metrics (num_classes:int,label=None, l=None,mat=None, test:bool=False):
    mat=confusion_matrix_t(label=label,l=l,num_classes=num_classes) if mat==None else mat
    precision, recall,F1,c_mat,acc=sum_and_find_dist(mat=mat, num_classes=num_classes)
    #F1=2*precision*recall/(precision+recall)
    #F1_c= 1/(0.5*(1/precision+1/recall)
    metrics_dict={}
    for i in range (num_classes):
        metrics_dict[f'Class {i} Recall']=recall[i]
        metrics_dict[f'Class {i} Precision']=precision[i]
        #metrics_dict[f'Class {i} F1_c']=F1_c[i]
        metrics_dict[f'Class {i} F1 Score']=F1[i]  
    metrics_dict['acc']=acc
    #recall,_=sum_and_find_dist(con_mat, num_classes=num_classes)
    if test:
        return c_mat, metrics_dict
    else:
        return metrics_dict

def sum_and_find_dist(mat:List[List[int]], num_classes:int):
    precision=torch.zeros([num_classes],dtype=torch.float)
    recall=torch.zeros([num_classes],dtype=torch.float)
    F1=torch.zeros([num_classes],dtype=torch.float)
    acc_count=0
    #print(mat)
    sum_mat=torch.sum(mat,1)
    c_mat=torch.transpose(mat,0,1)
    sum_c_mat=torch.sum(c_mat,1)
    
    for ind in range (num_classes):
        precision[ind]=mat[ind,ind]/sum_mat[ind] if sum_mat[ind] !=0 else 0
        recall[ind]=mat[ind,ind]/sum_c_mat[ind] if sum_c_mat[ind] !=0 else 0
        F1[ind]=2*precision[ind]*recall[ind]/(precision[ind]+recall[ind]) if precision[ind]+recall[ind]!=0 else 0
        acc_count+=mat[ind,ind]
    acc=acc_count/sum(sum_mat)
    return precision,recall,F1,c_mat, acc


