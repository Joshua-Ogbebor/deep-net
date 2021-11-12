import torch
from typing import List, Any
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

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
        #print(logits,loss,y)
        mat=confusion_matrix_t(label=y, l=logits, num_classes=self.num_classes)
        #print(mat)
        CM=self.logger.experiment.create_confusion_matrix()
        mat2=CM.compute_matrix(y_true=y.cpu(), y_predicted=logits.cpu())
        return {'val_loss_per_step': loss, 'val_accuracy_per_step': acc, 'mat_t':mat, 'c_mat':mat2}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack(
            [x['train_accuracy_per_step'] for x in outputs]).mean()
        if self.trainer.is_global_zero:
            self.logger.experiment.log_metric('train_loss', avg_loss, step=self.current_epoch)
            self.logger.experiment.log_metric('train_accuracy', avg_acc, step=self.current_epoch)
            #self.log("val_loss", avg_loss,rank_zero_only=True)

    def validation_epoch_end(self, outputs, **kwargs):
        avg_loss = torch.stack(
            [x['val_loss_per_step'] for x in outputs]).mean()
        avg_acc = torch.stack(
            [x['val_accuracy_per_step'] for x in outputs]).mean()
        #print(avg_loss)
        
        mat_t = torch.stack(
            [x['mat_t'] for x in outputs])#
        mat_t=torch.sum(mat_t,0)        
        #print(mat_t)
        self.logger.experiment.log_metrics(classification_metrics(mat=mat_t, num_classes=self.num_classes, **kwargs))
        if self.trainer.is_global_zero:
            self.logger.experiment.log_metric('val_loss', avg_loss, step=self.current_epoch)
            self.logger.experiment.log_metric('val_accuracy', avg_acc, step=self.current_epoch)
            self.log("val_loss", avg_loss,rank_zero_only=True,logger=False)
        
        
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x['val_loss_per_step'] for x in outputs]).mean()
        avg_acc = torch.stack(
            [x['val_accuracy_per_step'] for x in outputs]).mean()
        mat_t = torch.stack(
            [x['mat_t'] for x in outputs])#
        mat_t=torch.sum(mat_t,0)
        cmat = torch.stack(
            [x['c-mat'] for x in outputs])
        cmat=torch.sum(cmat,0)
        self.logger.experiment.log_confusion_matrix(cmat, title='comet_stacked')
        mat_t_C, metrics=classification_metrics(mat=mat_t, num_classes=self.num_classes, test=True)
        self.logger.experiment.log_confusion_matrix(mat_t_C, title='me_stacked')
        self.logger.experiment.log_metrics(metrics)
        if self.trainer.is_global_zero:
            self.logger.experiment.log_metric('test_loss', avg_loss, step=self.current_epoch)
            self.logger.experiment.log_metric('test_accuracy', avg_acc, step=self.current_epoch)
            self.log("val_loss", avg_loss,rank_zero_only=True,logger=False)
    
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
    precision, recall,c_mat,acc=sum_and_find_dist(mat=mat, num_classes=num_classes)
    F1=2*precision*recall/(precision+recall)
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
    acc_count=0
    #print(mat)
    sum_mat=torch.sum(mat,1)
    c_mat=torch.transpose(mat,0,1)
    sum_c_mat=torch.sum(c_mat,1)
    
    for ind in range (num_classes):
        precision[ind]=mat[ind,ind]/sum_mat[ind]
        recall[ind]=mat[ind,ind]/sum_c_mat[ind]
        acc_count+=mat[ind,ind]
    return precision,recall,c_mat, acc_count/sum(sum_mat)


