import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys


def show_fig(loss_filename,Loss_list,items,label='loss'):
    plt.figure(figsize=(10,6))
    plt.suptitle(loss_filename)
    train,val,train_face,val_face,train_ldm,val_ldm=items
    if(train in Loss_list.columns and val in Loss_list.columns):
        x=range(0,len(Loss_list[train]))
        y1=Loss_list[train]
        y2=Loss_list[val]        
        plt.subplot(221)
        if label=='loss':
            label1=train+' min:'+str(round(y1.min(),5))
            label2=val+' min:'+str(round(y2.min(),5))
        else:
            label1=train+' max:'+str(round(y1.max(),5))
            label2=val+' max:'+str(round(y2.max(),5))
        plt.semilogy(x,y1,c='r',marker='o',label=label1)
        plt.semilogy(x,y2,c='b',marker='o',label=label2)
        plt.ylabel('loss')
        plt.title(train+' and '+ val)
        plt.legend()
        
    if(train_face in Loss_list.columns and val_face in Loss_list.columns):
        x=range(0,len(Loss_list[train_face]))
        y1=Loss_list[train_face]
        y2=Loss_list[val_face]
        plt.subplot(223)
        if label=='loss':
            label1=train_face+' min:'+str(round(y1.min(),5))
            label2=val_face+' min:'+str(round(y2.min(),5))
        else:
            label1=train_face+' max:'+str(round(y1.max(),5))
            label2=val_face+' max:'+str(round(y2.max(),5))
        plt.semilogy(x,y1,c='r',marker='o',label=label1)
        plt.semilogy(x,y2,c='b',marker='o',label=label2)
        plt.xlabel('epoch times')
        plt.ylabel('loss')
        plt.title(train_face+' and '+ val_face)
        plt.legend()
        
    if(train_ldm in Loss_list.columns and val_ldm in Loss_list.columns):
        x=range(0,len(Loss_list[train_ldm]))
        y1=Loss_list[train_ldm]
        y2=Loss_list[val_ldm]
        plt.subplot(224)
        if label=='loss':
            label1=train_ldm+' min:'+str(round(y1.min(),5))
            label2=val_ldm+' min:'+str(round(y2.min(),5))
        else:
            label1=train_ldm+' max:'+str(round(y1.max(),5))
            label2=val_ldm+' max:'+str(round(y2.max(),5))
        plt.semilogy(x,y1,c='r',marker='o',label=label1)
        plt.semilogy(x,y2,c='b',marker='o',label=label2)
        plt.xlabel('epoch times')
        plt.title(train_ldm+' and '+ val_ldm)
        plt.legend()
    plt.show()

def show_list(loss_filename,path):
    '''
    face and landmarks training, 
    face classification accuracy of postive and negative samples
    '''
    
    Loss_list=pd.read_csv(path+loss_filename)
    if sorted(list(Loss_list.columns))!=['Unnamed: 0', 'train', 'train_acc', 'train_acc_neg', 'train_acc_neg_per', 'train_acc_per', 'train_acc_pos', 'train_acc_pos_per', 'train_face', 'train_face_per', 'train_ldm', 'train_ldm_per', 'train_per', 'val', 'val_acc', 'val_acc_neg', 'val_acc_neg_per', 'val_acc_per', 'val_acc_pos', 'val_acc_pos_per', 'val_face', 'val_face_per', 'val_ldm', 'val_ldm_per', 'val_per'] and sorted(list(Loss_list.columns))!=['Unnamed: 0', 'ldms_list', 'ldms_v_list','train', 'train_acc', 'train_acc_neg', 'train_acc_neg_per', 'train_acc_per', 'train_acc_pos', 'train_acc_pos_per', 'train_face', 'train_face_per', 'train_ldm', 'train_ldm_per', 'train_per', 'val', 'val_acc', 'val_acc_neg', 'val_acc_neg_per', 'val_acc_per', 'val_acc_pos', 'val_acc_pos_per', 'val_face', 'val_face_per', 'val_ldm', 'val_ldm_per', 'val_per']:
        return
    items=['train','val','train_face','val_face','train_ldm','val_ldm']
    show_fig(loss_filename,Loss_list,items)
    items=['train_per','val_per','train_face_per','val_face_per','train_ldm_per','val_ldm_per']
    show_fig(loss_filename,Loss_list,items)
    
    items=['train_acc','val_acc','train_acc_pos','val_acc_pos','train_acc_neg','val_acc_neg']
    show_fig(loss_filename,Loss_list,items,'acc')
    
    items=['train_acc_per','val_acc_per','train_acc_pos_per','val_acc_pos_per','train_acc_neg_per','val_acc_neg_per']
    show_fig(loss_filename,Loss_list,items,'acc')
    

filenames=os.listdir('./csv')
loss_list=[]
for filename in filenames:
    if(filename.find('loss_train_2019')>=0 and filename not in ['loss_train_20191212182418.csv',
                                                               'loss_train_20191212192018.csv']):
        loss_list.append(filename)
for i in loss_list:
    Loss_list=pd.read_csv(os.path.join('./csv',i))
    show_list(os.path.join('./csv',i),'')