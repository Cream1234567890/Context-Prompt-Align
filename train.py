from model import BotRGCN
from Dataset import Twibot22
import torch
from torch import nn
from utils import accuracy,init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, HeteroData
import logging
import pandas as pd
import numpy as np

import math
from torchsummary import summary

device = 'cuda:0'
embedding_size,dropout,lr,weight_decay=32,0.1,1e-2,5e-2
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("./logs/vicuna/k_for_5_fix.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


root='/root/autodl-tmp/processed_data/'
print("device: ", device)
dataset=Twibot22(root=root,device=device,process=True,save=True)
des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()

# print(des_tensor.shape)
# print(tweets_tensor.shape)
# print(num_prop.shape)
# print(category_prop.shape)
# print(edge_index.shape)
# print(edge_type.shape)

model=BotRGCN(cat_prop_size=3,embedding_dimension=embedding_size).to(device)
# model.load_state_dict(torch.load('./models/model_best.pth'))
loss=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                    lr=lr,weight_decay=weight_decay)

def train(epoch):
    model.train()
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_train = loss(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    logger.info('Epoch: {:04d}, loss_train: {:.4f}, acc_train: {:.4f}, acc_val: {:.4f}'.format(epoch+1, loss_train.item(), acc_train.item(), acc_val.item()))
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'acc_val: {:.4f}'.format(acc_val.item()),)
    return acc_train,loss_train

def test(epoch):
    global max_acc
    model.eval()
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_test = loss(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    # if acc_test.item() > max_acc:
    #     save_dir = f'./models/model_best_1.pth'
    #     max_acc = acc_test.item()
    #     torch.save(model.state_dict(), save_dir)
    output=output.max(1)[1].to('cpu').detach().numpy()
    label=labels.to('cpu').detach().numpy()
    f1=f1_score(label[test_idx],output[test_idx])
    #mcc=matthews_corrcoef(label[test_idx], output[test_idx])
    precision=precision_score(label[test_idx],output[test_idx])
    recall=recall_score(label[test_idx],output[test_idx])
    fpr, tpr, thresholds = roc_curve(label[test_idx], output[test_idx], pos_label=1)
    Auc=auc(fpr, tpr)
    logger.info("Test set results:")
    logger.info('test_loss= {:.4f}, test_accuracy= {:.4f}, precision= {:.4f}, recall= {:.4f}, f1_score= {:.4f}, auc= {:.4f}'.format(loss_test.item(), acc_test.item(), precision.item(), recall.item(), f1.item(), Auc.item()))
    print("Test set results:",
            "test_loss= {:.4f}".format(loss_test.item()),
            "test_accuracy= {:.4f}".format(acc_test.item()),
            "precision= {:.4f}".format(precision.item()),
            "recall= {:.4f}".format(recall.item()),
            "f1_score= {:.4f}".format(f1.item()),
            #"mcc= {:.4f}".format(mcc.item()),
            "auc= {:.4f}".format(Auc.item()),
            )
    
model.apply(init_weights)

from thop import profile
flops, params = profile(model, [des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type])
print('flops: ', flops, 'params: ', params)


from pytorch_model_summary import summary
print(summary(model, des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type, show_input=False, show_hierarchical=False))


# 获取模型参数
params = model.parameters()
# 计算参数量
num_params = 0
for param in params:
    num_params += torch.prod(torch.tensor(param.size()))
print('模型参数量：', num_params)



# max_acc = 0.0
# epochs=50000
# for epoch in range(epochs):
#     train(epoch)
#     if epoch % 50 == 0 and epoch != 0:
#         test(epoch)
# test()