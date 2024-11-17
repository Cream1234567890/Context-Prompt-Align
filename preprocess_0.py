import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt
from dataset_tool import fast_merge,df_to_mask
import os

print('loading raw data')
node=pd.read_json("/root/autodl-tmp/BotRGCN/node.json")
edge=pd.read_csv("/root/autodl-tmp/BotRGCN/edge.csv")
label=pd.read_csv("/root/autodl-tmp/BotRGCN/label.csv")
split=pd.read_csv("/root/autodl-tmp/BotRGCN/split.csv")
print('processing raw data')
user,tweet=fast_merge(node, label, split, dataset='Twibot-20')


user_index_to_uid = list(user.id)
np.save('/root/autodl-tmp/BotRGCN/twibot_20/processed_data/user_id',user_index_to_uid)