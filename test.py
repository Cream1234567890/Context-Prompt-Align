import torch
from tqdm import tqdm
import pandas as pd
from dataset_tool import fast_merge
import numpy as np
from transformers import pipeline
import os

user_id = np.load('/root/autodl-tmp/BotRGCN/twibot_20/processed_data/user_id.npy')
print(len(user_id))