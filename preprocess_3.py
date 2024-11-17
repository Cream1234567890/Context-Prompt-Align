import torch
from tqdm import tqdm
import pandas as pd
from dataset_tool import fast_merge
import numpy as np
from transformers import pipeline
import os
import xmltodict

# 错误
error = "had error!!!!!!!\n"

print('loading raw data')
node=pd.read_json("/root/autodl-tmp/BotRGCN/node.json")
train_df_data=pd.read_json("/root/autodl-tmp/BotRGCN/twibot_20/vicuna_processed_data/new_data/newtrain_5_fix.json")
test_df_data=pd.read_json("/root/autodl-tmp/BotRGCN/twibot_20/vicuna_processed_data/new_data/newtest_5_fix.json")
df_data=pd.concat([train_df_data,test_df_data],ignore_index=True)
edge=pd.read_csv("/root/autodl-tmp/BotRGCN/edge.csv")
label=pd.read_csv("/root/autodl-tmp/BotRGCN/label.csv")
split=pd.read_csv("/root/autodl-tmp/BotRGCN/split.csv")
print('processing raw data')
user,tweet=fast_merge(node, label, split, dataset='Twibot-20')

user_text=list(user['description'])
tweet_text = [text for text in tweet.text]

print('loading npy')
each_user_tweets=np.load('/root/autodl-tmp/BotRGCN/twibot_20/yangweibin/each_user_tweets.npy',allow_pickle=True)
each_user_tweets = each_user_tweets.tolist()
# each_user_tweets=np.load('/root/autodl-tmp/BotRGCN/twibot_20/processed_data/each_user_tweets.npy')
user_id = np.load('/root/autodl-tmp/BotRGCN/twibot_20/processed_data/user_id.npy')
print('processing npy')

print('loading roberta')
feature_extract=pipeline('feature-extraction',model='roberta-base',tokenizer='roberta-base',device=0,padding=True, truncation=True,max_length=500, add_special_tokens = True)
print('finish loading roberta')

def train_val_test_mask():
    root = "/root/autodl-tmp/processed_data/"
    print("load train_val_test_mask success!!!!!!!")
    train_idx=torch.load(root+'train_idx.pt').tolist()
    val_idx=torch.load(root+'val_idx.pt').tolist()
    test_idx=torch.load(root+'test_idx.pt').tolist()
    return train_idx,val_idx,test_idx

train_idx,val_idx,test_idx = train_val_test_mask()
train_user_id = [user_id[i] for i in train_idx]
val_user_id = [user_id[i] for i in val_idx]
test_user_id = [user_id[i] for i in test_idx]

def Des_embbeding_llama():
    print('Running feature1 embedding llama')
    path="/root/autodl-tmp/processed_data/des_tensor.pt"
    save_path = "/root/autodl-tmp/BotRGCN/twibot_20/vicuna_processed_data/temp_yang/des_tensor_json_5_fix.pt"
    des_tensor = torch.load(path)
    des_vec = torch.unbind(des_tensor, 0)
    des_vec = list(des_vec)
    for k,each in enumerate(tqdm(user_text)):
        uid = user_id[k]
        if not uid in train_user_id and not uid in val_user_id and not uid in test_user_id:
            continue
        # print(uid)
        uid = uid.replace('u','')
        uid = int(uid)
        index = None
        des = None
        try:
            index = df_data.loc[df_data['ID'] == uid].index.item()
        except Exception as e:
            index = None
        if index == None:
            continue
        des = df_data.loc[index]['profile']['description']
        each = des
        if each is None:
            des_vec.append(torch.zeros(768))
        else:
            feature=torch.Tensor(feature_extract(each))
            for (i,tensor) in enumerate(feature[0]):
                if i==0:
                    feature_tensor=tensor
                else:
                    feature_tensor+=tensor
            feature_tensor/=feature.shape[1]
            des_vec[k] = feature_tensor
    des_tensor=torch.stack(des_vec,0)
    torch.save(des_tensor,save_path)
    print('Finished')
    return des_tensor
    
        
def tweets_embedding_llama():
    print('Running feature2 embedding')
    path="/root/autodl-tmp/processed_data/tweets_tensor.pt"
    save_path = "/root/autodl-tmp/BotRGCN/twibot_20/vicuna_processed_data/temp_yang/tweets_tensor_json_5_fix.pt"
    tweets_tensor = torch.load(path)
    tweets_list = torch.unbind(tweets_tensor, 0)
    tweets_list = list(tweets_list)
    for i in tqdm(range(len(each_user_tweets))):
        flag = False
        tweet_list = list()
        uid = user_id[i]
        if not uid in train_user_id and not uid in val_user_id and not uid in test_user_id:
            continue
        uid = uid.replace('u','')
        uid = int(uid)
        index = None
        des = None
        try:
            index = df_data.loc[df_data['ID'] == uid].index.item()
        except Exception as e:
            index = None
        if index == None:
            continue
        tweet_list = df_data.loc[index]['tweet']
        if tweet_list == None:
            continue
        if len(tweet_list)==0:
            total_each_person_tweets=torch.zeros(768)
        else:
            for j in range(len(tweet_list)):
                each_tweet = tweet_list[j]
                if each_tweet is None:
                    total_word_tensor=torch.zeros(768)
                else:
                    try:
                        each_tweet_tensor=torch.tensor(feature_extract(each_tweet))
                    except Exception as e:
                        print(e)
                        print(type(each_tweet))
                        # print(each_tweet)
                    for k,each_word_tensor in enumerate(each_tweet_tensor[0]):
                        if k==0:
                            total_word_tensor=each_word_tensor
                        else:
                            total_word_tensor+=each_word_tensor
                    total_word_tensor/=each_tweet_tensor.shape[1]
                if j==0:
                    total_each_person_tweets=total_word_tensor
                elif j==20:
                    break
                else:
                    total_each_person_tweets+=total_word_tensor
            if (j==20):
                total_each_person_tweets/=20
            else:
                total_each_person_tweets/=len(tweet_list)
        tweets_list[i] = total_each_person_tweets

    tweet_tensor=torch.stack(tweets_list)
    torch.save(tweet_tensor,save_path)
    print('Finished')

# Des_embbeding()
# tweets_embedding()
Des_embbeding_llama()
tweets_embedding_llama()