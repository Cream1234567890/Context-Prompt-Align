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

file_root_train = '/root/autodl-tmp/BotRGCN/twibot_20/processed_data/sampled_path_yang_xml_train/'
file_root_test = '/root/autodl-tmp/BotRGCN/twibot_20/processed_data/sampled_path_yang_xml_test/'

print('loading raw data')
node=pd.read_json("/root/autodl-tmp/BotRGCN/node.json")
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


def xml_process(content,k):
    des,tweets = None, None
    des_start = content.find('<description>')
    des_end = content.find('</description>')
    tweets_start = content.find('<tweets>')
    tweets_end = content.find('</tweets>')
    if des_end == -1 and tweets_start != -1:
        content = content.replace('<tweets>','</description>\n<tweets>')
        des_start = content.find('<description>')
        des_end = content.find('</description>')
        tweets_start = content.find('<tweets>')
        tweets_end = content.find('</tweets>')
    if tweets_end == -1:
        pos = 0
        tweet_pos = 0
        count = 0
        for i in range(0,k):
            tweet_pos = content.find('</tweet>',pos+len('</tweet>'))
            if pos != tweet_pos:
                pos = tweet_pos
                count += 1
        if count == k:
            content = content[0:tweet_pos + len('</tweet>') + 1] + '\n</tweets>' + content[tweet_pos + len('</tweet>')+1:]
            des_start = content.find('<description>')
            des_end = content.find('</description>')
            tweets_start = content.find('<tweets>')
            tweets_end = content.find('</tweets>')
    if des_start == -1 or des_end == -1 or des_start >= des_end:
        return False, des, tweets
    else:
        des = content[des_start: des_end + len('</description>')]

    if tweets_start == -1 or tweets_end == -1 or tweets_start >= tweets_end:
        return False, des, tweets
    else:
        tweets = content[tweets_start: tweets_end + len('</tweets>')]
    des = des.replace('&','&amp;').replace('>','&gt;').replace('<','&lt;').replace('"','&quot;').replace('\'','&apos;')
    des = des.replace('&lt;description&gt;','<description>').replace('&lt;/description&gt;','</description>')
    tweets = tweets.replace('&','&amp;').replace('>','&gt;').replace('<','&lt;').replace('"','&quot;').replace('\'','&apos;')
    tweets = tweets.replace('&lt;tweets&gt;','<tweets>').replace('&lt;/tweets&gt;','</tweets>').replace('&lt;tweet&gt;','<tweet>').replace('&lt;/tweet&gt;','</tweet>').replace('&lt;text&gt;','<text>').replace('&lt;/text&gt;','</text>')
    return True, des, tweets

def Des_embbeding():
        print('Running feature1 embedding')
        print('/root/autodl-tmp/BotRGCN/twibot_20/processed_data/temp_yang/des_tensor_yang1.pt')
        des_vec_path = '/root/autodl-tmp/BotRGCN/twibot_20/processed_data/temp_yang/des_vec.npy'
        path="/root/autodl-tmp/BotRGCN/twibot_20/processed_data/temp_yang/des_tensor_yang1.pt"
        if not os.path.exists(path):
            des_vec=[]
            for k,each in enumerate(tqdm(user_text)):
                uid = user_id[k]
                if not uid in train_idx and not uid in val_idx and not uid in test_idx:
                    continue
                # print(uid)
                file_path_train = file_root_train + uid + '.txt'
                file_path_test = file_root_test + uid + '.txt'
                content = None
                flag = False
                des = None
                try:
                    with open(file_path_train, 'r', encoding='utf-8') as file:
                        content = file.read()
                except Exception as e:
                    # print(e)
                    flag = False
                try:
                    with open(file_path_test, 'r', encoding='utf-8') as file:
                        content = file.read()
                except Exception as e:
                    # print(e)
                    flag = False
                # print(content)
                if content != None and content != error:
                    is_legitimate, des_xml, tweets_xml = xml_process(content,2)
                    if is_legitimate:
                        try:
                            des_dict = xmltodict.parse(des_xml)
                            des = des_dict['description']
                            flag = True
                        except Exception as e:
                            # print(e)
                            flag = False
                if flag and des != None and des != 'None' and des != 'N/A' and len(des) > len(each):
                    # print(des)
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
                    des_vec.append(feature_tensor)
            np.save(des_vec,des_vec_path)
            des_tensor=torch.stack(des_vec,0)
            torch.save(des_tensor,path)
        else:
            des_tensor=torch.load(path)
        print('Finished')
        return des_tensor

def Des_embbeding_llama():
    print('Running feature1 embedding llama')
    des_vec_path = '/root/autodl-tmp/BotRGCN/twibot_20/processed_data/temp_yang/des_vec.npy'
    path="/root/autodl-tmp/BotRGCN/twibot_20/processed_data/temp_yang/des_tensor_yang1.pt"
    save_path = "/root/autodl-tmp/BotRGCN/twibot_20/processed_data/temp_yang/des_tensor_yang2.pt"
    des_tensor = torch.load(path)
    des_vec = torch.unbind(des_tensor, 0)
    des_vec = list(des_vec)
    for k,each in enumerate(tqdm(user_text)):
        uid = user_id[k]
        if not uid in train_user_id and not uid in val_user_id and not uid in test_user_id:
            continue
        # print(uid)
        file_path_train = file_root_train + uid + '.txt'
        file_path_test = file_root_test + uid + '.txt'
        content = None
        flag = False
        des = None
        try:
            with open(file_path_train, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            # print(e)
            flag = False
        try:
            with open(file_path_test, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            # print(e)
            flag = False
        # print(content)
        if content != None and content != error:
            is_legitimate, des_xml, tweets_xml = xml_process(content,2)
            if is_legitimate:
                try:
                    des_dict = xmltodict.parse(des_xml)
                    des = des_dict['description']
                    flag = True
                except Exception as e:
                    # print(e)
                    flag = False
        if (flag and des != None and des != 'None' and des != 'N/A') or (flag and each == None):
            # print(des)
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
    
def tweets_embedding():
        print('Running feature2 embedding')
        path="./processed_data/tweets_tensor.pt"
        if not os.path.exists(path):
            tweets_list=[]
            for i in tqdm(range(len(each_user_tweets))):
                if i == 10000:
                    break
                flag = False
                tweet_list = list()
                uid = user_id[i]
                file_path_train = file_root_train + uid + '.txt'
                file_path_test = file_root_test + uid + '.txt'
                content = None
                try:
                    with open(file_path_train, 'r', encoding='utf-8') as file:
                        content = file.read()
                except Exception as e:
                    # print(e)
                    flag = False
                try:
                    with open(file_path_test, 'r', encoding='utf-8') as file:
                        content = file.read()
                except Exception as e:
                    # print(e)
                    flag = False
                if content != None and content != error:
                    is_legitimate, des_xml, tweets_xml = xml_process(content,2)
                    if is_legitimate:
                        try:
                            des_dict = xmltodict.parse(des_xml)
                            des = des_dict['description']
                            tweets_dict = xmltodict.parse(tweets_xml)
                            tweets_list_xml = tweets_dict['tweets']['tweet']
                            tweet_list_temp = list()
                            for tweet in tweets_list_xml:
                                if type(tweet) == dict:
                                    tweet = tweet['text']
                                tweet_list_temp.append(tweet)
                            tweet_list = tweet_list_temp
                            flag = True
                        except Exception as e:
                            # print(e)
                            flag = False
                if len(each_user_tweets[i])==0:
                    total_each_person_tweets=torch.zeros(768)
                elif flag:
                    for j in range(len(tweet_list)):
                        each_tweet = tweet_list[j]
                        if each_tweet is None:
                            total_word_tensor=torch.zeros(768)
                        else:
                            try:
                                each_tweet_tensor=torch.tensor(feature_extract(each_tweet))
                            except Exception as e:
                                print(e)
                                print(each_tweet)
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
                else:
                    for j in range(len(each_user_tweets[i])):
                        each_tweet=tweet_text[each_user_tweets[i][j]]
                        if each_tweet is None:
                            total_word_tensor=torch.zeros(768)
                        else:
                            each_tweet_tensor=torch.tensor(feature_extract(each_tweet))
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
                        total_each_person_tweets/=len(each_user_tweets[i])
                        
                tweets_list.append(total_each_person_tweets)
                        
            tweet_tensor=torch.stack(tweets_list)
            torch.save(tweet_tensor,"/root/autodl-tmp/BotRGCN/twibot_20/processed_data/temp_yang/tweets_tensor_yang.pt")
            
        else:
            tweets_tensor=torch.load(path)
        print('Finished')
        
def tweets_embedding_llama():
    print('Running feature2 embedding')
    path="/root/autodl-tmp/processed_data/tweets_tensor.pt"
    save_path = "/root/autodl-tmp/BotRGCN/twibot_20/processed_data/temp_yang/des_tensor_new.pt"
    tweets_tensor = torch.load(path)
    tweets_list = torch.unbind(tweets_tensor, 0)
    tweets_list = list(tweets_list)
    for i in tqdm(range(len(each_user_tweets))):
        flag = False
        tweet_list = list()
        uid = user_id[i]
        if not uid in train_user_id and not uid in val_user_id and not uid in test_user_id:
            continue
        file_path_train = file_root_train + uid + '.txt'
        file_path_test = file_root_test + uid + '.txt'
        content = None
        try:
            with open(file_path_train, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            # print(e)
            flag = False
        try:
            with open(file_path_test, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            # print(e)
            flag = False
        if content != None and content != error:
            is_legitimate, des_xml, tweets_xml = xml_process(content,2)
            if is_legitimate:
                try:
                    des_dict = xmltodict.parse(des_xml)
                    des = des_dict['description']
                    tweets_dict = xmltodict.parse(tweets_xml)
                    tweets_list_xml = tweets_dict['tweets']['tweet']
                    tweet_list_temp = list()
                    for tweet in tweets_list_xml:
                        if type(tweet) == dict:
                            tweet = tweet['text']
                        tweet_list_temp.append(tweet)
                    tweet_list = tweet_list_temp
                    flag = True
                except Exception as e:
                    # print(e)
                    flag = False
        if len(each_user_tweets[i])==0:
            total_each_person_tweets=torch.zeros(768)
        elif flag:
            for j in range(len(tweet_list)):
                each_tweet = tweet_list[j]
                if type(each_tweet) == list:
                    each_tweet = each_tweet[0]
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
# Des_embbeding_llama()
tweets_embedding_llama()