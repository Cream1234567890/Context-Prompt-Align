import json
import os
from difflib import get_close_matches
from tqdm import tqdm

def find_closest_match(input_str, str_list):
    matches = get_close_matches(input_str, str_list)
    if matches:
        closest_match = matches[0]
        return closest_match
    else:
        return None
des_count = 0
none_count = 0
error_count = 0
count=0
un_list = []
# 读取train.json文件
with open('/root/autodl-tmp/BotRGCN/twibot_20/fangxx/twbot20/test.json', 'r') as train_file:
    train_data = json.load(train_file)

# 创建一个新的列表来保存修改后的用户条目
new_train_data = []
id = -1
# 遍历train.json中的每个用户条目
for user_entry in tqdm(train_data):
    uid = user_entry['ID']
    id += 1
    json_file_path = f'./processed_data/test_all_json/u{uid}.json'  # 替换为JSON文件的路径
    try:
        # 读取对应的JSON文件
        with open(json_file_path, 'r') as json_file:
            user_json_data = json.load(json_file)
        
        tweets = list()
        if user_entry['tweet'] == None:
            none_count += 1
            un_list.append(uid)
            continue
        for tweet in user_json_data['tweets']:                
            closest_tweet = find_closest_match(tweet, user_entry['tweet'])
            if closest_tweet != tweet:
                print(id)
                print(uid)
                print(tweet)
                print(closest_tweet)
            if closest_tweet == None:
                error_count += 1
                un_list.append(uid)
                continue
            tweets.append(closest_tweet)
        # 在JSON文件中进行内容替换
        user_entry['profile']['description'] = user_json_data['description']
        des_count += 1
        user_entry['tweet']=tweets
        count=count+1
    except FileNotFoundError as e:
        continue
        # print(f"找不到用户 {uid} 的JSON文件")
    
    # 将修改后的用户条目添加到新的列表中
    new_train_data.append(user_entry)

print("所有用户信息替换完成")

# # 将修改后的用户条目列表写入新的JSON文件中
# output_file_path = "./new_data/newtest_all_1.json"  # 新的JSON文件路径
# with open(output_file_path, 'w') as output_file:
#     json.dump(new_train_data, output_file, indent=4)

# print(f"新的test.json已保存到文件: {output_file_path}")
# print('des_count：',des_count)
# print('none_count：',none_count)
# print('error_count',error_count)
# print('count：',count)
# with open('./new_data/un_test_1.txt', 'w') as file:
#     file.write(str(un_list))
