import json
import os
count=0
# 读取train.json文件
with open('/root/autodl-tmp/BotRGCN/twibot_20/fangxx/twbot20/new_data/newtrain.json', 'r') as train_file:
    train_data = json.load(train_file)

# 创建一个新的列表来保存修改后的用户条目
new_train_data = []

# 遍历train.json中的每个用户条目
for user_entry in train_data:
    uid = user_entry['ID']
    json_file_path = f'./processed_data/train_json_qlora/u{uid}.json'  # 替换为JSON文件的路径
    try:
        # 读取对应的JSON文件
        with open(json_file_path, 'r') as json_file:
            user_json_data = json.load(json_file)
        
        # 在JSON文件中进行内容替换
        user_entry['profile']['description'] = user_json_data['description']
        user_entry['tweet']=user_json_data['tweets']
        count=count+1
    except FileNotFoundError:
        print(f"找不到用户 {uid} 的JSON文件")
    
    # 将修改后的用户条目添加到新的列表中
    new_train_data.append(user_entry)

print("所有用户信息替换完成")

# 将修改后的用户条目列表写入新的JSON文件中
output_file_path = "./new_data/newtrain_.json"  # 新的JSON文件路径
with open(output_file_path, 'w') as output_file:
    json.dump(new_train_data, output_file, indent=4)

print(f"新的train.json已保存到文件: {output_file_path}")
print(count)