import os
import shutil
import json

def copy_file_with_name(source_path, target_path):
    
    if not os.path.exists(target_path):
       os.makedirs(target_path)

    source_file = os.listdir(source_path)
    
    with open(f"./{target_path}_set_dict.json", "r", encoding="utf-8") as f:
        content = json.load(f)
    list_help=list(content.keys())

    for file in source_file:
        file_name_0=file.split("_")[0]

        if file_name_0 in list_help:
            source_file_path = os.path.join(source_path, file)
            
            if os.path.isfile(source_file_path):
                new_file_path = os.path.join(target_path, file)
                shutil.copy(source_file_path, new_file_path)
            else:
                new_file_path = os.path.join(target_path, file)
                shutil.copytree(source_file_path, new_file_path)
            
source_path =  './after_eval_process_pc' 
target_train = 'train'
target_test = 'test'

copy_file_with_name(source_path, target_train)
copy_file_with_name(source_path, target_test)