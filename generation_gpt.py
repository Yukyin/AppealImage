import os
import time
import numpy as np
from tqdm import tqdm
os.environ['TRANSFORMERS_CACHE'] = "../cache"
os.environ['HF_HOME'] = "../cache"
from PIL import Image
from utils import *
import requests
import torch
import json
import concurrent.futures
import argparse
import time

def query_model(sub_folder,i, args):
    try:
        if i%50 == 0:
            print(f"Processing {i}th image")
        image_path = os.path.join(args.data_path, str(sub_folder), f'{sub_folder}.jpg')
        sub_folder_path = os.path.join(args.data_path, str(sub_folder))
        json_file_path = os.path.join(sub_folder_path, f'{args.prompting_template}.json')
        if os.path.exists(json_file_path):
            print(f"Skipping sub_folder {sub_folder} as it already exists")
        else:
            prompt = prompt_dict[args.prompting_template]
            gpt_response = gpt4v(image_path, prompt)
            with open(json_file_path, 'w') as f:
                json.dump({"gpt_response":gpt_response}, f, indent=4)
    except Exception as e:
        time.sleep(20)
        print(e)
        print(f"Error in {key}")

def generation(args):
    for _,folders,_ in os.walk(args.data_path):
        break
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        for i,sub_folder in enumerate(folders):
            executor.submit(query_model, sub_folder,i,args)
            if i == args.count:
                break

def generation_loop(args):
    for _,folders,_ in os.walk(args.data_path):
        break
    for i,sub_folder in enumerate(folders):
        query_model(sub_folder,i,args)
        if i == args.count:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Data.')
    parser.add_argument('--data_path', type=str, default='./GQA', help='dataset name')
    parser.add_argument('--json_path', type=str, default='../DATA/val_sceneGraphs.json', help='json folder')
    parser.add_argument('--object_list_path', type=str, default='./object_list.txt', help='object list')
    parser.add_argument('--prompting_template', type=str, default='whole_image_prompt', help='prompting template')
    parser.add_argument('--count', type=int, default=100, help='number of samples')
    ### 
    ### gqa
    ### ../DATA/val_sceneGraphs.json
    ### gqa_object_sentence.json
    ### coco
    ### ./coco_summary.json
    ### coco_object_sentence.json
    argstring = ['--data_path', './GQA_train',
                 '--json_path', './coco_summary.json',
                 '--object_list_path', 'coco_object_sentence.json',
                 '--prompting_template', 'extract_object_prompt',
                 '--count', '74943']
    args = parser.parse_args(argstring)
    print("Starting Generation...")
    time1 = time.time()
    generation(args)
    time2 = time.time()
    print("Generation Completed")
    print(f"Generation Time: {time2-time1}")