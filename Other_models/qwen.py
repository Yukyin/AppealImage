import os
import time
import numpy as np
from tqdm import tqdm
os.environ['TRANSFORMERS_CACHE'] = "../../cache"
os.environ['HF_HOME'] = "../../cache"
from PIL import Image
from utils import *
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [-int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system("rm tmp")
    GPU_id = int(np.argmax(memory_available))
    print("using GPU{}".format(GPU_id))
    return GPU_id


if __name__ == "__main__":
    device = torch.device("cuda:{}".format(get_free_gpu()) if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen-VL-Chat"
    tokenizer  = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, fp16=True).eval().to(device)
    OUTPUT_FOLDER = "../GQA_train"
    for _,folder,_ in os.walk(OUTPUT_FOLDER):
        break
    print("Start generating")
    for index,f in enumerate(folder):
        if index == 1:
            print('start at {}'.format(f))
        temp_dict = json.load(open(os.path.join(OUTPUT_FOLDER,f,"response_qwen.json")))
        image_path = os.path.join(OUTPUT_FOLDER,f,f"{f}.jpg")
        image = Image.open(image_path)
        json_file = json.load(open(os.path.join(OUTPUT_FOLDER,f,"extract_object_prompt.json")))
        object_list = json_file['object_list']
        obeject_sentence = json_file['object_sentence']
        for key in easy_prompt_dict.keys():
            if 'multi' in key:
                response_list = []
                for obt in object_list:
                    prompt = easy_prompt_dict[key].format(object=obt,object_list=obeject_sentence)
                    query = tokenizer.from_list_format([
                        {'image': image_path},
                        {'text': prompt},
                    ])
                    generated_text, history = model.chat(tokenizer, query=query, history=None)
                    response_list.append((obt,generated_text))
            elif 'single' in key:
                response_list = []
                for obt in object_list:
                    prompt =easy_prompt_dict[key].format(object=obt)
                    query = tokenizer.from_list_format([
                        {'image': image_path},
                        {'text': prompt},
                    ])
                    generated_text, history = model.chat(tokenizer, query=query, history=None)
                    response_list.append((obt,generated_text))
            else:
                response_list = []
                prompt = easy_prompt_dict[key]
                query = tokenizer.from_list_format([
                    {'image': image_path},
                    {'text': prompt},
                ])
                generated_text, history = model.chat(tokenizer, query=query, history=None)
                response_list.append((generated_text))

            temp_dict[key] = response_list
        json.dump(temp_dict,open(os.path.join(OUTPUT_FOLDER,f,"response_qwen.json"),"w"),indent=4)
        if index % 10 == 0:
            print("Finish {} images".format(index))