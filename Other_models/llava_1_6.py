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
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import json

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [-int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system("rm tmp")
    GPU_id = int(np.argmax(memory_available))
    print("using GPU{}".format(GPU_id))
    return GPU_id


if __name__ == "__main__":
    OUTPUT_FOLDER = '../GQA_train'
    print(torch.cuda.is_available())
    device = torch.device("cuda:{}".format(get_free_gpu()) if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = "llava-hf/llava-v1.6-mistral-7b-hf"

    processor = LlavaNextProcessor.from_pretrained(model)

    model = LlavaNextForConditionalGeneration.from_pretrained(model, torch_dtype=torch.float16) 
    model.to(device)
    for _,folder,_ in os.walk(OUTPUT_FOLDER):
        break
    print("Start generating")
    for index,f in enumerate(folder):
        if index == 1:
            print('start at {}'.format(f))
        temp_dict = json.load(open(os.path.join(OUTPUT_FOLDER,f,"response_llava16.json")))
        image_path = os.path.join(OUTPUT_FOLDER,f,f"{f}.jpg")
        image = Image.open(image_path)
        json_file = json.load(open(os.path.join(OUTPUT_FOLDER,f,"extract_object_prompt.json")))
        object_list = json_file['object_list']
        obeject_sentence = json_file['object_sentence']
        for key in easy_prompt_dict.keys():
            if 'multi' in key:
                response_list = []
                for obt in object_list:
                    prompt = "[INST] <image>\n" + easy_prompt_dict[key].format(object=obt,object_list=obeject_sentence)+" [/INST]"
                    inputs = processor(prompt, image, return_tensors="pt").to(device)
                    output = model.generate(**inputs, max_new_tokens=200,pad_token_id=model.config.eos_token_id)
                    generated_text = processor.decode(output[0], skip_special_tokens=True).split("[/INST]")[1].strip(" *[]")

                    response_list.append((obt,generated_text))
            elif 'single' in key:
                response_list = []
                for obt in object_list:
                    prompt = "[INST] <image>\n" + easy_prompt_dict[key].format(object=obt) + "[/INST]"
                    inputs = processor(prompt, image, return_tensors="pt").to(device)
                    output = model.generate(**inputs, max_new_tokens=200,pad_token_id=model.config.eos_token_id)
                    generated_text = processor.decode(output[0], skip_special_tokens=True).split("[/INST]")[1].strip(" *[]")

                    response_list.append((obt,generated_text))
            else:
                prompt = "[INST] <image>\n" +easy_prompt_dict[key]+" [/INST]"
                inputs = processor(prompt, image, return_tensors="pt").to(device)
                output = model.generate(**inputs, max_new_tokens=200,pad_token_id=model.config.eos_token_id)
                generated_text = processor.decode(output[0], skip_special_tokens=True)
                response_list = generated_text.split("[/INST]")[1].strip(" *[]")

            temp_dict[key] = response_list
        json.dump(temp_dict,open(os.path.join(OUTPUT_FOLDER,f,"response_llava16.json"),"w"),indent=4)
        if index % 10 == 0:
            print("Finish {} images".format(index))