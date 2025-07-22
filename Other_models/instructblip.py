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
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import json

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [-int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system("rm tmp")
    GPU_id = int(np.argmax(memory_available))
    print("using GPU{}".format(GPU_id))
    return GPU_id


if __name__ == "__main__":
    OUTPUT_FOLDER = '../Output'
    device = torch.device("cuda:{}".format(get_free_gpu()) if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model_name = "Salesforce/instructblip-vicuna-7b"
    processor = InstructBlipProcessor.from_pretrained(model_name)
    model = InstructBlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    for _,folder,_ in os.walk(OUTPUT_FOLDER):
        break
    print("Start generating")
    for index,f in enumerate(folder):
        temp_dict = {}
        image_path = os.path.join(OUTPUT_FOLDER,f,"image.jpg")
        image = Image.open(image_path)
        json_file = json.load(open(os.path.join(OUTPUT_FOLDER,f,"response.json")))
        object_list = json_file['object_list']
        obeject_sentence = object_list[0]
        for obj in object_list[1:]:
            obeject_sentence += ", " + obj
        for key in easy_prompt_dict.keys():
            if 'multi' in key:
                response_list = []
                for obt in object_list:
                    prompt = easy_prompt_dict[key].format(object=obt,object_list=obeject_sentence)
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device=device, dtype=torch.float16)
                    generated_ids = model.generate(**inputs,max_length=1000)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                    response_list.append((obt,generated_text))
            elif 'single' in key:
                response_list = []
                for obt in object_list:
                    prompt = easy_prompt_dict[key].format(object=obt)
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device=device, dtype=torch.float16)
                    generated_ids = model.generate(**inputs,max_length=1000)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                    response_list.append((obt,generated_text))
            else:
                prompt = easy_prompt_dict[key]
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device=device, dtype=torch.float16)
                generated_ids = model.generate(**inputs,max_length=1000)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                response_list = generated_text

            temp_dict[key] = response_list
        json.dump(temp_dict,open(os.path.join(OUTPUT_FOLDER,f,"response_instructblip.json"),"w"),indent=4)
        if index % 10 == 0:
            print("Finish {} images".format(index))