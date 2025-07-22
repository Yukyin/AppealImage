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
from transformers import AutoProcessor, LlavaForConditionalGeneration
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
    device = torch.device("cuda:{}".format(get_free_gpu()) if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model_name = "llava-hf/llava-1.5-7b-hf"

    # prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
    # image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_name)
    for _,folder,_ in os.walk(OUTPUT_FOLDER):
        break
    print("Start generating")
    for index,f in enumerate(folder):
        if index == 1:
            print('start at {}'.format(f))
        temp_dict = json.load(open(os.path.join(OUTPUT_FOLDER,f,"response_llava15.json")))
        image_path = os.path.join(OUTPUT_FOLDER,f,f"{f}.jpg")
        image = Image.open(image_path)
        json_file = json.load(open(os.path.join(OUTPUT_FOLDER,f,"extract_object_prompt.json")))
        object_list = json_file['object_list']  
        obeject_sentence = json_file['object_sentence']
        for key in easy_prompt_dict.keys():
            if 'multi' in key:
                response_list = []
                for obt in object_list:
                    prompt = "USER: <image>\n" + easy_prompt_dict[key].format(object=obt,object_list=obeject_sentence)+"\nASSISTANT:"
                    inputs = processor(prompt, image, return_tensors='pt').to(device, torch.float16)
                    output = model.generate(**inputs, max_new_tokens=200, pad_token_id=model.config.eos_token_id,do_sample=False)
                    generated_text = processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[1].strip(" *[]")
                    response_list.append((obt,generated_text))
            elif 'single' in key:
                response_list = []
                for obt in object_list:
                    prompt = "USER: <image>\n" + easy_prompt_dict[key].format(object=obt) +"\nASSISTANT:"
                    inputs = processor(prompt, image, return_tensors='pt').to(device, torch.float16)
                    output = model.generate(**inputs, max_new_tokens=200,pad_token_id=model.config.eos_token_id, do_sample=False)
                    generated_text = processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[1].strip(" *[]")

                    response_list.append((obt,generated_text))
            else:
                prompt = "USER: <image>\n" + easy_prompt_dict[key]+"\nASSISTANT:"
                inputs = processor(prompt, image, return_tensors='pt').to(device, torch.float16)
                output = model.generate(**inputs, max_new_tokens=200,pad_token_id=model.config.eos_token_id, do_sample=False)
                generated_text = processor.decode(output[0], skip_special_tokens=True)
                response_list = generated_text.split("ASSISTANT:")[1].strip(" *[]")

            temp_dict[key] = response_list
        json.dump(temp_dict,open(os.path.join(OUTPUT_FOLDER,f,"response_llava15.json"),"w"),indent=4)
        if index % 10 == 0:
            print("Finish {} images".format(index))