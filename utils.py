import os
import time
import numpy as np
import openai
import base64
import requests
import openai
import time
from openai import AzureOpenAI


prompt_dict = {'whole_image_prompt':"""Generate one humorous and creative sentence for the given image using one of the following techniques:
1. Wordplay - Utilizing puns or idiomatic expressions to create a clever and amusing sentence, e.g. "I'm on a seafood diet. I see food and I eat it!"
2. Phonetic similarity - Employing words that sound alike to generate a humorous twist, e.g. "I'm feeling grape!"
3. Rhetorical devices  - Using literary techniques such as unexpected contrasts, irony, personification, or hyperbole to enhance the humor, e.g. "I'm not a complete idiot. Some parts are missing.""",
'single_object_prompt':"""Generate humorous and creative sentences for the given image using one of the following techniques:
1. Wordplay - Utilizing puns or idiomatic expressions to create a clever and amusing sentence, e.g. "I'm on a seafood diet. I see food and I eat it!"
2. Phonetic similarity - Employing words that sound alike to generate a humorous twist, e.g. "I'm feeling grape!"
3. Rhetorical devices  - Using literary techniques such as unexpected contrasts, irony, personification, or hyperbole to enhance the humor, e.g. "I'm not a complete idiot. Some parts are missing.

Here is a list of objects (including human) in the given image. 
List of objects: {object_list}
Provide an intereseting sentence for each object by strictly follow the template. Each object should be mentioned once, so there should be {number} sentences in total. Be careful and only use the object name in the list. The sentence should only focus on the object and shouldn't mention other objects. You should only consider object from the list.
- object: sentence
""",
'multiple_object_prompt':"""Generate humorous and creative sentences for the given image using one of the following techniques:
1. Wordplay - Utilizing puns or idiomatic expressions to create a clever and amusing sentence, e.g. "I'm on a seafood diet. I see food and I eat it!"
2. Phonetic similarity - Employing words that sound alike to generate a humorous twist, e.g. "I'm feeling grape!"
3. Rhetorical devices  - Using literary techniques such as unexpected contrasts, irony, personification, or hyperbole to enhance the humor, e.g. "I'm not a complete idiot. Some parts are missing.

Here is a list of objects (including human) in the given image. 
List of objects: {object_list}
Provide an intereseting sentence for each object by strictly follow the template and don't use any other notations. Each object should be mentioned as the main object at least once, so there should be {number} sentences in total. In the sentence of each object, also mention another object in the list to increase the humor. Here is a template example for two sentences:
- main object
- other objects
- sentence

- main object
- other objects
- sentence""",
"extract_object_prompt":"""Extract the objects from the image by strictly follow the template. The name of object should less than three words.
- object1
- object2
"""}



def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [-int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system("rm tmp")
    GPU_id = int(np.argmax(memory_available))
    print("using GPU{}".format(GPU_id))
    return GPU_id


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def azure(image_path, system_prompt):
  api_version = '2023-12-01-preview' # this might change in the future
  deployment_name = 'SWC_GPT4_VISION'
  client = AzureOpenAI(
      api_key=api_key,  
      api_version=api_version,
      base_url=f"{api_base}/openai/deployments/{deployment_name}",
  )

  base64_image = encode_image(image_path)
  response = client.chat.completions.create(
      model=deployment_name,
      messages=[
          { "role": "system", "content": "You are a helpful assistant." },
          { "role": "user", "content": [  
              { 
                  "type": "text", 
                  "text": system_prompt
              },
              { 
                  "type": "image_url",
                  "image_url": {
                      "url": f"data:image/jpeg;base64,{base64_image}"
                  }
              }
          ] } 
      ],
      max_tokens=2000 
  )

  return response.choices[0].message.content

def gpt4v(image_path, system_prompt):
  base64_image = encode_image(image_path)

  # text_content = [{
  #   "type": "text",
  #   "text": question
  # }]

  image_contents = [{
  "type": "image_url",
  "image_url": {
    "url": f"data:image/jpeg;base64,{base64_image}",
    "detail": "low"
  }
  }]
  

  headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
  }
  
  payload = {
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": system_prompt,
    },
    {
      "role": "user",
      "content": image_contents,
    }
  ],
  "max_tokens": 1000,
  'temperature': 0,
  }

  start = time.time()
  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  end = time.time()
  if end - start < 4:
      time.sleep(4 - (end - start))

  return response.json()["choices"][0]["message"]["content"]



text_to_number = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'thirteen':13, 'fourteen':14, 'fifteen':15, 'sixteen':16, 'seventeen':17, 'eighteen':18, 'nineteen':19, 'twenty':20}