import os
from os.path import join as opj
cur_pth = os.path.dirname(os.path.abspath(__file__))

import json
import re

from .base_vlm import BaseVLM
from openai import OpenAI

from .ram_vlm_utils import prepare_inputs_chats
from .ram_vlm_utils import ChatMessage

from utils.config import get_api_config

class RamVLM(BaseVLM):
    def __init__(self, api_key=None, base_url=None):
        api_config = get_api_config()

        self.api_key = api_key if api_key is not None else api_config.api_key
        self.base_url = base_url if base_url is not None else api_config.base_url
        self.model_name = api_config.model_name
        self.max_tokens = api_config.max_tokens
        self.temperature = api_config.temperature

        self.chat_history = {
            "model": self.model_name,
            "messages": [],
            "max_tokens": self.max_tokens
        }

        missing = [
            name for name, value in (
                ("api_key", self.api_key),
                ("base_url", self.base_url),
                ("model_name", self.model_name),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing VLM API configuration values in config/config.yaml: "
                + ", ".join(missing)
            )

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def clear_history(self):
        self.chat_history = {
            "model": self.model_name,
            "messages": [],
            "max_tokens": self.max_tokens
        }
        
    def inference(self, chat_messages:list, preload_message=[], use_history=False, max_tokens=2000):
        message = preload_message
        message.extend(chat_messages)

        if use_history:
            payload = prepare_inputs_chats(message, chat_history=self.chat_history)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=payload['messages'],
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
            res = response.choices[0].message.content
            self.chat_history['messages'].append({
                "role": "assistant",
                "content": res,
            })
        else:
            payload = prepare_inputs_chats(message)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=payload['messages'],
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
            res = response.choices[0].message.content
        return res
    
    def infer_task_decomp_with_ref(self, instruction, instruction_image, image, RAM_object=None, max_tokens=20000):
        message = []
        for j in range(3):
            prompt_path = opj(cur_pth, 'assets', 'task_decomp_example_ref')
            text_prompt_path = f'{prompt_path}/prompt{j + 1}.md'
            image_prompt_path = f'{prompt_path}/prompt{j + 1}.png'

            if os.path.exists(text_prompt_path):
                with open(f'{prompt_path}/prompt{j + 1}.md', 'r', encoding='utf-8') as f:
                    text_prompt = f.read()
                message.append(ChatMessage(content=text_prompt, type="text"))
            elif os.path.exists(image_prompt_path):
                image_prompt = image_prompt_path
                message.append(ChatMessage(content=image_prompt, type="image_url"))
            else:
                raise ValueError(f"Prompt file {text_prompt_path} or {image_prompt_path} does not exist")
        
        instruction_promt = ""
        if RAM_object is not None:
            instruction_promt = "In the concated image above. The left image is the observation. Now, we want to rearrange the tableware in the left image to like the right image.\n" 
            instruction_promt += "\n In the left image, we already known that: \n" + RAM_object
        
        chat_messages = [
            ChatMessage(content=image, type="image_url"),
            ChatMessage(content=instruction_promt, type="text"),
            ChatMessage(content="Please give me your chain of thought and final answer next to rearrange the tableware in the left image to like the right image.", type="text"),
        ]
        res = self.inference(chat_messages, preload_message=message, max_tokens=max_tokens)
        return res
    

    def decomp_longtask2subtask(self, instruction, obs_image, RAM_object=None, prompt_type='task1', max_tokens=5000):
        message = [ChatMessage(content=obs_image, type="image_url"),]

        prompt_path = opj(cur_pth, 'assets', 'longtask_decomp')
        prompt_file_num = 1


        for j in range(prompt_file_num):
            text_prompt_path = f'{prompt_path}/prompt{j + 1}.md'
            image_prompt_path = f'{prompt_path}/prompt{j + 1}.png'

            if os.path.exists(text_prompt_path):
                with open(f'{prompt_path}/prompt{j + 1}.md', 'r', encoding='utf-8') as f:
                    text_prompt = f.read()
                if RAM_object is not None:
                    text_prompt = text_prompt.replace("<RAM_object></RAM_object>", str(RAM_object))
                if instruction is not None:
                    text_prompt = text_prompt.replace("<instruction></instruction>", instruction)
                message.append(ChatMessage(content=text_prompt, type="text"))
            elif os.path.exists(image_prompt_path):
                image_prompt = image_prompt_path
                message.append(ChatMessage(content=image_prompt, type="image_url"))
            else:
                raise ValueError(f"Prompt file {text_prompt_path} or {image_prompt_path} does not exist")
        
        res = self.inference(message, preload_message=[], max_tokens=max_tokens)
        return res
    

    def decomp_longtask2subtask_reference(self, obs_ref_image, RAM_object=None, max_tokens=5000):
        message = [ChatMessage(content=obs_ref_image, type="image_url"),]
        for j in range(1):
            prompt_path = opj(cur_pth, 'assets', 'longtask_decomp_reference')
            text_prompt_path = f'{prompt_path}/prompt{j + 1}.md'
            image_prompt_path = f'{prompt_path}/prompt{j + 1}.png'

            if os.path.exists(text_prompt_path):
                with open(f'{prompt_path}/prompt{j + 1}.md', 'r', encoding='utf-8') as f:
                    text_prompt = f.read()
                if RAM_object is not None:
                    text_prompt = text_prompt.replace("<RAM_object></RAM_object>", str(RAM_object))
                message.append(ChatMessage(content=text_prompt, type="text"))
            elif os.path.exists(image_prompt_path):
                image_prompt = image_prompt_path
                message.append(ChatMessage(content=image_prompt, type="image_url"))
            else:
                raise ValueError(f"Prompt file {text_prompt_path} or {image_prompt_path} does not exist")
        
        res = self.inference(message, preload_message=[], max_tokens=max_tokens)
        return res
    
    
    def decomp_subtask2action(self, instruction, image, RAM_object=None, prompt_type='default', max_tokens=5000):
        message = []

        prompt_path = opj(cur_pth, 'assets', 'action_decomp')
        prompt_file_num = 1

        for j in range(prompt_file_num):
            text_prompt_path = f'{prompt_path}/prompt{j + 1}.md'
            image_prompt_path = f'{prompt_path}/prompt{j + 1}.png'

            if os.path.exists(text_prompt_path):
                with open(f'{prompt_path}/prompt{j + 1}.md', 'r', encoding='utf-8') as f:
                    text_prompt = f.read()
                message.append(ChatMessage(content=text_prompt, type="text"))
            elif os.path.exists(image_prompt_path):
                image_prompt = image_prompt_path
                message.append(ChatMessage(content=image_prompt, type="image_url"))
            else:
                raise ValueError(f"Prompt file {text_prompt_path} or {image_prompt_path} does not exist")
            
        
        instruction_promt = ""
        if RAM_object is not None:
            instruction_promt = "From this observation of this new task, we already know following object information:\n" + RAM_object
        
        instruction_promt += f"\nNew task instruction: {instruction}.\n" \
            + "Give your chain of thought and your final answer next.\n"

        chat_messages = [
            ChatMessage(content=image, type="image_url"),
            ChatMessage(content=instruction_promt, type="text"),
        ]
        res = self.inference(chat_messages, preload_message=message, max_tokens=max_tokens)
        return res
    
    def infer_objects(self, instruction, image, RAM_keys=None, max_tokens=5000):
        message = []
        for j in range(3):
            prompt_path = opj(cur_pth, 'assets', 'find_objects')
            text_prompt_path = f'{prompt_path}/prompt{j + 1}.md'
            image_prompt_path = f'{prompt_path}/prompt{j + 1}.png'

            if os.path.exists(text_prompt_path):
                with open(f'{prompt_path}/prompt{j + 1}.md', 'r') as f:
                    text_prompt = f.read()

                if RAM_keys is not None:
                    text_prompt = text_prompt.replace("<RAM_keys></RAM_keys>", str(RAM_keys))
                
                message.append(ChatMessage(content=text_prompt, type="text"))
            elif os.path.exists(image_prompt_path):
                image_prompt = image_prompt_path
                message.append(ChatMessage(content=image_prompt, type="image_url"))
            else:
                raise ValueError(f"Prompt file {text_prompt_path} or {image_prompt_path} does not exist")
            
        chat_messages = [
            ChatMessage(content=instruction, type="text"),
            ChatMessage(content=image, type="image_url")
        ]
        res = self.inference(chat_messages, preload_message=message, max_tokens=max_tokens)
        return res
    
    def infer_objects_label(self, image, object_items, max_tokens=5000):
        message = []
        for j in range(3):
            prompt_path = opj(cur_pth, 'assets', 'find_objects')
            text_prompt_path = f'{prompt_path}/prompt{j + 1}.md'
            image_prompt_path = f'{prompt_path}/prompt{j + 1}.png'

            if os.path.exists(text_prompt_path):
                with open(f'{prompt_path}/prompt{j + 1}.md', 'r', encoding='utf-8') as f:
                    text_prompt = f.read()
                message.append(ChatMessage(content=text_prompt, type="text"))
            elif os.path.exists(image_prompt_path):
                image_prompt = image_prompt_path
                message.append(ChatMessage(content=image_prompt, type="image_url"))
            else:
                raise ValueError(f"Prompt file {text_prompt_path} or {image_prompt_path} does not exist")

        instruction_prompt = (
            f"In this observation, we already knonw:\n"
            f"```json\n"
            f"{object_items}"
            f"\n```\n"
            f"help identify the label of all objects and give me your answer."
        )
        chat_messages = [
            ChatMessage(content=image, type="image_url"),
            ChatMessage(content=instruction_prompt, type="text")
        ]
        
        res = self.inference(chat_messages, preload_message=message, max_tokens=max_tokens)
        return res
    
    @staticmethod
    def read_RAM_categories():
        ram_folder = opj(cur_pth, 'assets', 'ram_categories')
        ram_files = [f for f in os.listdir(ram_folder) if f.endswith('.json')]
        ram_contents = {}
        for ram_file in ram_files:
            ram_file_name = os.path.splitext(ram_file)[0]
            ram_file_path = opj(ram_folder, ram_file)
            with open(ram_file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            primitives = content["primitives"]
            grasp_primitives = [k for k, v in primitives.items() if k.startswith("grasp_")]
            contact_point_primitives = [k for k, v in primitives.items() if k.startswith("contact_point_")]
            hinge_primitives = [k for k, v in primitives.items() if k.startswith("hinge")]
            plane_primitives = [k for k, v in primitives.items() if k.startswith("plane_")]

            if len(grasp_primitives) > 0 or len(contact_point_primitives) > 0 or len(hinge_primitives) > 0:
                summary = (
                    f"- For an object beloning to category '{content['name']}', "
                    f"we define {len(grasp_primitives) + len(contact_point_primitives) + len(hinge_primitives)} grasp/contact point/hinge and {len(plane_primitives)} plane in this object as follows: "
                )
            else:
                summary = (
                    f"- For an object beloning to category '{content['name']}', "
                    f"we define {len(plane_primitives)} plane in this object as follows: "
                )

            content["text_description"] = f"{summary}"
            for primitive, primitive_meta in primitives.items():
                content["text_description"] += f"\n    - '{primitive}': '{primitive_meta['description']}'"
            ram_contents[ram_file_name] = content
        return ram_contents

    @staticmethod
    def parse_json_response(response):
        try:
            json_pattern = r'```json\s*(.*?)\s*```'
            match = re.search(json_pattern, response, re.DOTALL)
            
            if match:
                json_str = match.group(1)
                json_data = json.loads(json_str)
                return json_data
            else:
                print("JSON code block not found")
                return None
                
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"Other error: {e}")
            return None
        
    @staticmethod
    def parse_python_response(response):
        try:
            python_pattern = r'```python\s*(.*?)\s*```'
            match = re.search(python_pattern, response, re.DOTALL)
            
            if match:
                python_code = match.group(1)
                return python_code
            else:
                print("Python code block not found")
                return None
                
        except Exception as e:
            print(f"Other error: {e}")
            return None
