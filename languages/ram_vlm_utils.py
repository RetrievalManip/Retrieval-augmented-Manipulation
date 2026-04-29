import base64
from io import BytesIO
from typing import NamedTuple

metaprompt = ''' 
You are a robot with a parallel jaw gripper. You are capable of understanding and processing images, and you can also help to decompse a long-hozrizon taks into a sequence of subtasks.
'''  

class ChatMessage(NamedTuple):
    content: str
    type: str


def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def prepare_inputs_chats(
    chat_messages: list, 
    chat_history={"model": "", "messages": [], "max_tokens": 2000},
):
    content = []
    for chat_msg in chat_messages:
        if chat_msg.type == 'text':
            content.append({"type": "text", "text": chat_msg.content})
        elif chat_msg.type == 'image_url':
            image_path = chat_msg.content
            encode_function = encode_image_from_file if isinstance(image_path, str) else encode_image_from_pil
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_function(image_path)}"}})
        else:
            raise ValueError(f"Unknown type: {chat_msg.type}")

    if len(chat_history["messages"]) == 0:
        payload = {
            "model": chat_history.get("model", ""),
            "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": metaprompt
                    }
                ]
            }, 
            {
                "role": "user",
                "content": content
            }
            ],
            "max_tokens": 2000
        }
    else:
        payload = chat_history
        payload['messages'].append({
            "role": "user",
            "content": content
        })
    
    return payload
