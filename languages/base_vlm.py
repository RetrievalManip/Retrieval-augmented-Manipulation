from abc import ABC, abstractmethod

class BaseVLM():
    def __init__(self, config:str):
        pass
    
    @abstractmethod
    def inference(self, text_promt, image):
        answer = ""
        return answer

    