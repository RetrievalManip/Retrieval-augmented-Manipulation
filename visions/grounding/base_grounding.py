from abc import ABC, abstractmethod


class BaseGrounding(ABC):
    @abstractmethod
    def grounding(self, image, text_prompt):
        raise NotImplementedError
