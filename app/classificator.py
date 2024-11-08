

#model 

# import torch  
#import onnxruntime as ort 

import random


class Classificator:
    def __init__(self):
        pass
    
    def __call__(self, image):
        return random.randint(0, 9)

    
    def batch_classificator(self, images):
        pass

classificator_instance = Classificator()