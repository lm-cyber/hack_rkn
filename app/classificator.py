

#model 

# import torch  
#import onnxruntime as ort 
import numpy as np
import random


class Classificator:
    def __init__(self):
        pass
    
    def __call__(self, image):
        return random.randint(0, 9)

    
    def predict_proba_class(self, image):
        return random.randint(0, 100)/100
    
    def predict(self, image):
        return random.randint(0, 9)
    def predict_embedding(self, image):
        return np.random.rand(3).astype(np.float32).tolist()
    
    def predict_probs(self, image):
        probs= np.random.rand(3)
        return (np.exp(probs) / np.sum(np.exp(probs), axis=0)).astype(np.float32).tolist()

    def predict_result(self, image):
        return {
            "class": random.randint(0, 9),
            "predict_prob": self.predict_probs(image),
            "probs_class": self.predict_proba_class(image),
            "embedding": self.predict_embedding(image)
        }


classificator_instance = Classificator()