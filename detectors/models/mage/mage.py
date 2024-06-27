import torch
import os
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from tqdm import tqdm

from .utils.utils import preprocess, detect

class Mage:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = "yaful/MAGE"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device)

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm(texts):
            text = preprocess(text)
            result = detect(text, self.tokenizer, self.model, self.device)
            machine_generated_prob = 0.0
            if result == "machine-generated":
                machine_generated_prob = 1.0
            predictions.append(machine_generated_prob)
        return predictions
