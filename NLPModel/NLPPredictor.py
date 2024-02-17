from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

class NLPPredictor():
    def __init__(self):
        model_path = "khygopole/NLP_HerbalMultilabelClassification"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path) 
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def predict_given_text(self, input_text):
        encoding = self.tokenizer(input_text, return_tensors="pt")
        encoding = {k: v.to(self.model.device) for k,v in encoding.items()}
        outputs = self.model(**encoding)
        logits = outputs.logits

        #Apply Sigmoid and Set Thresholding for Probabilities
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1

        return predictions

    def transform_predictions_result(self, predictions_arr):
        ans_cols = ["JACKFRUIT", "SAMBONG", "LEMON", "JASMINE", "MANGO", "MINT", "AMPALAYA", "MALUNGGAY", "GUAVA", "LAGUNDI"]
        ans_cols = [x.capitalize() for x in ans_cols]
        predicted_labels = [ans_cols[idx] for idx, label in enumerate(predictions_arr) if label == 1.0]

        return predicted_labels
