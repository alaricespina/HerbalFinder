from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

model_path = "khygopole/NLP_HerbalMultilabelClassification"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

inputText = input("Input your symptom to get the herbal prediction: ")

encoding = tokenizer(inputText, return_tensors="pt")
encoding = {k: v.to(model.device) for k,v in encoding.items()}
outputs = model(**encoding)
logits = outputs.logits
#Apply Sigmoid and Set Thresholding for Probabilities
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)
raw_predictions = predictions.copy()
predictions[np.where(probs >= 0.5)] = 1

#Convert Predicted Labels into Actual Names of Herbal
ans_cols = ["JACKFRUIT", "SAMBONG", "LEMON", "JASMINE", "MANGO", "MINT", "AMPALAYA", "MALUNGGAY", "GUAVA", "LAGUNDI"]
predicted_labels = [ans_cols[idx] for idx, label in enumerate(predictions) if label == 1.0]

print("Raw Predictions:")
print(probs)
print("Predictions:")
print(predictions)
print("Predicted Labels:")
print(predicted_labels)

