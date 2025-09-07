import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModelWithProjection


search_sentence = "a basketball player performing a slam dunk"

model = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
tokenizer = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")

inputs = tokenizer(text=search_sentence , return_tensors="pt")
outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

# Normalize embeddings for retrieval:
final_output = outputs[0] / outputs[0].norm(dim=-1, keepdim=True)
final_output = final_output.cpu().detach().numpy()
print("final output: ", final_output)
