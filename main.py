import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch.optim as optim

MODEL_ID = "google/gemma-3-1b-pt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to(torch.device("cpu"))



