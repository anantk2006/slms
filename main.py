import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch.nn as nn

from train import train_taid
from dataset import get_dataloader


STUDENT_MODEL_ID = "google/gemma-3-1b-pt"
TEACHER_MODEL_ID = "google/gemma-7b"

tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
student_model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
student_model = get_peft_model(student_model, lora_config)

teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_ID)

dataloader = get_dataloader(batch_size=4, context_size=1024)

if torch.cuda.is_available():
    student_model = student_model.to("cuda")
    teacher_model = teacher_model.to("cuda")
    train_taid(teacher_model, student_model, tokenizer, dataloader, epochs=3, lr=1e-4, beta=0.9, alpha=0.1)
else:
    raise EnvironmentError("CUDA is not available. Please run on a machine with a GPU.")





