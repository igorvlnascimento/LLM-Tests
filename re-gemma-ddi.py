from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer#, GenerationConfig, TrainingArguments, Trainer
import torch
# import time
# import evaluate
# import pandas as pd
# import numpy as np
from copy import copy

file_txt = open('results.txt', 'w')

huggingface_dataset_name = "YBXL/DDI2013_test"

dataset = load_dataset(huggingface_dataset_name)

model_name='google/gemma-2b-it'

original_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(original_model))

file_txt.write(print_number_of_trainable_model_parameters(original_model)+'\n')


correct = 0
for index in range(int(len(dataset['test'])*.1)):
    query = dataset['test'][index]['query']
    answer = dataset['test'][index]['answer']

    prompt = copy(query)

    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        original_model.generate(
            inputs["input_ids"], 
            max_new_tokens=200,
        )[0], 
        skip_special_tokens=True
    )

    dash_line = '-'.join('' for x in range(100))
    if output.lower() == answer.lower():
        correct += 1

    file_txt.write(dash_line+'\n')
    print(dash_line)
    file_txt.write(f'INPUT PROMPT:\n{prompt}'+'\n')
    print(f'INPUT PROMPT:\n{prompt}')
    file_txt.write(dash_line+'\n')
    print(dash_line)
    file_txt.write(f'BASELINE HUMAN ANSWER:\n{answer}'+'\n')
    print(f'BASELINE HUMAN ANSWER:\n{answer}\n')
    file_txt.write(dash_line+'\n')
    print(dash_line)
    file_txt.write(f'MODEL GENERATION - ZERO SHOT:\n{output}'+'\n')
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

accuracy = correct / len(dataset['test'])
file_txt.write(f'ACCURACY:\n{accuracy}')
print(f"Accuracy: {accuracy}")
file_txt.close()