from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
from copy import copy
from peft import LoraConfig, get_peft_model, TaskType

file_txt = open('results.txt', 'w')

huggingface_dataset_name = "YBXL/DDI2013_test"

dataset = load_dataset(huggingface_dataset_name)

model_name='google/flan-t5-base'

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    example['input_ids'] = tokenizer(example["query"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["answer"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
    return example

# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'query', 'answer', 'choices', 'gold'])

lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

peft_model = get_peft_model(original_model, 
                            lora_config)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(peft_model))

file_txt.write(print_number_of_trainable_model_parameters(peft_model)+'\n')

output_dir = f'./peft-re-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=1,
    logging_steps=1,
    max_steps=1    
)
    
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)

peft_trainer.train()

peft_model_path="./peft-re-checkpoint-local"

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

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