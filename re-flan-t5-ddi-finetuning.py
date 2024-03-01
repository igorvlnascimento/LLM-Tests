from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback, IntervalStrategy
import torch
import time
import numpy as np
from copy import copy
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def prompt_results(model_name, model, tokenizer, dataset):
    correct = 0
    file_txt = open(f'results.txt', 'a')
    file_txt.write(f'{model_name}\n')
    file_txt.write(print_number_of_trainable_model_parameters(model)+'\n')
    for index in range(int(len(dataset['test'])*.1)):
        query = dataset['test'][index]['query']
        answer = dataset['test'][index]['answer']

        prompt = copy(query)

        inputs = tokenizer(prompt, return_tensors='pt')
        output = tokenizer.decode(
            model.generate(
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

def compute_metrics(p):    
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    micro_f1 = f1_score(y_true=labels, y_pred=pred, average='micro')
    macro_f1 = f1_score(y_true=labels, y_pred=pred, average='macro')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "micro_f1": micro_f1, "macro_f1": macro_f1}

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

output_dir = f'./full-flan-re-checkpoint'

training_args = TrainingArguments(
    evaluation_strategy = IntervalStrategy.STEPS, # "steps"
    eval_steps = 1, # Evaluation and Save happens every 50 step
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=1,
    metric_for_best_model = 'micro_f1',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['valid'],
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

output_dir = f'./lora-flan-re-checkpoint'

training_args = TrainingArguments(
    evaluation_strategy = IntervalStrategy.STEPS, # "steps"
    eval_steps = 1, # Evaluation and Save happens every 50 step
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=1,
    metric_for_best_model = 'micro_f1',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['valid'],
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

instruct_model = AutoModelForSeq2SeqLM.from_pretrained("./full-flan-re-checkpoint", torch_dtype=torch.bfloat16)
lora_instruct_model = AutoModelForSeq2SeqLM.from_pretrained("./lora-flan-re-checkpoint", torch_dtype=torch.bfloat16)

prompt_results("Original Model", original_model, tokenizer, dataset)
prompt_results("Full finetuning", instruct_model, tokenizer, dataset)
prompt_results("Lora finetuning", lora_instruct_model, tokenizer, dataset)

