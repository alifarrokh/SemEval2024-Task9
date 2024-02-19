"""
Fine-tune T5-based models
"""
import re
from math import floor
import argparse
import numpy as np
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
from Levenshtein import ratio as sim_ratio
from load_datasets import DatasetManager
from custom_trainer_callback import CustomTrainerCallback


# Args
parser = argparse.ArgumentParser(description="Fine-tune T5-based models on BrainTeaser")
parser.add_argument("--dataset", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--tokenizer")
parser.add_argument("--name", required=True)
parser.add_argument(
    "--log_steps",
    type=float,
    required=True,
    help="A float number in range [0,1] specifying a ratio of epochs"
)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--accumulation_steps", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--early_stopping_patience", type=int, default=10)
args = parser.parse_args()
args.tokenizer = args.checkpoint if args.tokenizer is None else args.tokenizer
assert 0 < args.log_steps <= 1, "Invalid value for log_steps"

# Load dataset
dataset_manager = DatasetManager(ignore_case=True, force_4_choices=False, ds_format='t5')
if '|' in args.dataset:
    assert args.dataset.count('|') == 1, "Invalid number of datasets"
    primary_ds, secondary_ds = args.dataset.split('|')
    raw_dataset = dataset_manager.load_combined_datasets(primary_ds, secondary_ds)
else:
    raw_dataset = dataset_manager.load_ds(args.dataset)

# Calculate the log steps based on the number of steps in each epoch
effective_batch_size = args.batch_size * args.accumulation_steps
args.log_steps = floor(args.log_steps * len(raw_dataset["train"]) / effective_batch_size)
args.log_steps = max(args.log_steps, 1)

# Load tokenizer of FLAN-t5
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, legacy=False)

# Find max source/target length
max_source_length = 512
max_target_length = None

if max_source_length is None:
    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([raw_dataset["train"], raw_dataset["test"]]).map(
        lambda x: tokenizer(x["text"], truncation=True),
        batched=True,
        remove_columns=['text', 'label'],
    )
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])

if max_target_length is None:
    # The maximum total sequence length for target text after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([raw_dataset["train"], raw_dataset["test"]]).map(
        lambda x: tokenizer(x["label"], truncation=True),
        batched=True,
        remove_columns=['text', 'label'],
    )
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])

print(f"Max source length: {max_source_length}")
print(f"Max target length: {max_target_length}")


def preprocess_function(sample: Dataset, padding: str = "max_length") -> dict:
    """Preprocess the dataset"""

    # add prefix to the input for t5
    inputs = list(sample["text"])

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=sample["label"],
        max_length=max_target_length,
        padding=padding,
        truncation=True
    )

    # Replace tokenizer.pad_token_id by -100 to ignore padding in the loss
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Prepare dataset
dataset = raw_dataset.map(preprocess_function, batched=True, remove_columns=['text', 'label'])


def compute_metrics(eval_preds):
    """Compute evaluationn metrics"""
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    n = len(decoded_labels)

    # Compute accuracy using Levenshtein distance
    test_ds = raw_dataset["test"]
    acc_sum = 0
    for i in range(n):
        if test_ds[i]['label'].strip().lower() != decoded_labels[i].strip().lower():
            print(test_ds[i]['label'].strip().lower())
            print(decoded_labels[i].strip().lower())
        assert test_ds[i]['label'].strip().lower() == decoded_labels[i].strip().lower()
        _q, choices = [p.strip() for p in test_ds[i]["text"].split('\\n')]
        choices = re.split(r'\([abcdeABCDE]\)', choices)
        choices = filter(lambda c: len(c) > 0, choices)
        choices = list(map(lambda c: c.strip().lower(), choices))
        true_key = np.argmax([c == test_ds[i]["label"].strip().lower() for c in choices])
        gt_label = decoded_preds[i].strip().lower()
        sim_scores = [sim_ratio(gt_label, c, score_cutoff=0.5) for c in choices]
        pred_key = np.argmax(sim_scores)
        acc_sum += int(true_key == pred_key)

    acc = acc_sum / n
    result = {"accuracy": acc}
    return result


# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

# Callbacks
callback = CustomTrainerCallback(vars(args))
early_stopping = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
callbacks = [callback, early_stopping]

# Training args
training_args = Seq2SeqTrainingArguments(
    # Saving
    output_dir=args.name,
    logging_dir=f"{args.name}/logs",
    save_strategy="steps",
    save_steps=args.log_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",

    # Logging
    logging_strategy="steps",
    logging_steps=args.log_steps,

    # Training
    learning_rate=args.learning_rate,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.accumulation_steps,
    predict_with_generate=True,

    # Evaluation
    evaluation_strategy="steps",
    eval_steps=args.log_steps,
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)

# Train
trainer.train()
