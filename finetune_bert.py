"""
Fine-tune BERT-based models
"""
from math import floor
import argparse
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from custom_data_collator import DataCollatorForMultipleChoice
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


# Process examples
def preprocess(examples):
    """Tokenize and group the given examples"""
    n_choices = 4
    n_examples = len(examples['label'])
    first_sentences = [[context] * n_choices for context in examples["text"]]
    second_sentences = [[examples[f'choice{c}'][i] for c in range(n_choices)] for i in range(n_examples)]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, max_length=512)
    return {k: [v[i : i + n_choices] for i in range(0, len(v), n_choices)] for k, v in tokenized_examples.items()}


# Load dataset
dataset_manager = DatasetManager(ignore_case=False, force_4_choices=True, ds_format='bert')
if '|' in args.dataset:
    assert args.dataset.count('|') == 1, "Invalid number of datasets"
    primary_ds, secondary_ds = args.dataset.split('|')
    dataset = dataset_manager.load_combined_datasets(primary_ds, secondary_ds)
else:
    dataset = dataset_manager.load_ds(args.dataset)

# Calculate the log steps based on the number of steps in each epoch
effective_batch_size = args.batch_size * args.accumulation_steps
args.log_steps = floor(args.log_steps * len(dataset["train"]) / effective_batch_size)
args.log_steps = max(args.log_steps, 1)

# Load tokenizer and process dataset
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
dataset = dataset.map(preprocess, batched=True)

# Evaluation metrics
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    """Calculate accuracy metric"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    result = accuracy.compute(predictions=predictions, references=labels)
    return result


# Load model & start training
callback = CustomTrainerCallback(vars(args))
early_stopping = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
callbacks = [callback, early_stopping]
model = AutoModelForMultipleChoice.from_pretrained(args.checkpoint)

# Training args
training_args = TrainingArguments(
    # Saving
    output_dir=args.name,
    logging_dir=f"{args.name}/logs",
    save_strategy="steps",
    save_steps=args.log_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",

    # Loggging
    logging_strategy="steps",
    logging_steps=args.log_steps,

    # Training
    learning_rate=args.learning_rate,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.accumulation_steps,

    # Evaluation
    evaluation_strategy="steps",
    eval_steps=args.log_steps,
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    callbacks=callbacks
)

# Train
trainer.train()
