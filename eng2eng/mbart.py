from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from torch.optim import AdamW
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from huggingface_hub import login 
from datasets import Dataset 
import torch
import tqdm as tqdm
import numpy as np
import optuna
from evaluate import load

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_function(examples):
    model_inputs = tokenizer(examples["source"], padding="max_length", truncation=True, max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target"], padding="max_length", truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


bertscore_metric = load("bertscore")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    bertscore = bertscore_metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")

    return {
        "bertscore_precision": np.mean(bertscore["precision"]),
        "bertscore_recall": np.mean(bertscore["recall"]),
        "bertscore_f1": np.mean(bertscore["f1"]),
    }

login("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")

file = open("train.modern.txt", "r")
lines = file.readlines()
for i in lines:
    i = i.strip()
modern = lines
file.close()
file = open("train.original.txt", "r")
lines = file.readlines()
for i in lines:
    i = i.strip()
original = lines
file.close()
file = open("test.modern.txt", "r")
lines = file.readlines()
for i in lines:
    i = i.strip()
test_mod = lines
file.close()
file = open("test.original.txt", "r")
lines = file.readlines()
for i in lines:
    i = i.strip()
test_og = lines
file.close()
file = open("valid.modern.txt", "r")
lines = file.readlines()
for i in lines:
    i = i.strip()
val_mod = lines
file.close()
file = open("valid.original.txt", "r")
lines = file.readlines()
for i in lines:
    i = i.strip()
val_og = lines
file.close()

data = [{"source": modern, "target": original} for modern, original in zip(modern, original)]
test = [{"source": test_mod, "target": test_og} for test_mod, test_og in zip(test_mod, test_og)]
val = [{"source": val_mod, "target": val_og} for val_mod, val_og in zip(val_mod, val_og)]

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
model.to("cuda")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="en_XX")

# optimizer = AdamW(model.parameters(), lr=0.01)

# model_inputs = tokenizer(modern, text_target=original, return_tensors="pt", padding=True)

dataset = Dataset.from_list(data)
train_dataset = dataset.map(tokenize_function, batched=True)
td = Dataset.from_list(test)
test_dataset = td.map(tokenize_function, batched=True)
vd = Dataset.from_list(val)
val_dataset = vd.map(tokenize_function, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="mbart/2e-5",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bertscore_f1",
    greater_is_better=True,
    push_to_hub=True,
    predict_with_generate=True,
    save_total_limit=3,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("mbart/2e-5/best")
eval = trainer.evaluate()
print("Evaluation Results:")
for key, value in eval.items():
    print(f"{key}: {value}")