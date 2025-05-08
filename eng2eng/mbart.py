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

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_function(examples):
    model_inputs = tokenizer(examples["source"], padding="max_length", truncation=True, max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target"], padding="max_length", truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


login("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")

file = open("test.modern.txt", "r")
lines = file.readlines()
for i in lines: 
    i = i.strip() 
modern = lines[:500]
test_mod = lines[101]
file.close()
file = open("test.original.txt", "r")
lines = file.readlines()
for i in lines: 
    i = i.strip()
original = lines[:500]
test_og = lines[101]
file.close()

data = [{"source": modern, "target": original} for modern, original in zip(modern, original)]
test = [{"source": test_mod, "target": test_og}] 

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
model.to(device)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="en_XX")

optimizer = AdamW(model.parameters(), lr=0.01)

# model_inputs = tokenizer(modern, text_target=original, return_tensors="pt", padding=True)

dataset = Dataset.from_list(data)
train_dataset = dataset.map(tokenize_function, batched=True)
td = Dataset.from_list(test)
test_dataset = td.map(tokenize_function, batched=True) 

training_args = Seq2SeqTrainingArguments(
    output_dir="mbart",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch", 
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
    predict_with_generate=True, 
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()


"""model.train()

for i in tqdm.tqdm(range(len(modern))):
    model_inputs = tokenizer(modern[i], return_tensors="pt", padding=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(original[i], return_tensors="pt", padding=True).input_ids
    optimizer.zero_grad()
    output = model(**model_inputs, labels=labels) # forward pass
    loss = output.loss
    loss.backward()
    optimizer.step()"""

batch = tokenizer(test_mod, return_tensors="pt", padding=True)
batch = {k: v.to(device) for k, v in batch.items()}
generated_tokens = model.generate(**batch, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

print(test_mod)
print(test_og)
print(translation)

"""for i in range(10):
    print(">", modern[i])
    print("=", original[i])
    print("<", translation[i])"""



