from datasets import Dataset 
from evaluate import load 
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration 
import torch 
import numpy as np 
import random 

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bertscore_metric = load("bertscore")

model = MBartForConditionalGeneration.from_pretrained("./mbart/2e-5/best")
model.to("cuda") 
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="en_XX")

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

batch = tokenizer(test_mod, return_tensors="pt", padding=True)
batch = {k: v.to(device) for k, v in batch.items()}
generated_tokens = model.generate(**batch, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

bertscore = bertscore_metric.compute(predictions=translation, references=test_og, lang="en")
print(f'bertscore_precision: {np.mean(bertscore["precision"])}, bertscore_recall: {np.mean(bertscore["recall"])}, bertscore_f1: {np.mean(bertscore["f1"])}')

