from datasets import Dataset
import evaluate
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import numpy as np

import random


def read_zh_data(fname):
    with open(fname, encoding="utf-8") as f:
        lines = f.read().splitlines()

    return lines


def load_dataset(src_fname, tgt_fname):
    src = read_zh_data(src_fname)
    tgt = read_zh_data(tgt_fname)
    dataset = Dataset.from_dict({"src": src, "tgt": tgt})
    return dataset


TEST_SRC_FNAME = "dev.src"
TEST_TGT_FNAME = "dev.tgt"

batch_size = 16  # change to 64 for full evaluation
max_length = 30

tokenizer = MBart50TokenizerFast.from_pretrained(
    "facebook/mbart-large-50", src_lang="zh_CN", tgt_lang="zh_CN"
)
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

model = MBartForConditionalGeneration.from_pretrained("./checkpoint-71500")
model.to("cuda")

bertscore = evaluate.load("bertscore")

test_data = load_dataset(TEST_SRC_FNAME, TEST_TGT_FNAME)

# only use 16 training examples for notebook - DELETE LINE FOR FULL TRAINING
# test_data = test_data.select(range(2000))


# map data correctly
def generate_pred(batch):
    inputs = tokenizer(
        batch["src"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
    )

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch


results = test_data.map(generate_pred, batched=True, batch_size=batch_size)

orig_str = results["src"]
pred_str = results["pred"]
label_str = results["tgt"]

print()
for i in random.sample(range(len(orig_str)), 10):
    print(orig_str[i])
    print(pred_str[i])
    print(label_str[i])
    print()

bertscore_output = bertscore.compute(
    predictions=pred_str, references=label_str, lang="zh"
)

print("Precision", np.mean(bertscore_output["precision"]))
print("Recall", np.mean(bertscore_output["recall"]))
print("F1", np.mean(bertscore_output["f1"]))
