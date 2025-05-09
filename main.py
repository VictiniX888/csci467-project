from datasets import Dataset
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import evaluate
import numpy as np


TEST_SRC_FNAME = "test_short.src"
TEST_TGT_FNAME = "test_short.tgt"

batch_size = 16  # change to 64 for full evaluation


def read_data(fname):
    with open(fname, encoding="utf-8") as f:
        lines = f.read().splitlines()

    return lines


def load_dataset(src_fname, tgt_fname):
    src = read_data(src_fname)
    tgt = read_data(tgt_fname)
    dataset = Dataset.from_dict({"src": src, "tgt": tgt})
    return dataset


def zh2zh(dataset):
    max_input_length = 30

    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50", src_lang="zh_CN", tgt_lang="zh_CN"
    )
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    model = MBartForConditionalGeneration.from_pretrained("./checkpoint-zh2zh")
    model.to("cuda")

    def generate_pred(batch):
        inputs = tokenizer(
            batch["src"],
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
        )

        # all special tokens including will be removed
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        batch["zh_mod"] = output_str

        return batch

    results = dataset.map(generate_pred, batched=True, batch_size=batch_size)
    return results


def zh2en(dataset):
    max_input_length = 30

    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt", src_lang="zh_CN", tgt_lang="en_XX"
    )
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    model.to("cuda")

    def generate_pred(batch):
        inputs = tokenizer(
            batch["zh_mod"],
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
        )

        # all special tokens including will be removed
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        batch["en_mod"] = output_str

        return batch

    results = dataset.map(generate_pred, batched=True, batch_size=batch_size)
    return results


def en2en(dataset):
    max_input_length = 120

    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50", src_lang="en_XX", tgt_lang="en_XX"
    )
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    model = MBartForConditionalGeneration.from_pretrained("./checkpoint-en2en")
    model.to("cuda")

    def generate_pred(batch):
        inputs = tokenizer(
            batch["en_mod"],
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
        )

        # all special tokens including will be removed
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        batch["en_old"] = output_str

        return batch

    results = dataset.map(generate_pred, batched=True, batch_size=batch_size)
    return results


def main():
    test_data = load_dataset(TEST_SRC_FNAME, TEST_TGT_FNAME)
    # test_data = test_data.shuffle()
    # test_data = test_data.select(range(10))

    data_zh_mod = zh2zh(test_data)
    data_en_mod = zh2en(data_zh_mod)
    results = en2en(data_en_mod)

    # orig_str = results["src"]
    pred_str = results["en_old"]
    label_str = results["tgt"]

    print()
    for i in range(10):
        print(results["src"][i])
        print(results["zh_mod"][i])
        print(results["en_mod"][i])
        print(results["en_old"][i])
        print(results["tgt"][i])
        # print(label_str[i])
        print()

    bertscore = evaluate.load("bertscore")
    bertscore_output = bertscore.compute(
        predictions=pred_str, references=label_str, lang="en"
    )
    print("Precision", np.mean(bertscore_output["precision"]))
    print("Recall", np.mean(bertscore_output["recall"]))
    print("F1", np.mean(bertscore_output["f1"]))


if __name__ == "__main__":
    main()
