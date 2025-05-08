from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import Dataset
import evaluate


batch_size = 16  # change to 16 for full training
max_length = 30

TRAIN_SRC_FNAME = "train.src"
TRAIN_TGT_FNAME = "train.tgt"
DEV_SRC_FNAME = "dev.src"
DEV_TGT_FNAME = "dev.tgt"


def read_zh_data(fname):
    with open(fname, encoding="utf-8") as f:
        lines = f.read().splitlines()

    return lines


def load_dataset(src_fname, tgt_fname):
    src = read_zh_data(src_fname)
    tgt = read_zh_data(tgt_fname)
    dataset = Dataset.from_dict({"src": src, "tgt": tgt})
    return dataset


def main():
    # Adapted from https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing#scrollTo=H1F58j028eTV
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50", src_lang="zh_CN", tgt_lang="zh_CN"
    )
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    model.to("cuda")

    # load sacrebleu for validation
    sacrebleu = evaluate.load("sacrebleu")

    def tokenize_function(batch):
        model_inputs = tokenizer(
            batch["src"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["tgt"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        sacrebleu_score = sacrebleu.compute(
            predictions=pred_str, references=label_str, tokenize="zh"
        )["score"]

        return {
            "score": round(sacrebleu_score, 4),
        }

    train_data = load_dataset(TRAIN_SRC_FNAME, TRAIN_TGT_FNAME)
    val_data = load_dataset(DEV_SRC_FNAME, DEV_TGT_FNAME)

    train_data = train_data.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
    )

    val_data = val_data.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
    )

    # set training arguments - these params are not really tuned, feel free to change
    training_args = Seq2SeqTrainingArguments(
        output_dir="./",
        eval_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        learning_rate=0.00005,
        weight_decay=0,
        num_train_epochs=3,
        logging_steps=1000,  # set to 1000 for full training
        save_steps=500,  # set to 500 for full training
        eval_steps=8000,  # set to 8000 for full training
        warmup_steps=2000,  # set to 2000 for full training
        # max_steps=16,  # delete for full training
        overwrite_output_dir=True,
        save_total_limit=3,
        fp16=True,
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    trainer.train()


if __name__ == "__main__":
    main()
