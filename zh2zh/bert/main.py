from transformers import (
    EncoderDecoderModel,
    BertTokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import Dataset
import evaluate


batch_size = 16  # change to 16 for full training
encoder_max_length = 30
decoder_max_length = 30

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
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    # load sacrebleu for validation
    sacrebleu = evaluate.load("sacrebleu")

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(
            batch["src"],
            padding="max_length",
            truncation=True,
            max_length=encoder_max_length,
            return_tensors="pt",
        )
        outputs = tokenizer(
            batch["tgt"],
            padding="max_length",
            truncation=True,
            max_length=decoder_max_length,
            return_tensors="pt",
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        output_ids = outputs.input_ids
        shifted_input_ids = output_ids.new_zeros(output_ids.shape)
        shifted_input_ids[:, :-1] = output_ids[:, 1:].clone()  # del CLS token
        shifted_input_ids[:, -1] = tokenizer.pad_token_id  # append [PAD] token
        batch["labels"] = shifted_input_ids

        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        sacrebleu_score = sacrebleu.compute(
            predictions=pred_str, references=label_str, tokenize="zh"
        )["score"]

        return {
            "score": round(sacrebleu_score, 4),
        }

    train_data = load_dataset(TRAIN_SRC_FNAME, TRAIN_TGT_FNAME)
    val_data = load_dataset(DEV_SRC_FNAME, DEV_TGT_FNAME)

    # only use 32 training examples for notebook - DELETE LINE FOR FULL TRAINING
    # train_data = train_data.select(range(32))
    # print(train_data)

    train_data = train_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["tgt", "src"],
    )
    train_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # only use 16 training examples for notebook - DELETE LINE FOR FULL TRAINING
    # val_data = val_data.select(range(16))

    val_data = val_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["tgt", "src"],
    )
    val_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(
        "bert-base-chinese", "bert-base-chinese"
    )

    # set special tokens
    bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
    bert2bert.config.eos_token_id = tokenizer.eos_token_id
    bert2bert.config.pad_token_id = tokenizer.pad_token_id
    bert2bert.generation_config.decoder_start_token_id = tokenizer.bos_token_id

    # sensible parameters for beam search
    bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size
    bert2bert.config.max_length = 30
    bert2bert.config.min_length = 5
    bert2bert.config.no_repeat_ngram_size = 3
    bert2bert.config.early_stopping = True
    bert2bert.config.length_penalty = 2.0
    bert2bert.config.num_beams = 4

    # set training arguments - these params are not really tuned, feel free to change
    training_args = Seq2SeqTrainingArguments(
        output_dir="./",
        eval_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
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
        model=bert2bert,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    trainer.train()


if __name__ == "__main__":
    main()
