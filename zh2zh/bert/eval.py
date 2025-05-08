from datasets import Dataset
import evaluate
from transformers import BertTokenizer, EncoderDecoderModel


def read_zh_data(fname):
    with open(fname, encoding="utf-8") as f:
        lines = f.read().splitlines()

    return lines


def load_dataset(src_fname, tgt_fname):
    src = read_zh_data(src_fname)
    tgt = read_zh_data(tgt_fname)
    dataset = Dataset.from_dict({"src": src, "tgt": tgt})
    return dataset


TEST_SRC_FNAME = "test.src"
TEST_TGT_FNAME = "test.tgt"

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

model = EncoderDecoderModel.from_pretrained("./checkpoint-71500")
model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
model.to("cuda")

sacrebleu = evaluate.load("sacrebleu")

test_data = load_dataset(TEST_SRC_FNAME, TEST_TGT_FNAME)

# only use 16 training examples for notebook - DELETE LINE FOR FULL TRAINING
test_data = test_data.select(range(2000))

batch_size = 16  # change to 64 for full evaluation


# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(
        batch["src"],
        padding="max_length",
        truncation=True,
        max_length=30,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch


results = test_data.map(generate_summary, batched=True, batch_size=batch_size)

orig_str = results["src"]
pred_str = results["pred"]
label_str = results["tgt"]

print()
for i in range(10):
    print(orig_str[i])
    print(pred_str[i])
    print(label_str[i])
    print()

sacrebleu_output = sacrebleu.compute(
    predictions=pred_str, references=label_str, tokenize="zh"
)

print(sacrebleu_output)
