import numpy as np
from datasets import load_dataset
import tokenizers.decoders
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import TemplateProcessing
import tokenizers
from tqdm import tqdm


# Prepares Wikitext-103 dataset
# Tokenizer vocab size: 20000
# --------------------
# Train dataset info:
# Maximum token id in train.bin: 19999
# Number of tokens: 120386279
# --------------------
# Val dataset info:
# Maximum token id in val.bin: 19998
# Number of tokens: 251964

# === Tokenizer parameters ===
VOCAB_SIZE = 20000
SPECIAL_TOKENS = ["<|endoftext|>", "[UNK]"]
TOKENIZER_PATH = "wikitext103_tokenizer.json"

# === Load dataset ===
train_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
val_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
# test_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")


# Prepare the training data for the tokenizer
def get_training_corpus():
    for sample in train_dataset:
        yield sample['text']


# === Train custom BPE tokenizer ===
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
tokenizer.train_from_iterator(get_training_corpus(), trainer)

tokenizer.post_processor = TemplateProcessing(
single="$A " + "<|endoftext|>",
special_tokens=[
    ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>"))
],
)   
tokenizer.decoder = tokenizers.decoders.ByteLevel()
tokenizer.save(TOKENIZER_PATH)

# === Load trained tokenizer in transformers format ===
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_PATH,
    eos_token="<|endoftext|>",
    unk_token="[UNK]",
    pad_token="[PAD]"  # Optional, but can be useful
)
# tokenizer._tokenizer.decoder = tokenizers.decoders.ByteLevel()
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")


# === Tokenize and save dataset ===
def tokenize_and_save(dataset, tokenizer: PreTrainedTokenizerFast, filename: str, dtype=np.uint16):
    token_ids = []

    for sample in tqdm(dataset):
        text = sample["text"]
        if text.strip() != "":
            ids = tokenizer.encode(text)
            token_ids.extend(ids)
        else:
            continue

    all_tokens = np.array(token_ids, dtype=dtype)
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # Magic number
    header[1] = 1         # Version
    header[2] = all_tokens.size  # Number of tokens

    with open(filename, 'wb') as f:
        f.write(header.tobytes())
        all_tokens.tofile(f)


tokenize_and_save(train_dataset, tokenizer, "train.bin")
tokenize_and_save(val_dataset, tokenizer, "val.bin")

# === Verify the maximum token ID is within the vocabulary size ===
def verify_tokens(filename):
    with open(filename, 'rb') as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    max_token_id = tokens.max()
    print(f"Maximum token id in {filename}: {max_token_id}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Number of tokens: {header[2]}")

print("Train dataset info:")
verify_tokens("train.bin")
print("-" * 20)
print("Val dataset info:")
verify_tokens("val.bin")
print("Data preparation is complete.")
