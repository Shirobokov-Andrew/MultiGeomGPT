import numpy as np
from datasets import load_dataset
import tokenizers.decoders
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import TemplateProcessing
import tokenizers


# Prepares Wikitext2 dataset
# Tokenizer vocab size: 1000
# --------------------
# Train dataset info:
# Maximum token id in train.bin: 999
# Number of tokens: 4467969
# --------------------
# Val dataset info:
# Maximum token id in val.bin: 999
# Number of tokens: 695869

# === Tokenizer parameters ===
VOCAB_SIZE = 1000
SPECIAL_TOKENS = ["<|endoftext|>", "[UNK]"]
TOKENIZER_PATH = "wikitext2_tokenizer.json"

# === Load dataset ===
train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")


# === Train custom BPE tokenizer ===
def train_tokenizer(texts, vocab_size, tokenizer_path):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer)

    tokenizer.post_processor = TemplateProcessing(
    single="$A " + "<|endoftext|>",
    special_tokens=[
        ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>"))
    ],
)   
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    tokenizer.save(tokenizer_path)


texts = [t for t in train_dataset["text"] if t.strip() != ""] + [t for t in test_dataset["text"][:len(test_dataset["text"]) // 2] if t.strip() != ""]
train_tokenizer(texts, VOCAB_SIZE, TOKENIZER_PATH)


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
def tokenize_and_save(dataset, test_dataset, tokenizer: PreTrainedTokenizerFast, filename: str, split: str, dtype=np.uint16):
    if split == "train":
        texts = [t for t in dataset["text"] if t.strip() != ""] + [t for t in test_dataset["text"][:len(test_dataset["text"]) // 2] if t.strip() != ""]
    else:
        texts = [t for t in dataset["text"] if t.strip() != ""] + [t for t in test_dataset["text"][len(test_dataset["text"]) // 2:] if t.strip() != ""]
    token_ids = []

    for text in texts:
        ids = tokenizer.encode(text)
        token_ids.extend(ids)

    all_tokens = np.array(token_ids, dtype=dtype)
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # Magic number
    header[1] = 1         # Version
    header[2] = all_tokens.size  # Number of tokens

    with open(filename, 'wb') as f:
        f.write(header.tobytes())
        all_tokens.tofile(f)


tokenize_and_save(train_dataset, test_dataset, tokenizer, "train.bin", "train")
tokenize_and_save(val_dataset, test_dataset, tokenizer, "val.bin", "val")

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
