import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import torch


def create_tokenizer(path):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"])
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train([path], trainer)
    return tokenizer


def get_dataloaders(dataset_path, batch_size, seq_len=8):
    with open(f"{dataset_path}/train.csv") as f:
        lines = f.readlines()[1:]
        train_corpus = "\n".join(lines)

    with open(f"{dataset_path}/test.csv") as f:
        lines = f.readlines()[1:]
        test_corpus = "\n".join(lines)

    tokenizer = create_tokenizer(f"{dataset_path}/train.csv")

    train_tokens = tokenizer.encode(train_corpus)
    test_tokens = tokenizer.encode(test_corpus)

    train_idxs = torch.arange(0, len(train_tokens) - seq_len - 1, seq_len).long()
    test_idxs = torch.arange(0, len(test_tokens) - seq_len - 1, seq_len).long()

    def collate_fn_train(batch):
        input_ids = []
        labels = []

        for idx in batch:
            input_ids.append(train_tokens.ids[idx:idx+seq_len])
            labels.append(train_tokens.ids[idx+1:idx+seq_len+1])

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }
    
    def collate_fn_test(batch):
        input_ids = []
        labels = []

        for idx in batch:
            input_ids.append(test_tokens.ids[idx:idx+seq_len])
            labels.append(test_tokens.ids[idx+1:idx+seq_len+1])

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }
    

    train_dl = DataLoader(train_idxs, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_train)
    test_dl = DataLoader(test_idxs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_test)

    return (train_dl, test_dl), tokenizer


if __name__=="__main__":
    (train_dl, test_dl), tokenizer = get_dataloaders("./data", 8)

    print(next(iter(train_dl)))
    