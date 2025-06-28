import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch


def create_tokenizer(train_ds):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(train_ds["Text"], trainer)

    return tokenizer


def get_dataloaders(dataset_path, max_len, batch_size):
    df = pd.read_csv(f"{dataset_path}/train.csv")
    train_ds = Dataset.from_pandas(df)

    df = pd.read_csv(f"{dataset_path}/test.csv")
    test_ds = Dataset.from_pandas(df)

    tokenizer = create_tokenizer(train_ds)
    tokenizer.enable_truncation(max_length=max_len)

    def collate_fn(batch):
        input_ids = []
        attention_masks = []
        labels = []

        tokens = tokenizer.encode_batch([example['Text'] for example in batch])

        pad_id = tokenizer.token_to_id("[PAD]")
        max_len = max(len(token.ids) for token in tokens) - 1
        for token in tokens:
            ids = token.ids[:-1]
            lbl = token.ids[1:]

            pad_len = max_len - len(ids)

            input_ids.append(ids + [pad_id] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)
            labels.append(lbl + [-100] * pad_len)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }
    

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return (train_loader, test_loader), tokenizer


if __name__=="__main__":
    (train_dl, test_dl), tokenizer = get_dataloaders('./data', 256, 8)

    x = next(iter(train_dl))
    print(x)