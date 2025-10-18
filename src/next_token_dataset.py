import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class LanguageModelDataset(Dataset):
    def __init__(self, tokenized_texts, max_len=128):
        self.texts = []
        self.labels = []

        for tokens in tokenized_texts:
            if len(tokens) < 2:
                continue
            input_tokens = tokens[:-1][:max_len]
            target_tokens = tokens[1:][:max_len]
            min_len = min(len(input_tokens), len(target_tokens))
            if min_len > 0:
                self.texts.append(input_tokens[:min_len])
                self.labels.append(target_tokens[:min_len])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.texts[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def create_data_loaders(train_tokens, val_tokens, test_tokens, batch_size=32):
    """Создание DataLoader'ов"""
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id

    def collate_fn(batch):
        pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=pad_token)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (padded_inputs != pad_token).long()

        return {
            'input_ids': padded_inputs,
            'attention_mask': attention_mask,
            'labels': padded_labels
        }

    train_dataset = LanguageModelDataset(train_tokens)
    val_dataset = LanguageModelDataset(val_tokens)
    test_dataset = LanguageModelDataset(test_tokens)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"DataLoader'ы созданы: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")

    return train_loader, val_loader, test_loader