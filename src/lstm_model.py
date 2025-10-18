import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_size=128, pad_token_id=0):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        # Слои модели
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(emb_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        # Эмбеддинг
        x = self.embedding(input_ids)  # [batch_size, seq_len, emb_dim]

        # LSTM
        rnn_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size]

        # Выходные логиты
        logits = self.fc(rnn_out)  # [batch_size, seq_len, vocab_size]

        return logits

    @torch.no_grad()
    def generate_next_tokens(self, input_ids, max_new_tokens=50):
        """Простая генерация - всегда выбираем самый вероятный токен"""
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = self.forward(generated)  # [batch_size, seq_len, vocab_size]

            # Берем логиты для последнего токена и выбираем самый вероятный
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch_size, 1]

            # Добавляем к сгенерированной последовательности
            generated = torch.cat([generated, next_token], dim=1)

            # Останавливаемся если достигли pad token
            if (next_token == self.pad_token_id).all():
                break

        return generated