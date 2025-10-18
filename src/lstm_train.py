import torch
import torch.nn as nn
from tqdm import tqdm
import os
from .lstm_model import LSTMModel


def train_one_epoch(model, loader, optimizer, criterion, device, epoch_num, max_grad_norm=1.0):
    """Тренировка одной эпохи"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num} Training")

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        V = logits.size(-1)
        loss = criterion(logits.view(-1, V), labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return total_loss / len(loader)


@torch.no_grad()
def validate_loss(model, loader, criterion, device):
    """Валидация loss"""
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Validation Loss"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        logits = model(input_ids)
        V = logits.size(-1)
        loss = criterion(logits.view(-1, V), labels.view(-1))
        total_loss += loss.item()
    return total_loss / len(loader)


def train_model(model, train_loader, val_loader, config, device):
    """Полный цикл обучения модели"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float('inf')
    patience_counter = 0
    MODEL_SAVE_PATH = "models/trained_lstm_model.pth"
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\\n{'=' * 60}")
        print(f"ЭПОХА {epoch}/{config['training']['epochs']}")
        print(f"{'=' * 60}")

        # Тренировка
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            config['training']['max_grad_norm']
        )
        print(f"Train Loss: {train_loss:.4f}")

        # Валидация
        val_loss = validate_loss(model, val_loader, criterion, device)
        print(f"Val Loss:   {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Модель улучшилась! Сохранена в {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"🚫 Patience counter: {patience_counter}/{config['training']['patience']}")

        if patience_counter >= config["training"]["patience"]:
            print("🛑 Early stopping triggered!")
            break

    print("Обучение завершено!")
    return MODEL_SAVE_PATH