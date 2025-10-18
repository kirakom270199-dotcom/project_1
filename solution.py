# solution.py - Главный файл, объединяющий все модули

import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer
import warnings

warnings.filterwarnings('ignore')

# Добавляем путь к src для импорта модулей
sys.path.append('src')

# Импорт всех модулей
from data_utils import download_dataset, preprocess_data, split_data, save_datasets
from next_token_dataset import create_data_loaders
from lstm_model import LSTMModel
from lstm_train import train_model
from eval_lstm import evaluate_lstm_generation
from eval_transformer_pipeline import (
    load_transformer_model,
    evaluate_transformer_generation,
    compare_models
)


def main():
    """Главная функция, объединяющая все этапы проекта"""

    print("🚀 ЗАПУСК ПРОЕКТА АВТОДОПОЛНЕНИЯ ТЕКСТОВ")
    print("=" * 60)

    # Конфигурация (можно вынести в отдельный файл)
    config = {
        'data': {
            'url': "https://code.s3.yandex.net/deep-learning/tweets.txt",
            'train_size': 0.8,
            'val_size': 0.1,
            'test_size': 0.1,
            'random_state': 42,
            'max_length': 128
        },
        'model': {
            'embedding_dim': 64,
            'hidden_dim': 64,
            'vocab_size': 50257,
            'pad_token_id': 0
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 1,
            'max_grad_norm': 1.0,
            'patience': 3
        },
        'evaluation': {
            'max_examples': 200,
            'print_examples': 3,
            'max_new_tokens': 50
        },
        'transformer': {
            'model_name': "distilgpt2",
            'temperature': 0.7
        }
    }

    # Определение устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Используемое устройство: {device}")

    # ЭТАП 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
    print("\n" + "=" * 60)
    print("ЭТАП 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("=" * 60)

    # Загрузка данных
    print("📥 Загрузка датасета...")
    raw_dataset = download_dataset(config['data']['url'])

    if raw_dataset.empty:
        print("❌ Не удалось загрузить данные")
        return

    # Предобработка данных
    print("🧹 Предобработка данных...")
    processed_df = preprocess_data(raw_dataset)

    # Разделение на train/val/test
    print("📊 Разделение данных...")
    train_df, val_df, test_df = split_data(processed_df, config)

    # Сохранение датасетов
    print("💾 Сохранение датасетов...")
    save_datasets(train_df, val_df, test_df, raw_dataset, processed_df)

    # Токенизация
    print("🔤 Токенизация текстов...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    def tokenize_texts(text_list):
        return tokenizer(text_list, truncation=True, padding=False)["input_ids"]

    train_tokens = tokenize_texts(train_df['cleaned_text'].tolist())
    val_tokens = tokenize_texts(val_df['cleaned_text'].tolist())
    test_tokens = tokenize_texts(test_df['cleaned_text'].tolist())

    print(f"✅ Токенизация завершена: Train={len(train_tokens)}, Val={len(val_tokens)}, Test={len(test_tokens)}")

    # Создание DataLoader'ов
    print("📦 Создание DataLoader'ов...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_tokens, val_tokens, test_tokens, config['training']['batch_size']
    )

    # ЭТАП 2: ОБУЧЕНИЕ LSTM МОДЕЛИ
    print("\n" + "=" * 60)
    print("ЭТАП 2: ОБУЧЕНИЕ LSTM МОДЕЛИ")
    print("=" * 60)

    # Создание модели
    print("🧠 Создание LSTM модели...")
    model = LSTMModel(
        vocab_size=len(tokenizer),
        emb_dim=config['model']['embedding_dim'],
        hidden_size=config['model']['hidden_dim'],
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    ).to(device)

    print(f"📏 Размер модели: {sum(p.numel() for p in model.parameters()):,} параметров")

    # Обучение модели
    print("🎯 Начало обучения...")
    model_path = train_model(model, train_loader, val_loader, config, device)

    # ЭТАП 3: ОЦЕНКА LSTM МОДЕЛИ
    print("\n" + "=" * 60)
    print("ЭТАП 3: ОЦЕНКА LSTM МОДЕЛИ")
    print("=" * 60)

    print("📊 Оценка качества LSTM модели...")
    lstm_rouge, lstm_preds, lstm_refs = evaluate_lstm_generation(
        model, test_loader, tokenizer, device,
        max_examples=config['evaluation']['max_examples'],
        print_examples=config['evaluation']['print_examples']
    )

    if lstm_rouge:
        print("\n✅ LSTM МЕТРИКИ:")
        for metric, score in lstm_rouge.items():
            print(f"   {metric}: {score:.4f}")
    else:
        print("❌ Не удалось вычислить метрики для LSTM модели")

    # ЭТАП 4: ОЦЕНКА TRANSFORMER МОДЕЛИ
    print("\n" + "=" * 60)
    print("ЭТАП 4: ОЦЕНКА TRANSFORMER МОДЕЛИ")
    print("=" * 60)

    print("🤖 Загрузка предобученной Transformer модели...")
    transformer_tokenizer, transformer_model = load_transformer_model(config['transformer']['model_name'])

    if transformer_model is None:
        print("❌ Не удалось загрузить Transformer модель")
        return

    print("📊 Оценка качества Transformer модели...")
    transformer_rouge, transformer_preds, transformer_refs = evaluate_transformer_generation(
        test_loader, tokenizer, transformer_tokenizer, transformer_model,
        max_examples=config['evaluation']['max_examples'],
        print_examples=config['evaluation']['print_examples']
    )

    if transformer_rouge:
        print("\n✅ TRANSFORMER МЕТРИКИ:")
        for metric, score in transformer_rouge.items():
            print(f"   {metric}: {score:.4f}")
    else:
        print("❌ Не удалось вычислить метрики для Transformer модели")

    # ЭТАП 5: СРАВНЕНИЕ И РЕКОМЕНДАЦИИ
    print("\n" + "=" * 60)
    print("ЭТАП 5: СРАВНЕНИЕ МОДЕЛЕЙ И РЕКОМЕНДАЦИИ")
    print("=" * 60)

    # Расчет параметров
    lstm_params = sum(p.numel() for p in model.parameters())
    transformer_params = sum(p.numel() for p in transformer_model.parameters())

    # Сравнение моделей
    comparison_results = compare_models(
        {'rouge': lstm_rouge},
        {'rouge': transformer_rouge},
        lstm_params,
        transformer_params
    )

    # Вывод рекомендаций
    print_recommendations(comparison_results)

    # Демонстрация работы моделей
    demonstrate_models(model, transformer_model, tokenizer, transformer_tokenizer, device)

    print("\n🎉 ПРОЕКТ УСПЕШНО ЗАВЕРШЕН!")


def print_recommendations(comparison_results):
    """Вывод рекомендаций для разработчиков"""
    print("\n" + "=" * 80)
    print("💡 РЕКОМЕНДАЦИИ ДЛЯ РАЗРАБОТЧИКОВ")
    print("=" * 80)

    lstm_rouge1 = comparison_results['lstm']['rouge'].get('rouge1', 0)
    transformer_rouge1 = comparison_results['transformer']['rouge'].get('rouge1', 0)

    lstm_params = comparison_results['lstm']['params']
    transformer_params = comparison_results['transformer']['params']

    print(f"\n📈 КАЧЕСТВО (ROUGE-1):")
    print(f"   LSTM: {lstm_rouge1:.4f}")
    print(f"   Transformer: {transformer_rouge1:.4f}")

    print(f"\n💾 РАЗМЕР МОДЕЛЕЙ:")
    print(f"   LSTM: {lstm_params:,} параметров")
    print(f"   Transformer: {transformer_params:,} параметров")

    quality_ratio = transformer_rouge1 / lstm_rouge1 if lstm_rouge1 > 0 else float('inf')
    size_ratio = transformer_params / lstm_params

    print(f"\n⚖️  СООТНОШЕНИЕ:")
    print(f"   Качество: {quality_ratio:.2f}x")
    print(f"   Размер: {size_ratio:.1f}x")


def demonstrate_models(lstm_model, transformer_model, lstm_tokenizer, transformer_tokenizer, device):
    """Демонстрация работы обеих моделей на примерах"""
    print("\n" + "=" * 80)
    print("🎭 ДЕМОНСТРАЦИЯ РАБОТЫ МОДЕЛЕЙ")
    print("=" * 80)

    test_phrases = [
        "I love this",
        "The weather is",
        "I want to",
        "This is amazing",
        "I think that"
    ]

    print("\nТестируем на фразах:")
    for i, phrase in enumerate(test_phrases, 1):
        print(f"\n{i}. '{phrase}...'")
        print("-" * 40)

        # LSTM генерация
        try:
            input_tokens = lstm_tokenizer.encode(phrase, return_tensors="pt").to(device)
            lstm_model.eval()
            with torch.no_grad():
                lstm_output = lstm_model.generate_next_tokens(input_tokens, max_new_tokens=20)
                lstm_text = lstm_tokenizer.decode(lstm_output[0].cpu().tolist(), skip_special_tokens=True)
                lstm_completion = lstm_text[len(phrase):].strip()
            print(f"   LSTM: {lstm_completion}")
        except Exception as e:
            print(f"   LSTM: ошибка генерации")

        # Transformer генерация
        try:
            inputs = transformer_tokenizer.encode(phrase, return_tensors="pt")
            with torch.no_grad():
                outputs = transformer_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 20,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=transformer_tokenizer.eos_token_id
                )
            transformer_text = transformer_tokenizer.decode(outputs[0], skip_special_tokens=True)
            transformer_completion = transformer_text[len(phrase):].strip()
            print(f"   Transformer: {transformer_completion}")
        except Exception as e:
            print(f"   Transformer: ошибка генерации")


if __name__ == "__main__":
    main()