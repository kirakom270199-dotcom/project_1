import re
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_dataset(url):
    """Загрузка датасета по URL"""
    try:
        response = requests.get(url)
        response.encoding = 'utf-8'
        response.raise_for_status()
        tweets = response.text.splitlines()
        raw_dataset = pd.DataFrame({'raw_text': tweets})
        print(f"Загружено твитов: {len(raw_dataset)}")
        return raw_dataset
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при загрузке данных: {e}")
        return pd.DataFrame()


def clean_string(text):
    """Функция для очищения и предобрабатывания текста"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_data(raw_dataset):
    """Предобработка данных"""
    raw_dataset['cleaned_text'] = raw_dataset['raw_text'].apply(clean_string)
    raw_dataset['cleaned_text'] = raw_dataset['cleaned_text'].fillna('')
    df_processed = raw_dataset[raw_dataset['cleaned_text'].str.len() > 0].copy()
    print(f"Осталось твитов после обработки: {len(df_processed)}")
    return df_processed


def split_data(df_processed, config):
    """Разделение данных на train/val/test"""
    train_df, temp_df = train_test_split(
        df_processed,
        test_size=config['data']['val_size'] + config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    val_size = config['data']['val_size'] / (config['data']['val_size'] + config['data']['test_size'])
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        random_state=config['data']['random_state']
    )

    print(f"Размеры выборок - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def save_datasets(train_df, val_df, test_df, raw_dataset, processed_df):
    """Сохранение датасетов"""
    raw_dataset.to_csv('data/raw_dataset.csv', index=False)
    processed_df.to_csv('data/dataset_processed.csv', index=False)
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    print("Датасеты сохранены в папку data/")