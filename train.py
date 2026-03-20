#!/usr/bin/env python
"""
Скрипт для обучения модели на подготовленных данных
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Добавляем путь к приложению
sys.path.insert(0, str(Path(__file__).parent))

from app.ml_model import URLClassifier
from app.feature_extractor import URLFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(output_path: str = "data/train.csv", num_samples: int = 1000):
    """
    Создает пример данных для обучения
    """
    import random
    
    # Список безопасных URL
    safe_urls = [
        "https://google.com",
        "https://github.com",
        "https://stackoverflow.com",
        "https://python.org",
        "https://youtube.com",
        "https://reddit.com",
        "https://wikipedia.org",
        "https://amazon.com",
        "https://microsoft.com",
        "https://apple.com",
    ]
    
    # Список подозрительных паттернов
    suspicious_patterns = [
        "http://secure-login-verify.com",
        "http://paypal-account-verification.xyz",
        "http://bank-of-america-update.ru",
        "http://facebook-security-alert.net",
        "http://apple-id-confirm.cn",
        "http://microsoft-account-unlock.info",
        "http://amazon-order-confirmation.xyz",
        "http://netflix-account-suspended.biz",
        "http://dropbox-verify-login.ru",
        "http://instagram-verify-account.top",
    ]
    
    # Генерация случайных URL
    safe_list = []
    malicious_list = []
    
    # Безопасные URL
    for i in range(num_samples // 2):
        if i < len(safe_urls):
            url = safe_urls[i % len(safe_urls)]
        else:
            # Генерация безопасного URL
            domains = ["example.com", "test.org", "demo.net", "sample.io", "company.ru"]
            paths = ["", "/about", "/contact", "/products", "/blog"]
            url = f"https://{random.choice(domains)}{random.choice(paths)}"
        safe_list.append(url)
    
    # Вредоносные URL
    for i in range(num_samples // 2):
        if i < len(suspicious_patterns):
            url = suspicious_patterns[i % len(suspicious_patterns)]
        else:
            # Генерация подозрительного URL
            domains = ["secure-login.xyz", "verify-account.ru", "bank-update.net", "paypal-confirm.biz"]
            paths = ["/login", "/verify", "/confirm", "/update", "/secure"]
            params = ["?id=123", "?token=abc", "?redirect=home"]
            url = f"http://{random.choice(domains)}{random.choice(paths)}{random.choice(params)}"
        malicious_list.append(url)
    
    # Создание DataFrame
    data = []
    for url in safe_list:
        data.append({"url": url, "label": 0})
    for url in malicious_list:
        data.append({"url": url, "label": 1})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Сохранение
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Created sample data with {len(df)} samples at {output_path}")
    
    return df


def train_model(data_path: str, model_type: str = "random_forest", 
                output_path: str = "data/model.pkl"):
    """
    Обучение модели
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    if 'url' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'url' and 'label' columns")
    
    logger.info(f"Extracting features from {len(df)} URLs...")
    extractor = URLFeatureExtractor()
    
    features_list = []
    valid_indices = []
    
    for idx, url in enumerate(df['url']):
        try:
            features = extractor.extract_as_array(url)
            features_list.append(features)
            valid_indices.append(idx)
        except Exception as e:
            logger.warning(f"Error extracting features from {url}: {e}")
            continue
    
    X = np.array(features_list)
    y = df.iloc[valid_indices]['label'].values
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Обучение
    logger.info(f"Training {model_type} model...")
    classifier = URLClassifier(model_type=model_type)
    metrics = classifier.fit(X, y, validation_split=0.2)
    
    logger.info(f"Training complete!")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Cross-validation mean: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    
    # Сохранение
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    classifier.save(output_path)
    logger.info(f"Model saved to {output_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train URL classifier model")
    parser.add_argument("--data", type=str, default="data/train.csv",
                        help="Path to training data CSV")
    parser.add_argument("--model-type", type=str, default="random_forest",
                        choices=["random_forest", "xgboost"],
                        help="Type of model to train")
    parser.add_argument("--output", type=str, default="data/model.pkl",
                        help="Path to save the model")
    parser.add_argument("--create-sample", action="store_true",
                        help="Create sample data before training")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of samples for sample data")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data(args.data, args.samples)
    
    train_model(args.data, args.model_type, args.output)


if __name__ == "__main__":
    main()
