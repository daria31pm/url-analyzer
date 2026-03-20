"""
Модуль для обучения и использования ML модели
"""
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from .feature_extractor import URLFeatureExtractor


class URLClassifier:
    """
    Классификатор URL с использованием Random Forest и XGBoost
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Инициализация классификатора
        
        Args:
            model_type: тип модели ('random_forest' или 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = URLFeatureExtractor()
        self.is_fitted = False
        
    def _create_model(self):
        """Создает модель выбранного типа"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_split: float = 0.2) -> Dict:
        """
        Обучение модели
        
        Args:
            X: матрица признаков
            y: целевая переменная (0 - безопасный, 1 - фишинг/опасный)
            validation_split: доля данных для валидации
            
        Returns:
            словарь с метриками
        """
        # Разделение на train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Масштабирование
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Создание и обучение модели
        self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Предсказания на валидации
        y_pred = self.model.predict(X_val_scaled)
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        # Метрики
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'classification_report': classification_report(y_val, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        self.is_fitted = True
        
        return metrics
    
    def predict(self, url: str) -> Dict:
        """
        Предсказание для одного URL
        
        Args:
            url: URL для анализа
            
        Returns:
            словарь с предсказанием и вероятностями
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Извлечение признаков
        features = self.feature_extractor.extract_as_array(url)
        features_array = np.array(features).reshape(1, -1)
        
        # Масштабирование
        features_scaled = self.scaler.transform(features_array)
        
        # Предсказание
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Извлечение признаков в читаемом виде для объяснения
        feature_dict = self.feature_extractor.extract(url)
        
        return {
            'url': url,
            'is_malicious': bool(prediction),
            'confidence': float(probability[1] if prediction == 1 else probability[0]),
            'probability': {
                'safe': float(probability[0]),
                'malicious': float(probability[1])
            },
            'features': feature_dict
        }
    
    def predict_batch(self, urls: List[str]) -> List[Dict]:
        """Предсказание для списка URL"""
        return [self.predict(url) for url in urls]
    
    def explain_prediction(self, url: str) -> Dict:
        """
        Объяснение предсказания с указанием наиболее влиятельных признаков
        
        Args:
            url: URL для анализа
            
        Returns:
            словарь с объяснением
        """
        prediction = self.predict(url)
        
        # Получаем важность признаков из модели
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.feature_extractor.get_feature_names()
            
            # Сортируем по важности
            important_features = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Определяем, какие признаки повышают риск
            risk_factors = []
            features = prediction['features']
            
            for name, importance in important_features:
                value = features.get(name, 0)
                if value > 0.3:  # Порог для выделения риск-факторов
                    risk_factors.append({
                        'feature': name,
                        'value': value,
                        'importance': importance,
                        'explanation': self._get_feature_explanation(name, value)
                    })
            
            prediction['explanation'] = {
                'risk_factors': risk_factors,
                'summary': self._generate_summary(prediction, risk_factors)
            }
        
        return prediction
    
    def _get_feature_explanation(self, feature_name: str, value: float) -> str:
        """Возвращает человеко-читаемое объяснение для признака"""
        explanations = {
            'url_length': 'Длина URL превышает норму',
            'num_dots': 'Много точек в URL (возможно много поддоменов)',
            'has_ip': 'URL содержит IP-адрес вместо доменного имени',
            'is_shortened': 'Использован сервис сокращения ссылок',
            'num_suspicious_keywords': 'Обнаружены подозрительные ключевые слова',
            'has_at_symbol': 'Символ @ может указывать на попытку маскировки',
            'entropy': 'Высокая энтропия (случайный набор символов)',
            'percent_encoded_chars': 'Много закодированных символов',
            'num_redirects': 'Обнаружены множественные редиректы'
        }
        return explanations.get(feature_name, f'Признак {feature_name} имеет значение {value:.2f}')
    
    def _generate_summary(self, prediction: Dict, risk_factors: list) -> str:
        """Генерирует краткое резюме"""
        if prediction['is_malicious']:
            if risk_factors:
                factors_str = ', '.join([f['feature'] for f in risk_factors[:3]])
                return f"⚠️ URL классифицирован как подозрительный. Основные риск-факторы: {factors_str}. Уверенность: {prediction['confidence']:.1%}"
            else:
                return f"⚠️ URL классифицирован как подозрительный с уверенностью {prediction['confidence']:.1%}"
        else:
            return f"✅ URL классифицирован как безопасный с уверенностью {prediction['confidence']:.1%}"
    
    def save(self, path: str):
        """Сохраняет модель и скейлер"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_extractor.get_feature_names()
        }, path)
    
    def load(self, path: str):
        """Загружает модель и скейлер"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.model_type = data.get('model_type', 'random_forest')
        self.is_fitted = True


# Функции для удобства использования
def load_model(model_path: str = 'data/model.pkl') -> URLClassifier:
    """Загружает сохраненную модель"""
    classifier = URLClassifier()
    classifier.load(model_path)
    return classifier
