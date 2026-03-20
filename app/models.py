"""
Pydantic модели для API
"""
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional, Any


class URLRequest(BaseModel):
    """Запрос на анализ URL"""
    url: str = Field(..., description="URL для анализа", example="https://example.com")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com"
            }
        }


class URLBatchRequest(BaseModel):
    """Пакетный запрос на анализ URL"""
    urls: List[str] = Field(..., description="Список URL для анализа", max_items=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "urls": ["https://example.com", "https://google.com"]
            }
        }


class FeatureResponse(BaseModel):
    """Признаки URL"""
    url_length: float
    num_dots: float
    num_hyphens: float
    num_underscores: float
    num_slashes: float
    num_equals: float
    num_question_marks: float
    num_and: float
    num_digits: float
    num_letters: float
    has_ip: float
    is_shortened: float
    num_subdomains: float
    domain_length: float
    path_length: float
    query_length: float
    num_suspicious_keywords: float
    has_at_symbol: float
    has_double_slash: float
    entropy: float
    num_redirects_in_path: float
    has_https: float
    has_http_only: float
    percent_encoded_chars: float
    path_depth: float


class RiskFactor(BaseModel):
    """Фактор риска с объяснением"""
    feature: str
    value: float
    importance: float
    explanation: str


class Explanation(BaseModel):
    """Объяснение предсказания"""
    risk_factors: List[RiskFactor]
    summary: str


class URLPrediction(BaseModel):
    """Результат предсказания для URL"""
    url: str
    is_malicious: bool
    confidence: float
    probability: Dict[str, float]
    features: FeatureResponse
    explanation: Optional[Explanation] = None


class HealthResponse(BaseModel):
    """Ответ на проверку здоровья сервиса"""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None


class TrainRequest(BaseModel):
    """Запрос на обучение модели"""
    data_path: str = Field(..., description="Путь к CSV файлу с данными")
    model_type: str = Field("random_forest", description="Тип модели: random_forest или xgboost")
    
    class Config:
        json_schema_extra = {
            "example": {
                "data_path": "data/train.csv",
                "model_type": "random_forest"
            }
        }


class TrainResponse(BaseModel):
    """Ответ на обучение модели"""
    success: bool
    metrics: Dict[str, Any]
    message: str
