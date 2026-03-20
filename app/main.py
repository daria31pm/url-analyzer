"""
FastAPI приложение для анализа URL
"""
import os
import logging
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np

from .models import (
    URLRequest, URLBatchRequest, URLPrediction, 
    HealthResponse, TrainRequest, TrainResponse
)
from .ml_model import URLClassifier
from .feature_extractor import URLFeatureExtractor

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание приложения
app = FastAPI(
    title="URL Analyzer API",
    description="Интеллектуальный анализатор URL для выявления фишинговых и вредоносных ссылок",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Шаблоны
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Глобальные переменные
classifier: URLClassifier = None
feature_extractor = URLFeatureExtractor()
model_path = Path(__file__).parent.parent / "data" / "model.pkl"

# Создание директории для данных
data_dir = Path(__file__).parent.parent / "data"
data_dir.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Загрузка модели при старте"""
    global classifier
    try:
        if model_path.exists():
            classifier = URLClassifier()
            classifier.load(str(model_path))
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning("No pre-trained model found. Train the model first.")
            classifier = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        classifier = None


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Главная страница с веб-интерфейсом"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "title": "URL Analyzer"}
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния сервиса"""
    return HealthResponse(
        status="healthy",
        model_loaded=classifier is not None and classifier.is_fitted,
        model_type=classifier.model_type if classifier and classifier.is_fitted else None
    )


@app.post("/api/analyze", response_model=URLPrediction)
async def analyze_url(request: URLRequest):
    """
    Анализ одного URL
    
    - **url**: URL для анализа
    """
    if classifier is None or not classifier.is_fitted:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        prediction = classifier.explain_prediction(request.url)
        return URLPrediction(**prediction)
    except Exception as e:
        logger.error(f"Error analyzing URL {request.url}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/analyze/batch", response_model=List[URLPrediction])
async def analyze_batch(request: URLBatchRequest):
    """
    Пакетный анализ URL (до 100 URL)
    """
    if classifier is None or not classifier.is_fitted:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first."
        )
    
    if len(request.urls) > 100:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 100 URLs per batch request"
        )
    
    predictions = []
    for url in request.urls:
        try:
            pred = classifier.explain_prediction(url)
            predictions.append(URLPrediction(**pred))
        except Exception as e:
            logger.error(f"Error analyzing URL {url}: {e}")
            predictions.append(
                URLPrediction(
                    url=url,
                    is_malicious=False,
                    confidence=0.0,
                    probability={"safe": 0.0, "malicious": 0.0},
                    features=feature_extractor.extract(url),
                    explanation=None
                )
            )
    
    return predictions


@app.get("/api/features/{url:path}")
async def get_features(url: str):
    """
    Получение признаков URL без классификации
    """
    try:
        features = feature_extractor.extract(url)
        return {"url": url, "features": features}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    Обучение модели на новых данных
    
    Ожидается CSV файл с колонками:
    - url: строка URL
    - label: 0 (безопасный) или 1 (вредоносный)
    """
    global classifier
    
    try:
        # Загрузка данных
        df = pd.read_csv(request.data_path)
        
        if 'url' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'url' and 'label' columns")
        
        # Извлечение признаков
        logger.info(f"Extracting features from {len(df)} URLs...")
        features_list = []
        for url in df['url']:
            try:
                features = feature_extractor.extract_as_array(url)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Error extracting features from {url}: {e}")
                continue
        
        X = np.array(features_list)
        y = df.loc[:len(X)-1, 'label'].values
        
        # Обучение модели
        logger.info(f"Training {request.model_type} model...")
        classifier = URLClassifier(model_type=request.model_type)
        metrics = classifier.fit(X, y, validation_split=0.2)
        
        # Сохранение модели
        classifier.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        return TrainResponse(
            success=True,
            metrics=metrics,
            message=f"Model trained successfully with accuracy: {metrics['accuracy']:.4f}"
        )
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/train/upload")
async def train_from_upload(
    file: UploadFile = File(...),
    model_type: str = Form("random_forest")
):
    """
    Обучение модели из загруженного CSV файла
    """
    # Сохраняем загруженный файл
    file_path = data_dir / "uploaded_train.csv"
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Вызываем обучение
    request = TrainRequest(data_path=str(file_path), model_type=model_type)
    return await train_model(request)


@app.get("/api/model/info")
async def model_info():
    """Информация о текущей модели"""
    if classifier is None or not classifier.is_fitted:
        return {"status": "no_model"}
    
    return {
        "status": "loaded",
        "model_type": classifier.model_type,
        "feature_names": feature_extractor.get_feature_names(),
        "num_features": len(feature_extractor.get_feature_names())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
