import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from loguru import logger

# Параметры из scoring.py
CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 0,
    "thread_count": 1,
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
}


def evaluate_feature(X: pd.DataFrame, y: pd.Series, feature_name: str = None) -> float:
    """
    Оценивает один признак через 5-fold CV на CatBoost.
    Возвращает mean ROC-AUC.
    """
    try:
        # Заполняем пропуски (CatBoost не любит NaN)
        X_filled = X.fillna(-999)
        
        # Определяем категориальные признаки
        cat_features = [
            i for i, col in enumerate(X_filled.columns) 
            if X_filled[col].dtype == "object"
        ] or None
        
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        
        # 5-fold CV
        scores = cross_val_score(
            model, X_filled, y, 
            cv=5, 
            scoring="roc_auc",
            n_jobs=1  # важно: 1 поток, чтобы не конфликтовало с thread_count в CatBoost
        )
        
        mean_auc = float(scores.mean())
        
        if feature_name:
            logger.debug(f"{feature_name}: CV AUC = {mean_auc:.4f} ± {scores.std():.4f}")
        
        return mean_auc
        
    except Exception as e:
        logger.warning(f"Ошибка оценки признака {feature_name}: {e}")
        return 0.5  # fallback: случайный классификатор


def select_top_5(
    candidates: dict[str, pd.Series], 
    train_df: pd.DataFrame, 
    target_col: str,
    min_auc: float = 0.55
) -> list[str]:
    """
    Оценивает все кандидаты и возвращает топ-5 по ROC-AUC.
    
    candidates: dict {feature_name: pd.Series с признаком}
    train_df: исходный train для извлечения target
    target_col: имя целевой колонки
    min_auc: если лучший признак < этого порога → возвращаем 5 исходных числовых колонок
    """
    if not candidates:
        logger.warning("Нет кандидатов для оценки, возвращаем фоллбэк")
        return _get_fallback_features(train_df, target_col)
    
    scores = {}
    y = train_df[target_col]
    
    for name, series in candidates.items():
        # Пропускаем, если признак полностью NaN
        series_pd = pd.Series(series) if not isinstance(series, pd.Series) else series
        if series_pd.isna().all():
            continue
            
        X = pd.DataFrame({name: series})
        scores[name] = evaluate_feature(X, y, feature_name=name)
    
    if not scores:
        logger.warning("Все признаки отбракованы, возвращаем фоллбэк")
        return _get_fallback_features(train_df, target_col)
    
    # Если лучший признак слабее порога → фоллбэк
    best_score = max(scores.values())
    if best_score < min_auc:
        logger.info(f"Лучший признак {max(scores, key=scores.get)} имеет AUC={best_score:.3f} < {min_auc}, используем фоллбэк")
        return _get_fallback_features(train_df, target_col)
    
    # Сортируем и берём топ-5
    top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"Топ-5 признаков: {[name for name, _ in top_5]}")
    
    return [name for name, _ in top_5]


def _get_fallback_features(train_df: pd.DataFrame, target_col: str, n: int = 5) -> list[str]:
    """Возвращает первые n числовых колонок (кроме target) как фоллбэк."""
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c != target_col][:n]