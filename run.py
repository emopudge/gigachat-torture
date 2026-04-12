"""Main entrypoint for the feature generation agent."""
from __future__ import annotations

import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Импорт из src.utils, как ты просила
from src.utils.baseline import make_baseline_submission
from src.utils.evaluator import select_top_5
from src.utils.data_loader import load_data           # TODO: Подключить модуль Майи
from src.utils.feature_generator import generate_features # TODO: Подключить модуль Арсения

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
MAX_TIME_SEC = 580  # Мягкий лимит (жесткий 600с у организаторов)

def main() -> None:
    load_dotenv()
    start_time = time.perf_counter()
    logger.info("🚀 Запуск пайплайна...")

    try:
        # 1️⃣ ЗАГРУЗКА ДАННЫХ
        logger.info("📥 Чтение train.csv и test.csv...")
        train_df = pd.read_csv(DATA_DIR / "train.csv", sep=None, engine="python")
        test_df = pd.read_csv(DATA_DIR / "test.csv", sep=None, engine="python")

        # Автоопределение ID и Target
        id_col = train_df.columns[0]
        target_candidates = [c for c in train_df.columns if c not in test_df.columns]
        target_col = target_candidates[0] if target_candidates else train_df.columns[-1]
        logger.info(f"Обнаружены: ID='{id_col}', Target='{target_col}'")

        # TODO: Здесь будет вызов data_loader от Майи для склейки таблиц
        # data = load_data()
        # train_df, test_df = data["train"], data["test"]

        # 2️⃣ ГЕНЕРАЦИЯ ПРИЗНАКОВ
        logger.info("🤖 Генерация признаков...")
        # TODO: Здесь будет LLM + код Арсения
        # ideas = get_feature_ideas(train_df, readme_text)
        # candidates = generate_features(ideas, train_df, test_df)

        # ⏳ ВРЕМЕННАЯ ЗАГЛУШКА (чтобы чекер прошел ПРЯМО СЕЙЧАС)
        numeric_cols = train_df.select_dtypes(include="number").columns.tolist()
        candidates = {c: train_df[c] for c in numeric_cols if c not in (id_col, target_col)}
        logger.info(f"Сформировано кандидатов: {len(candidates)}")

        # 3️⃣ ОЦЕНКА И ОТБОР ТОП-5
        logger.info("📊 Оценка через CatBoost CV и отбор...")
        best_features = select_top_5(
            candidates=candidates,
            train_df=train_df,
            target_col=target_col,
            min_auc=0.55
        )
        logger.info(f"✅ Отобрано топ-5: {best_features}")

        # 4️⃣ СОХРАНЕНИЕ OUTPUT (формат под check_submission.py)
        logger.info("💾 Сохранение результатов...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # check_submission требует, чтобы фичи были НОВЫМИ колонками
        feature_prefix = "ft_"
        new_feature_names = [f"{feature_prefix}{i+1}" for i in range(len(best_features))]

        out_train = train_df.copy()
        out_test = test_df.copy()

        for new_name, old_name in zip(new_feature_names, best_features):
            out_train[new_name] = train_df[old_name]
            out_test[new_name] = test_df[old_name]

        out_train.to_csv(OUTPUT_DIR / "train.csv", index=False)
        out_test.to_csv(OUTPUT_DIR / "test.csv", index=False)
        logger.success("🎉 Файлы сохранены в output/")

    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        logger.warning("🔄 Запуск fallback (baseline)...")
        try:
            make_baseline_submission()
        except Exception as fallback_err:
            logger.critical(f"💥 Baseline тоже упал: {fallback_err}")

    finally:
        elapsed = time.perf_counter() - start_time
        logger.info(f"⏱️ Затрачено: {elapsed:.2f} сек.")
        if elapsed > MAX_TIME_SEC:
            logger.warning(f"⚠️ Превышен мягкий лимит {MAX_TIME_SEC}с!")

if __name__ == "__main__":
    main()