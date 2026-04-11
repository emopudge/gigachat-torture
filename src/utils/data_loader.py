import os
from dotenv import load_dotenv
from gigachat import GigaChat
import json
import pandas as pd
from pathlib import Path

# Абсолютный путь к корню проекта (поднимаемся из src/utils/ на 2 уровня вверх)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
README_PATH = PROJECT_ROOT / "data" / "readme.txt"

load_dotenv()  # автоматически находит и читает .env в корне проекта
key = os.getenv("GIGACHAT_CREDENTIALS")
scope = os.getenv("GIGACHAT_SCOPE")

readme = README_PATH.read_text(encoding="utf-8")


def get_features(readme):
    
    giga = GigaChat(credentials=key,
    verify_ssl_certs=False, 
    profanity_check=False,
    scope='GIGACHAT_API_CORP')

    PROMPT = f"""Ты — эксперт по анализу табличных данных.
Тебе дано описание датасета:
*{readme}*
Задача — бинарная классификация таргета (целевого столбца) с помощью CatBoost.

Ответь ТОЛЬКО в JSON формате. Никакого другого текста, комментариев, пояснений или markdown-разметки.
Вывод должен начинаться с "[" и заканчиваться "]".

Опиши каждый столбец по формату:
Формат JSON:
[
  {{
    "column": "название столбца или группы (только из описания)",
    "reason": "почему этот столбец/группа может влиять на таргет (причинно-следственная связь, без общих фраз)",
    "feature_ideas": [
      {{
        "operation": "операция (например: group_agg, binning, rolling, diff, ratio, one_hot_encoding)",
        "groupby": "существующий_столбец (если нужен, иначе null)",
        "column": "существующий_столбец (над которым делаем операцию)",
        "agg": "агрегатор (mean, median, std, count, min, max, sum — только если operation = group_agg или rolling)"
      }}
    ],
    "priority": "целое число от 1 до 10 (10 — самый важный признак, который предположительно сильнее всего влияет)"
  }}
]

Правила:
- ВАЖНО! НЕ придумывай столбцы, которых нет в описании. НЕ создавай новых колонок в JSON! feature_ideas != feature_идеи != feature_idea.
- Учитывай возможные утечки таргета (например, не использовать признаки, которые вычисляются из будущего).
- Используй ТОЛЬКО названия операций из operation. Не пиши иначе, используй только указанный спеллинг.
- feature_ideas может содержать только 1 объект. Все feature_ideas должны быть заполнены. ТОЛЬКО ЕСЛИ это столбец target оставь там пустой list.
- Если операция не требует groupby или agg, ставь null.
- Пример допустимого значения "operation": "group_agg", "binning", "rolling_mean", "rolling_std", "diff", "ratio", "count_occurrences".

ВАЖНО! Проверь свой вывод на валидность JSON (кавычки, запятые, скобки, отсутствие комментариев).
ВАЖНО! Проверь, что там, где ты указал операцию, ты указал и столбец, на котором ее нужно применить.
Никаких фраз вроде "Вот ваш JSON" — только чистый JSON. Он не должен быть пустым.
"""
    request_giga = PROMPT
    response = giga.chat(request_giga)
    response = response.choices[0].message.content
    return response


data_json = json.loads(get_features(readme))
data = pd.json_normalize(data_json)
if len(data):
  output_path = PROJECT_ROOT / "features_ranking.csv"
  data.to_csv(output_path, index=False, encoding="utf-8")
else:
   raise "Не записался df, запусти еще раз"