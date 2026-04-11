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
Задача — бинарная классификация таргета (целевого столбца)

Ответь ТОЛЬКО в JSON формате. Никакого другого текста, комментариев, пояснений или markdown-разметки.
Вывод должен начинаться с "[" и заканчиваться "]".

Формат JSON:
[
  {{
    "column": "название столбца или группы (только из описания)",
    "reason": "почему этот столбец/группа может влиять на таргет (причинно-следственная связь, без общих фраз)",
    "feature_ideas": [
      {{
        "operation": "операция (например: group_agg, binning, rolling, diff, ratio)",
        "groupby": "существующий_столбец (если нужен, иначе null)",
        "column": "существующий_столбец (над которым делаем операцию)",
        "agg": "агрегатор (mean, median, std, count, min, max, sum — только если operation = group_agg или rolling)"
      }}
    ],
    "priority": "целое число от 1 до 10 (10 — самый важный признак)"
  }}
]

Правила:
- НЕ придумывай столбцы, которых нет в описании. НЕ создавай новых колонок в JSON!
- Учитывай возможные утечки таргета (например, не использовать признаки, которые вычисляются из будущего).
- feature_ideas может содержать от 1 до 3 объектов. Заполни feature_ideas как минимум в 3-х записях.
- Если операция не требует groupby или agg, ставь null.
- Пример допустимого значения "operation": "group_agg", "binning", "rolling_mean", "rolling_std", "diff", "ratio", "count_occurrences".

Проверь свой вывод на валидность JSON (кавычки, запятые, скобки, отсутствие комментариев).
Никаких фраз вроде "Вот ваш JSON" — только чистый JSON. Он не должен быть пустым.

КРИТИЧНО: В финальном ответе должно быть РОВНО 5 записей с НЕ-пустыми feature_ideas.
- Если идея не требует сложной операции, используй: "operation": "keep_original", "column": "имя_колонки"
- Заполни feature_ideas во всех 5 записях (минимум 1 объект в массиве)
- Верни только топ-5 самых важных признаков по приоритету (отсортируй по priority убыванию)
- Если сомневаешься — используй простые операции: binning, one_hot_encoding, keep_original
"""
    request_giga = PROMPT
    response = giga.chat(request_giga)
    response = response.choices[0].message.content
    return response


data_json = json.loads(get_features(readme))
data = pd.json_normalize(data_json)
if len(data):
  data.to_csv('features_ranking.csv')
else:
   raise "Не записался df, запусти еще раз"