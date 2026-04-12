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
    scope='GIGACHAT_API_CORP',
    timeout=120,
    model = 'GigaChat-2-Max')

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
        "operation": "операция (только group_agg или null)",
        "groupby": "существующий_столбец (если нужен, иначе null). ЕСЛИ УМЕСТНО, группируй по столбцам ID. НЕ группируй по target",
        "column": "существующий_столбец (над которым делаем операцию)",
        "agg": "агрегаторы ТОЛЬКО реально существующие в Python. ВСЕГДА должен быть заполнен, если заполнен groupby. НЕ ПРИДУМЫВАЙ АГРЕГАТОРЫ."
      }}
    ],
    "priority": "целое число от 1 до 10 (10 — самый важный признак, который предположительно сильнее всего влияет)"
  }}
]

УДАЛИ ДУБЛИКАТЫ в feature_ideas. Проверь, существуют ли агрегаторы из agg в Python. Например, в Python нет агрегатора mode.

Правила:
- ВАЖНО! НЕ придумывай столбцы, которых нет в описании. НЕ создавай новых колонок в JSON! feature_ideas != feature_идеи != feature_idea.
- Учитывай возможные утечки таргета (например, не использовать признаки, которые вычисляются из будущего).
- Используй ТОЛЬКО названия операций из operation. Не пиши иначе, используй только указанный спеллинг.
- ВСЕ feature_ideas должны быть заполнены. Для одной записи может быть несколько feature_ideas. ТОЛЬКО ЕСЛИ это столбец target оставь там пустой list.
- ТОЛЬКО ЕСЛИ groupby и agg могут помешать CatBoost, став null.
- Пример допустимого значения "operation" - ТОЛЬКО "group_agg".

ВАЖНО! Проверь свой вывод на валидность JSON (кавычки, запятые, скобки, отсутствие комментариев).
ВАЖНО! Проверь, что там, где ты указал операцию, ты указал и столбец, на котором ее нужно применить.
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