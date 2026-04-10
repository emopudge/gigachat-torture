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
    
    giga = GigaChat(credentials=key, scope=scope, verify_ssl_certs=False)

    PROMPT = """Ты — эксперт по анализу табличных данных.
Тебе дано описание датасета:
{readme}
Задача — бинарная классификация.
Однозначно определи таргет.
Ответь ТОЛЬКО в JSON формате:
[
  {{
    "column": "название столбца или группы",
    "reason": "почему может влиять на таргет",
    "feature_ideas": "["операция ()": "параметры (подробнее описано в "правилах")"]",
    "priority": "от 1 до 10"
  }}
]

Правила:
- в feature ideas ответ должен выглядеть как признаки в строгом формате? например: <
                                                                          "operation": "операция",
                                                                          "groupby": "существующий_столбец",
                                                                          "column": "существующий_столбец",
                                                                          "agg": "агрегатор">
- НЕ придумывай столбцы, которых нет в описании
- учитывай возможные утечки таргета
- делай упор на причинно-логическую связь
- избегай общих фраз
- НЕ пиши ничего кроме json
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