import os
from dotenv import load_dotenv
from gigachat import GigaChat
import json
import pandas as pd
from pathlib import Path

load_dotenv()  # автоматически находит и читает .env в корне проекта
key = os.getenv("GIGACHAT_CREDENTIALS")
scope = os.getenv("GIGACHAT_SCOPE")

readme = Path("data/readme.txt").read_text(encoding="utf-8")

def get_features(readme):
    
    giga = GigaChat(credentials=key, scope=scope, verify_ssl_certs=False)

    PROMPT = """Ты — эксперт по анализу табличных данных.
Тебе дано описание датасета:
{readme}
Задача — бинарная классификация.
Ответь ТОЛЬКО в JSON формате:
[
  {{
    "column": "название столбца или группы",
    "reason": "почему может влиять на таргет",
    "feature_ideas": [
      "какие признаки можно построить"
    ],
    "priority": "от 1 до 10"
  }}
]
Правила:
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
print(data)
data.to_csv('features_ranking.csv')