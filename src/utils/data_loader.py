from gigachat import GigaChat
import json
import pandas as pd

with open('gigachat-torture\\secrets.txt', 'r') as f:
    key = f.readline().strip()

readme = open(r"gigachat-torture\data\readme.txt").read()

def get_features(readme):
    
    giga = GigaChat(credentials=key, verify_ssl_certs=False)

    PROMPT = """Ты — эксперт по анализу табличных данных.

Тебе дано описание датасета:

{readme.txt}

Задача — бинарная классификация.
Однозначно определи таргет.
Ответь ТОЛЬКО в JSON формате:

[
  {
    "column": "название столбца или группы",
    "reason": "почему может влиять на таргет",
    "feature_ideas": "["операция ()": "параметры (подробнее описано в "правилах")"]",
    "priority": "от 1 до 10"
  }
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
  data.to_csv('gigachat-torture/features_ranking.csv')
else:
   raise "Не записался df, запусти еще раз"