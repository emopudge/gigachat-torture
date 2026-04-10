import pandas as pd
features_ranked = pd.read_csv('gigachat-torture\\features_ranking.csv')
print(features_ranked.columns)
features_ranked = features_ranked[features_ranked.priority > 5]
print(features_ranked)