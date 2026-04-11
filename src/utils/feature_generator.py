import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
README_PATH = PROJECT_ROOT / 'features_ranking.csv'
MERGED_TABLE_PATH = PROJECT_ROOT / "data" / "merged_table.csv"
features_ranked = pd.read_csv(README_PATH)
data = pd.read_csv(MERGED_TABLE_PATH)

def clean_features_df(df, ideas_col='feature_ideas'):
    df = df.copy()

    # убираем NaN и пустые строки
    df = df[df[ideas_col].notna()]
    df = df[df[ideas_col].astype(str).str.strip() != ""]
    df = df[df[ideas_col].astype(str).str.strip() != "[]"]

    # безопасный парсинг JSON
    def safe_parse(x):
        if isinstance(x, list):
            return x
        if not isinstance(x, str):
            return []

        try:
            return json.loads(x)
        except json.JSONDecodeError:
            try:
                # костыль: заменяем одинарные кавычки на двойные
                x_fixed = x.replace("'", '"')
                return json.loads(x_fixed)
            except Exception:
                return []

    df[ideas_col] = df[ideas_col].apply(safe_parse)

    # оставляем только непустые списки
    df = df[df[ideas_col].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    return df

class FeatureGeneratorMVP:

    def __init__(self, features_df, ideas_col='feature_ideas'):
        self.features_df = clean_features_df(features_df, ideas_col)
        self.ideas_col = ideas_col
        self.encoder = None

    def _extract_onehot_columns(self):
        cols = []

        for _, row in self.features_df.iterrows():
            ideas = row[self.ideas_col]

            if not isinstance(ideas, list):
                continue

            for idea in ideas:
                if isinstance(idea, dict) and idea.get("operation") == "one_hot_encoding":
                    cols.append(row["column"])

        return list(set(cols))  # убираем дубли

    def implement_onehotenc(self, df_raw):
        df_result = df_raw.copy()

        onehot_cols = self._extract_onehot_columns()
        onehot_cols = [c for c in onehot_cols if c in df_result.columns]

        if not onehot_cols:
            return df_result

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        encoded_array = encoder.fit_transform(df_result[onehot_cols])

        # имена новых колонок
        encoded_cols = encoder.get_feature_names_out(onehot_cols)

        df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=df_result.index)

        return pd.concat(
            [df_result.drop(columns=onehot_cols), df_encoded],
            axis=1
        )
    
    def fit(self, df_raw):
        onehot_cols = self._extract_onehot_columns()
        onehot_cols = [c for c in onehot_cols if c in df_raw.columns]

        self.onehot_cols = onehot_cols

        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.encoder.fit(df_raw[onehot_cols])

    def transform(self, df_raw):
        df_result = df_raw.copy()

        encoded_array = self.encoder.transform(df_result[self.onehot_cols])
        encoded_cols = self.encoder.get_feature_names_out(self.onehot_cols)

        df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=df_result.index)

        return pd.concat(
            [df_result.drop(columns=self.onehot_cols), df_encoded],
            axis=1
        )
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)










