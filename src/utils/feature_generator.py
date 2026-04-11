import pandas as pd
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_RANK_PATH = PROJECT_ROOT / 'features_ranking.csv'
MERGED_TABLE_PATH = PROJECT_ROOT / "data" / "merged_table.csv"
features_ranked = pd.read_csv(CSV_RANK_PATH)
data = pd.read_csv(MERGED_TABLE_PATH)


def clean_features_df(df, ideas_col='feature_ideas'):
    df = df.copy()

    df = df[df[ideas_col].notna()]
    df = df[df[ideas_col].astype(str).str.strip() != ""]
    df = df[df[ideas_col].astype(str).str.strip() != "[]"]

    def safe_parse(x):
        if isinstance(x, list):
            return x
        if not isinstance(x, str):
            return []

        try:
            return json.loads(x)
        except json.JSONDecodeError:
            try:
                x_fixed = x.replace("'", '"')
                return json.loads(x_fixed)
            except Exception:
                return []

    df[ideas_col] = df[ideas_col].apply(safe_parse)

    df = df[df[ideas_col].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    return df

class FeatureGeneratorMVP:

    def __init__(self, features_df, ideas_col='feature_ideas'):
        self.features_df = clean_features_df(features_df, ideas_col)
        self.ideas_col = ideas_col
        self.aggregations = []

    def _extract_groupby_features(self):
        aggs = []

        for _, row in self.features_df.iterrows():
            ideas = row[self.ideas_col]

            for idea in ideas:
                if isinstance(idea, dict) and idea.get("operation") == "group_agg":
                    aggs.append({
                        "groupby": idea.get("groupby"),
                        "column": idea.get("column"),
                        "agg": idea.get("agg")
                    })
        return aggs

    def fit(self, df):
        self.aggregations = self._extract_groupby_features()
        self.groupby_tables = []

        for agg in self.aggregations:
            group_col = agg["groupby"]
            target_col = agg["column"]

            if group_col not in df.columns or target_col not in df.columns:
                print(f"Skipping aggregation: missing column(s) - groupby='{group_col}', target='{target_col}'")
                continue 
            agg_func = agg["agg"]

            if not isinstance(group_col, list):
                group_col = [group_col]

            grouped = (
                df.groupby(group_col)[target_col]
                .agg(agg_func)
                .reset_index()
            )

            # имя новой фичи
            new_col = "_".join(group_col) + f"_{target_col}_{agg_func}"
            grouped = grouped.rename(columns={target_col: new_col})

            self.groupby_tables.append((group_col, grouped))

    def transform(self, df):
        df_result = df.copy()

        for group_cols, grouped_df in self.groupby_tables:
            df_result = df_result.merge(
                grouped_df,
                on=group_cols,
                how="left"
            )

        return df_result

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
    
generator = FeatureGeneratorMVP(features_ranked)
data_transformed = generator.fit_transform(data)
print(data_transformed.columns)
print(data_transformed.columns.size)
data_transformed.to_csv('data_transformed.csv')


