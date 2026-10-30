"""Pipelines Scikit-learn da Fase 3.

Todo o pre-processamento fica DENTRO do Pipeline do modelo. Assim imputacao,
escala e encoding sao ajustados so com os dados de treino de cada fold da
validacao cruzada, e o mesmo objeto e aplicado em teste e em producao.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE


# ---------------------------------------------------------------------------
# Definicao das features de cada modelo
# ---------------------------------------------------------------------------
FEATURES_ALUNO = {
    "numericas": [
        "hist_taxa_mesma_rede", "hist_media_lp_mesma_rede",
        "hist_publica_taxa", "hist_publica_media_lp",
        "meta_municipal_ano", "meta_municipal_2030",
        "log_populacao", "part_agropecuaria", "part_industria", "part_servicos",
    ],
    "assimetricas": ["pib_per_capita"],
    "binarias": ["capital_uf", "amazonia_legal", "regiao_metropolitana"],
    "categoricas": ["rede_aluno", "serie", "nome_regiao", "sigla_uf"],
    "derivadas": ["gap_meta_historico"],
}

FEATURES_MUNICIPIO = {
    "numericas": [
        "taxa_anterior", "media_lp_anterior", "meta_ano", "meta_2030",
        "log_populacao", "part_agropecuaria", "part_industria", "part_servicos",
    ],
    "assimetricas": ["pib_per_capita"],
    "binarias": ["capital_uf", "amazonia_legal", "regiao_metropolitana"],
    "categoricas": ["nome_regiao", "sigla_uf"],
    "derivadas": ["esforco_relativo"],
}


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Cria variaveis derivadas linha a linha (sem estatistica do conjunto,
    portanto sem risco de vazamento entre treino e teste)."""

    def __init__(self, tipo: str = "aluno"):
        self.tipo = tipo

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.tipo == "aluno":
            # quanto o municipio precisa avancar sobre o proprio historico
            X["gap_meta_historico"] = X["meta_municipal_ano"] - X["hist_publica_taxa"]
        else:
            X["esforco_relativo"] = (X["meta_ano"] - X["taxa_anterior"]) / X["taxa_anterior"].replace(0, np.nan)
        return X


def colunas_usadas(features: dict) -> list[str]:
    return [c for grupo in ("numericas", "assimetricas", "binarias", "categoricas") for c in features[grupo]]


def criar_preprocessador(features: dict) -> ColumnTransformer:
    numerico = Pipeline([
        ("imputacao", SimpleImputer(strategy="median", add_indicator=True)),
        ("escala", StandardScaler()),
    ])
    assimetrico = Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("escala", StandardScaler()),
    ])
    binario = SimpleImputer(strategy="most_frequent")
    categorico = Pipeline([
        ("imputacao", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numerico, features["numericas"] + features["derivadas"]),
        ("assim", assimetrico, features["assimetricas"]),
        ("bin", binario, features["binarias"]),
        ("cat", categorico, features["categoricas"]),
    ], verbose_feature_names_out=False)


def criar_pipeline(features: dict, tipo: str, modelo: str) -> Pipeline:
    if modelo == "logistica":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    elif modelo == "gradient_boosting":
        clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    else:
        raise ValueError(modelo)
    return Pipeline([
        ("features", FeatureEngineering(tipo=tipo)),
        ("preprocessamento", criar_preprocessador(features)),
        ("modelo", clf),
    ])


# Espacos de busca da otimizacao (RandomizedSearchCV)
ESPACOS_BUSCA = {
    "logistica": {
        "modelo__C": np.logspace(-3, 1, 20),
    },
    "gradient_boosting": {
        "modelo__learning_rate": [0.03, 0.05, 0.1],
        "modelo__max_depth": [3, 4, 6, None],
        "modelo__max_leaf_nodes": [15, 31, 63],
        "modelo__min_samples_leaf": [20, 50, 100, 200],
        "modelo__l2_regularization": [0.0, 0.1, 1.0],
        "modelo__max_iter": [200, 400],
    },
}


def selecionar_X(df: pd.DataFrame, features: dict) -> pd.DataFrame:
    return df[colunas_usadas(features)].copy()
