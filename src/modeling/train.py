"""Treino, otimizacao e validacao dos modelos supervisionados.

Modelo principal (aluno): preve se o aluno avaliado em 2024 sera
alfabetizado, usando so informacao disponivel antes da prova.

Estrategia contra vazamento e overfitting:
  - teste separado ANTES de qualquer ajuste, por escola (GroupShuffleSplit):
    nenhuma escola aparece em treino e teste ao mesmo tempo;
  - validacao cruzada tambem por escola (GroupKFold), com o
    pre-processamento dentro do Pipeline (ajustado so no fold de treino);
  - hiperparametros escolhidos por RandomizedSearchCV na validacao cruzada;
  - limiar de decisao escolhido com previsoes fora-da-amostra do treino;
  - teste usado uma unica vez, no fim.

Uso:
    python -m src.modeling.train
"""
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    GroupKFold, GroupShuffleSplit, RandomizedSearchCV, StratifiedKFold, cross_val_predict,
    train_test_split,
)

from src import config
from src.evaluation.interpretability import interpretar
from src.evaluation.metrics import (
    calcular_metricas, grafico_comparacao_modelos, graficos_avaliacao, limiar_youden,
)
from src.modeling.pipeline import ESPACOS_BUSCA, FEATURES_ALUNO, criar_pipeline, selecionar_X

log = logging.getLogger(__name__)

N_ITER = {"logistica": 10, "gradient_boosting": 20}


def dividir(df: pd.DataFrame, alvo: str, grupo: str | None):
    if grupo:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
        idx_treino, idx_teste = next(gss.split(df, df[alvo], groups=df[grupo]))
        treino, teste = df.iloc[idx_treino], df.iloc[idx_teste]
        sobreposicao = set(treino[grupo]) & set(teste[grupo])
        assert not sobreposicao, "vazamento: grupo presente em treino e teste"
    else:
        treino, teste = train_test_split(df, test_size=0.2, stratify=df[alvo],
                                         random_state=config.RANDOM_STATE)
    return treino.reset_index(drop=True), teste.reset_index(drop=True)


def treinar_e_avaliar(df: pd.DataFrame, features: dict, tipo: str, alvo: str,
                      grupo: str | None, prefixo: str, rotulo: str, n_iter: dict | None = None) -> dict:
    n_iter = n_iter or N_ITER
    treino, teste = dividir(df, alvo, grupo)
    X_tr, y_tr = selecionar_X(treino, features), treino[alvo].values
    X_te, y_te = selecionar_X(teste, features), teste[alvo].values
    g_tr = treino[grupo].values if grupo else None
    cv = GroupKFold(n_splits=5) if grupo else StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)

    relatorio = {
        "alvo": alvo,
        "linhas_treino": len(treino), "linhas_teste": len(teste),
        "divisao": f"por {grupo}" if grupo else "estratificada",
        "prevalencia_treino": round(float(y_tr.mean()), 4),
        "modelos": {},
    }

    base = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    relatorio["baseline"] = calcular_metricas(y_te, base.predict_proba(X_te)[:, 1], 0.5)

    buscas = {}
    for nome in ("logistica", "gradient_boosting"):
        log.info("[%s] otimizando %s", prefixo, nome)
        busca = RandomizedSearchCV(
            criar_pipeline(features, tipo, nome), ESPACOS_BUSCA[nome], n_iter=n_iter[nome],
            scoring="roc_auc", cv=cv, return_train_score=True, n_jobs=-1,
            random_state=config.RANDOM_STATE, refit=True,
        )
        busca.fit(X_tr, y_tr, groups=g_tr)
        i = busca.best_index_
        relatorio["modelos"][nome] = {
            "cv_auc_media": round(float(busca.cv_results_["mean_test_score"][i]), 4),
            "cv_auc_desvio": round(float(busca.cv_results_["std_test_score"][i]), 4),
            "cv_auc_treino": round(float(busca.cv_results_["mean_train_score"][i]), 4),
            "melhores_parametros": {k: (v.item() if hasattr(v, "item") else v)
                                    for k, v in busca.best_params_.items()},
        }
        buscas[nome] = busca
        log.info("[%s] %s: %s", prefixo, nome, relatorio["modelos"][nome])

    escolhido = max(relatorio["modelos"], key=lambda n: relatorio["modelos"][n]["cv_auc_media"])
    modelo = buscas[escolhido].best_estimator_
    relatorio["modelo_escolhido"] = escolhido
    grafico_comparacao_modelos(relatorio["modelos"], prefixo)

    # limiar com previsao fora-da-amostra no treino (o teste continua intocado)
    oof = cross_val_predict(modelo, X_tr, y_tr, groups=g_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    limiar = limiar_youden(y_tr, oof)
    relatorio["validacao_oof"] = calcular_metricas(y_tr, oof, limiar)

    proba_te = modelo.predict_proba(X_te)[:, 1]
    relatorio["teste"] = calcular_metricas(y_te, proba_te, limiar)
    relatorio["gap_generalizacao_auc"] = round(
        relatorio["modelos"][escolhido]["cv_auc_media"] - relatorio["teste"]["roc_auc"], 4)
    graficos_avaliacao(y_te, proba_te, limiar, prefixo, rotulo)

    importancia = interpretar(modelo, X_te, y_te, prefixo, rotulo)
    relatorio["top_variaveis"] = importancia.head(10)[
        ["variavel", "shap_medio_abs", "efeito_quando_maior"]].round(4).to_dict("records")

    joblib.dump({"modelo": modelo, "limiar": limiar, "features": features},
                config.REPORTS / f"{prefixo}_modelo.joblib")
    teste.assign(proba=proba_te, previsto=(proba_te >= limiar).astype(int)).to_parquet(
        config.DATA_PROCESSED / f"{prefixo}_previsoes_teste.parquet", index=False)
    with open(config.REPORTS / f"{prefixo}_metricas.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2, default=str)
    return relatorio


def main() -> dict:
    df = pd.read_parquet(config.DATA_PROCESSED / "base_aluno.parquet")
    return treinar_e_avaliar(df, FEATURES_ALUNO, "aluno", "alfabetizado", "id_escola",
                             "aluno", "alfabetizado")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    np.seterr(all="ignore")
    main()
