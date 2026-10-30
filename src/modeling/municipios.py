"""Inteligencia por municipio (perguntas de negocio do desafio).

1. Risco de nao atingir a meta: modelo treinado com 2024 (features de 2023,
   alvo = resultado 2024 abaixo da meta 2024) e aplicado aos dados de 2024
   para estimar o risco de nao atingir a meta de 2025.
2. Municipios de maior risco educacional: ranking pela probabilidade acima.
3. Regioes com padroes semelhantes: K-Means sobre o perfil educacional e
   socioeconomico, com k escolhido pelo coeficiente de silhueta.

Uso:
    python -m src.modeling.municipios
"""
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.modeling.pipeline import FEATURES_MUNICIPIO, selecionar_X
from src.modeling.train import treinar_e_avaliar
from src.visualization.plots import BRANCO, CINZA, ROSA, ROSA_CLARO, plt, salvar

log = logging.getLogger(__name__)

VARS_CLUSTER = ["taxa_anterior", "esforco_necessario", "pib_per_capita", "log_populacao", "part_agropecuaria"]
CORES_CLUSTER = [ROSA, BRANCO, CINZA, ROSA_CLARO, "#7A0A30", "#4D4D4D"]


def risco_metas(base: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    treino = base[base["ano_ref"] == config.ANO_ALVO].copy()
    futuro = base[base["ano_ref"] == config.ANO_ALVO + 1].copy()

    relatorio = treinar_e_avaliar(treino, FEATURES_MUNICIPIO, "municipio", "nao_atingiu_meta",
                                  None, "municipio", "não atingiu meta")
    salvo = joblib.load(config.REPORTS / "municipio_modelo.joblib")
    modelo, limiar = salvo["modelo"], salvo["limiar"]

    futuro["prob_nao_atingir"] = modelo.predict_proba(selecionar_X(futuro, FEATURES_MUNICIPIO))[:, 1]
    # alto = o modelo classifica como "nao atinge"; medio = ate 15 p.p. abaixo do limiar
    p = futuro["prob_nao_atingir"]
    futuro["risco"] = np.select([p >= limiar, p >= limiar - 0.15], ["alto", "medio"], "baixo")
    cols = ["id_municipio", "nome_municipio", "sigla_uf", "nome_regiao", "taxa_anterior",
            "meta_ano", "esforco_necessario", "prob_nao_atingir", "risco"]
    ranking = futuro.sort_values("prob_nao_atingir", ascending=False)[cols].rename(columns={
        "taxa_anterior": f"taxa_{config.ANO_ALVO}", "meta_ano": f"meta_{config.ANO_ALVO + 1}"})
    ranking.to_csv(config.REPORTS / f"ranking_risco_meta_{config.ANO_ALVO + 1}.csv", index=False)

    relatorio["projecao"] = {
        "ano_meta": config.ANO_ALVO + 1,
        "municipios_avaliados": int(len(ranking)),
        "risco_alto": int((ranking["risco"] == "alto").sum()),
        "pct_risco_alto": round(float((ranking["risco"] == "alto").mean() * 100), 1),
        "risco_alto_por_regiao": ranking[ranking["risco"] == "alto"]["nome_regiao"].value_counts().to_dict(),
        "top10": ranking.head(10)[["nome_municipio", "sigla_uf", "prob_nao_atingir"]].round(3).to_dict("records"),
    }

    por_uf = ranking.groupby("sigla_uf")["prob_nao_atingir"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(por_uf.index, por_uf.values * 100, color=[ROSA if v >= 0.5 else CINZA for v in por_uf.values])
    ax.set_xlabel(f"Risco médio de não atingir a meta {config.ANO_ALVO + 1} (%)")
    ax.set_title("Risco por UF (média dos municípios)")
    ax.grid(axis="y", visible=False)
    salvar(fig, "municipio_risco_por_uf")
    return relatorio, futuro


def clusters(futuro: pd.DataFrame) -> dict:
    dados = futuro.dropna(subset=["taxa_anterior"]).copy()
    X = dados[VARS_CLUSTER].copy()
    X["pib_per_capita"] = np.log10(X["pib_per_capita"])
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Xs = prep.fit_transform(X)

    silhuetas = {}
    for k in range(3, 7):
        rotulos = KMeans(k, n_init=10, random_state=config.RANDOM_STATE).fit_predict(Xs)
        silhuetas[k] = float(silhouette_score(Xs, rotulos, sample_size=min(3000, len(Xs)),
                                              random_state=config.RANDOM_STATE))
    k = max(silhuetas, key=silhuetas.get)
    km = KMeans(k, n_init=10, random_state=config.RANDOM_STATE).fit(Xs)
    dados["cluster"] = km.labels_

    perfil = dados.groupby("cluster").agg(
        municipios=("id_municipio", "size"),
        taxa_media=("taxa_anterior", "mean"),
        esforco_medio=("esforco_necessario", "mean"),
        pib_pc_mediano=("pib_per_capita", "median"),
        log_pop_medio=("log_populacao", "mean"),
        part_agro_media=("part_agropecuaria", "mean"),
        risco_medio=("prob_nao_atingir", "mean"),
        regiao_predominante=("nome_regiao", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
    ).sort_values("taxa_media")
    # renumera do pior para o melhor resultado (cluster 1 = mais vulneravel)
    ordem = {c: i + 1 for i, c in enumerate(perfil.index)}
    dados["cluster"] = dados["cluster"].map(ordem)
    perfil.index = perfil.index.map(ordem)
    perfil.round(3).to_csv(config.REPORTS / "clusters_perfil.csv")
    dados[["id_municipio", "nome_municipio", "sigla_uf", "nome_regiao", "cluster"]].to_csv(
        config.REPORTS / "clusters_municipios.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for c in sorted(dados["cluster"].unique()):
        d = dados[dados["cluster"] == c]
        ax.scatter(np.log10(d["pib_per_capita"]), d["taxa_anterior"], s=7, alpha=0.6,
                   color=CORES_CLUSTER[(c - 1) % len(CORES_CLUSTER)], label=f"grupo {c}")
    ax.set_xlabel("PIB per capita (log10)")
    ax.set_ylabel(f"Taxa de alfabetização {config.ANO_ALVO} (%)")
    ax.set_title(f"Grupos de municípios com perfil semelhante (k={k})")
    ax.legend(markerscale=3)
    salvar(fig, "municipio_clusters")

    return {"k": k, "silhuetas": {str(a): round(b, 3) for a, b in silhuetas.items()},
            "perfil": perfil.round(2).reset_index().to_dict("records")}


def main() -> dict:
    base = pd.read_parquet(config.DATA_PROCESSED / "base_municipio.parquet")
    relatorio, futuro = risco_metas(base)
    relatorio["clusters"] = clusters(futuro)
    with open(config.REPORTS / "municipio_metricas.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2, default=str)
    return relatorio


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
