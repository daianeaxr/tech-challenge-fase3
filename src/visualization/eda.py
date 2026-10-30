"""Analise exploratoria. Gera os graficos em images/ e um resumo numerico em
reports/eda_resumo.json, usado no README e na apresentacao.

Uso:
    python -m src.visualization.eda
"""
import json
import logging

import numpy as np
import pandas as pd

from src import config
from src.visualization.plots import BRANCO, CINZA, ROSA, plt, salvar

log = logging.getLogger(__name__)


def _ler():
    aluno = pd.read_parquet(config.DATA_PROCESSED / "base_aluno.parquet")
    mun = pd.read_parquet(config.DATA_PROCESSED / "base_municipio.parquet")
    return aluno, mun[mun["ano_ref"] == config.ANO_ALVO].copy()


def taxa_por_grupo(aluno: pd.DataFrame, coluna: str) -> pd.DataFrame:
    return (aluno.groupby(coluna)["alfabetizado"]
            .agg(taxa="mean", alunos="size")
            .assign(taxa=lambda d: d["taxa"] * 100)
            .sort_values("taxa"))


def grafico_barras(df: pd.DataFrame, titulo: str, nome: str, media: float) -> None:
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(df) + 1)))
    cores = [ROSA if v < media else CINZA for v in df["taxa"]]
    ax.barh(df.index.astype(str), df["taxa"], color=cores)
    ax.axvline(media, color=BRANCO, linestyle="--", linewidth=1)
    ax.text(media, -0.9, f" média {media:.1f}%", color=BRANCO, fontsize=9)
    ax.set_xlabel("% de alunos alfabetizados")
    ax.set_title(titulo)
    ax.grid(axis="y", visible=False)
    salvar(fig, nome)


def main() -> dict:
    aluno, mun = _ler()
    resumo = {}

    media = aluno["alfabetizado"].mean() * 100
    resumo["alunos_modelagem"] = int(len(aluno))
    resumo["escolas"] = int(aluno["id_escola"].nunique())
    resumo["municipios_alunos"] = int(aluno["id_municipio"].nunique())
    resumo["taxa_alfabetizados_amostra"] = round(media, 1)

    # 1. distribuicao do alvo por regiao, UF e rede
    por_regiao = taxa_por_grupo(aluno, "nome_regiao")
    por_uf = taxa_por_grupo(aluno, "sigla_uf")
    por_rede = taxa_por_grupo(aluno, "rede_aluno")
    grafico_barras(por_regiao, "Alfabetização por região (amostra de alunos)", "eda_01_regiao", media)
    grafico_barras(por_uf, "Alfabetização por UF (amostra de alunos)", "eda_02_uf", media)
    grafico_barras(por_rede, "Alfabetização por rede de ensino", "eda_03_rede", media)
    resumo["taxa_por_regiao"] = por_regiao["taxa"].round(1).to_dict()
    resumo["taxa_por_rede"] = por_rede["taxa"].round(1).to_dict()
    resumo["uf_menor_taxa"] = por_uf.index[0]
    resumo["uf_maior_taxa"] = por_uf.index[-1]

    # 2. historico do municipio x resultado do aluno
    aluno["faixa_hist"] = pd.qcut(aluno["hist_publica_taxa"], 5, duplicates="drop")
    por_faixa = aluno.groupby("faixa_hist", observed=True)["alfabetizado"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    rotulos = [f"{iv.left:.0f} a {iv.right:.0f}%" for iv in por_faixa.index]
    ax.bar(rotulos, por_faixa.values, color=ROSA)
    ax.set_xlabel(f"Taxa do município em {config.ANO_HISTORICO} (quintis, rede pública)")
    ax.set_ylabel(f"% alfabetizados em {config.ANO_ALVO}")
    ax.set_title("Histórico do município antecipa o resultado do aluno")
    salvar(fig, "eda_04_historico_vs_aluno")
    resumo["taxa_quintil_hist_baixo"] = round(float(por_faixa.iloc[0]), 1)
    resumo["taxa_quintil_hist_alto"] = round(float(por_faixa.iloc[-1]), 1)

    # 3. municipio: taxa x PIB per capita
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(np.log10(mun["pib_per_capita"]), mun["taxa_realizada"], s=6, alpha=0.4, color=ROSA)
    ax.set_xlabel("PIB per capita (log10, R$)")
    ax.set_ylabel(f"Taxa de alfabetização {config.ANO_ALVO} (%)")
    ax.set_title("Renda do município x alfabetização")
    salvar(fig, "eda_05_pib_vs_taxa")

    # 4. correlacoes (Spearman, robusto a assimetria)
    cols = ["taxa_realizada", "taxa_anterior", "media_lp_anterior", "meta_ano",
            "esforco_necessario", "pib_per_capita", "log_populacao",
            "part_agropecuaria", "part_industria", "part_servicos"]
    corr = mun[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="RdPu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="black" if corr.iloc[i, j] > 0.4 else "white")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Correlação de Spearman (municípios)")
    salvar(fig, "eda_06_correlacao")
    resumo["corr_taxa_historico"] = round(float(corr.loc["taxa_realizada", "taxa_anterior"]), 2)
    resumo["corr_taxa_pib"] = round(float(corr.loc["taxa_realizada", "pib_per_capita"]), 2)
    resumo["corr_taxa_agro"] = round(float(corr.loc["taxa_realizada", "part_agropecuaria"]), 2)

    # 5. meta x realizado
    resumo["municipios_com_meta"] = int(len(mun))
    resumo["pct_nao_atingiu_meta"] = round(float(mun["nao_atingiu_meta"].mean() * 100), 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    gap = mun["taxa_realizada"] - mun["meta_ano"]
    ax.hist(gap, bins=50, color=ROSA)
    ax.axvline(0, color=BRANCO, linestyle="--")
    ax.set_xlabel(f"Resultado {config.ANO_ALVO} menos meta {config.ANO_ALVO} (p.p.)")
    ax.set_ylabel("Municípios")
    ax.set_title("Distância até a meta (à esquerda de 0: não atingiu)")
    salvar(fig, "eda_07_gap_meta")

    # 6. dados faltantes
    faltantes = (aluno.isna().mean() * 100).round(1)
    resumo["faltantes_pct"] = faltantes[faltantes > 0].sort_values(ascending=False).to_dict()

    with open(config.REPORTS / "eda_resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2, default=str)
    log.info("EDA concluida: %s", resumo)
    return resumo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
