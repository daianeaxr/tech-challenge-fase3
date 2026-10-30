"""Monta as bases analiticas da Fase 3 a partir de data/raw/.

Gera duas bases em data/processed/:

1. base_aluno.parquet
   Uma linha por aluno avaliado em ANO_ALVO (2024). Alvo: alfabetizado (0/1).
   Todas as variaveis explicativas sao conhecidas ANTES da prova:
   caracteristicas do aluno/escola, historico do municipio em 2023, meta
   municipal definida para 2024, dados territoriais e socioeconomicos (IBGE).

2. base_municipio.parquet
   Uma linha por municipio (rede municipal) e ano de referencia.
   - ano_ref=2024: alvo nao_atingiu_meta (resultado 2024 < meta 2024),
     features com dados de 2023. Usada para treinar.
   - ano_ref=2025: mesmas features calculadas com dados de 2024 e meta 2025.
     Sem alvo (o resultado de 2025 ainda nao existe). Usada para o ranking
     de risco.

Uso:
    python -m src.preprocessing.build_dataset
"""
import logging

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)

# Codigo da rede nas tabelas de resultado (dicionario INEP):
# 0=Total 1=Federal 2=Estadual 3=Municipal 4=Privada 5=Publica(Est+Mun)
REDE_RESULTADO_PUBLICA = "5"

REGIAO_POR_DIGITO = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}


def ler(nome: str) -> pd.DataFrame:
    return pd.read_parquet(config.DATA_RAW / f"{nome}.parquet")


def _bool_para_int(serie: pd.Series) -> pd.Series:
    return serie.map({True: 1, False: 0, "true": 1, "false": 0, "1": 1, "0": 0, 1: 1, 0: 0}).astype("float")


def socioeconomico(ano_limite: int) -> pd.DataFrame:
    """Ultimo dado de populacao e PIB disponivel ate ano_limite (inclusive).
    Usar so anos <= ano_limite evita usar informacao do futuro."""
    va_cols = ["va_agropecuaria", "va_industria", "va_servicos"]
    pop_all = ler("populacao")[["id_municipio", "ano", "populacao"]]
    pib = ler("pib")
    for col in ["pib"] + va_cols:
        pib[col] = pd.to_numeric(pib[col], errors="coerce").astype("float")
    pib = pib[pib["ano"] <= ano_limite]

    pop = pop_all[pop_all["ano"] <= ano_limite].sort_values("ano").groupby("id_municipio").tail(1)

    # PIB per capita: ultimo ano com PIB, dividido pela populacao do mesmo ano (PIB em R$)
    pib_pc = pib.dropna(subset=["pib"]).sort_values("ano").groupby("id_municipio").tail(1)
    pib_pc = pib_pc.merge(pop_all, on=["id_municipio", "ano"], how="left")
    pib_pc["pib_per_capita"] = pib_pc["pib"] / pib_pc["populacao"]

    # composicao da economia: ultimo ano em que o IBGE publicou o valor adicionado
    # por setor (o ano mais recente da tabela costuma vir so com o PIB total)
    va = pib.dropna(subset=va_cols).sort_values("ano").groupby("id_municipio").tail(1).copy()
    va_total = va[va_cols].sum(axis=1).replace(0, np.nan)
    for setor in ("agropecuaria", "industria", "servicos"):
        va[f"part_{setor}"] = va[f"va_{setor}"] / va_total

    saida = (pop[["id_municipio", "populacao"]]
             .merge(pib_pc[["id_municipio", "pib_per_capita"]], on="id_municipio", how="outer")
             .merge(va[["id_municipio", "part_agropecuaria", "part_industria", "part_servicos"]],
                    on="id_municipio", how="outer"))
    saida["log_populacao"] = np.log10(saida["populacao"])
    return saida.drop(columns="populacao")


def territorio() -> pd.DataFrame:
    mun = ler("municipio").copy()
    if "nome_regiao" not in mun.columns or mun["nome_regiao"].isna().all():
        mun["nome_regiao"] = mun["id_municipio"].str[0].map(REGIAO_POR_DIGITO)
    for col in ("capital_uf", "amazonia_legal", "regiao_metropolitana"):
        if col in mun.columns:
            mun[col] = _bool_para_int(mun[col])
    cols = ["id_municipio", "nome_municipio", "sigla_uf", "nome_regiao",
            "capital_uf", "amazonia_legal", "regiao_metropolitana"]
    return mun[[c for c in cols if c in mun.columns]]


def resultado(ano: int, rede: str, prefixo: str) -> pd.DataFrame:
    res = ler("resultado_municipio")
    res = res[(res["ano"] == ano) & (res["rede"] == rede)]
    res = res[["id_municipio", "taxa_alfabetizacao", "media_portugues"]]
    return res.drop_duplicates("id_municipio").rename(columns={
        "taxa_alfabetizacao": f"{prefixo}_taxa",
        "media_portugues": f"{prefixo}_media_lp",
    })


def meta(ano_meta: int, nome: str) -> pd.DataFrame:
    metas = ler("metas_municipio")
    metas = metas[metas["ano_meta"] == ano_meta][["id_municipio", "meta_valor"]]
    return metas.drop_duplicates("id_municipio").rename(columns={"meta_valor": nome})


def construir_base_aluno() -> pd.DataFrame:
    alunos = ler("alunos")
    alunos = alunos[
        (alunos["ano"] == config.ANO_ALVO)
        & (alunos["presenca"] == "1")
        & (alunos["preenchimento_caderno"] == "1")
        & alunos["alfabetizado"].isin(["0", "1"])
    ].copy()
    log.info("Alunos %d presentes e com prova preenchida: %d", config.ANO_ALVO, len(alunos))

    alunos["alfabetizado"] = alunos["alfabetizado"].astype(int)
    alunos["rede_aluno"] = alunos["rede"].map(config.REDE_ALUNO).fillna("Outra")

    hist = config.ANO_HISTORICO
    # historico do municipio na MESMA rede do aluno (2=Estadual, 3=Municipal...)
    res_hist = ler("resultado_municipio")
    res_hist = res_hist[res_hist["ano"] == hist][["id_municipio", "rede", "taxa_alfabetizacao", "media_portugues"]]
    res_hist = res_hist.drop_duplicates(["id_municipio", "rede"]).rename(columns={
        "taxa_alfabetizacao": "hist_taxa_mesma_rede", "media_portugues": "hist_media_lp_mesma_rede",
    })
    base = alunos.merge(res_hist, on=["id_municipio", "rede"], how="left")
    base = base.merge(resultado(hist, REDE_RESULTADO_PUBLICA, "hist_publica"), on="id_municipio", how="left")
    base = base.merge(meta(config.ANO_ALVO, "meta_municipal_ano"), on="id_municipio", how="left")
    base = base.merge(meta(2030, "meta_municipal_2030"), on="id_municipio", how="left")
    base = base.merge(socioeconomico(hist), on="id_municipio", how="left")
    base = base.merge(territorio(), on="id_municipio", how="left")

    # proficiencia NAO entra: alfabetizado = proficiencia >= 743 (vazamento direto)
    # caderno NAO entra: e sorteado, nao descreve o aluno
    base = base.drop(columns=["proficiencia", "caderno", "presenca", "preenchimento_caderno", "rede"],
                     errors="ignore")
    return base


def _features_municipio(ano_hist: int, ano_meta: int) -> pd.DataFrame:
    base = resultado(ano_hist, config.REDE_MUNICIPAL, "taxa_anterior")
    base = base.rename(columns={"taxa_anterior_taxa": "taxa_anterior",
                                "taxa_anterior_media_lp": "media_lp_anterior"})
    base = base.merge(meta(ano_meta, "meta_ano"), on="id_municipio", how="inner")
    base = base.merge(meta(2030, "meta_2030"), on="id_municipio", how="left")
    base["esforco_necessario"] = base["meta_ano"] - base["taxa_anterior"]
    base = base.merge(socioeconomico(ano_hist), on="id_municipio", how="left")
    base = base.merge(territorio(), on="id_municipio", how="left")
    base["ano_ref"] = ano_meta
    return base


def construir_base_municipio() -> pd.DataFrame:
    treino = _features_municipio(config.ANO_HISTORICO, config.ANO_ALVO)
    real = resultado(config.ANO_ALVO, config.REDE_MUNICIPAL, "real")
    treino = treino.merge(real[["id_municipio", "real_taxa"]], on="id_municipio", how="inner")
    treino["nao_atingiu_meta"] = (treino["real_taxa"] < treino["meta_ano"]).astype(int)
    treino = treino.rename(columns={"real_taxa": "taxa_realizada"})

    futuro = _features_municipio(config.ANO_ALVO, config.ANO_ALVO + 1)
    return pd.concat([treino, futuro], ignore_index=True)


def main() -> None:
    base_aluno = construir_base_aluno()
    base_aluno.to_parquet(config.DATA_PROCESSED / "base_aluno.parquet", index=False)
    log.info("base_aluno: %d linhas, %d colunas", *base_aluno.shape)

    base_mun = construir_base_municipio()
    base_mun.to_parquet(config.DATA_PROCESSED / "base_municipio.parquet", index=False)
    log.info("base_municipio: %d linhas", len(base_mun))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
