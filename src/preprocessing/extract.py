"""Extracao dos dados usados na Fase 3.

Duas fontes possiveis para as tabelas da Fase 2:

  --fonte athena    le as tabelas Silver/Gold catalogadas no Glue pela pipeline
                    da Fase 2 (precisa da infra AWS da Fase 2 de pe).
  --fonte bigquery  le as mesmas tabelas de origem direto da Base dos Dados,
                    com o mesmo tratamento da Silver (chave de 7 digitos e
                    metas despivotadas). Use se a AWS ja foi desmontada.

O enriquecimento externo (IBGE: populacao e PIB municipal) e o diretorio
territorial completo vem sempre do BigQuery, porque nao fizeram parte da
Fase 2.

Saida: arquivos Parquet em data/raw/.

Uso:
    python -m src.preprocessing.extract --fonte bigquery
"""
import argparse
import logging

import pandas as pd

from src import config

log = logging.getLogger(__name__)

META_YEAR_COLUMNS = [f"meta_alfabetizacao_{ano}" for ano in range(2024, 2031)]

BQ_QUERIES = {
    "alunos": f"""
        SELECT ano, id_municipio, id_escola, id_aluno, serie, rede, presenca,
               preenchimento_caderno, alfabetizado, proficiencia, peso_aluno
        FROM `basedosdados.br_inep_avaliacao_alfabetizacao.alunos`
        TABLESAMPLE SYSTEM ({config.ALUNOS_SAMPLE_PERCENT} PERCENT)
    """,
    "resultado_municipio": """
        SELECT * FROM `basedosdados.br_inep_avaliacao_alfabetizacao.municipio`
    """,
    "metas_municipio_wide": """
        SELECT * FROM `basedosdados.br_inep_avaliacao_alfabetizacao.meta_alfabetizacao_municipio`
    """,
    "municipio": """
        SELECT id_municipio, nome AS nome_municipio, sigla_uf, nome_regiao,
               capital_uf, amazonia_legal,
               id_regiao_metropolitana IS NOT NULL AS regiao_metropolitana
        FROM `basedosdados.br_bd_diretorios_brasil.municipio`
    """,
    "populacao": """
        SELECT id_municipio, ano, populacao
        FROM `basedosdados.br_ibge_populacao.municipio`
        WHERE ano >= 2019
    """,
    "pib": """
        SELECT id_municipio, ano, pib, va_agropecuaria, va_industria, va_servicos
        FROM `basedosdados.br_ibge_pib.municipio`
        WHERE ano >= 2017
    """,
}

ATHENA_QUERIES = {
    "alunos": "SELECT * FROM silver_alunos",
    "resultado_municipio": "SELECT * FROM silver_resultado_municipio",
    "metas_municipio": "SELECT id_municipio, rede, ano_meta, meta_valor FROM silver_metas_municipio",
}


def normalizar_chaves(df: pd.DataFrame) -> pd.DataFrame:
    """Mesmo tratamento de chave da Silver da Fase 2."""
    if "id_municipio" in df.columns:
        df["id_municipio"] = df["id_municipio"].astype(str).str.split(".").str[0].str.zfill(7)
    if "sigla_uf" in df.columns:
        df["sigla_uf"] = df["sigla_uf"].astype(str).str.strip().str.upper()
    for col in ("rede", "serie", "presenca", "preenchimento_caderno", "alfabetizado"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.split(".").str[0]
    return df


def despivotar_metas(df: pd.DataFrame) -> pd.DataFrame:
    """Wide (meta_alfabetizacao_2024..2030) -> long (ano_meta, meta_valor),
    mantendo a versao mais recente de cada meta (mesma regra da Silver)."""
    cols = [c for c in META_YEAR_COLUMNS if c in df.columns]
    df = df.rename(columns={"ano": "ano_base"})
    id_cols = [c for c in ("id_municipio", "rede", "ano_base") if c in df.columns]
    longo = df.melt(id_vars=id_cols, value_vars=cols, var_name="ano_meta", value_name="meta_valor")
    longo["ano_meta"] = longo["ano_meta"].str[-4:].astype(int)
    if "ano_base" in longo.columns:
        longo = (
            longo.sort_values("ano_base")
            .drop_duplicates(subset=["id_municipio", "rede", "ano_meta"], keep="last")
            .drop(columns="ano_base")
        )
    return longo


def _cliente_bigquery():
    from google.cloud import bigquery
    from google.oauth2 import service_account

    if config.GCP_SERVICE_ACCOUNT_JSON:
        cred = service_account.Credentials.from_service_account_file(config.GCP_SERVICE_ACCOUNT_JSON)
        return bigquery.Client(project=config.GCP_BILLING_PROJECT, credentials=cred)
    return bigquery.Client(project=config.GCP_BILLING_PROJECT)


def extrair_bigquery(entidades: list[str]) -> dict[str, pd.DataFrame]:
    cliente = _cliente_bigquery()
    saida = {}
    for nome in entidades:
        log.info("BigQuery: extraindo %s", nome)
        saida[nome] = cliente.query(BQ_QUERIES[nome]).to_dataframe()
    return saida


def extrair_athena(entidades: list[str]) -> dict[str, pd.DataFrame]:
    import awswrangler as wr

    saida = {}
    for nome in entidades:
        log.info("Athena: extraindo %s", nome)
        saida[nome] = wr.athena.read_sql_query(
            ATHENA_QUERIES[nome],
            database=config.GLUE_DATABASE,
            workgroup=config.ATHENA_WORKGROUP,
            ctas_approach=False,
        )
    return saida


def salvar(tabelas: dict[str, pd.DataFrame]) -> None:
    for nome, df in tabelas.items():
        df = normalizar_chaves(df)
        caminho = config.DATA_RAW / f"{nome}.parquet"
        df.to_parquet(caminho, index=False)
        log.info("%s: %d linhas -> %s", nome, len(df), caminho)


def main(fonte: str) -> None:
    if fonte == "athena":
        tabelas = extrair_athena(["alunos", "resultado_municipio", "metas_municipio"])
        externas = extrair_bigquery(["municipio", "populacao", "pib"])
    else:
        tabelas = extrair_bigquery(["alunos", "resultado_municipio", "metas_municipio_wide"])
        tabelas["metas_municipio"] = despivotar_metas(
            normalizar_chaves(tabelas.pop("metas_municipio_wide"))
        )
        externas = extrair_bigquery(["municipio", "populacao", "pib"])
    tabelas.update(externas)
    salvar(tabelas)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonte", choices=["bigquery", "athena"], default="bigquery")
    main(parser.parse_args().fonte)
