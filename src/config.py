"""Configuracao central da Fase 3.

Os valores de AWS/GCP seguem os mesmos da Fase 2 (repositorio
TECH_CHALLENGE_FASE2_PIPELINE). Ajuste por variavel de ambiente se precisar.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# TC3_SAIDA permite rodar os testes em outra pasta sem sujar data/, reports/ e images/
SAIDA = Path(os.environ.get("TC3_SAIDA", ROOT))
DATA_RAW = SAIDA / "data" / "raw"
DATA_PROCESSED = SAIDA / "data" / "processed"
REPORTS = SAIDA / "reports"
IMAGES = SAIDA / "images"

for _p in (DATA_RAW, DATA_PROCESSED, REPORTS, IMAGES):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Fonte 1: camada Gold/Silver da Fase 2 (AWS Athena)
# ---------------------------------------------------------------------------
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
GLUE_DATABASE = os.environ.get("GLUE_DATABASE", "alfabetizacao_db")
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "tech-challenge-alfabetizacao")

# ---------------------------------------------------------------------------
# Fonte 2: mesmas tabelas da Fase 2 direto no BigQuery (Base dos Dados).
# Usada quando a infraestrutura AWS da Fase 2 ja foi desmontada (teardown).
# ---------------------------------------------------------------------------
GCP_BILLING_PROJECT = os.environ.get("GCP_BILLING_PROJECT", "techchallenge-505723")
GCP_SERVICE_ACCOUNT_JSON = os.environ.get("GCP_SERVICE_ACCOUNT_JSON", "")

# Mesma amostra de alunos usada na Bronze da Fase 2 (TABLESAMPLE 2%).
ALUNOS_SAMPLE_PERCENT = int(os.environ.get("ALUNOS_SAMPLE_PERCENT", "2"))

# Codigos INEP (tabela "dicionario" do dataset br_inep_avaliacao_alfabetizacao)
REDE_ALUNO = {"1": "Federal", "2": "Estadual", "3": "Municipal", "4": "Privada"}
REDE_MUNICIPAL = "3"

# Ponto de corte do Saeb que define "alfabetizado" (INEP, Alfabetiza Brasil 2023)
PONTO_CORTE_SAEB = 743

# Ano usado para treinar o modelo do aluno. As features vem do ano anterior
# (ANO_ALVO - 1) para que nenhuma informacao do proprio ano da prova entre no
# modelo (ver README, secao "Data leakage").
ANO_ALVO = 2024
ANO_HISTORICO = ANO_ALVO - 1
