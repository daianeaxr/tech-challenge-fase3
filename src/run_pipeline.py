"""Executa a Fase 3 de ponta a ponta.

    python -m src.run_pipeline --extrair bigquery   # baixa os dados e roda tudo
    python -m src.run_pipeline --extrair athena     # le a Gold/Silver da Fase 2 na AWS
    python -m src.run_pipeline                      # usa o que ja esta em data/raw/

Etapas: extracao -> bases analiticas -> EDA -> modelo do aluno ->
risco por municipio e clusters -> apresentacao com os resultados.
"""
import argparse
import logging
import time

import numpy as np

from src.modeling import municipios, train
from src.preprocessing import build_dataset, extract
from src.visualization import eda, relatorio, slides

log = logging.getLogger("pipeline")


def main(extrair: str | None) -> None:
    np.seterr(all="ignore")
    etapas = []
    if extrair:
        etapas.append(("extracao", lambda: extract.main(extrair)))
    etapas += [
        ("bases analiticas", build_dataset.main),
        ("analise exploratoria", eda.main),
        ("modelo do aluno", train.main),
        ("risco por municipio e clusters", municipios.main),
        ("resultados no README", relatorio.main),
        ("apresentacao", slides.main),
    ]
    for nome, funcao in etapas:
        inicio = time.time()
        log.info(">>> %s", nome)
        funcao()
        log.info("<<< %s (%.0fs)", nome, time.time() - inicio)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--extrair", choices=["bigquery", "athena"], default=None)
    main(parser.parse_args().extrair)
