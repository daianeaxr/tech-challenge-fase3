"""Teste de fumaca: roda a pipeline inteira com dados sinteticos numa pasta
temporaria e confere as garantias contra vazamento.

    TC3_SAIDA=/tmp/tc3 python -m pytest tests -q
"""
import importlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="module")
def saida(tmp_path_factory):
    pasta = tmp_path_factory.mktemp("tc3")
    os.environ["TC3_SAIDA"] = str(pasta)
    import src.config
    importlib.reload(src.config)
    for mod in ["src.preprocessing.build_dataset", "src.visualization.plots", "src.visualization.eda",
                "src.evaluation.metrics", "src.evaluation.interpretability", "src.modeling.train",
                "src.modeling.municipios", "src.visualization.slides", "src.run_pipeline"]:
        importlib.reload(importlib.import_module(mod))

    import dados_sinteticos
    dados_sinteticos.gerar(pasta / "data" / "raw")

    import src.modeling.train as train
    train.N_ITER.update({"logistica": 2, "gradient_boosting": 2})
    from src import run_pipeline
    run_pipeline.main(None)
    return pasta


def test_base_aluno_sem_vazamento(saida):
    base = pd.read_parquet(saida / "data" / "processed" / "base_aluno.parquet")
    assert "proficiencia" not in base.columns
    assert set(base["ano"]) == {2024}


def test_teste_separado_por_escola(saida):
    teste = pd.read_parquet(saida / "data" / "processed" / "aluno_previsoes_teste.parquet")
    treino_escolas = set(pd.read_parquet(saida / "data" / "processed" / "base_aluno.parquet")["id_escola"]) - set(teste["id_escola"])
    assert not (treino_escolas & set(teste["id_escola"]))


def test_saidas(saida):
    m = json.loads((saida / "reports" / "aluno_metricas.json").read_text())
    assert 0.5 <= m["teste"]["roc_auc"] <= 1
    assert (saida / "reports" / "ranking_risco_meta_2025.csv").exists()
    assert (saida / "reports" / "apresentacao_fase3.pptx").exists()
