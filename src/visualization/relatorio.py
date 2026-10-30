"""Escreve a secao de resultados do README com os numeros da ultima execucao.

O texto fica entre os marcadores <!-- RESULTADOS:INICIO --> e
<!-- RESULTADOS:FIM --> do README.md, entao README e apresentacao sempre
mostram os mesmos numeros.
"""
import json
import logging

from src import config
from src.visualization.plots import NOMES

log = logging.getLogger(__name__)

INICIO, FIM = "<!-- RESULTADOS:INICIO -->", "<!-- RESULTADOS:FIM -->"


def _p(v, casas=1):
    return f"{v:.{casas}f}".replace(".", ",")


def _ler(nome):
    with open(config.REPORTS / nome, encoding="utf-8") as f:
        return json.load(f)


def montar() -> str:
    eda, aluno, mun = _ler("eda_resumo.json"), _ler("aluno_metricas.json"), _ler("municipio_metricas.json")
    t, b = aluno["teste"], aluno["baseline"]
    linhas = [
        "### Base analisada",
        "",
        f"- {eda['alunos_modelagem']:,} alunos de {config.ANO_ALVO} (presentes e com prova preenchida), "
        f"{eda['escolas']:,} escolas, {eda['municipios_alunos']:,} municípios.".replace(",", "."),
        f"- {_p(eda['taxa_alfabetizados_amostra'])}% dos alunos da amostra foram considerados alfabetizados.",
        f"- {_p(eda['pct_nao_atingiu_meta'])}% dos {eda['municipios_com_meta']} municípios com meta ficaram abaixo da meta de {config.ANO_ALVO}.",
        "",
        "### Análise exploratória",
        "",
        f"- Municípios no quintil mais baixo de resultado em {config.ANO_HISTORICO} tiveram {_p(eda['taxa_quintil_hist_baixo'])}% "
        f"de alunos alfabetizados em {config.ANO_ALVO}; no quintil mais alto, {_p(eda['taxa_quintil_hist_alto'])}%.",
        f"- Correlação de Spearman entre a taxa de um ano e a do seguinte: {_p(eda['corr_taxa_historico'], 2)}.",
        f"- Correlação com PIB per capita: {_p(eda['corr_taxa_pib'], 2)}; com peso da agropecuária: {_p(eda['corr_taxa_agro'], 2)}.",
        "- Taxa por região (amostra): " + ", ".join(f"{k} {_p(v)}%" for k, v in eda["taxa_por_regiao"].items()) + ".",
        "- Taxa por rede: " + ", ".join(f"{k} {_p(v)}%" for k, v in eda["taxa_por_rede"].items()) + ".",
        "",
        "![Histórico x aluno](images/eda_04_historico_vs_aluno.png)",
        "",
        "### Modelo do aluno",
        "",
        "| Modelo | AUC treino (CV) | AUC validação (CV) | Desvio |",
        "|---|---|---|---|",
    ]
    for nome, m in aluno["modelos"].items():
        linhas.append(f"| {nome} | {_p(m['cv_auc_treino'], 3)} | {_p(m['cv_auc_media'], 3)} | {_p(m['cv_auc_desvio'], 3)} |")
    linhas += [
        "",
        f"Modelo escolhido: **{aluno['modelo_escolhido']}** (maior AUC na validação cruzada).",
        "",
        "| Métrica (teste) | Modelo | Baseline (classe mais comum) |",
        "|---|---|---|",
        f"| ROC AUC | {_p(t['roc_auc'], 3)} | {_p(b['roc_auc'], 3)} |",
        f"| PR AUC | {_p(t['pr_auc'], 3)} | {_p(b['pr_auc'], 3)} |",
        f"| Acurácia balanceada | {_p(t['acuracia_balanceada'], 3)} | {_p(b['acuracia_balanceada'], 3)} |",
        f"| Recall alfabetizados | {_p(t['recall'], 3)} | {_p(b['recall'], 3)} |",
        f"| Recall não alfabetizados | {_p(t['especificidade'], 3)} | {_p(b['especificidade'], 3)} |",
        f"| Brier | {_p(t['brier'], 3)} | {_p(b['brier'], 3)} |",
        "",
        f"Diferença de AUC entre validação cruzada e teste: {_p(aluno['gap_generalizacao_auc'], 3)} "
        "(perto de zero indica que o modelo generaliza para escolas que não viu no treino).",
        "",
        "![Avaliação](images/aluno_avaliacao.png)",
        "",
        "Variáveis de maior influência (SHAP médio absoluto):",
        "",
    ]
    for i, v in enumerate(aluno["top_variaveis"][:6], 1):
        efeito = f", valores maiores {v['efeito_quando_maior']} a chance" if v.get("efeito_quando_maior") in ("aumenta", "diminui") else ""
        linhas.append(f"{i}. {NOMES.get(v['variavel'], v['variavel'])}{efeito}")
    proj, cl, tm = mun["projecao"], mun["clusters"], mun["teste"]
    linhas += [
        "",
        "![SHAP](images/aluno_shap_importancia.png)",
        "",
        "### Municípios: risco de não atingir a meta",
        "",
        f"Modelo escolhido: **{mun['modelo_escolhido']}**, AUC no teste {_p(tm['roc_auc'], 3)} "
        f"(baseline {_p(mun['baseline']['roc_auc'], 3)}).",
        "",
        f"Aplicado aos dados de {config.ANO_ALVO}, o modelo aponta **{proj['risco_alto']} municípios "
        f"({_p(proj['pct_risco_alto'])}%) com risco alto** de não atingir a meta de {proj['ano_meta']}. "
        f"Lista completa: `reports/ranking_risco_meta_{proj['ano_meta']}.csv`.",
        "",
        "Risco alto por região: " + ", ".join(f"{k} {v}" for k, v in proj["risco_alto_por_regiao"].items()) + ".",
        "",
        "![Risco por UF](images/municipio_risco_por_uf.png)",
        "",
        "### Grupos de municípios semelhantes",
        "",
        f"K-Means com k={cl['k']} (maior silhueta entre 3 e 6). Grupo 1 é o de menor taxa.",
        "",
        "| Grupo | Municípios | Taxa média | Esforço até a meta (p.p.) | Risco médio | Região mais comum |",
        "|---|---|---|---|---|---|",
    ]
    for p in cl["perfil"]:
        linhas.append(f"| {p['cluster']} | {p['municipios']} | {_p(p['taxa_media'])}% | {_p(p['esforco_medio'])} | "
                      f"{_p(p['risco_medio'] * 100, 0)}% | {p['regiao_predominante']} |")
    linhas += ["", "![Clusters](images/municipio_clusters.png)"]
    return "\n".join(linhas)


def main() -> None:
    readme = config.ROOT / "README.md"
    if config.SAIDA != config.ROOT:  # execucao de teste: nao mexe no README real
        readme = config.REPORTS / "README_resultados.md"
        readme.write_text(f"{INICIO}\n{FIM}\n", encoding="utf-8")
    texto = readme.read_text(encoding="utf-8")
    if INICIO not in texto:
        log.warning("Marcadores de resultado nao encontrados em %s", readme)
        return
    antes, resto = texto.split(INICIO, 1)
    depois = resto.split(FIM, 1)[1]
    readme.write_text(f"{antes}{INICIO}\n{montar()}\n{FIM}{depois}", encoding="utf-8")
    log.info("Resultados atualizados em %s", readme)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
