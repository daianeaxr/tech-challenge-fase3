"""Interpretabilidade: Permutation Importance e SHAP values.

As duas tecnicas sao reportadas por variavel ORIGINAL (ex.: "sigla_uf"),
somando as colunas criadas pelo one-hot e pelos indicadores de faltante.
"""
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src import config
from src.visualization.plots import CINZA, NOMES, ROSA, plt, salvar


def _variavel_original(nome: str, originais: list[str]) -> str:
    if nome.startswith("missingindicator_"):
        nome = nome[len("missingindicator_"):]
    candidatos = [o for o in originais if nome == o or nome.startswith(o + "_")]
    return max(candidatos, key=len) if candidatos else nome


def permutacao(pipeline, X: pd.DataFrame, y, n_max: int = 20000) -> pd.DataFrame:
    if len(X) > n_max:
        idx = np.random.RandomState(config.RANDOM_STATE).choice(len(X), n_max, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    r = permutation_importance(pipeline, X, y, scoring="roc_auc", n_repeats=5,
                               random_state=config.RANDOM_STATE, n_jobs=-1)
    return (pd.DataFrame({"variavel": X.columns, "perm_importancia": r.importances_mean,
                          "perm_desvio": r.importances_std})
            .sort_values("perm_importancia", ascending=False))


def valores_shap(pipeline, X: pd.DataFrame, originais: list[str], n_max: int = 2000):
    amostra = X.sample(min(n_max, len(X)), random_state=config.RANDOM_STATE)
    transformador = pipeline[:-1]
    modelo = pipeline[-1]
    Xt = pd.DataFrame(transformador.transform(amostra),
                      columns=transformador[-1].get_feature_names_out())

    if modelo.__class__.__name__ == "LogisticRegression":
        explicador = shap.LinearExplainer(modelo, Xt)
    else:
        explicador = shap.TreeExplainer(modelo)
    valores = explicador.shap_values(Xt)
    if isinstance(valores, list):  # versoes antigas devolvem uma matriz por classe
        valores = valores[1]
    valores = np.asarray(valores)
    if valores.ndim == 3:
        valores = valores[:, :, 1]

    mapa = {c: _variavel_original(c, originais) for c in Xt.columns}
    abs_agregado = pd.DataFrame(np.abs(valores), columns=Xt.columns).T.groupby(mapa).sum().T.mean()

    # direcao do efeito (so variaveis numericas): correlacao valor x SHAP
    direcao = {}
    for col in Xt.columns:
        if col in originais and Xt[col].nunique() > 2:
            c = np.corrcoef(Xt[col], valores[:, list(Xt.columns).index(col)])[0, 1]
            direcao[col] = "aumenta" if c > 0 else "diminui"

    tabela = (pd.DataFrame({"variavel": abs_agregado.index, "shap_medio_abs": abs_agregado.values})
              .assign(efeito_quando_maior=lambda d: d["variavel"].map(direcao))
              .sort_values("shap_medio_abs", ascending=False))
    return tabela, valores, Xt


def interpretar(pipeline, X_teste, y_teste, prefixo: str, alvo: str) -> pd.DataFrame:
    originais = list(X_teste.columns) + ["gap_meta_historico", "esforco_relativo"]
    perm = permutacao(pipeline, X_teste, y_teste)
    shap_tab, valores, Xt = valores_shap(pipeline, X_teste, originais)
    tabela = shap_tab.merge(perm, on="variavel", how="outer").sort_values("shap_medio_abs", ascending=False)
    tabela.to_csv(config.REPORTS / f"{prefixo}_importancia.csv", index=False)

    top = tabela.dropna(subset=["shap_medio_abs"]).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["variavel"].map(lambda v: NOMES.get(v, v)), top["shap_medio_abs"], color=ROSA)
    ax.set_xlabel(f"Impacto médio |SHAP| na previsão de '{alvo}'")
    ax.set_title("Variáveis com maior influência (SHAP)")
    ax.grid(axis="y", visible=False)
    salvar(fig, f"{prefixo}_shap_importancia")

    top_perm = tabela.dropna(subset=["perm_importancia"]).sort_values("perm_importancia").tail(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_perm["variavel"].map(lambda v: NOMES.get(v, v)), top_perm["perm_importancia"], xerr=top_perm["perm_desvio"],
            color=CINZA, ecolor=ROSA)
    ax.set_xlabel("Queda no ROC AUC ao embaralhar a variável")
    ax.set_title("Permutation Importance (teste)")
    ax.grid(axis="y", visible=False)
    salvar(fig, f"{prefixo}_permutation_importance")

    plt.figure()
    shap.summary_plot(valores, Xt, max_display=12, show=False, plot_size=(8, 5.5))
    fig = plt.gcf()
    fig.axes[0].set_title("Direção do efeito (SHAP, amostra do teste)")
    salvar(fig, f"{prefixo}_shap_beeswarm")
    return tabela
