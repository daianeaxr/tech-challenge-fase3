"""Metricas e graficos de avaliacao."""
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

from src.visualization.plots import BRANCO, CINZA, ROSA, plt, salvar


def limiar_youden(y_true, proba) -> float:
    """Limiar que maximiza sensibilidade + especificidade. Calculado nas
    previsoes fora-da-amostra do TREINO, nunca no teste."""
    fpr, tpr, thr = roc_curve(y_true, proba)
    return float(thr[np.argmax(tpr - fpr)])


def calcular_metricas(y_true, proba, limiar: float) -> dict:
    pred = (proba >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "acuracia": round(accuracy_score(y_true, pred), 4),
        "acuracia_balanceada": round(balanced_accuracy_score(y_true, pred), 4),
        "precisao": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "especificidade": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "f1": round(f1_score(y_true, pred), 4),
        "brier": round(brier_score_loss(y_true, proba), 4),
        "limiar": round(limiar, 4),
        "matriz_confusao": {"vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp)},
        "n": int(len(y_true)),
        "prevalencia": round(float(np.mean(y_true)), 4),
    }


def graficos_avaliacao(y_true, proba, limiar: float, prefixo: str, rotulo_positivo: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    ax.plot(fpr, tpr, color=ROSA, linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=CINZA, linestyle="--")
    ax.set_xlabel("Falso positivo")
    ax.set_ylabel("Verdadeiro positivo")
    ax.set_title("Curva ROC (teste)")
    ax.legend(loc="lower right")

    ax = axes[1]
    frac, media = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    ax.plot(media, frac, marker="o", color=ROSA)
    ax.plot([0, 1], [0, 1], color=CINZA, linestyle="--")
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Frequência observada")
    ax.set_title("Calibração (teste)")

    ax = axes[2]
    cm = confusion_matrix(y_true, (proba >= limiar).astype(int), labels=[0, 1])
    ax.imshow(cm, cmap="RdPu")
    ax.grid(False)
    rotulos = [f"não {rotulo_positivo}", rotulo_positivo]
    ax.set_xticks([0, 1], rotulos)
    ax.set_yticks([0, 1], rotulos)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center",
                    color="black" if cm[i, j] > cm.max() / 2 else BRANCO, fontsize=12)
    ax.set_title(f"Matriz de confusão (limiar {limiar:.2f})")
    salvar(fig, f"{prefixo}_avaliacao")


def grafico_comparacao_modelos(resultados: dict, prefixo: str) -> None:
    nomes = list(resultados)
    medias = [resultados[n]["cv_auc_media"] for n in nomes]
    desvios = [resultados[n]["cv_auc_desvio"] for n in nomes]
    treino = [resultados[n]["cv_auc_treino"] for n in nomes]
    x = np.arange(len(nomes))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x - 0.18, treino, width=0.36, color=CINZA, label="treino (CV)")
    ax.bar(x + 0.18, medias, width=0.36, yerr=desvios, color=ROSA, ecolor=BRANCO, capsize=4,
           label="validação (CV)")
    ax.set_xticks(x, nomes)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Comparação de modelos (validação cruzada por grupo)")
    ax.legend()
    salvar(fig, f"{prefixo}_comparacao_modelos")
