"""Estilo unico dos graficos (preto e rosa, mesmo padrao da apresentacao)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402

PRETO = "#000000"
CINZA_ESCURO = "#1A1A1A"
CINZA = "#8C8C8C"
BRANCO = "#FFFFFF"
ROSA = "#ED145B"
ROSA_CLARO = "#F7769F"


# nomes legiveis das variaveis (graficos e apresentacao)
NOMES = {
    "hist_taxa_mesma_rede": "Resultado anterior do município na mesma rede",
    "hist_media_lp_mesma_rede": "Média de português anterior (mesma rede)",
    "hist_publica_taxa": "Resultado anterior da rede pública do município",
    "hist_publica_media_lp": "Média de português anterior (rede pública)",
    "meta_municipal_ano": "Meta do município para o ano",
    "meta_municipal_2030": "Meta do município para 2030",
    "gap_meta_historico": "Distância entre meta e histórico",
    "log_populacao": "Tamanho da população",
    "pib_per_capita": "PIB per capita",
    "part_agropecuaria": "Peso da agropecuária na economia",
    "part_industria": "Peso da indústria na economia",
    "part_servicos": "Peso dos serviços na economia",
    "capital_uf": "Capital do estado",
    "amazonia_legal": "Amazônia Legal",
    "regiao_metropolitana": "Região metropolitana",
    "rede_aluno": "Rede de ensino do aluno",
    "serie": "Série",
    "nome_regiao": "Região do país",
    "sigla_uf": "Estado (UF)",
    "taxa_anterior": "Taxa de alfabetização do ano anterior",
    "media_lp_anterior": "Média de português do ano anterior",
    "meta_ano": "Meta do ano",
    "meta_2030": "Meta 2030",
    "esforco_relativo": "Esforço relativo até a meta",
}


def aplicar_estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": PRETO,
        "axes.facecolor": PRETO,
        "savefig.facecolor": PRETO,
        "axes.edgecolor": CINZA,
        "axes.labelcolor": BRANCO,
        "axes.titlecolor": BRANCO,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": CINZA_ESCURO,
        "xtick.color": BRANCO,
        "ytick.color": BRANCO,
        "text.color": BRANCO,
        "legend.frameon": False,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=[ROSA, BRANCO, CINZA, ROSA_CLARO]),
    })


def salvar(fig, nome: str) -> str:
    caminho = config.IMAGES / f"{nome}.png"
    fig.tight_layout()
    fig.savefig(caminho, dpi=160)
    plt.close(fig)
    return str(caminho)


aplicar_estilo()
