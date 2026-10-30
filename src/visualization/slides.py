"""Gera a apresentacao executiva (preto e rosa) a partir dos resultados.

Todos os numeros vem de reports/*.json e os graficos de images/, entao a
apresentacao sempre bate com a ultima execucao da pipeline.

Uso:
    python -m src.visualization.slides
"""
import json
import logging

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from src import config
from src.visualization.plots import NOMES

log = logging.getLogger(__name__)

PRETO = RGBColor(0x00, 0x00, 0x00)
CARTAO = RGBColor(0x1A, 0x1A, 0x1A)
BRANCO = RGBColor(0xFF, 0xFF, 0xFF)
CINZA = RGBColor(0xA6, 0xA6, 0xA6)
ROSA = RGBColor(0xED, 0x14, 0x5B)
FONTE = "Arial"

W, H = Inches(13.333), Inches(7.5)
MARGEM = Inches(0.6)



def _num(v, casas=1) -> str:
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _inteiro(v) -> str:
    return f"{int(v):,}".replace(",", ".")


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.layout = self.prs.slide_layouts[6]

    def slide(self, titulo: str | None = None, notas: str = ""):
        s = self.prs.slides.add_slide(self.layout)
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = PRETO
        if titulo:
            self.texto(s, titulo, MARGEM, Inches(0.45), W - 2 * MARGEM, Inches(0.9), 32, ROSA, negrito=True)
        if notas:
            s.notes_slide.notes_text_frame.text = notas
        return s

    def texto(self, s, conteudo, x, y, w, h, tamanho=16, cor=BRANCO, negrito=False, alinhar=PP_ALIGN.LEFT):
        caixa = s.shapes.add_textbox(x, y, w, h)
        tf = caixa.text_frame
        tf.word_wrap = True
        linhas = conteudo if isinstance(conteudo, list) else [conteudo]
        for i, linha in enumerate(linhas):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = alinhar
            p.space_after = Pt(8)
            r = p.add_run()
            r.text = linha
            r.font.size, r.font.color.rgb, r.font.bold, r.font.name = Pt(tamanho), cor, negrito, FONTE
        return caixa

    def cartao(self, s, x, y, w, h):
        forma = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
        forma.adjustments[0] = 0.06
        forma.fill.solid()
        forma.fill.fore_color.rgb = CARTAO
        forma.line.fill.background()
        return forma

    def numero(self, s, valor, rotulo, x, y, w):
        self.cartao(s, x, y, w, Inches(1.9))
        self.texto(s, valor, x + Inches(0.25), y + Inches(0.2), w - Inches(0.5), Inches(0.9), 40, ROSA, True)
        self.texto(s, rotulo, x + Inches(0.25), y + Inches(1.05), w - Inches(0.5), Inches(0.8), 13, CINZA)

    def imagem(self, s, nome, x, y, w=None, h=None):
        caminho = config.IMAGES / f"{nome}.png"
        if caminho.exists():
            s.shapes.add_picture(str(caminho), x, y, width=w, height=h)

    def salvar(self, caminho):
        self.prs.save(caminho)


def _ler(nome):
    with open(config.REPORTS / nome, encoding="utf-8") as f:
        return json.load(f)


def main(autora: str = "Daiane Aparecida Xavier Ribeiro") -> str:
    eda = _ler("eda_resumo.json")
    aluno = _ler("aluno_metricas.json")
    mun = _ler("municipio_metricas.json")
    t, base = aluno["teste"], aluno["baseline"]
    proj = mun["projecao"]
    top = [NOMES.get(v["variavel"], v["variavel"]) for v in aluno["top_variaveis"][:3]]
    top_mun = [NOMES.get(v["variavel"], v["variavel"]) for v in mun["top_variaveis"][:3]]
    nao_alf = 100 - eda["taxa_alfabetizados_amostra"]
    d = Deck()
    larg = W - 2 * MARGEM

    # 1. capa
    s = d.slide(notas="Abertura: o objetivo e mostrar como os dados ajudam a agir antes do resultado ruim aparecer.")
    d.texto(s, "TECH CHALLENGE  |  FASE 3", MARGEM, Inches(2.2), larg, Inches(0.5), 16, CINZA, True)
    d.texto(s, "Alfabetização no Brasil: prever para agir antes", MARGEM, Inches(2.75), Inches(10), Inches(1.6),
            44, BRANCO, True)
    d.texto(s, "Modelo preditivo e inteligência analítica sobre o Indicador Criança Alfabetizada",
            MARGEM, Inches(4.4), Inches(10), Inches(0.6), 18, ROSA)
    d.texto(s, f"{autora}  |  Pós Tech FIAP", MARGEM, Inches(6.4), larg, Inches(0.5), 14, CINZA)

    # 2. problema
    s = d.slide("O problema", "Contextualizar: a meta nacional é 2030 e hoje uma parte grande das crianças termina o 2o ano sem estar alfabetizada.")
    d.numero(s, f"{_num(nao_alf)}%", f"dos alunos avaliados em {config.ANO_ALVO} não atingiram o ponto de corte de 743 do Saeb",
             MARGEM, Inches(1.7), Inches(3.8))
    d.numero(s, f"{_num(eda['pct_nao_atingiu_meta'])}%", f"dos municípios ficaram abaixo da própria meta de {config.ANO_ALVO}",
             MARGEM + Inches(4.1), Inches(1.7), Inches(3.8))
    d.numero(s, "2030", "prazo do Compromisso Nacional Criança Alfabetizada", MARGEM + Inches(8.2), Inches(1.7), Inches(3.8))
    d.texto(s, ["O resultado só aparece depois da prova, quando o ano letivo já acabou.",
                "A pergunta deste trabalho: dá para saber antes onde o risco e maior?"],
            MARGEM, Inches(4.3), larg, Inches(1.5), 20)

    # 3. dados
    s = d.slide("De onde vem os dados", "Explicar que a base e a Gold da Fase 2, com enriquecimento do IBGE.")
    itens = [("Camada Gold da Fase 2", "Resultado por município, metas 2024 a 2030, dados territoriais e microdados de alunos"),
             ("IBGE", "População e PIB municipal, com participação de agro, indústria e serviços"),
             ("Base de modelagem", f"{_inteiro(eda['alunos_modelagem'])} alunos, {_inteiro(eda['escolas'])} escolas e "
                                   f"{_inteiro(eda['municipios_alunos'])} municípios em {config.ANO_ALVO}")]
    for i, (tit, desc) in enumerate(itens):
        x = MARGEM + i * Inches(4.1)
        d.cartao(s, x, Inches(1.8), Inches(3.8), Inches(3.2))
        d.texto(s, tit, x + Inches(0.3), Inches(2.0), Inches(3.2), Inches(0.9), 20, ROSA, True)
        d.texto(s, desc, x + Inches(0.3), Inches(2.9), Inches(3.2), Inches(2.0), 15)
    d.texto(s, f"Regra de ouro: o modelo só usa o que se sabe antes da prova de {config.ANO_ALVO} "
               f"(histórico de {config.ANO_HISTORICO}, metas já publicadas e dados do IBGE).",
            MARGEM, Inches(5.5), larg, Inches(1), 16, CINZA)

    # 4. EDA historico
    s = d.slide("O passado do município diz muito", "O histórico é o sinal mais forte encontrado na exploração.")
    d.imagem(s, "eda_04_historico_vs_aluno", MARGEM, Inches(1.5), w=Inches(7.4))
    d.texto(s, [f"Nos municípios com pior resultado em {config.ANO_HISTORICO}, {_num(eda['taxa_quintil_hist_baixo'])}% "
                f"dos alunos se alfabetizaram em {config.ANO_ALVO}.",
                f"Nos de melhor resultado, {_num(eda['taxa_quintil_hist_alto'])}%.",
                f"Correlação entre o resultado de um ano e do seguinte: {_num(eda['corr_taxa_historico'], 2)}."],
            Inches(8.3), Inches(2.0), Inches(4.4), Inches(4), 17)

    # 5. EDA territorio
    s = d.slide("A desigualdade tem endereço", "Diferença entre estados e regiões; renda também pesa.")
    d.imagem(s, "eda_02_uf", MARGEM, Inches(1.4), w=Inches(6.0))
    regs = sorted(eda["taxa_por_regiao"].items(), key=lambda kv: kv[1])
    d.texto(s, [f"Menor taxa na amostra: {eda['uf_menor_taxa']}. Maior: {eda['uf_maior_taxa']}.",
                f"Por região, vai de {_num(regs[0][1])}% ({regs[0][0]}) a {_num(regs[-1][1])}% ({regs[-1][0]}).",
                f"Correlação com PIB per capita: {_num(eda['corr_taxa_pib'], 2)}. "
                f"Com peso da agropecuária: {_num(eda['corr_taxa_agro'], 2)}."],
            Inches(7.0), Inches(2.0), Inches(5.7), Inches(4), 17)

    # 6. como o modelo foi feito
    s = d.slide("Como o modelo foi construído", "Mostrar a pipeline e, principalmente, como evitamos vazamento.")
    etapas = ["Dados faltantes: mediana + indicador", "Numéricas: escala e log no PIB",
              "Categóricas: one-hot", "Modelo: regressão logística x gradient boosting",
              "Ajuste: busca aleatória com validação cruzada"]
    for i, e in enumerate(etapas):
        x = MARGEM + i * Inches(2.45)
        d.cartao(s, x, Inches(1.7), Inches(2.25), Inches(1.9))
        d.texto(s, str(i + 1), x + Inches(0.2), Inches(1.8), Inches(1), Inches(0.6), 26, ROSA, True)
        d.texto(s, e, x + Inches(0.2), Inches(2.45), Inches(1.9), Inches(1.1), 13)
    d.texto(s, "Contra vazamento de dados", MARGEM, Inches(4.0), larg, Inches(0.5), 20, ROSA, True)
    d.texto(s, ["A proficiência ficou de fora: ela define o alvo (743 pontos).",
                "Treino e teste separados por escola. Nenhuma escola aparece nos dois.",
                "Todo o tratamento roda dentro do Pipeline, ajustado só com o treino de cada rodada.",
                "O teste foi usado uma única vez, no final."],
            MARGEM, Inches(4.6), larg, Inches(2.5), 16)

    # 7. resultados
    esc = aluno["modelo_escolhido"].replace("_", " ")
    s = d.slide("Quanto o modelo acerta", f"Modelo escolhido: {esc}. Comparar com o baseline que chuta sempre a classe mais comum.")
    d.numero(s, _num(t["roc_auc"], 2), f"AUC no teste (baseline: {_num(base['roc_auc'], 2)})", MARGEM, Inches(1.7), Inches(3.8))
    d.numero(s, f"{_num(t['especificidade'] * 100, 0)}%", "dos não alfabetizados identificados corretamente",
             MARGEM + Inches(4.1), Inches(1.7), Inches(3.8))
    d.numero(s, f"{_num(abs(aluno['gap_generalizacao_auc']), 3)}", "diferença de AUC entre validação e teste (generaliza bem se pequena)",
             MARGEM + Inches(8.2), Inches(1.7), Inches(3.8))
    d.imagem(s, "aluno_avaliacao", Inches(1.4), Inches(3.9), h=Inches(3.3))

    # 8. o que pesa
    s = d.slide("O que mais pesa na previsão", "SHAP mostra quanto cada variável move a probabilidade de alfabetização.")
    d.imagem(s, "aluno_shap_importancia", MARGEM, Inches(1.4), w=Inches(7.0))
    efeitos = []
    for v in aluno["top_variaveis"][:3]:
        if v.get("efeito_quando_maior") in ("aumenta", "diminui"):
            efeitos.append(f"Quanto maior {NOMES.get(v['variavel'], v['variavel']).lower()}, "
                           f"{'maior' if v['efeito_quando_maior'] == 'aumenta' else 'menor'} a chance de alfabetização.")
    d.texto(s, ["As três variáveis de maior influência:"] + [f"{i + 1}. {v}" for i, v in enumerate(top)] + efeitos[:2],
            Inches(8.0), Inches(2.0), Inches(4.7), Inches(4.5), 16)

    # 9. risco 2025
    s = d.slide(f"Municípios em risco para {proj['ano_meta']}",
                "Modelo de município: treinado com 2024, aplicado aos dados de 2024 para a meta de 2025.")
    d.numero(s, _inteiro(proj["risco_alto"]), f"municípios com risco alto de não atingir a meta de {proj['ano_meta']}",
             MARGEM, Inches(1.7), Inches(4.2))
    d.numero(s, f"{_num(proj['pct_risco_alto'])}%", "do total avaliado", MARGEM, Inches(3.9), Inches(4.2))
    d.texto(s, ["Variáveis que mais pesam aqui:"] + top_mun, MARGEM, Inches(6.0), Inches(4.4), Inches(1.4), 13, CINZA)
    d.imagem(s, "municipio_risco_por_uf", Inches(5.4), Inches(1.4), h=Inches(5.8))

    # 10. clusters
    cl = mun["clusters"]
    s = d.slide("Grupos de municípios parecidos", "K-Means com perfil educacional e socioeconômico; grupo 1 é o mais vulnerável.")
    d.imagem(s, "municipio_clusters", MARGEM, Inches(1.4), w=Inches(7.0))
    linhas = [f"{cl['k']} grupos com perfil semelhante."]
    for p in cl["perfil"]:
        linhas.append(f"Grupo {p['cluster']}: {_inteiro(p['municipios'])} municípios, taxa média {_num(p['taxa_media'])}%, "
                      f"risco {_num(p['risco_medio'] * 100, 0)}%, mais comum no {p['regiao_predominante']}")
    d.texto(s, linhas, Inches(8.0), Inches(1.6), Inches(4.8), Inches(5.5), 13)

    # 11. politica publica
    s = d.slide("Como usar na gestão pública", "Fechar com o valor prático para quem decide.")
    usos = [("Priorizar", f"Lista dos municípios com maior risco para {proj['ano_meta']}, antes da prova."),
            ("Agrupar ações", "Municípios do mesmo grupo podem receber o mesmo tipo de apoio."),
            ("Acompanhar", "Rodar de novo a cada resultado do INEP e medir quem saiu da zona de risco.")]
    for i, (tit, desc) in enumerate(usos):
        x = MARGEM + i * Inches(4.1)
        d.cartao(s, x, Inches(1.8), Inches(3.8), Inches(2.6))
        d.texto(s, tit, x + Inches(0.3), Inches(2.0), Inches(3.2), Inches(0.6), 22, ROSA, True)
        d.texto(s, desc, x + Inches(0.3), Inches(2.7), Inches(3.2), Inches(1.6), 15)
    d.texto(s, "Limites", MARGEM, Inches(4.8), larg, Inches(0.5), 18, ROSA, True)
    d.texto(s, ["Só existem dois anos de avaliação (2023 e 2024) e a base de alunos é uma amostra de "
                f"{config.ALUNOS_SAMPLE_PERCENT}%.",
                "O modelo aponta risco, não causa. Próximo passo: Censo Escolar, Atlas (IDHM) e Cadastro Único."],
            MARGEM, Inches(5.35), larg, Inches(1.6), 15, CINZA)

    caminho = config.REPORTS / "apresentacao_fase3.pptx"
    d.salvar(caminho)
    log.info("Apresentação salva em %s", caminho)
    return str(caminho)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
