# Predição e Inteligência Analítica para Alfabetização no Brasil

Tech Challenge, Fase 3 (Pós Tech FIAP). Continuação da pipeline de dados da Fase 2
([TECH_CHALLENGE_FASE2_PIPELINE](https://github.com/daianeaxr/TECH_CHALLENGE_FASE2_PIPELINE)):
os dados da camada Gold agora alimentam análise exploratória, modelos de Machine
Learning e respostas para perguntas de gestão pública.

## Sumário

1. [Contexto do problema](#contexto-do-problema)
2. [Objetivo analítico](#objetivo-analítico)
3. [Base utilizada](#base-utilizada)
4. [Etapas de modelagem](#etapas-de-modelagem)
5. [Data leakage: o que foi feito](#data-leakage-o-que-foi-feito)
6. [Escolha do algoritmo](#escolha-do-algoritmo)
7. [Métricas de avaliação](#métricas-de-avaliação)
8. [Resultados, interpretação e insights](#resultados-interpretação-e-insights)
9. [Perguntas de negócio](#perguntas-de-negócio)
10. [Aplicação prática para políticas públicas](#aplicação-prática-para-políticas-públicas)
11. [Limitações](#limitações)
12. [Evoluções futuras](#evoluções-futuras)
13. [Estrutura do repositório](#estrutura-do-repositório)
14. [Como rodar](#como-rodar)

## Contexto do problema

O Compromisso Nacional Criança Alfabetizada quer todas as crianças alfabetizadas ao fim
do 2º ano do ensino fundamental até 2030. O INEP definiu 743 pontos na escala do Saeb
como o ponto a partir do qual a criança é considerada alfabetizada, e cada município
recebeu uma trajetória de metas de 2024 a 2030.

O resultado da avaliação só aparece depois que o ano letivo acabou. Para o gestor, isso
chega tarde: a escola que precisava de reforço já passou o ano sem ele. Este projeto
tenta antecipar esse risco com o que já se sabe antes da prova.

## Objetivo analítico

**Principal:** modelo supervisionado que prevê se um aluno avaliado em 2024 será
considerado alfabetizado, usando variáveis educacionais, territoriais e socioeconômicas
conhecidas antes da prova.

**Complementares** (perguntas de negócio do desafio):

- modelo por município que prevê quem não vai atingir a meta do ano seguinte;
- ranking de municípios por risco para a meta de 2025;
- agrupamento de municípios com perfil semelhante (K-Means);
- interpretação com Permutation Importance e SHAP.

## Base utilizada

| Origem | Tabela | Uso |
|---|---|---|
| Fase 2 (Silver/Gold) | `alunos` | microdados da avaliação, uma linha por aluno; alvo `alfabetizado` |
| Fase 2 (Silver/Gold) | `resultado_municipio` | taxa de alfabetização e média de português por município, ano e rede |
| Fase 2 (Silver/Gold) | `metas_municipio` | metas 2024 a 2030, já no formato longo (`ano_meta`, `meta_valor`) |
| Fase 2 (Silver) + diretório | `municipio` | UF, região, capital, Amazônia Legal, região metropolitana |
| Fonte externa: IBGE | `br_ibge_populacao.municipio` | população |
| Fonte externa: IBGE | `br_ibge_pib.municipio` | PIB, PIB per capita e peso de agro, indústria e serviços |

A extração (`src/preprocessing/extract.py`) tem dois modos:

- `athena`: lê as tabelas Silver/Gold catalogadas no Glue pela pipeline da Fase 2;
- `bigquery`: lê as mesmas tabelas de origem na Base dos Dados e aplica o mesmo tratamento
  da Silver (chave de município com 7 dígitos e metas despivotadas). Serve para quando a
  infraestrutura AWS da Fase 2 já foi desmontada para não gerar custo.

A tabela de alunos tem cerca de 3,9 milhões de linhas. Assim como na Bronze da Fase 2,
usamos uma amostra de 2% (`TABLESAMPLE`), ajustável em `ALUNOS_SAMPLE_PERCENT`.

Duas bases analíticas são montadas em `src/preprocessing/build_dataset.py`:

- `base_aluno`: alunos de 2024 presentes e com prova preenchida;
- `base_municipio`: uma linha por município (rede municipal). Com `ano_ref=2024` tem alvo
  (`nao_atingiu_meta`) e features de 2023; com `ano_ref=2025` tem as mesmas features
  calculadas com 2024, usadas para projetar o risco.

## Etapas de modelagem

1. **Análise exploratória** (`src/visualization/eda.py`): distribuição do alvo por região,
   UF e rede; relação entre histórico do município e resultado do aluno; correlações de
   Spearman; distância até a meta; percentual de dados faltantes. Os achados definiram as
   features: o histórico do município apareceu como o sinal mais forte, e PIB per capita
   e peso da agropecuária mostraram relação com a taxa.
2. **Engenharia de atributos** (`FeatureEngineering` em `src/modeling/pipeline.py`):
   distância entre a meta e o histórico do município (`gap_meta_historico`) e esforço
   relativo até a meta (`esforco_relativo`). São cálculos linha a linha, sem estatística
   do conjunto, então não vazam informação entre treino e teste.
3. **Pré-processamento dentro do Pipeline** (`ColumnTransformer`):
   - numéricas: imputação pela mediana com indicador de faltante (`add_indicator=True`)
     e padronização;
   - PIB per capita (assimétrico): imputação, `log1p` e padronização;
   - binárias: imputação pela moda;
   - categóricas (rede, região, UF, série): imputação pela moda e one-hot, com categorias
     raras agrupadas (`min_frequency=20`) e categorias novas tratadas (`handle_unknown`).
4. **Treino e otimização** (`src/modeling/train.py`): regressão logística e
   HistGradientBoosting, cada um com `RandomizedSearchCV` em validação cruzada de 5 dobras.
5. **Validação final**: o conjunto de teste é usado uma única vez, com o modelo e o limiar
   já escolhidos.
6. **Interpretação** (`src/evaluation/interpretability.py`): Permutation Importance no
   teste e SHAP values (TreeExplainer ou LinearExplainer), somados por variável original.

## Data leakage: o que foi feito

| Risco | Tratamento |
|---|---|
| `proficiencia` define o alvo (`alfabetizado` = proficiência ≥ 743) | removida da base |
| Taxa do município no mesmo ano inclui o próprio aluno | só entra o resultado de **2023** para prever **2024** |
| Metas de 2024 foram calculadas com o resultado de 2023 | usamos só alunos de 2024; para eles, a meta já estava publicada antes da prova |
| Dados do IBGE posteriores à prova | só o último ano disponível até 2023 |
| Alunos da mesma escola no treino e no teste | `GroupShuffleSplit` e `GroupKFold` por `id_escola`; o teste confere que nenhuma escola aparece nos dois |
| Imputação e escala ajustadas com dados de validação | todo o pré-processamento fica dentro do `Pipeline`, reajustado em cada dobra |
| Limiar de decisão ajustado no teste | limiar escolhido (Youden) com previsões fora da amostra do treino (`cross_val_predict`) |
| `caderno` e `peso_aluno` | fora das features: o caderno é sorteado e o peso é desenho amostral |

## Escolha do algoritmo

- **Regressão logística:** ponto de partida interpretável, com `class_weight="balanced"`
  e regularização L2 ajustada (`C`).
- **HistGradientBoosting:** captura interações (por exemplo rede de ensino x histórico do
  município) e lida bem com dezenas de milhares de linhas. Regularização via
  `min_samples_leaf`, `max_depth`, `max_leaf_nodes` e `l2_regularization`.

Ganha o que tiver maior AUC média na validação cruzada. A distância entre AUC de treino e
de validação, e entre validação e teste, é reportada para mostrar se há overfitting.

## Métricas de avaliação

- **ROC AUC** (métrica de seleção): mede a capacidade de ordenar alunos por risco, que é
  o uso real do modelo (priorizar), e não depende do limiar.
- **PR AUC, recall das duas classes e acurácia balanceada:** as classes são próximas de
  50/50, mas o erro que mais custa é deixar de apontar um aluno que não vai se alfabetizar.
- **Brier e curva de calibração:** a probabilidade precisa fazer sentido para ser usada
  como "nível de risco".
- **Baseline:** classificador que sempre prevê a classe mais frequente.

## Resultados, interpretação e insights

Esta seção é preenchida automaticamente por `python -m src.run_pipeline`, com os números
da última execução sobre os dados reais. Os mesmos números alimentam a apresentação
(`reports/apresentacao_fase3.pptx`).

<!-- RESULTADOS:INICIO -->
_Rode `python -m src.run_pipeline --extrair bigquery` para preencher._
<!-- RESULTADOS:FIM -->

## Perguntas de negócio

| Pergunta | Onde está a resposta |
|---|---|
| Quais fatores mais impactam a alfabetização? | SHAP e Permutation Importance do modelo do aluno (`reports/aluno_importancia.csv`, `images/aluno_shap_*.png`) |
| Quais municípios apresentam maior risco educacional? | `reports/ranking_risco_meta_2025.csv` |
| Quais regiões possuem padrões semelhantes? | K-Means (`reports/clusters_perfil.csv`, `reports/clusters_municipios.csv`) |
| Como prever municípios que podem não atingir metas futuras? | modelo de município treinado com 2024 e aplicado aos dados de 2024 para a meta de 2025 |
| Quais variáveis possuem maior influência nos modelos? | `reports/aluno_importancia.csv` e `reports/municipio_importancia.csv` |

## Aplicação prática para políticas públicas

- **Priorizar antes da prova:** o ranking de risco para 2025 sai com dados que a secretaria
  já tem em mãos, então dá para direcionar formação de professores, material e
  acompanhamento para onde a chance de ficar abaixo da meta é maior.
- **Desenhar ações por grupo:** municípios no mesmo cluster têm perfil parecido de
  resultado, renda, porte e economia; faz sentido testar a mesma estratégia neles e
  comparar.
- **Monitorar:** a pipeline é reproduzível, então a cada nova divulgação do INEP basta
  rodar de novo e ver quem saiu ou entrou na zona de risco.
- **Usar com cuidado:** o modelo aponta onde olhar, não diz por que o resultado é baixo
  nem deve servir para punir rede ou escola.

## Limitações

- Só existem duas edições da avaliação (2023 e 2024). O modelo de município aprende com
  uma única transição de ano e a projeção para 2025 assume que o padrão se repete.
- A base de alunos é uma amostra de 2%; escolas pequenas podem aparecer com poucos alunos.
- Não há variáveis individuais socioeconômicas do aluno nos microdados; as variáveis
  socioeconômicas são do município, o que limita o poder de prever o aluno.
- `peso_aluno` não foi usado como peso de treino; as taxas da amostra não substituem o
  indicador oficial do INEP.
- Correlação não é causa: renda e economia agrícola aparecem associadas ao resultado, mas
  o modelo não mostra o mecanismo.

## Evoluções futuras

- Incluir Censo Escolar (infraestrutura, formação docente, alunos por turma), Atlas do
  Desenvolvimento Humano (IDHM), Cadastro Único e FUNDEB, já previstos na Silver da Fase 2
  pela chave `id_municipio`.
- Com a edição de 2025, validar o modelo de município de forma temporal (treina 2024,
  testa 2025).
- Levar a etapa de modelagem para a AWS (Glue ou SageMaker) disparada pela chegada de
  dados novos na Gold, como já acontece na Fase 2.
- Painel para as secretarias com o ranking e os grupos.

## Estrutura do repositório

```
tech-challenge-fase3
├── data/                     # raw/ e processed/ (gerados, fora do git)
├── notebooks/
│   └── fase3_alfabetizacao.ipynb   # roteiro: EDA -> modelo -> interpretação -> negócio
├── src/
│   ├── config.py
│   ├── run_pipeline.py       # executa tudo de ponta a ponta
│   ├── preprocessing/        # extract.py, build_dataset.py
│   ├── modeling/             # pipeline.py, train.py, municipios.py
│   ├── evaluation/           # metrics.py, interpretability.py
│   └── visualization/        # plots.py, eda.py, relatorio.py, slides.py
├── reports/                  # métricas, rankings, clusters, apresentação
├── images/                   # gráficos
├── tests/                    # teste de ponta a ponta com dados sintéticos
├── requirements.txt
├── README.md
└── .gitignore
```

## Como rodar

```bash
pip install -r requirements.txt

# credencial do GCP (a mesma service account da Fase 2)
export GCP_SERVICE_ACCOUNT_JSON=/caminho/para/service-account.json
export GCP_BILLING_PROJECT=techchallenge-505723

# tudo de uma vez: extrai, monta as bases, EDA, modelos, README e apresentação
python -m src.run_pipeline --extrair bigquery

# alternativa com a infra da Fase 2 de pé (lê Silver/Gold pelo Athena)
python -m src.run_pipeline --extrair athena
```

Cada etapa também roda sozinha (`python -m src.preprocessing.build_dataset`,
`python -m src.modeling.train`, etc.). O notebook `notebooks/fase3_alfabetizacao.ipynb`
percorre as mesmas etapas mostrando tabelas e gráficos.

Teste de ponta a ponta (gera dados falsos numa pasta temporária, não mexe em `data/`):

```bash
python -m pytest tests -q
```

Reprodutibilidade: `RANDOM_STATE = 42` em toda divisão, busca de hiperparâmetros e
K-Means; versões mínimas em `requirements.txt`.

## Vídeo executivo

Link: _adicionar após a gravação_.
