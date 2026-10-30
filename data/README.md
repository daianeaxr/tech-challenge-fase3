# data/

- `raw/`: tabelas extraídas por `python -m src.preprocessing.extract` (Parquet).
- `processed/`: bases analíticas (`base_aluno`, `base_municipio`) e previsões do teste.

Os arquivos não vão para o git; qualquer pessoa recria tudo com
`python -m src.run_pipeline --extrair bigquery`.
