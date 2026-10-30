"""Gera dados FALSOS no mesmo schema das tabelas reais, so para testar se o
codigo roda de ponta a ponta. Nenhum numero daqui vai para o README ou para a
apresentacao.
"""
import numpy as np
import pandas as pd

UFS = {"1": ["AM", "PA", "RO"], "2": ["BA", "CE", "PE", "MA"], "3": ["SP", "MG", "RJ"],
       "4": ["PR", "RS", "SC"], "5": ["GO", "MT", "MS"]}
REGIAO = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}


def gerar(pasta, n_mun: int = 300, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    pasta.mkdir(parents=True, exist_ok=True)

    regs = rng.choice(list(UFS), n_mun)
    mun = pd.DataFrame({
        "id_municipio": [f"{r}{i:06d}" for i, r in enumerate(regs)],
        "sigla_uf": [rng.choice(UFS[r]) for r in regs],
        "nome_regiao": [REGIAO[r] for r in regs],
    })
    mun["nome_municipio"] = "Municipio " + mun.index.astype(str)
    mun["capital_uf"] = rng.random(n_mun) < 0.03
    mun["amazonia_legal"] = mun["nome_regiao"].eq("Norte")
    mun["regiao_metropolitana"] = rng.random(n_mun) < 0.2
    efeito_reg = mun["nome_regiao"].map({"Norte": -10, "Nordeste": -6, "Sudeste": 4, "Sul": 6, "Centro-Oeste": 2})
    log_pib = rng.normal(4.4, 0.3, n_mun) + efeito_reg / 40
    qualidade = 50 + efeito_reg + 15 * (log_pib - 4.4) + rng.normal(0, 8, n_mun)

    pop = pd.concat([pd.DataFrame({"id_municipio": mun["id_municipio"], "ano": a,
                                   "populacao": (10 ** rng.normal(4.2, 0.6, n_mun)).astype(int)})
                     for a in (2021, 2022, 2023, 2024)])
    pop_2021 = pop[pop["ano"] == 2021].set_index("id_municipio")["populacao"]
    pib = pd.DataFrame({"id_municipio": mun["id_municipio"], "ano": 2021})
    pib["pib"] = 10 ** log_pib * pib["id_municipio"].map(pop_2021).values / 1000
    agro = rng.uniform(0.02, 0.6, n_mun)
    ind = rng.uniform(0.05, 0.4, n_mun)
    pib["va_agropecuaria"] = pib["pib"] * agro
    pib["va_industria"] = pib["pib"] * ind
    pib["va_servicos"] = pib["pib"] * (1 - agro - ind).clip(0.1)
    qualidade = qualidade - 10 * agro

    res = []
    for ano, delta in ((2023, 0), (2024, 3)):
        for rede, ef in (("0", 0), ("2", 3), ("3", -2), ("5", 0)):
            taxa = (qualidade + delta + ef + rng.normal(0, 4, n_mun)).clip(5, 98)
            res.append(pd.DataFrame({"ano": ano, "id_municipio": mun["id_municipio"], "serie": "2",
                                     "rede": rede, "taxa_alfabetizacao": taxa,
                                     "media_portugues": 700 + taxa * 1.2}))
    res = pd.concat(res)
    sem_hist = rng.choice(n_mun, n_mun // 20, replace=False)  # municipios sem dado em 2023
    res = res[~((res["ano"] == 2023) & res["id_municipio"].isin(mun["id_municipio"].iloc[sem_hist]))]

    t23 = res[(res["ano"] == 2023) & (res["rede"] == "3")].set_index("id_municipio")["taxa_alfabetizacao"]
    metas = pd.DataFrame({"ano": 2023, "id_municipio": mun["id_municipio"], "rede": "Municipal"})
    base = metas["id_municipio"].map(t23).fillna(50)
    for i, a in enumerate(range(2024, 2031)):
        metas[f"meta_alfabetizacao_{a}"] = (base + 2.5 * (i + 1)).clip(upper=100)
    metas = metas.melt(id_vars=["id_municipio", "rede", "ano"], var_name="c", value_name="meta_valor")
    metas["ano_meta"] = metas["c"].str[-4:].astype(int)
    metas = metas[["id_municipio", "rede", "ano_meta", "meta_valor"]]

    alunos = []
    q = dict(zip(mun["id_municipio"], qualidade))
    for ano in (2023, 2024):
        for m in mun["id_municipio"]:
            for e in range(4):
                n = 15
                rede = rng.choice(["2", "3", "4"], p=[0.3, 0.6, 0.1])
                ef = {"2": 3, "3": -2, "4": 15}[rede] + rng.normal(0, 6)
                p = 1 / (1 + np.exp(-(q[m] + ef - 50) / 10))
                alf = rng.random(n) < p
                alunos.append(pd.DataFrame({
                    "ano": ano, "id_municipio": m, "id_escola": f"{m}{e}", "id_aluno": range(n),
                    "caderno": rng.integers(1, 20, n).astype(str), "serie": "2", "rede": rede,
                    "presenca": np.where(rng.random(n) < 0.95, "1", "0"), "preenchimento_caderno": "1",
                    "alfabetizado": np.where(alf, "1", "0"),
                    "proficiencia": np.where(alf, 760, 700) + rng.normal(0, 10, n),
                    "peso_aluno": 1.0}))
    alunos = pd.concat(alunos)

    for nome, df in {"municipio": mun, "populacao": pop, "pib": pib, "resultado_municipio": res,
                     "metas_municipio": metas, "alunos": alunos}.items():
        df.to_parquet(pasta / f"{nome}.parquet", index=False)
