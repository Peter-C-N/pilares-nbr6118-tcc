"""Réplica do cálculo do TQS nos pilares com 90 < λ ≤ 140 (pilar-padrão acoplado a diagramas N, M, 1/r) e
decomposição das causas da diferença para a rotina, nos 9 pilares em que o TQS adotou menos armadura que a rotina
exige com as mesmas barras (camada2_nm1r_tqs.csv) e no P165.

Dados do TQS lidos nas telas do visualizador de efeitos locais (combinação 10, PP+PERM+ACID), capturas de
04/11/2026 (P165 às 11:37; demais às 13:06 a 13:12): rigidez secante EI das duas direções, coeficiente de correção
da fluência (dica "Fluência", combinação 11) e momentos do diagrama da envoltória no topo e no meio do lance.

Cálculo do TQS decifrado no P165 (registro calibracao_tqs_P165_2026-11-04):
  e_cc = e1 · [exp(φ N_sg / (N_e − N_sg)) − 1], sem e_a, com N_sg no topo do lance (sem o peso próprio) e
         e1 = M_1d / N_d (conferido nos 9 pilares: o e_cc implícito no topo do diagrama difere de e1 · c em
         menos de 0,2 %, contra 4 a 9 % com M_sg / N_sg no topo; nos pilares da matriz, M_sg/N_sg = e1);
  M_1c = M_1d + N_d · e_cc  (momento do topo e da base no diagrama);
  M_tot = M_1c + M_1d · [1 / (1 − N_d ℓe² / (10 EI_sec)) − 1]  (e_cc não amplificado; EI_sec constante).

Variantes da decomposição, cada uma somando uma hipótese da rotina à anterior, com as barras do TQS (seção bruta):
  A  réplica do TQS;
  B  A com e_a na fórmula da fluência (NBR 6118:2023, 15.8.4);
  C  B com N_sg e M_sg da matriz (N_gk + ψ2 N_qk, com o peso próprio do lance, como na rotina);
  D  C com e_cc somado a e1 antes da 2ª ordem (amplificado, 15.8.4: "como se fosse um efeito imediato");
  E  rotina completa: D com o equilíbrio na curva M, N, 1/r em vez da rigidez secante constante;
  A' (isolado) A só com e_cc amplificado, para medir esse efeito sozinho.
Em cada uma, a relação Sd/Rd na direção de b_w (raio da origem em (N, M), seção bruta) mostra em que passo a
armadura do TQS deixa de atender.

Execução:  python3 replica_tqs_nm1r.py
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "analise"))
import pilares_nbr6118 as rotina  # noqa: E402
import comparar_tqs_lote as comparacao  # noqa: E402
from ler_pilar_rep import ler_pilar_rep, pilares_por_nome  # noqa: E402

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — dados, constantes e fórmulas
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [D01] Arquivos.  Usada em: principal
PASTA_TQS = Path(__file__).resolve().parent
ARQUIVO_ENTRADA_TQS = PASTA_TQS / "entrada_tqs_lote.csv"
ARQUIVO_MATRIZ = Path(__file__).resolve().parent.parent / "saida" / "resultados_matriz.csv"
ARQUIVO_SAIDA = PASTA_TQS / "replica_tqs_nm1r.csv"

# [D02] Dados do visualizador do TQS (combinação 10), por pilar: EI_sec na direção de b_w (kN·m², "Direção z" do
#   TQS, a de λ > 90), coeficiente de correção da fluência nessa direção, momentos do diagrama no topo e no meio
#   (kN·m, lidos com a precisão visível na tela) e Sd/Rd do meio. Fonte: capturas de tela de 04/11/2026 (arquivo).
#   Usada em: principal
TELAS_TQS = {
    "P165": dict(ei=16142.13, coeficiente=0.74, topo=100.681, meio=251.584, sdrd=0.95,
                 captura="11:37 (registro do P165)", momento_base=57.862),
    "P021": dict(ei=2317.68, coeficiente=0.65, topo=21.282560825, meio=59.034032822, sdrd=0.90, captura="130606"),
    "P098": dict(ei=2495.89, coeficiente=0.47, topo=26.615633965, meio=64.456982613, sdrd=0.99, captura="130745"),
    "P067": dict(ei=2020.40, coeficiente=0.51, topo=11.838902235, meio=50.414714813, sdrd=0.98, captura="130823"),
    "P068": dict(ei=2294.89, coeficiente=0.51, topo=23.254985809, meio=65.250358582, sdrd=0.98, captura="130928"),
    "P237": dict(ei=53352.83, coeficiente=0.61, topo=221.1317, meio=527.98, sdrd=0.86, captura="131010"),
    "P238": dict(ei=67268.95, coeficiente=0.61, topo=442.158, meio=774.71, sdrd=0.92, captura="131050"),
    "P273": dict(ei=64735.25, coeficiente=0.72, topo=math.nan, meio=638.52, sdrd=0.84, captura="131128"),
    "P309": dict(ei=76601.79, coeficiente=0.83, topo=math.nan, meio=747.35, sdrd=0.84, captura="131159"),
}

# [D04] Nota do P165: as telas são do processamento manual das 11:13, com M_1d = 57,862 kN·m (resíduo de 0,001 tf·m
#   no momento lançado); no lote, M_1d = 57,82 kN·m. Para conferir a réplica com a tela, usa-se o M_1d da tela
#   (chave "momento_base"); nos demais pilares, o do Pilar.rep do lote.  Usada em: principal

# [D03] Coeficientes: fluência φ = 2,5 e ψ2 = 0,3 (critérios do TQS e da rotina); γf = 1,4; unidades.
#   Usada em: excentricidade_fluencia_tqs; principal
COEFICIENTE_FLUENCIA = 2.5
PSI_2 = 0.3
GAMA_F = 1.4
KN_POR_TF = 10.0
CM_POR_M = 100.0
FYK_MPA = 500.0


# [R01] Módulo de elasticidade inicial (kN/m²).  Variáveis: fck (MPa).  Fórmula: E_ci = 5600 √fck · 1000
#   (NBR 6118:2023, 8.2.8, α_E = 1).  Usada em: forca_euler
def modulo_inicial_kn_m2(fck_mpa: float) -> float:
    return 5600.0 * math.sqrt(fck_mpa) * 1000.0


# [R02] Força de Euler para a fluência (kN).  Variáveis: modulo (kN/m²), inercia (m⁴), comprimento (m).
#   Fórmula: N_e = 10 E_ci I_c / ℓe²  (15.8.4).  Usada em: principal
def forca_euler(modulo: float, inercia: float, comprimento: float) -> float:
    return 10.0 * modulo * inercia / comprimento ** 2


# [R03] Coeficiente de correção da fluência.  Variáveis: forca_quase_permanente (N_sg), forca_euler_valor (N_e).
#   Fórmula: c = exp(φ N_sg / (N_e − N_sg)) − 1  (15.8.4).  Usada em: principal
def coeficiente_fluencia(forca_quase_permanente: float, forca_euler_valor: float) -> float:
    return math.exp(COEFICIENTE_FLUENCIA * forca_quase_permanente
                    / (forca_euler_valor - forca_quase_permanente)) - 1.0


# [R04] Excentricidade de fluência (m).  Variáveis: momento_qp (M_sg, kN·m), forca_qp (N_sg, kN), e_a (m),
#   coeficiente.  Fórmula: e_cc = (M_sg / N_sg + e_a) · c  (15.8.4; o TQS usa e_a = 0).  Usada em: principal
def excentricidade_fluencia(momento_qp: float, forca_qp: float, excentricidade_acidental: float,
                            coeficiente: float) -> float:
    return (momento_qp / forca_qp + excentricidade_acidental) * coeficiente


# [R05] Fator de amplificação com rigidez secante constante.  Variáveis: forca_normal (N_d, kN), comprimento (m),
#   rigidez (EI, kN·m²).  Fórmula: f = 1 / (1 − N_d ℓe² / (10 EI))  (pilar-padrão, curvatura senoidal).
#   Usada em: momento_total_secante
def fator_amplificacao(forca_normal: float, comprimento: float, rigidez: float) -> float:
    return 1.0 / (1.0 - forca_normal * comprimento ** 2 / (10.0 * rigidez))


# [R06] Momento total com rigidez secante. Variáveis: momento_base (M_1d), momento_fluencia (N_d e_cc), fator,
#   amplifica_fluencia. Fórmulas: TQS M_tot = M_1d + N_d e_cc + M_1d (f − 1); com e_cc amplificado,
#   M_tot = (M_1d + N_d e_cc) f.  Usada em: principal (variantes A a D)
def momento_total_secante(momento_base: float, momento_fluencia: float, fator: float,
                          amplifica_fluencia: bool) -> float:
    if amplifica_fluencia:
        return (momento_base + momento_fluencia) * fator
    return momento_base + momento_fluencia + momento_base * (fator - 1.0)


# [R08] Rigidez secante da curva M, N, 1/r da rotina (N_d/γf3, 1,1 f_cd; [F30] de pilares_nbr6118.py) até o ponto B,
#   para comparar com o EI_sec da tela do TQS: EI = M_B / (1/r)_B, com M_B = M_Rd/γf3 (forma da Seção 15.3.1 e de
#   Kimura, usada em validacao.py) ou M_B = M_Rd (alternativa), e (1/r)_B a curvatura da curva em M_B.
#   Variáveis: momento_ponto (M_B, kN·cm), curvatura_ponto ((1/r)_B, 1/cm).  Fórmula: EI = M_B / (1/r)_B (kN·cm²)
#   Usada em: rigidez_rotina
def rigidez_secante_ponto(momento_ponto: float, curvatura_ponto: float) -> float:
    return momento_ponto / curvatura_ponto


# [R09] Conversão de rigidez: kN·cm² -> kN·m².  Variáveis: rigidez (kN·cm²).  Fórmula: EI / 100²
#   Usada em: principal
def kn_cm2_para_kn_m2(rigidez: float) -> float:
    return rigidez / CM_POR_M ** 2


# [R07] Diferença percentual da réplica em relação ao valor da tela.  Variáveis: valor, referencia.
#   Fórmula: Δ = 100 (X − X_tela) / X_tela.  Usada em: principal
def diferenca_percentual(valor: float, referencia: float) -> float:
    return 100.0 * (valor - referencia) / referencia


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


def sdrd_direcao_b(largura, altura, x, y, area, material, forca_normal_kn, momento_kn_m) -> float:
    """Sd/Rd pelo raio da origem em (N, M_b), seção bruta, com as barras do TQS."""
    rotina.DESCONTAR_AREA_BARRAS = False
    valor = comparacao.relacao_sd_rd(largura, altura, x, y, area, material, forca_normal_kn,
                                     momento_kn_m * CM_POR_M, 1.0e-9)
    rotina.DESCONTAR_AREA_BARRAS = True
    return valor


def rigidez_rotina(secao, material, area, forca_normal_kn, momento_ponto_kn_cm) -> float:
    """EI secante da curva de deformação da rotina (seção bruta) até o ponto B de momento dado ([R08]); nan se a curva
    não chega a esse momento."""
    rotina.DESCONTAR_AREA_BARRAS = False
    forca_deformacao = rotina.forca_normal_curva_deformacao(forca_normal_kn)
    limite = rotina.curvatura_ultima(secao, material, area, forca_deformacao, True)
    curvaturas, momentos = rotina.curva_momento_curvatura(secao, material, area, forca_deformacao, True)
    rotina.DESCONTAR_AREA_BARRAS = True
    if momentos.max() < momento_ponto_kn_cm:
        return math.nan
    rotina.DESCONTAR_AREA_BARRAS = False
    curvatura_b = brentq(lambda curvatura: rotina.momento_curva_deformacao(secao, material, area, forca_deformacao,
                                                                           curvatura) - momento_ponto_kn_cm,
                         rotina.CURVATURA_NULA, float(curvaturas[int(np.argmax(momentos))]), xtol=1e-14)
    rotina.DESCONTAR_AREA_BARRAS = True
    return rigidez_secante_ponto(momento_ponto_kn_cm, curvatura_b)


def principal() -> None:
    with open(ARQUIVO_ENTRADA_TQS, encoding="utf-8") as arquivo:
        entrada = {linha["id"]: linha for linha in csv.DictReader(arquivo, delimiter=";")
                   if linha["edificio"].startswith("TCCL")}
    with open(ARQUIVO_MATRIZ, encoding="utf-8-sig") as arquivo:
        matriz = {linha["ID"]: linha for linha in csv.DictReader(arquivo, delimiter=";")}
    repositorios = {}
    linhas_saida = []
    for identificador, tela in TELAS_TQS.items():
        dados = entrada[identificador]
        rotina_linha = matriz[identificador]
        edificio = dados["edificio"]
        if edificio not in repositorios:
            repositorios[edificio] = pilares_por_nome(ler_pilar_rep(comparacao.PASTA_RESULTADOS_LOTE / edificio
                                                                    / "PILAR" / "Pilar.rep"))
        trecho = repositorios[edificio][identificador]["SPAN[1]"]
        caso = next(item for item in comparacao.casos_distintos(trecho)
                    if item["CASE_NAME"] == comparacao.CASO_PERMANENTE_VARIAVEL)
        largura = float(dados["b_w_cm"])
        altura = float(dados["h_cm"])
        fck = float(rotina_linha["fck (MPa)"])
        comprimento = float(dados["pe_direito_m"])
        forca_normal = comparacao.newton_para_kn(caso["TOP_NSD"])
        momento_base = tela.get("momento_base", abs(caso["MIDDLE_M1DAx"]) / 1000.0)
        excentricidade_primeira_ordem = momento_base / forca_normal
        # força quase permanente no topo do lance (TQS): cargas lançadas, arredondadas a 0,01 tf
        forca_qp_tqs = KN_POR_TF * (comparacao.arredondar_centesimo(float(dados["fz_permanente_tf"]))
                                    + PSI_2 * comparacao.arredondar_centesimo(float(dados["fz_acidental_tf"])))
        # quase permanente da matriz (rotina): N_gk + ψ2 N_qk com o peso próprio, e M_sg com as mesmas parcelas
        forca_qp_matriz = float(rotina_linha["N_sg b_w (kN)"])
        momento_qp_matriz = float(rotina_linha["M_sg b_w (kN·m)"])
        excentricidade_acidental = float(rotina_linha["e_a b_w (cm)"]) / CM_POR_M
        inercia = (altura / CM_POR_M) * (largura / CM_POR_M) ** 3 / 12.0
        forca_euler_valor = forca_euler(modulo_inicial_kn_m2(fck), inercia, comprimento)
        coeficiente_tqs = coeficiente_fluencia(forca_qp_tqs, forca_euler_valor)
        coeficiente_matriz = coeficiente_fluencia(forca_qp_matriz, forca_euler_valor)
        fator = fator_amplificacao(forca_normal, comprimento, tela["ei"])
        excentricidades = {
            "A": excentricidade_fluencia(excentricidade_primeira_ordem * forca_qp_tqs, forca_qp_tqs, 0.0,
                                         coeficiente_tqs),
            "B": excentricidade_fluencia(excentricidade_primeira_ordem * forca_qp_tqs, forca_qp_tqs,
                                         excentricidade_acidental, coeficiente_tqs),
            "C": excentricidade_fluencia(momento_qp_matriz, forca_qp_matriz, excentricidade_acidental,
                                         coeficiente_matriz),
        }
        excentricidades["D"] = excentricidades["C"]
        excentricidades["A'"] = excentricidades["A"]
        momentos = {chave: momento_total_secante(momento_base, forca_normal * excentricidades[chave], fator,
                                                  chave in ("D", "A'")) for chave in ("A", "B", "C", "D", "A'")}
        x, y, _, area = comparacao.barras_tqs(trecho)
        material = rotina.Material(fck, FYK_MPA)
        secao = rotina.secao_na_direcao(largura, altura, x, y, rotina.DIRECAO_B)
        rotina.DESCONTAR_AREA_BARRAS = False
        equilibrio = rotina.equilibrio_pilar_padrao_acoplado(secao, material, area, forca_normal,
                                                             CM_POR_M * (momento_base + forca_normal
                                                                         * excentricidades["C"]),
                                                             1.0, comprimento * CM_POR_M)
        rotina.DESCONTAR_AREA_BARRAS = True
        momento_e = math.nan if equilibrio is None else rotina.kn_cm_para_kn_m(equilibrio[1])
        momento_topo_a = momento_base + forca_normal * excentricidades["A"]
        saida = {"id": identificador, "edificio": edificio, "seção": "%sx%s" % (dados["b_w_cm"], dados["h_cm"]),
                 "C": "C%d" % round(fck), "λ": rotina_linha["λ (dir. b_w)"], "ν": rotina_linha["ν"],
                 "e1 (cm)": rotina_linha["e1 (cm)"], "barras TQS": "%d φ %.1f" % (len(x), 1000 * trecho["FI_AS"]),
                 "Nd (kN)": "%.3f" % forca_normal, "M1d (kN·m)": "%.4f" % momento_base,
                 "N_sg TQS (kN)": "%.3f" % forca_qp_tqs, "N_sg matriz (kN)": "%.3f" % forca_qp_matriz,
                 "coef. réplica": "%.4f" % coeficiente_tqs, "coef. tela": "%.2f" % tela["coeficiente"],
                 "M topo réplica (kN·m)": "%.4f" % momento_topo_a, "M topo tela (kN·m)": "%.4f" % tela["topo"],
                 "dif. topo (%)": "" if math.isnan(tela["topo"])
                 else "%.3f" % diferenca_percentual(momento_topo_a, tela["topo"]),
                 "M meio réplica A (kN·m)": "%.4f" % momentos["A"], "M meio tela (kN·m)": "%.4f" % tela["meio"],
                 "dif. meio (%)": "%.3f" % diferenca_percentual(momentos["A"], tela["meio"]),
                 "Sd/Rd tela": "%.2f" % tela["sdrd"]}
        for chave in ("A", "B", "C", "D", "A'"):
            saida["M_tot %s (kN·m)" % chave] = "%.3f" % momentos[chave]
            saida["Sd/Rd %s" % chave] = "%.4f" % sdrd_direcao_b(largura, altura, x, y, area, material, forca_normal,
                                                                momentos[chave])
        rotina.DESCONTAR_AREA_BARRAS = False
        momento_resistente = rotina.momento_resistente(secao, material, area, forca_normal)
        rotina.DESCONTAR_AREA_BARRAS = True
        saida["EI_sec tela (kN·m²)"] = "%.2f" % tela["ei"]
        for rotulo, momento_ponto in (("MRd/γf3", momento_resistente / rotina.GAMA_F3), ("MRd", momento_resistente)):
            rigidez = kn_cm2_para_kn_m2(rigidez_rotina(secao, material, area, forca_normal, momento_ponto))
            saida["EI rotina B = %s (kN·m²)" % rotulo] = "%.2f" % rigidez
            saida["dif. EI B = %s (%%)" % rotulo] = "" if math.isnan(rigidez) else "%.2f" % diferenca_percentual(
                rigidez, tela["ei"])
        saida["M_tot E rotina (kN·m)"] = "%.3f" % momento_e
        saida["Sd/Rd E"] = "sem equilíbrio" if math.isnan(momento_e) else "%.4f" % sdrd_direcao_b(
            largura, altura, x, y, area, material, forca_normal, momento_e)
        linhas_saida.append(saida)
        print(identificador, {chave: saida[chave] for chave in ("coef. réplica", "coef. tela", "dif. topo (%)",
                                                                 "dif. meio (%)", "Sd/Rd A", "Sd/Rd tela",
                                                                 "Sd/Rd B", "Sd/Rd C", "Sd/Rd D", "Sd/Rd E",
                                                                 "Sd/Rd A'")})
    with open(ARQUIVO_SAIDA, "w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas_saida[0]), delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas_saida)
    print("gravado em %s" % ARQUIVO_SAIDA)


if __name__ == "__main__":
    principal()
