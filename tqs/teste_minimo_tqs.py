"""Teste do momento mínimo no TQS (pilares M01 a M09, edifícios TCCM01 a TCCM09, lote de 04/11/2026, 13:29 a 13:33).

Compara, em cada pilar do conjunto complementar (Seção 3.6 do TCC; e1 = 0, nu = 0,4, C30), o que o TQS grava no
relatório (Pilar.rep) e nas listagens (SELEC.LST, LISPIL.LST) com os resultados da rotina
(pilar/saida/resultados_teste_minimo.csv, Tabela do teste do momento mínimo no Capítulo 4):
  - momento mínimo de 1ª ordem nas duas direções (MIDDLE_MSD_IMPx e MIDDLE_MSD_IMPy, caso PERMACID; nos pilares sem
    2ª ordem, o caso não grava TOP_MSD_BDMIMX);
  - envoltória mínima com 2ª ordem gravada pelo TQS (MINIMUMENVELOPEBDMOM[0..7]): semieixos e posição dos 8 pontos;
  - momento da seção intermediária do caso (MIDDLE_MSDX, MIDDLE_MSDY), com M1d,A = 0;
  - armadura adotada pelo TQS, a da rotina pelas duas análises isoladas e o primeiro arranjo que engloba a elipse;
  - com as barras do TQS, o fator eta da envoltória mínima da rotina (verificar_envoltoria_minima), com seção bruta
    e líquida, e o fator nos 8 pontos do próprio TQS.

Também lê a envoltória mínima gravada pelo TQS nos 212 pilares do lote (conferência dos semieixos com a curvatura
aproximada e η nos 8 pontos com as barras do TQS).
Grava teste_minimo_tqs.csv e envoltoria_minima_lote_tqs.csv.  Execução:  python3 teste_minimo_tqs.py
"""
import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "analise"))
import pilares_nbr6118 as rotina  # noqa: E402
import processar_matriz as matriz  # noqa: E402
import comparar_tqs_lote as comparacao  # noqa: E402
from ler_pilar_rep import ler_pilar_rep, pilares_por_nome  # noqa: E402

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — constantes e fórmulas
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [E01] Arquivos.  Usada em: principal
ARQUIVO_ROTINA = Path(__file__).resolve().parent.parent / "saida" / "resultados_teste_minimo.csv"
ARQUIVO_SAIDA = Path(__file__).resolve().parent / "teste_minimo_tqs.csv"
ARQUIVO_ENTRADA = Path(__file__).resolve().parent / "entrada_tqs_lote.csv"
ARQUIVO_COMPARACAO = Path(__file__).resolve().parent / "comparacao_tqs_lote.csv"
ARQUIVO_SAIDA_LOTE = Path(__file__).resolve().parent / "envoltoria_minima_lote_tqs.csv"
PREFIXO_EDIFICIO = "TCCM"

# [E02] Número de pontos da envoltória mínima gravada pelo TQS (MINIMUMENVELOPEBDMOM[0..7]) e índices dos
#   semieixos: [0] na direção de b_w (MSDX) e [2] na de h (MSDY).  Usada em: pontos_envoltoria_tqs, avaliar
PONTOS_ENVOLTORIA_TQS = 8

# [E03] Milímetros por centímetro (a rotina recebe a bitola em cm; o relatório do TQS dá mm).  Usada em:
#   envoltoria_rotina
MM_POR_CM = 10.0
INDICE_SEMIEIXO_B = 0
INDICE_SEMIEIXO_H = 2


# [E04] Processos paralelos (envoltória dos 212 pilares do lote).  Usada em: principal
PROCESSOS = 2


# [E14] Semieixo da envoltória mínima com 2ª ordem pela curvatura aproximada (kN·cm), para conferir o que o TQS
#   grava: a = M1d,mín + N_d ℓe²/10 · 1/r, com 1/r = 0,005 / [b (ν + 0,5)] ≤ 0,005/b ([F07] e [F28] de
#   pilares_nbr6118.py, curvatura_aproximada e momento_total_pilar_padrao, com α_b = 1).
#   Variáveis: forca_normal (N_d, kN), dimensao (b na direção, cm), comprimento (ℓe, cm), area_concreto (cm²), fcd.
#   Usada em: avaliar_lote
def semieixo_curvatura_aproximada(forca_normal: float, dimensao: float, comprimento: float, area_concreto: float,
                                  fcd: float) -> float:
    curvatura = rotina.curvatura_aproximada(rotina.forca_normal_adimensional(forca_normal, area_concreto, fcd),
                                            dimensao)
    return rotina.momento_total_pilar_padrao(1.0, rotina.momento_minimo_primeira_ordem(forca_normal, dimensao),
                                             forca_normal, comprimento, curvatura)


# [E10] Diferença percentual com sinal, da rotina em relação ao TQS (a [C11] de comparar_tqs_lote.py).
#   Variáveis: valor_rotina, valor_tqs.  Fórmula: Δ = 100 · (X_rotina − X_TQS) / X_TQS
#   Usada em: avaliar
def diferenca_percentual(valor_rotina: float, valor_tqs: float) -> float:
    return 100.0 * (valor_rotina - valor_tqs) / valor_tqs


# [E11] Posição de um ponto (M_b, M_h) em relação à elipse de semieixos (a_b, a_h): valor 1 sobre a elipse.
#   Variáveis: momento_b, momento_h, semieixo_b, semieixo_h.  Fórmula: q = (M_b/a_b)² + (M_h/a_h)²
#   Usada em: avaliar (conferência de que os 8 pontos do TQS estão sobre a elipse da Figura 15.2)
def posicao_na_elipse(momento_b: float, momento_h: float, semieixo_b: float, semieixo_h: float) -> float:
    return (momento_b / semieixo_b) ** 2 + (momento_h / semieixo_h) ** 2


# [E12] Ângulo paramétrico do ponto na elipse (graus): θ = atan2(M_h/a_h, M_b/a_b).
#   Variáveis: momento_b, momento_h, semieixo_b, semieixo_h.  Usada em: avaliar
def angulo_parametrico(momento_b: float, momento_h: float, semieixo_b: float, semieixo_h: float) -> float:
    return math.degrees(math.atan2(momento_h / semieixo_h, momento_b / semieixo_b))


# [E13] Fator da envoltória resistente da rotina num ponto (M_b, M_h) com N = N_d fixo: η = r_R(ψ) / r_S, com
#   r_S = √(M_b² + M_h²) e r_R pelo contorno resistente da rotina ([F71] de pilares_nbr6118.py).
#   Variáveis: raio_resistente, momento_b, momento_h.  Usada em: fatores_pontos_tqs
def fator_ponto(raio_resistente: float, momento_b: float, momento_h: float) -> float:
    return raio_resistente / math.hypot(momento_b, momento_h)


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


def pontos_envoltoria_tqs(trecho: dict) -> list[tuple[float, float]]:
    """Os 8 pontos (M_b, M_h) da envoltória mínima do TQS, em kN·cm."""
    return [(comparacao.newton_metro_para_kn_cm(trecho["MINIMUMENVELOPEBDMOM[%d]" % indice]["MSDX"]),
             comparacao.newton_metro_para_kn_cm(trecho["MINIMUMENVELOPEBDMOM[%d]" % indice]["MSDY"]))
            for indice in range(PONTOS_ENVOLTORIA_TQS)]


def caso_permanente_variavel(trecho: dict) -> dict:
    return next(caso for chave, caso in trecho.items() if chave.startswith("INTERNALFORCES_CASE[")
                and caso.get("CASE_NAME") == comparacao.CASO_PERMANENTE_VARIAVEL)


def fatores_pontos_tqs(caso_matriz, material, x, y, area, pontos) -> list[float]:
    """η da rotina (contorno resistente com as barras do TQS, N = Nd) nos 8 pontos do TQS."""
    pilar = caso_matriz.pilar
    momentos_resistentes = {direcao: rotina.momento_resistente(
        rotina.secao_na_direcao(pilar.largura_bw, pilar.altura_h, x, y, direcao), material, area,
        pilar.forca_normal_calculo) for direcao in rotina.ORDEM_DIRECOES}
    momentos_b, momentos_h = rotina.envoltoria_resistente(pilar.largura_bw, pilar.altura_h, x, y, area, material,
                                                          pilar.forca_normal_calculo,
                                                          momentos_resistentes[rotina.DIRECAO_B],
                                                          momentos_resistentes[rotina.DIRECAO_H])
    direcoes_contorno = rotina.direcao_momento(momentos_b, momentos_h)
    fatores = []
    for momento_b, momento_h in pontos:
        direcao = float(rotina.direcao_momento(abs(momento_b), abs(momento_h)))
        segmentos, seguintes = rotina.segmento_do_contorno(direcoes_contorno, np.array([direcao]))
        raio = float(rotina.raio_contorno(momentos_b[segmentos], momentos_h[segmentos], momentos_b[seguintes],
                                          momentos_h[seguintes], np.array([direcao]))[0])
        fatores.append(fator_ponto(raio, momento_b, momento_h))
    return fatores


def envoltoria_rotina(caso_matriz, material, x, y, area, diametro):
    pilar = caso_matriz.pilar
    forca = pilar.forca_normal_calculo
    comprimento = pilar.comprimento_equivalente
    solicitacoes = {rotina.DIRECAO_B: rotina.solicitacao_direcao(pilar.largura_bw, pilar.altura_h, material, forca,
                                                                 comprimento, rotina.DIRECAO_B,
                                                                 rotina.EXCENTRICIDADE_ENVOLTORIA_MINIMA),
                    rotina.DIRECAO_H: rotina.solicitacao_direcao(pilar.altura_h, pilar.largura_bw, material, forca,
                                                                 comprimento, rotina.DIRECAO_H,
                                                                 rotina.EXCENTRICIDADE_ENVOLTORIA_MINIMA)}
    diametro_cm = diametro / MM_POR_CM
    candidato = rotina.Candidato(diametro_cm, rotina.diametro_estribo(diametro_cm), 0.0, 0, x, y, x.size, area)
    momentos = {direcao: rotina.momento_resistente(rotina.secao_na_direcao(pilar.largura_bw, pilar.altura_h, x, y,
                                                                           direcao), material, area, forca)
                for direcao in rotina.ORDEM_DIRECOES}
    return rotina.verificar_envoltoria_minima(pilar, material, candidato, solicitacoes, momentos)


def avaliar(caso_matriz, linha_rotina: dict, entrada: dict) -> dict:
    identificador = caso_matriz.identificador
    edificio = PREFIXO_EDIFICIO + identificador[1:]
    pasta = comparacao.PASTA_RESULTADOS_LOTE / edificio / "PILAR"
    if entrada["edificio"] != edificio:
        raise ValueError("%s: edifício da entrada %s, esperado %s" % (identificador, entrada["edificio"], edificio))
    tep_confere, tep_problemas = comparacao.conferir_tep(comparacao.ler_tep(pasta / "PRJ-0001.TEP")[identificador],
                                                         entrada)
    pe_direito_duplo = comparacao.ler_peddup(pasta / "PEDDUP.DAT")[identificador]
    selec = comparacao.ler_selec(pasta / "SELEC.LST")[identificador]
    lispil = comparacao.ler_lispil(pasta / "LISPIL.LST")[identificador]
    trecho = pilares_por_nome(ler_pilar_rep(pasta / "Pilar.rep"))[identificador]["SPAN[1]"]
    caso = caso_permanente_variavel(trecho)
    pontos = pontos_envoltoria_tqs(trecho)
    semieixo_b = abs(pontos[INDICE_SEMIEIXO_B][0])
    semieixo_h = abs(pontos[INDICE_SEMIEIXO_H][1])
    posicoes = [posicao_na_elipse(momento_b, momento_h, semieixo_b, semieixo_h) for momento_b, momento_h in pontos]
    angulos = [angulo_parametrico(momento_b, momento_h, semieixo_b, semieixo_h) for momento_b, momento_h in pontos]
    x, y, diametro, area = comparacao.barras_tqs(trecho)
    material = rotina.Material(caso_matriz.fck_mpa, rotina.RESISTENCIA_ACO_MPA)
    resultado = {"pilar": identificador, "edificio": edificio, "TEP confere": "sim" if tep_confere else "não",
                 "TEP problemas": tep_problemas, "pé direito duplo (x,y)": pe_direito_duplo,
                 "b_w (cm)": caso_matriz.largura_bw,
                 "h (cm)": caso_matriz.altura_h, "λ b_w": caso_matriz.esbeltez,
                 "λ TQS (SELEC)": selec["lambda"], "método TQS": selec["metodo"],
                 "método rotina b_w": linha_rotina["Método b_w"], "método rotina h": linha_rotina["Método h"],
                 "Nd rotina (kN)": float(linha_rotina["Nd (kN)"]),
                 "Nd TQS (kN)": comparacao.newton_para_kn(caso["TOP_NSD"]),
                 "M1d,mín b_w rotina (kN·m)": float(linha_rotina["M1d,mín b_w (kN·m)"]),
                 "M1d,mín b_w TQS (kN·m)": comparacao.newton_metro_para_kn_cm(caso["MIDDLE_MSD_IMPx"]) / 100.0,
                 "M1d,mín h rotina (kN·m)": float(linha_rotina["M1d,mín h (kN·m)"]),
                 "M1d,mín h TQS (kN·m)": comparacao.newton_metro_para_kn_cm(caso["MIDDLE_MSD_IMPy"]) / 100.0,
                 "Md,tot,mín b_w rotina (kN·m)": float(linha_rotina["Md,tot,mín b_w (kN·m)"]),
                 "semieixo b_w TQS (kN·m)": semieixo_b / 100.0,
                 "Md,tot,mín h rotina (kN·m)": float(linha_rotina["Md,tot,mín h (kN·m)"]),
                 "semieixo h TQS (kN·m)": semieixo_h / 100.0,
                 "M meio b_w TQS (kN·m)": comparacao.newton_metro_para_kn_cm(caso["MIDDLE_MSDX"]) / 100.0
                 if "MIDDLE_MSDX" in caso else math.nan,
                 "M meio h TQS (kN·m)": comparacao.newton_metro_para_kn_cm(caso["MIDDLE_MSDY"]) / 100.0
                 if "MIDDLE_MSDY" in caso else math.nan,
                 "pontos na elipse: q mín": min(posicoes), "pontos na elipse: q máx": max(posicoes),
                 "ângulos paramétricos (°)": " ".join("%.1f" % angulo for angulo in angulos),
                 "barras TQS": "%d φ %.1f" % (selec["numero_barras"], selec["diametro_mm"]),
                 "As,adot TQS (cm²)": area,
                 "ASnec TQS arranjo adotado (cm²)": lispil["adotado"]["area_necessaria"],
                 "barras rotina (isoladas)": linha_rotina["Barras"],
                 "As,adot rotina (cm²)": float(linha_rotina["As,adot (cm²)"]),
                 "As,calc rotina (cm²)": float(linha_rotina["As,calc (cm²)"]),
                 "barras rotina (elipse)": linha_rotina["Barras elipse"],
                 "As,adot elipse rotina (cm²)": float(linha_rotina["As,adot elipse (cm²)"]),
                 "η rotina (barras próprias)": float(linha_rotina["η envoltória mínima"])}
    for descontar, rotulo in ((False, "bruta"), (True, "líquida")):
        rotina.DESCONTAR_AREA_BARRAS = descontar
        envoltoria = envoltoria_rotina(caso_matriz, material, x, y, area, diametro)
        resultado["η rotina barras TQS %s" % rotulo] = envoltoria.fator
        resultado["ψ crítico barras TQS %s (°)" % rotulo] = envoltoria.direcao_critica
        resultado["Md,tot,mín b_w rotina barras TQS %s (kN·m)" % rotulo] = rotina.kn_cm_para_kn_m(
            envoltoria.direcao_b.momento_total)
        fatores = fatores_pontos_tqs(caso_matriz, material, x, y, area, pontos)
        resultado["η nos 8 pontos do TQS %s: mín" % rotulo] = min(fatores)
    rotina.DESCONTAR_AREA_BARRAS = True
    resultado["dif. M1d,mín b_w (%)"] = diferenca_percentual(resultado["M1d,mín b_w rotina (kN·m)"],
                                                             resultado["M1d,mín b_w TQS (kN·m)"])
    resultado["dif. M1d,mín h (%)"] = diferenca_percentual(resultado["M1d,mín h rotina (kN·m)"],
                                                           resultado["M1d,mín h TQS (kN·m)"])
    resultado["dif. semieixo b_w (%)"] = diferenca_percentual(resultado["Md,tot,mín b_w rotina (kN·m)"],
                                                              resultado["semieixo b_w TQS (kN·m)"])
    resultado["dif. semieixo h (%)"] = diferenca_percentual(resultado["Md,tot,mín h rotina (kN·m)"],
                                                            resultado["semieixo h TQS (kN·m)"])
    resultado["dif. As,adot isoladas (%)"] = diferenca_percentual(resultado["As,adot rotina (cm²)"], area)
    resultado["dif. As,adot elipse (%)"] = diferenca_percentual(resultado["As,adot elipse rotina (cm²)"], area)
    return resultado


def avaliar_lote(dados: dict) -> dict:
    """Envoltória mínima gravada pelo TQS num pilar do lote: semieixos, conferência com a curvatura aproximada e η nos
    8 pontos com as barras do TQS (seção bruta, contorno resistente da rotina a N = N_d)."""
    caso_matriz = dados["caso"]
    trecho = dados["trecho"]
    pilar = caso_matriz.pilar
    material = rotina.Material(caso_matriz.fck_mpa, rotina.RESISTENCIA_ACO_MPA)
    forca_tqs = comparacao.newton_para_kn(caso_permanente_variavel(trecho)["TOP_NSD"])
    area_concreto = rotina.area_secao_retangular(pilar.largura_bw, pilar.altura_h)
    pontos = pontos_envoltoria_tqs(trecho)
    semieixo_b = abs(pontos[INDICE_SEMIEIXO_B][0])
    semieixo_h = abs(pontos[INDICE_SEMIEIXO_H][1])
    aproximado_b = semieixo_curvatura_aproximada(forca_tqs, pilar.largura_bw, pilar.comprimento_equivalente,
                                                 area_concreto, material.fcd)
    aproximado_h = semieixo_curvatura_aproximada(forca_tqs, pilar.altura_h, pilar.comprimento_equivalente,
                                                 area_concreto, material.fcd)
    x, y, _, area = comparacao.barras_tqs(trecho)
    rotina.DESCONTAR_AREA_BARRAS = False
    fatores = fatores_pontos_tqs(caso_matriz, material, x, y, area, pontos)
    rotina.DESCONTAR_AREA_BARRAS = True
    return {"pilar": caso_matriz.identificador, "edificio": dados["edificio"], "método TQS": dados["metodo"],
            "λ b_w": caso_matriz.esbeltez, "ν": caso_matriz.nu, "semieixo b_w TQS (kN·m)": semieixo_b / 100.0,
            "semieixo h TQS (kN·m)": semieixo_h / 100.0,
            "dif. semieixo b_w × curvatura aproximada (%)": diferenca_percentual(semieixo_b, aproximado_b),
            "dif. semieixo h × curvatura aproximada (%)": diferenca_percentual(semieixo_h, aproximado_h),
            "η nos 8 pontos do TQS bruta: mín": min(fatores)}


def principal_lote() -> None:
    casos = {caso.identificador: caso for caso in matriz.montar_matriz()}
    with ARQUIVO_COMPARACAO.open(encoding="utf-8-sig") as arquivo:
        linhas = list(csv.DictReader(arquivo, delimiter=";"))
    repositorios = {}
    tarefas = []
    for linha in linhas:
        edificio = linha["edificio"]
        if edificio not in repositorios:
            repositorios[edificio] = pilares_por_nome(ler_pilar_rep(comparacao.PASTA_RESULTADOS_LOTE / edificio
                                                                    / "PILAR" / "Pilar.rep"))
        tarefas.append({"caso": casos[linha["id"]], "trecho": repositorios[edificio][linha["id"]]["SPAN[1]"],
                        "edificio": edificio, "metodo": linha["método TQS"]})
    with ProcessPoolExecutor(max_workers=PROCESSOS) as executor:
        resultados = list(executor.map(avaliar_lote, tarefas))
    with ARQUIVO_SAIDA_LOTE.open("w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(resultados[0]), delimiter=";")
        escritor.writeheader()
        for resultado in resultados:
            escritor.writerow({chave: ("%.4f" % valor if isinstance(valor, float) else valor)
                               for chave, valor in resultado.items()})
    print("%d pilares do lote gravados em %s" % (len(resultados), ARQUIVO_SAIDA_LOTE))


def principal() -> None:
    with ARQUIVO_ROTINA.open(encoding="utf-8-sig") as arquivo:
        linhas_rotina = {linha["ID"]: linha for linha in csv.DictReader(arquivo, delimiter=";")}
    with ARQUIVO_ENTRADA.open(encoding="utf-8") as arquivo:
        entradas = {linha["id"]: linha for linha in csv.DictReader(arquivo, delimiter=";")}
    resultados = [avaliar(caso, linhas_rotina[caso.identificador], entradas[caso.identificador])
                  for caso in matriz.montar_teste_minimo()]
    with ARQUIVO_SAIDA.open("w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(resultados[0]), delimiter=";")
        escritor.writeheader()
        for resultado in resultados:
            escritor.writerow({chave: ("%.4f" % valor if isinstance(valor, float) else valor)
                               for chave, valor in resultado.items()})
    print("%d pilares gravados em %s" % (len(resultados), ARQUIVO_SAIDA))


if __name__ == "__main__":
    principal()
    principal_lote()
