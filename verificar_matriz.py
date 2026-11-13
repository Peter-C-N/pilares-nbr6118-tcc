# -*- coding: utf-8 -*-
"""
verificar_matriz.py
===================
Confere, pilar por pilar, se a matriz gerada por processar_matriz.py e os resultados da rotina seguem
TODAS as orientações recebidas para a matriz de pilares-teste:

  PP    — Projeto de Pesquisa assinado em 01/07/2026 (Seções 1.4.2, 3.5, 3.6, 3.7, 3.9; Tabela 3.1; Figura 3.1);
  MSF   — análises do orientador, Prof. Milton Soares Filho (PIT de 28/05/2026 e PP de 22/06/2026);
  ARD   — anotações do coorientador, Prof. Alberto Rodrigues Dalmaso, no PP (notas 5 a 8);
  NBR   — ABNT NBR 6118:2023 (itens citados em cada requisito).

(As anotações do Prof. Claudio Cruz Nunes — "Revisão CCN" — tratam do texto e da validação, não da matriz;
 constam da lista de pendências do texto.)

A conferência tem duas etapas:
  1) entradas (conferir_entradas): composição da matriz e entradas do Módulo 1, sem depender de resultados;
     é executada por processar_matriz.py ANTES do processamento, que é interrompido se algum requisito falhar;
  2) resultados (conferir_resultados): procedimento normativo, armadura, arranjo e planilha, lidos dos
     arquivos gravados por processar_matriz.py.
Este programa refaz as duas etapas e grava o relatório completo.

Saídas: saida/verificacao_orientacoes.csv e a aba "Conferencia_orientacoes" em saida/resultados_matriz.xlsx.
Uso:  python3 verificar_matriz.py [pasta_de_saida]      (requer processar_matriz.py já executado)
"""
from __future__ import annotations

import csv
import dataclasses
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

import pilares_nbr6118 as nucleo
import processar_matriz as matriz

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — INÍCIO
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [N01] Tolerância numérica das comparações de valores recalculados (relativa).
#   Variáveis: TOLERANCIA_RELATIVA.
#   Usada em: iguais (l. 85)
TOLERANCIA_RELATIVA = 1e-9

# [R06] Etapas da conferência: entradas (antes do processamento) e resultados (após o processamento).
#   Variáveis: ETAPA_ENTRADAS, ETAPA_RESULTADOS.
#   Usada em: conferir_entradas (l. 236); conferir_resultados (l. 344); main (l. 641, 644)
ETAPA_ENTRADAS = "1 - entradas (antes do processamento)"
ETAPA_RESULTADOS = "2 - resultados (após o processamento)"

# [R01] Níveis exigidos pelas orientações (Tabela 3.1 do PP, Quadro 1 do TCC, e item 7 da análise do orientador):
#   seções como pares ordenados (b_w, h) = (menor, maior) dimensão, fck, lambda, nu e número de níveis de e1/b_w.
#   Variáveis: SECOES_EXIGIDAS, FCK_EXIGIDOS, LAMBDAS_EXIGIDOS, NUS_EXIGIDOS, NIVEIS_E1_EXIGIDOS, TOTAL_EXIGIDO.
#   Usada em: nível do módulo (l. 67); conferir_entradas (l. 241, 246, 249, 251, 256, 261, 262)
SECOES_EXIGIDAS = {(20.0, 30.0), (30.0, 50.0), (40.0, 80.0)}
FCK_EXIGIDOS = {25.0, 30.0, 35.0}
LAMBDAS_EXIGIDOS = {30.0, 60.0, 90.0, 110.0, 130.0, 140.0}
NUS_EXIGIDOS = {0.20, 0.40, 0.60}
NIVEIS_E1_EXIGIDOS = 2
TOTAL_EXIGIDO = len(SECOES_EXIGIDAS) * len(FCK_EXIGIDOS) * len(LAMBDAS_EXIGIDOS) * len(NUS_EXIGIDOS) * NIVEIS_E1_EXIGIDOS

# [R02] Colunas exigidas na tabela completa dos casos (análise do orientador de 22/06/2026, itens 3.5, 6 e
#   13: "tabela completa dos casos, com dimensões, fck, lambda, Nd, e1, comprimento equivalente e critérios
#   de vinculação") e nas entradas do Módulo 1 (Figura 3 do TCC, Figura 3.1 do PP).
#   Variáveis: COLUNAS_TABELA_CASOS, CAMPOS_MODULO_1, CAMPOS_PROIBIDOS_MODULO_1.
#   Usada em: conferir_entradas (l. 272, 274, 275); conferir_resultados (l. 430, 431)
COLUNAS_TABELA_CASOS = ("b_w (cm)", "h (cm)", "fck (MPa)", "λ (dir. b_w)", "Nd (kN)", "e1 (cm)", "ℓe (cm)",
                        "Vinculação")
CAMPOS_MODULO_1 = ("largura_bw", "altura_h", "fck_mpa", "fyk_mpa", "forca_normal_calculo",
                   "excentricidade_primeira_ordem", "comprimento_equivalente", "vinculacao")
CAMPOS_PROIBIDOS_MODULO_1 = ("esbeltez", "esbeltez_b", "esbeltez_h", "nu")

# [F01] Igualdade numérica com tolerância relativa: |a - b| <= tol max(1, |b|).
#   Variáveis: valor, referencia.
#   Usada em: conferir_entradas (l. 277, 287, 307, 308);
#       conferir_resultados (l. 349, 350, 351, 368, 412, 422, 445, 446, 447, 448, 449, 451, 493, 516, 533, 534, 571, 572)
def iguais(valor: float, referencia: float) -> bool:
    return abs(valor - referencia) <= TOLERANCIA_RELATIVA * max(1.0, abs(referencia))


# [F02] Excentricidade mínima de 1ª ordem: e1,min = M1d,min / Nd ([F21] do núcleo dividido por Nd).
#   Variáveis: momento_minimo, forca_normal.  Fonte: NBR 6118:2023, 11.3.3.4.3.
#   Usada em: conferir_entradas (l. 328)
def excentricidade_minima(momento_minimo: float, forca_normal: float) -> float:
    return momento_minimo / forca_normal


# [F03] Razão entre o momento de 1ª ordem aplicado e o mínimo: Nd e1 / M1d,min (> 1 = flexão representativa).
#   Variáveis: momento_aplicado, momento_minimo.
#   Usada em: conferir_entradas (l. 330)
def razao_momento_minimo(momento_aplicado: float, momento_minimo: float) -> float:
    return momento_aplicado / momento_minimo


# [R03] Método de 2ª ordem esperado pela faixa de lambda ([R01] do núcleo, recalculado de forma independente
#   com os limites da NBR): < lambda_1 dispensa (15.8.2: "menor que o valor-limite"); <= 90 curvatura aproximada;
#   <= 140 M, N, 1/r com fluência; nas fronteiras, tolerância absoluta TOLERANCIA_FRONTEIRA (erro de ponto flutuante
#   do lambda recalculado de l_e exato, [R10] de processar_matriz).
#   Variáveis: esbeltez, esbeltez_limite, TOLERANCIA_FRONTEIRA.
#   Fonte: NBR 6118:2023, 15.8.2, 15.8.3.3.2, 15.8.3.3.4 e 15.8.4.
#   Usada em: metodo_esperado (l. 113, 115); conferir_entradas (l. 310, 311); conferir_resultados (l. 377, 517)
TOLERANCIA_FRONTEIRA = 1e-9


def metodo_esperado(esbeltez: float, esbeltez_limite: float) -> str:
    if esbeltez < esbeltez_limite - TOLERANCIA_FRONTEIRA:
        return nucleo.METODO_DISPENSADO
    if esbeltez <= 90.0 + TOLERANCIA_FRONTEIRA:
        return nucleo.METODO_CURVATURA
    return nucleo.METODO_MN1R


# [R04] Simetria do arranjo em relação aos dois eixos da seção: os conjuntos de coordenadas espelhadas
#   (b_w - x, y) e (x, h - y), ordenados, coincidem com o original a menos de TOLERANCIA_COORDENADA
#   (coordenadas exportadas com 4 casas decimais; erro de exportação <= 0,00005 cm por coordenada).
#   Variáveis: x, y, largura, altura, TOLERANCIA_COORDENADA (cm).  Fonte: PP, Seções 1.1 e 3.6 (armadura simétrica).
#   Usada em: conjuntos_coincidem (l. 130); arranjo_simetrico (l. 138); conferir_resultados (l. 390)
TOLERANCIA_COORDENADA = 1e-3


def conjuntos_coincidem(pontos_a: list[tuple], pontos_b: list[tuple]) -> bool:
    return len(pontos_a) == len(pontos_b) and all(
        abs(xa - xb) <= TOLERANCIA_COORDENADA and abs(ya - yb) <= TOLERANCIA_COORDENADA
        for (xa, ya), (xb, yb) in zip(sorted(pontos_a), sorted(pontos_b)))


def arranjo_simetrico(x: list[float], y: list[float], largura: float, altura: float) -> bool:
    original = list(zip(x, y))
    espelho_x = [(largura - valor_x, valor_y) for valor_x, valor_y in original]
    espelho_y = [(valor_x, altura - valor_y) for valor_x, valor_y in original]
    return conjuntos_coincidem(original, espelho_x) and conjuntos_coincidem(original, espelho_y)


# [F04] Comparação "não menor que" com tolerância absoluta de arredondamento (1e-12): a >= b - tol.
#   Variáveis: valor, referencia.
#   Usada em: nao_menor (l. 148); conferir_resultados (l. 364, 491)
TOLERANCIA_ABSOLUTA = 1e-12


def nao_menor(valor: float, referencia: float) -> bool:
    return valor >= referencia - TOLERANCIA_ABSOLUTA


# [F05] Excentricidade mínima exigida para "flexão representativa": e1 > margem x e1,min ([R02] de
#   processar_matriz.py, MARGEM_EXCENTRICIDADE_MINIMA).  Variáveis: excentricidade_minima_cm, margem.
#   Usada em: conferir_entradas (l. 328)
def excentricidade_exigida(excentricidade_minima_cm: float, margem: float) -> float:
    return margem * excentricidade_minima_cm


# [F06] Número de pilares viáveis com armadura definida pelo equilíbrio (direção b_w ou h) numa faixa.
#   Variáveis: governa (contagem por faixa e parcela), faixa.
#   Usada em: conferir_resultados (l. 475, 477)
def casos_equilibrio(governa: Counter, faixa: str) -> int:
    return governa[(faixa, "direção b_w")] + governa[(faixa, "direção h")]


# [F13] Entradas exatas recalculadas de forma independente ([R10] de processar_matriz): l_e = lambda b_w / raiz(12)
#   e Nd = nu b_w h fck / (10 gama_c) (fck em MPa; 1 kN/cm² = 10 MPa).
#   Variáveis: esbeltez, largura (b_w, cm), altura (h, cm), nu, fck_mpa; nucleo.GAMA_C.
#   Fonte: NBR 6118:2023, 15.8.2 e 12.3.3; Quadro 1 do TCC (Tabela 3.1 do PP).
#   Usada em: conferir_entradas (l. 307, 308)
def comprimento_exato(esbeltez: float, largura: float) -> float:
    return esbeltez * largura / math.sqrt(12.0)


def forca_exata(nu: float, largura: float, altura: float, fck_mpa: float) -> float:
    return nu * largura * altura * fck_mpa / (10.0 * nucleo.GAMA_C)


# [F10] Desvio de um valor recalculado em relação ao nível da matriz: valor - nível (relatado na evidência).
#   Variáveis: valor, nivel.
#   Usada em: conferir_entradas (l. 278, 288)
def desvio_do_nivel(valor: float, nivel: float) -> float:
    return valor - nivel


# [F11] Comparação "não menor que" com a tolerância da busca da armadura (TOLERANCIA_AREA do núcleo):
#   a >= b - tol, para áreas obtidas por busca de raiz em geometrias diferentes.
#   Variáveis: valor, referencia; nucleo.TOLERANCIA_AREA.
#   Usada em: conferir_resultados (l. 414, 415)
def nao_menor_que_area(valor: float, referencia: float) -> bool:
    return valor >= referencia - nucleo.TOLERANCIA_AREA


# [F12] Distância d' esperada na geometria de referência ([R11] do núcleo), recalculada de forma
#   independente: d' = c + φt,mín + φl,mín / 2, com φl,mín = menor bitola longitudinal e φt,mín = menor
#   bitola de estribo (5 mm >= φl,mín / 4 para φl,mín = 10 mm).
#   Variáveis: cobrimento, diametro_estribo, diametro_longitudinal (cm).
#   Usada em: conferir_resultados (l. 410)
def distancia_referencia_esperada(cobrimento: float, diametro_estribo: float, diametro_longitudinal: float) -> float:
    return cobrimento + diametro_estribo + diametro_longitudinal / 2.0


# [R05] Faixas de esbeltez da análise das hipóteses (Seção 3.9 c do PP): H1 em lambda <= 90 e H2 em
#   90 < lambda <= 140.  Variáveis: esbeltez.
#   Usada em: conferir_resultados (l. 468, 470)
def faixa_esbeltez(esbeltez: float) -> str:
    return "λ ≤ 90" if esbeltez <= 90.0 else "90 < λ ≤ 140"

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


def numero(texto: str) -> float:
    return float(texto) if texto not in ("", None) else math.nan


def ler_resultados(pasta: Path) -> list[dict]:
    with (pasta / "resultados_matriz.csv").open(encoding="utf-8") as arquivo:
        return list(csv.DictReader(arquivo, delimiter=";"))


def duas_faces_isoladas(pilar: nucleo.PilarTeste) -> str:
    return nucleo.dimensionar_pilar(dataclasses.replace(pilar, arranjo=nucleo.ARRANJO_DUAS_FACES)).status


def registrador(itens: list[dict], etapa: str):
    def registrar(codigo: str, fonte: str, requisito: str, aprovado: bool, evidencia: str) -> None:
        itens.append({"Etapa": etapa, "Código": codigo, "Fonte": fonte, "Requisito": requisito,
                      "Resultado": "ATENDE" if aprovado else "NÃO ATENDE", "Evidência": evidencia})
    return registrar


def conferir_entradas(casos: list) -> list[dict]:
    """Etapa 1: composição da matriz e entradas do Módulo 1 (não depende de resultados)."""
    itens: list[dict] = []
    registrar = registrador(itens, ETAPA_ENTRADAS)

    # ------------------------------------------------------------------ composição da matriz
    secoes = {tuple(sorted((caso.largura_bw, caso.altura_h))) for caso in casos}
    registrar("M01", "PP 3.6 / Tab. 3.1; MSF 22/06 item 7", "Três seções: 20×30, 30×50 e 40×80 cm",
              secoes == SECOES_EXIGIDAS, "seções na matriz: %s" % sorted(secoes))
    menor_e_bw = all(caso.largura_bw <= caso.altura_h for caso in casos)
    registrar("M02", "PP Tab. 3.1 (seção b_w × h); decisão do autor (justificada em [R01] de processar_matriz e [H07] "
              "do núcleo)", "b_w = menor dimensão (direção de e1, da flexão de 1ª ordem e do λ da matriz)", menor_e_bw,
              "b_w ≤ h em %d de %d pilares" % (sum(caso.largura_bw <= caso.altura_h for caso in casos), len(casos)))
    registrar("M03", "PP 3.6 / Tab. 3.1", "fck = 25, 30 e 35 MPa", {caso.fck_mpa for caso in casos} == FCK_EXIGIDOS,
              "fck: %s" % sorted({caso.fck_mpa for caso in casos}))
    registrar("M04", "PP 3.6 / Tab. 3.1", "λ = 30, 60, 90, 110, 130 e 140",
              {caso.esbeltez for caso in casos} == LAMBDAS_EXIGIDOS, "λ: %s" % sorted({caso.esbeltez for caso in casos}))
    registrar("M05", "PP 3.6 / Tab. 3.1; MSF 22/06 item 7", "ν = Nd/(Ac fcd): baixo 0,20, médio 0,40, alto 0,60",
              {caso.nu for caso in casos} == NUS_EXIGIDOS, "ν: %s" % sorted({caso.nu for caso in casos}))
    niveis_por_base = Counter((caso.largura_bw, caso.fck_mpa, caso.esbeltez, caso.nu) for caso in casos)
    niveis_e1 = {caso.excentricidade_relativa for caso in casos}
    registrar("M06", "PP 3.6 (e1: dois níveis) / Tab. 3.1; MSF PIT e 22/06; PP 1.4.2 d",
              "e1 variável em dois níveis (pequena e moderada) para cada combinação-base e cada ν",
              len(niveis_e1) == NIVEIS_E1_EXIGIDOS and set(niveis_por_base.values()) == {NIVEIS_E1_EXIGIDOS},
              "e1/b_w: %s; níveis por (seção, fck, λ, ν): %s" % (sorted(niveis_e1), sorted(set(niveis_por_base.values()))))
    combinacoes = Counter((caso.largura_bw, caso.fck_mpa, caso.esbeltez, caso.nu, caso.excentricidade_relativa)
                          for caso in casos)
    registrar("M07", "PP 3.6 / Tab. 3.1 (54 base × 3 ν × 2 e1)",
              "Total de %d pilares, cada combinação exatamente uma vez (fatorial completo)" % TOTAL_EXIGIDO,
              len(casos) == TOTAL_EXIGIDO and set(combinacoes.values()) == {1} and len(combinacoes) == TOTAL_EXIGIDO,
              "%d pilares, %d combinações distintas" % (len(casos), len(combinacoes)))
    registrar("M08", "PP 3.6; ARD nota 7", "Arranjo da armadura NÃO é fator da matriz (é regra)",
              all(caso.pilar.arranjo == nucleo.ARRANJO_REGRA for caso in casos),
              "arranjo de entrada = regra do PP em %d pilares" % sum(caso.pilar.arranjo == nucleo.ARRANJO_REGRA
                                                                      for caso in casos))

    # ------------------------------------------------------------------ entradas do Módulo 1
    campos = [campo.name for campo in dataclasses.fields(nucleo.PilarTeste)]
    registrar("E01", "PP Figura 3.1 (Módulo 1)", "Entradas do programa: b_w, h, fck, fyk, Nd, e1, ℓe, vinculação",
              all(campo in campos for campo in CAMPOS_MODULO_1), "campos de PilarTeste: %s" % ", ".join(campos))
    registrar("E02", "ARD nota 5 (λ 'é função do ℓe e vinculação')", "λ e ν não são entradas; são calculados",
              not any(campo in campos for campo in CAMPOS_PROIBIDOS_MODULO_1),
              "campos proibidos presentes: %s" % [campo for campo in CAMPOS_PROIBIDOS_MODULO_1 if campo in campos])
    esbeltezes = [nucleo.indice_esbeltez(caso.pilar.comprimento_equivalente, caso.largura_bw) for caso in casos]
    lambdas = [iguais(esbeltez, caso.esbeltez) for esbeltez, caso in zip(esbeltezes, casos)]
    desvios_lambda = [desvio_do_nivel(esbeltez, caso.esbeltez) for esbeltez, caso in zip(esbeltezes, casos)]
    registrar("E03", "PP 3.6 (ℓe = β ℓ); NBR 15.8.2; [R10] de processar_matriz (valores exatos)",
              "λ calculado de ℓe ([F19] do núcleo) igual ao nível da matriz (ℓe exato, sem arredondamento)",
              all(lambdas), "atende em %d de %d; λ calculado − nível de %.1e a %.1e" % (
                  sum(lambdas), len(lambdas), min(desvios_lambda), max(desvios_lambda)))
    forcas_adimensionais = [nucleo.forca_normal_adimensional(caso.pilar.forca_normal_calculo,
                                                             nucleo.area_secao_retangular(caso.largura_bw,
                                                                                          caso.altura_h),
                                                             nucleo.Material(caso.fck_mpa).fcd) for caso in casos]
    nus = [iguais(valor, caso.nu) for valor, caso in zip(forcas_adimensionais, casos)]
    desvios_nu = [desvio_do_nivel(valor, caso.nu) for valor, caso in zip(forcas_adimensionais, casos)]
    registrar("E04", "PP 3.6 (Nd por ν, 'ν ≈ 0,20 / 0,40 / 0,60'); [R10] de processar_matriz (valores exatos)",
              "ν calculado de Nd ([F25] do núcleo) igual ao nível da matriz (Nd exato, sem arredondamento)", all(nus),
              "atende em %d de %d; ν calculado − nível de %.1e a %.1e" % (
                  sum(nus), len(nus), min(desvios_nu), max(desvios_nu)))
    registrar("E05", "PP 3.6 (biapoiado, ℓe = ℓ)", "Vinculação biapoiada, β = 1, ℓe = ℓ em todos",
              all(caso.pilar.vinculacao == nucleo.VINCULACAO_BIAPOIADO for caso in casos)
              and nucleo.FATOR_COMPRIMENTO_BIAPOIADO == 1.0, "β = %.1f" % nucleo.FATOR_COMPRIMENTO_BIAPOIADO)
    registrar("E06", "PP 3.6 (parâmetros fixos)", "Aço CA-50 (fyk = 500 MPa) e cobrimento nominal 25 mm",
              all(caso.pilar.fyk_mpa == 500.0 and caso.pilar.cobrimento == 2.5 for caso in casos),
              "fyk = %s MPa; c = %s cm" % ({caso.pilar.fyk_mpa for caso in casos}, {caso.pilar.cobrimento for caso in casos}))
    registrar("E07", "NBR 13.2.3", "Menor dimensão ≥ 19 cm e Ac ≥ 360 cm²",
              all(min(caso.largura_bw, caso.altura_h) >= nucleo.DIMENSAO_MINIMA_PILAR
                  and nucleo.area_secao_retangular(caso.largura_bw, caso.altura_h) >= nucleo.AREA_MINIMA_PILAR
                  for caso in casos), "menor dimensão mínima = %.0f cm" % min(caso.largura_bw for caso in casos))
    exatas = []
    for caso in casos:
        pilar = caso.pilar
        esbeltez_calculada = nucleo.indice_esbeltez(pilar.comprimento_equivalente, pilar.largura_bw)
        exatas.append(iguais(pilar.comprimento_equivalente, comprimento_exato(caso.esbeltez, caso.largura_bw))
                      and iguais(pilar.forca_normal_calculo, forca_exata(caso.nu, caso.largura_bw, caso.altura_h,
                                                                         caso.fck_mpa))
                      and metodo_esperado(esbeltez_calculada, nucleo.LAMBDA1_MIN)
                      == metodo_esperado(caso.esbeltez, nucleo.LAMBDA1_MIN))
    registrar("E08", "Decisão do autor (22/10/2026: valores exatos, sem arredondamento por conveniência de digitação); "
              "PP 1.4.2 e e 3.6; [R10] de processar_matriz",
              "Entradas exatas: ℓe = λ b_w/√12 e Nd = ν Ac fcd recalculados de forma independente iguais aos da rotina; "
              "o λ calculado de ℓe dá o mesmo método de 2ª ordem do nível (fronteiras 35, 90 e 140)",
              all(exatas), "atende em %d de %d" % (sum(exatas), len(exatas)))

    # ------------------------------------------------------------------ excentricidade e1
    e1_ea, e1_min, razoes = [], [], []
    for caso in casos:
        pilar = caso.pilar
        comprimento = nucleo.comprimento_real(pilar.comprimento_equivalente, nucleo.FATOR_COMPRIMENTO_BIAPOIADO)
        excentricidade_a = nucleo.excentricidade_acidental(nucleo.desaprumo_local(comprimento), comprimento)
        momento_minimo = nucleo.momento_minimo_primeira_ordem(pilar.forca_normal_calculo, pilar.largura_bw)
        momento_aplicado = nucleo.momento_excentricidade(pilar.forca_normal_calculo, pilar.excentricidade_primeira_ordem)
        e1_ea.append(pilar.excentricidade_primeira_ordem >= excentricidade_a)
        e1_min.append(pilar.excentricidade_primeira_ordem
                      > excentricidade_exigida(excentricidade_minima(momento_minimo, pilar.forca_normal_calculo),
                                               matriz.MARGEM_EXCENTRICIDADE_MINIMA))
        razoes.append(razao_momento_minimo(momento_aplicado, momento_minimo))
    registrar("X01", "PP 3.6 (e1 ≥ ea, ea pela NBR 11.3.3.4.2)", "e1 ≥ ea em todos os pilares", all(e1_ea),
              "atende em %d de %d" % (sum(e1_ea), len(e1_ea)))
    registrar("X02", "PP Tab. 3.1 ('evitar casos sem flexão representativa'); NBR 11.3.3.4.3",
              "e1 > e1,mín (Nd·e1 > M1d,mín): nenhum nível absorvido pelo momento mínimo", all(e1_min),
              "atende em %d de %d; Nd·e1/M1d,mín de %.2f a %.2f" % (sum(e1_min), len(e1_min), min(razoes), max(razoes)))
    return itens


def conferir_resultados(pasta: Path, casos: list) -> list[dict]:
    """Etapa 2: resultados gravados por processar_matriz.py (procedimento, armadura, arranjo e planilha)."""
    por_id = {caso.identificador: caso for caso in casos}
    linhas = ler_resultados(pasta)
    itens: list[dict] = []
    registrar = registrador(itens, ETAPA_RESULTADOS)

    # ------------------------------------------------------------------ correspondência com a matriz
    mesmas_entradas = (len(linhas) == len(casos) and all(
        linha["ID"] == caso.identificador
        and iguais(numero(linha["Nd (kN)"]), caso.pilar.forca_normal_calculo)
        and iguais(numero(linha["ℓe (cm)"]), caso.pilar.comprimento_equivalente)
        and iguais(numero(linha["e1 (cm)"]), caso.pilar.excentricidade_primeira_ordem)
        for linha, caso in zip(linhas, casos)))
    registrar("P07", "PP 3.6 (registro em planilha); etapa 1 desta conferência",
              "Os resultados lidos correspondem à matriz conferida na etapa 1 (mesmos pilares, na mesma ordem, "
              "com Nd, ℓe e e1 iguais)", mesmas_entradas,
              "%d linhas em resultados_matriz.csv; %d pilares na matriz" % (len(linhas), len(casos)))

    # ------------------------------------------------------------------ excentricidade e1 (resultados)
    so_bw = all(numero(linha["Nd·e1 h (kN·m)"]) == 0.0 and numero(linha["Nd·e1 b_w (kN·m)"]) > 0.0 for linha in linhas)
    registrar("X03", "PP título e 1.1 (flexão normal composta reta); NBR 15.3.2",
              "Momento de 1ª ordem aplicado só na direção de b_w; direção de h só com M1d,mín (análise isolada)",
              so_bw, "e1 aplicado só em b_w em %d de %d" % (
                  sum(numero(linha["Nd·e1 h (kN·m)"]) == 0.0 for linha in linhas), len(linhas)))
    momento_ok = all(nao_menor(numero(linha["M1d,A %s (kN·m)" % direcao]), numero(linha["M1d,mín %s (kN·m)" % direcao]))
                     for linha in linhas for direcao in ("b_w", "h") if linha["M1d,A %s (kN·m)" % direcao])
    registrar("X04", "PP 3.6 (M1d ≥ Md,mín); NBR 11.3.3.4.3", "M1d,A ≥ M1d,mín nas duas direções", momento_ok,
              "verificado em todas as linhas com M1d,A finito")
    alfa_ok = all(iguais(numero(linha["αb %s" % direcao]), 1.0) for linha in linhas for direcao in ("b_w", "h"))
    registrar("X05", "PP 3.6 (αb em função dos momentos de extremidade); NBR 15.8.2 a) e d)",
              "αb = 0,60 + 0,40 MB/MA = 1,0 (MA = MB) em b_w; αb = 1,0 em h (momentos < mínimo)", alfa_ok,
              "αb = 1,0 em todas as direções de todos os pilares")

    # ------------------------------------------------------------------ procedimento normativo
    metodos_ok, fluencia_ok, lambda_h_ok = True, True, True
    for linha in linhas:
        for direcao, coluna_lambda in (("b_w", "λ b_w calc"), ("h", "λ h calc")):
            esperado = metodo_esperado(numero(linha[coluna_lambda]), numero(linha["λ1 %s" % direcao]))
            metodos_ok &= linha["Método %s" % direcao] == esperado
            if esperado == nucleo.METODO_MN1R:
                fluencia_ok &= linha["e_a %s (cm)" % direcao] != "" and linha["N_sg %s (kN)" % direcao] != ""
        lambda_h_ok &= numero(linha["λ h calc"]) <= nucleo.LAMBDA_LIMITE_PILAR_PADRAO
    registrar("N01", "PP 3.5; NBR 15.8.2, 15.8.3.3.2, 15.8.3.3.4", "Método de 2ª ordem conforme a faixa de λ, em cada direção",
              metodos_ok, "método recalculado independentemente = método da rotina em todas as direções")
    registrar("N02", "PP 3.5; NBR 15.8.4", "Fluência (ecc com ea e combinação quase permanente) em todo λ > 90",
              fluencia_ok, "ea e Nsg presentes em todas as direções com M, N, 1/r")
    registrar("N03", "PP 1.1 (λ ≤ 140); ARD nota 6 ('teste se λ < 140')", "λ ≤ 140 também na direção de h (e teste na entrada)",
              lambda_h_ok, "máximo λ_h = %.1f" % max(numero(linha["λ h calc"]) for linha in linhas))

    # ------------------------------------------------------------------ armadura e arranjo
    simetria = [arranjo_simetrico([float(valor) for valor in linha["x barras (cm)"].split()],
                                  [float(valor) for valor in linha["y barras (cm)"].split()],
                                  numero(linha["b_w (cm)"]), numero(linha["h (cm)"]))
                for linha in linhas if linha["x barras (cm)"]]
    registrar("A01", "PP 1.1 e 3.6 (armadura simétrica)", "Armadura adotada simétrica nos dois eixos", all(simetria),
              "simétrica em %d de %d pilares detalhados" % (sum(simetria), len(simetria)))
    perimetrais = [linha for linha in linhas if linha["Arranjo adotado"] == nucleo.ARRANJO_PERIMETRO]
    tentativa_ok = all(linha["Arranjos avaliados"].startswith(nucleo.ARRANJO_DUAS_FACES) for linha in linhas)
    inviaveis_duas = [linha for linha in linhas if linha["Situação"] != nucleo.STATUS_OK]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        status_duas = list(executor.map(duas_faces_isoladas, [por_id[linha["ID"]].pilar for linha in
                                                              [*perimetrais, *inviaveis_duas]]))
    perimetro_justificado = all(status != nucleo.STATUS_OK for status in status_duas)
    tentou_perimetro = all(linha["Arranjos avaliados"] == "%s, %s" % nucleo.ORDEM_ARRANJOS_REGRA for linha in inviaveis_duas)
    registrar("A02", "PP 3.6; ARD nota 7", "Duas faces opostas sempre tentadas primeiro; perímetro só se duas faces for inviável",
              tentativa_ok and perimetro_justificado,
              "%d pilares com perímetro adotado; duas faces confirmado inviável em todos os %d casos recalculados" % (
                  len(perimetrais), len(status_duas)))
    registrar("A03", "PP 3.6; ARD nota 7", "Nos casos inviáveis em duas faces, o perímetro foi efetivamente tentado",
              tentou_perimetro, "%d casos inviáveis, todos com duas faces e perímetro avaliados" % len(inviaveis_duas))
    distancia_esperada = distancia_referencia_esperada(nucleo.COBRIMENTO_NOMINAL, min(nucleo.BITOLAS_ESTRIBO),
                                                       min(nucleo.BITOLAS_LONGITUDINAIS))
    distancias_ok = all(iguais(numero(linha["d' ref (cm)"]), distancia_esperada) for linha in linhas)
    em_duas_faces = [linha for linha in linhas if linha["Arranjo adotado"] == nucleo.ARRANJO_DUAS_FACES]
    limite_ok = all(nao_menor_que_area(numero(linha["As,calc (cm²)"]), numero(linha["As,calc ref (cm²)"]))
                    and nao_menor_que_area(numero(linha["As,nec b_w (cm²)"]), numero(linha["As,nec b_w ref (cm²)"]))
                    for linha in em_duas_faces)
    registrar("A04", "PP 3.9 a (comparação isolando o método); [R11] do núcleo",
              "Geometria de referência com d' único em todos os pilares; em duas faces, As,calc ≥ As,calc ref e "
              "As,nec b_w ≥ As,nec b_w ref (a referência, com as barras mais próximas das faces, é a mais eficiente)",
              distancias_ok and limite_ok and bool(em_duas_faces),
              "d' ref = %.2f cm em %d de %d; As,calc ≥ As,calc ref nos %d pilares detalhados em duas faces" % (
                  distancia_esperada, sum(iguais(numero(linha["d' ref (cm)"]), distancia_esperada) for linha in linhas),
                  len(linhas), len(em_duas_faces)))

    # ------------------------------------------------------------------ planilha, comparação e análise
    livro = load_workbook(pasta / "resultados_matriz.xlsx", read_only=True)
    cabecalho = [celula.value for celula in next(livro["Matriz_entradas"].iter_rows(min_row=1, max_row=1))]
    registrar("P01", "PP 3.6 (registro em planilha); MSF 22/06 itens 3.5, 6 e 13",
              "Tabela completa dos casos com dimensões, fck, λ, Nd, e1, ℓe e vinculação",
              all(coluna in cabecalho for coluna in COLUNAS_TABELA_CASOS), "aba Matriz_entradas: %s" % ", ".join(
                  coluna for coluna in COLUNAS_TABELA_CASOS if coluna in cabecalho))
    linhas_soft = list(livro["Entrada_softwares"].iter_rows(values_only=True))
    cab_soft = next(linha for linha in linhas_soft if linha and linha[0] == "ID")
    colunas_soft = {coluna: indice for indice, coluna in enumerate(cab_soft)}
    dados_soft = [linha for linha in linhas_soft[linhas_soft.index(cab_soft) + 1:] if linha and linha[0]]
    exigidas_soft = ("ℓ = ℓe (cm)", "ℓ (m)", "Nk (kN)", "Ngk (kN)", "Nqk (kN)", "e1 (cm)", "Mgk topo = base (kN·m)",
                     "Mqk topo = base (kN·m)", "Mk topo = base (kN·m)")
    identicas = []
    if all(coluna in colunas_soft for coluna in exigidas_soft):
        for linha in dados_soft:
            pilar = por_id[linha[colunas_soft["ID"]]].pilar
            forca_k, forca_gk, forca_qk = nucleo.parcelas_forca_caracteristica(pilar.forca_normal_calculo,
                                                                               nucleo.FRACAO_PERMANENTE)
            identicas.append(
                iguais(linha[colunas_soft["ℓ = ℓe (cm)"]], pilar.comprimento_equivalente)
                and iguais(linha[colunas_soft["Nk (kN)"]], forca_k) and iguais(linha[colunas_soft["Ngk (kN)"]], forca_gk)
                and iguais(linha[colunas_soft["Nqk (kN)"]], forca_qk)
                and iguais(linha[colunas_soft["e1 (cm)"]], pilar.excentricidade_primeira_ordem)
                and iguais(linha[colunas_soft["Mgk topo = base (kN·m)"]], nucleo.kn_cm_para_kn_m(
                    nucleo.momento_excentricidade(forca_gk, pilar.excentricidade_primeira_ordem)))
                and iguais(linha[colunas_soft["Mqk topo = base (kN·m)"]], nucleo.kn_cm_para_kn_m(
                    nucleo.momento_excentricidade(forca_qk, pilar.excentricidade_primeira_ordem))))
    registrar("P02", "PP 1.4.2 e (mesmos materiais, cobrimento, bitolas, vinculação e combinações de ações); MSF 22/06 3.5",
              "Entradas para os softwares com ℓ, cargas características (Ngk, Nqk) e momentos de topo/base por caso de "
              "carga, idênticos aos valores usados na rotina Python",
              bool(identicas) and all(identicas) and len(identicas) == len(linhas),
              "aba Entrada_softwares: %d linhas; idênticas à rotina em %d" % (len(dados_soft), sum(identicas)))
    desvios_exigidos = ["Δ%% %s %s" % (grandeza, software) for software in matriz.SOFTWARES_COMPARADOS
                        for grandeza in ("As,calc", "Md,tot b_w", "Md,tot h")]
    registrar("P03", "PP 3.9 a/b (As,calc preferencial; Δ% da Eq. 3.2); proposta A do autor (22/10/2026)",
              "Comparação de As,calc, As,adot e Md,tot (direções de b_w e de h) com Δ%",
              all(coluna in cab_soft for coluna in desvios_exigidos),
              "colunas %s por fórmula idêntica a [F50]" % ", ".join(desvios_exigidos))
    hipoteses = list(livro["Hipoteses"].iter_rows(min_row=2, values_only=True))
    registrar("P04", "MSF 22/06 item 3.5 ('quadro com entradas fixas e variáveis e sua fonte normativa')",
              "Todos os parâmetros fixos com fonte declarada", all(linha[2] for linha in hipoteses),
              "%d parâmetros na aba Hipoteses, todos com fonte" % len(hipoteses))
    contagem = Counter((faixa_esbeltez(por_id[linha["ID"]].esbeltez), linha["Situação"] == nucleo.STATUS_OK)
                       for linha in linhas)
    governa = Counter((faixa_esbeltez(por_id[linha["ID"]].esbeltez), linha["Governa"]) for linha in linhas
                      if linha["Situação"] == nucleo.STATUS_OK)
    faixas = ("λ ≤ 90", "90 < λ ≤ 140")
    registrar("P05", "PP 3.9 c (análise separada λ ≤ 90 e 90 < λ ≤ 140 para H1 e H2)",
              "Casos comparáveis com armadura definida pelo equilíbrio nas duas faixas",
              all(casos_equilibrio(governa, faixa) > 0 for faixa in faixas),
              "; ".join("%s: %d viáveis (%d pelo equilíbrio, %d por As,mín), %d inviáveis" % (
                  faixa, contagem[(faixa, True)], casos_equilibrio(governa, faixa),
                  governa[(faixa, "As,mín")], contagem[(faixa, False)]) for faixa in faixas))
    modelagem_ok = all((linha["Modelar nos softwares"] == "sim") == (linha["Situação"] == nucleo.STATUS_OK)
                       for linha in linhas)
    registrar("P06", "PP Tab. 3.1 ('casos com baixa representatividade podem ser excluídos'); MSF 22/06 item 7",
              "Critério explícito de exclusão da modelagem nos softwares (inviáveis só na rotina Python)", modelagem_ok,
              "%d para modelar; %d só na rotina Python" % (
                  sum(linha["Modelar nos softwares"] == "sim" for linha in linhas),
                  sum(linha["Modelar nos softwares"] != "sim" for linha in linhas)))

    # ------------------------------------------------------------------ envoltória mínima com 2ª ordem (15.3.2)
    viaveis = [linha for linha in linhas if linha["Situação"] == nucleo.STATUS_OK]
    fatores = [numero(linha["η envoltória mínima"]) for linha in viaveis]
    semieixos_ok = all(
        nao_menor(numero(linha["Md,tot b_w (kN·m)"]), numero(linha["Md,tot,mín b_w (kN·m)"]))
        and (linha["Método h"] == nucleo.METODO_MN1R
             or iguais(numero(linha["Md,tot,mín h (kN·m)"]), numero(linha["Md,tot h (kN·m)"])))
        for linha in viaveis)
    englobadas = sum(fator >= nucleo.FATOR_MINIMO_ENVOLTORIA for fator in fatores)
    registrar("X06", "NBR 11.3.3.4.3 e 15.3.2; [H06] e [R12] do núcleo",
              "Envoltória mínima com 2ª ordem verificada com a armadura adotada em todos os pilares viáveis "
              "(semieixos só com M1d,mín; na direção de b_w, Md,tot,mín ≤ Md,tot com e1; na de h, igual ao Md,tot "
              "da análise isolada, exceto no M, N, 1/r, que depende da armadura)",
              bool(fatores) and all(math.isfinite(fator) for fator in fatores) and semieixos_ok,
              "η de %.3f a %.2f; elipse englobada em %d de %d pilares viáveis" % (
                  min(fatores), max(fatores), englobadas, len(viaveis)))

    # ------------------------------------------------------------------ teste do momento mínimo (conjunto [R13])
    with (pasta / "resultados_teste_minimo.csv").open(encoding="utf-8") as arquivo:
        linhas_teste = list(csv.DictReader(arquivo, delimiter=";"))
    casos_teste = matriz.montar_teste_minimo()
    niveis_ok = (len(linhas_teste) == len(casos_teste)
                 and [linha["ID"] for linha in linhas_teste] == [caso.identificador for caso in casos_teste]
                 and {(numero(linha["b_w (cm)"]), numero(linha["h (cm)"])) for linha in linhas_teste} == set(matriz.SECOES)
                 and {numero(linha["λ (dir. b_w)"]) for linha in linhas_teste} == set(matriz.ESBELTEZES_TESTE_MINIMO)
                 and all(numero(linha["fck (MPa)"]) == matriz.FCK_TESTE_MINIMO
                         and numero(linha["ν"]) == matriz.NU_TESTE_MINIMO and numero(linha["e1 (cm)"]) == 0.0
                         for linha in linhas_teste))
    so_minimo_ok = all(numero(linha["Nd·e1 %s (kN·m)" % direcao]) == 0.0
                       and iguais(numero(linha["αb %s" % direcao]), 1.0)
                       and linha["Método %s" % direcao] == metodo_esperado(numero(linha["λ %s calc" % direcao]),
                                                                           numero(linha["λ1 %s" % direcao]))
                       for linha in linhas_teste for direcao in ("b_w", "h"))
    registrar("T01", "Pedido do autor (22/10/2026); NBR 11.3.3.4.3, 15.8.2 d) e 15.8.3; [R13] de processar_matriz",
              "Teste do momento mínimo: %d pilares fora da matriz, só com M1d,mín (e1 = 0) nas duas direções, nas três "
              "seções, com λ = 30, 90 e 110, fck = 30 MPa e ν = 0,40; αb = 1 e método de 2ª ordem da faixa de λ" % len(
                  casos_teste), niveis_ok and so_minimo_ok,
              "resultados_teste_minimo.csv: %d pilares (%s); %d viáveis" % (
                  len(linhas_teste), ", ".join(linha["ID"] for linha in linhas_teste),
                  sum(linha["Situação"] == nucleo.STATUS_OK for linha in linhas_teste)))
    linhas_aba = list(livro["Teste_momento_minimo"].iter_rows(values_only=True))
    cab_teste = next(linha for linha in linhas_aba if linha and linha[0] == "ID")
    colunas_teste = {coluna: indice for indice, coluna in enumerate(cab_teste)}
    dados_teste = [linha for linha in linhas_aba[linhas_aba.index(cab_teste) + 1:] if linha and linha[0]]
    por_id_teste = {caso.identificador: caso for caso in casos_teste}
    entradas_teste_ok = len(dados_teste) == len(casos_teste) and all(
        iguais(linha[colunas_teste["ℓ = ℓe (cm)"]], por_id_teste[linha[0]].pilar.comprimento_equivalente)
        and iguais(linha[colunas_teste["Nk (kN)"]], nucleo.parcelas_forca_caracteristica(
            por_id_teste[linha[0]].pilar.forca_normal_calculo, nucleo.FRACAO_PERMANENTE)[0])
        and linha[colunas_teste["e1 (cm)"]] == 0.0 for linha in dados_teste)
    desvios_teste = ["Δ%% %s %s" % (grandeza, software) for software in matriz.SOFTWARES_COMPARADOS
                     for grandeza in ("M1d,mín b_w", "M1d,mín h", "Md,tot b_w", "Md,tot h")]
    viaveis_teste = [linha for linha in linhas_teste if linha["Situação"] == nucleo.STATUS_OK]
    envoltoria_teste_ok = all(math.isfinite(numero(linha["η envoltória mínima"])) and linha["As,adot elipse (cm²)"]
                              for linha in viaveis_teste)
    registrar("T02", "Pedido do autor (22/10/2026); NBR 15.3.2 (Figura 15.2)",
              "Aba Teste_momento_minimo com as entradas idênticas às da rotina, Δ% de M1d,mín e Md,tot nas duas "
              "direções e, para cada pilar viável, η da elipse e o primeiro arranjo que a engloba",
              entradas_teste_ok and all(coluna in cab_teste for coluna in desvios_teste) and envoltoria_teste_ok,
              "%d linhas; η de %s; elipse englobada pela armadura das duas análises isoladas em %d de %d viáveis" % (
                  len(dados_teste), " / ".join("%s %.3f" % (linha["ID"], numero(linha["η envoltória mínima"]))
                                               for linha in viaveis_teste),
                  sum(numero(linha["η envoltória mínima"]) >= nucleo.FATOR_MINIMO_ENVOLTORIA
                      for linha in viaveis_teste), len(viaveis_teste)))
    governa_matriz = Counter(linha["Governa"] for linha in viaveis)
    governa_teste = Counter(linha["Governa"] for linha in viaveis_teste)
    registrar("T03", "Justificativa do teste do momento mínimo ([R13] de processar_matriz)",
              "Na matriz, o momento mínimo não define a armadura (nenhum pilar viável governado pela direção de h, a "
              "única só com M1d,mín); no teste, com e1 = 0, a armadura de todos os pilares vem só do mínimo (parcelas "
              "na evidência)",
              governa_matriz["direção h"] == 0,
              "matriz (%d viáveis): %s; teste (%d viáveis): %s" % (
                  len(viaveis), ", ".join("%s %d" % item for item in sorted(governa_matriz.items())),
                  len(viaveis_teste), ", ".join("%s %d" % item for item in sorted(governa_teste.items()))))

    # ------------------------------------------------------------------ inviáveis: critério e cenários ([R17] a [R19])
    with (pasta / "resultados_inviaveis.csv").open(encoding="utf-8") as arquivo:
        linhas_inviaveis = list(csv.DictReader(arquivo, delimiter=";"))
    por_id_linha = {linha["ID"]: linha for linha in linhas}
    ids_inviaveis = [linha["ID"] for linha in linhas if linha["Situação"] != nucleo.STATUS_OK]
    prova_ok = [linha["ID"] for linha in linhas_inviaveis] == ids_inviaveis
    for linha in linhas_inviaveis:
        principal = por_id_linha[linha["ID"]]
        prova_ok &= (linha["Situação (momento em b_w)"] == principal["Situação"]
                     and iguais(numero(linha["Nd (kN)"]), numero(principal["Nd (kN)"]))
                     and iguais(numero(linha["e1 (cm)"]), numero(principal["e1 (cm)"])))
        criterio = linha["Critério"]
        if criterio == "A":
            prova_ok &= numero(linha["As,calc (cm²)"]) > numero(linha["As,máx 4 % (cm²)"])
        elif criterio == "E":
            prova_ok &= (math.isnan(numero(linha["As,calc (cm²)"]))
                         and (math.isnan(numero(linha["As,nec b_w (cm²)"])) or math.isnan(numero(linha["As,nec h (cm²)"]))))
        elif criterio == "D":
            prova_ok &= numero(linha["As,calc (cm²)"]) <= numero(linha["As,máx 4 % (cm²)"])
            for prefixo in ("Duas faces", "Perímetro"):
                if linha["%s: situação" % prefixo] == "D" and linha["%s: candidatos" % prefixo] not in ("", "0"):
                    prova_ok &= (numero(linha["%s: As,adot do maior (cm²)" % prefixo])
                                 < numero(linha["%s: As,calc do maior (cm²)" % prefixo]))
        else:
            prova_ok = False
    contagens = {chave: Counter(linha[chave] for linha in linhas_inviaveis)
                 for chave in ("Critério", "Situação com As,máx = 8 %", "Critério (momento em h)",
                               "Situação, momento em h, As,máx = 8 %")}
    registrar("I01", "Pedido do autor (Pessoal 1, itens 4 e 5; 22/10/2026); [R05] e [R09] do núcleo; [R17] a [R19] de "
              "processar_matriz",
              "Tabela dos pilares inviáveis com o critério que falhou e a prova numérica (A: As,calc > As,máx; E: sem "
              "equilíbrio ou > 8 % Ac na direção indicada; D: o maior candidato que cabe tem As,adot < As,calc da "
              "própria geometria), sensibilidade com As,máx = 8 % e cenário com a força na maior dimensão",
              prova_ok and bool(linhas_inviaveis),
              "%d inviáveis; critério: %s; com 8 %%: %s; momento em h: %s; momento em h com 8 %%: %s" % (
                  len(linhas_inviaveis), *(", ".join("%s %d" % item for item in sorted(contagem.items()))
                                           for contagem in contagens.values())))
    return itens


def escrever(pasta: Path, itens: list[dict]) -> None:
    with (pasta / "verificacao_orientacoes.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(itens[0].keys()), delimiter=";")
        escritor.writeheader()
        escritor.writerows(itens)
    livro = load_workbook(pasta / "resultados_matriz.xlsx")
    if "Conferencia_orientacoes" in livro.sheetnames:
        del livro["Conferencia_orientacoes"]
    planilha = livro.create_sheet("Conferencia_orientacoes", 0)
    planilha.append(list(itens[0].keys()))
    for item in itens:
        planilha.append(list(item.values()))
    for celula in planilha[1]:
        celula.font = Font(name=matriz.FONTE_PADRAO, bold=True, size=10)
        celula.fill = matriz.PREENCHIMENTO_CABECALHO
    verde, vermelho = PatternFill("solid", fgColor="C6EFCE"), PatternFill("solid", fgColor="FFC7CE")
    for linha in planilha.iter_rows(min_row=2):
        for celula in linha:
            celula.font = Font(name=matriz.FONTE_PADRAO, size=10)
            celula.alignment = Alignment(wrap_text=True, vertical="top")
        linha[4].fill = verde if linha[4].value == "ATENDE" else vermelho
    for letra, largura in (("A", 20), ("B", 8), ("C", 42), ("D", 52), ("E", 13), ("F", 70)):
        planilha.column_dimensions[letra].width = largura
    livro.save(pasta / "resultados_matriz.xlsx")


def imprimir(itens: list[dict]) -> list[dict]:
    """Mostra os itens e devolve os não atendidos."""
    for item in itens:
        print("%-4s %-10s %s\n      %s" % (item["Código"], item["Resultado"], item["Requisito"], item["Evidência"]))
    falhas = [item for item in itens if item["Resultado"] != "ATENDE"]
    print("requisitos: %d | atendidos: %d | não atendidos: %d" % (
        len(itens), sum(item["Resultado"] == "ATENDE" for item in itens), len(falhas)))
    return falhas


def main() -> int:
    pasta = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "saida"
    casos = matriz.montar_matriz()
    print("Etapa %s" % ETAPA_ENTRADAS)
    entradas = conferir_entradas(casos)
    falhas = imprimir(entradas)
    print("\nEtapa %s" % ETAPA_RESULTADOS)
    resultados = conferir_resultados(pasta, casos)
    falhas += imprimir(resultados)
    todos = [*entradas, *resultados]
    escrever(pasta, todos)
    print("\ntotal: %d requisitos | não atendidos: %d" % (len(todos), len(falhas)))
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
