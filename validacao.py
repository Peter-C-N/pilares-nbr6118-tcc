# -*- coding: utf-8 -*-
"""
validacao.py
============
Validação da rotina pilares_nbr6118.py em dois níveis (Seção 3.7 do TCC).

Nível 1 — validação matemática: casos com solução analítica exata (compressão centrada, inclusive com a
seção líquida da hipótese [H05] do núcleo, tração pura, resultante e centroide do bloco
parábola-retângulo, rigidez inicial da seção e continuidade dos domínios), conferência da integração
em forma fechada por um modelo de fibras independente e conferência do solver do método geral com material
elástico linear contra a solução exata (fórmula da secante), para N/Ne = 0,2, 0,5 e 0,8. Na flexão oblíqua
(envoltória mínima com 2ª ordem, [R12] do núcleo): resultantes por faixas de Gauss x modelo de fibras em grade,
limites alfa -> 0 e alfa -> 90° x flexão normal composta, simetria da seção quadrada, monotonicidade de N_R,
raio do contorno num polígono regular, elipse, refinamento da discretização, semieixos recalculados e pontos da
envoltória resistente refeitos ponta a ponta com fibras e equilíbrio próprio.

Nível 2 — validação normativa: (a) memorial de cálculo de quatro pilares da matriz (V1 a V4), com o
equilíbrio da seção no estado final verificado à mão com o bloco parábola-retângulo e com o diagrama
retangular simplificado; nos dois casos com fluência (V3 e V4), as grandezas da Seção 15.8.4 e o equilíbrio
do pilar-padrão acoplado a diagramas M, N, 1/r são refeitos com expressões independentes e com o modelo de
fibras, inclusive a verificação de que a curvatura de equilíbrio é a menor; (b) exemplos resolvidos de
Bastos (2017, itens 15.2.1, 15.2.2, 16.2.1 e 16.2.3), refeitos pelo motor de cálculo e comparados etapa por
etapa com os valores publicados (Nd, λ, M1d,mín, λ1, αb, método, ν, 1/r, Md,tot e taxa mecânica ω do
ábaco).

Comparações gravadas à parte ("comparacoes" de validacao.json), que não fazem parte da validação (revisão de
escopo de 23/10/2026; no TCC, Seções 4.2 e 4.4): pilar-padrão acoplado a diagramas M, N, 1/r x método geral
(integração numérica da linha elástica ao longo do pilar, Seção 15.8.3.2 da NBR 6118:2023) com a mesma relação
momento-curvatura, com registro da situação de cada análise (convergiu, instável ou não convergiu), da área que o
método geral exige nas mesmas posições das barras e, nos pilares sem equilíbrio, do maior fator de carga com
equilíbrio ([R09]); e curvatura aproximada x rigidez kappa aproximada. O término das análises do método geral é conferido no Nível 1, com o teste
do algoritmo contra a fórmula da secante.

Uso:  python3 validacao.py [pasta_de_saida]   (grava validacao.json e validacao.txt)
"""
from __future__ import annotations

import contextlib
import json
import math
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq

import pilares_nbr6118 as nucleo
import processar_matriz

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — INÍCIO
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [N01] Tolerâncias das verificações e origem de cada uma (critério de [N05]; apresentação como concordância
#   = 100 % - Δ%, [F48]; quadro-resumo com o maior desvio observado em [R06]).
#   TOLERANCIA_EXATA = 1e-9, relativa (concordância mínima 99,9999999 %): calculado e esperado são a mesma
#     grandeza obtida por expressões exatas diferentes (forma fechada da rotina x forma fechada independente,
#     identidade algébrica, simetria, mesma fórmula em outra ordem de operações). A única diferença possível é
#     o arredondamento da aritmética de ponto flutuante de 64 bits (épsilon da máquina 2,2e-16 por operação),
#     que, acumulado nas centenas de operações de cada teste, fica abaixo de 1e-12 (desvios observados da ordem
#     de 1e-15; o maior, 1,0e-12, na diferença central da rigidez tangente, [N07]). O valor 1e-9 é a média
#     geométrica entre o ruído típico (1e-15) e o menor efeito que os testes precisam detectar (1e-3:
#     arredondamentos das referências, como 3,46 no lugar de raiz(12) = 3,4641, e a precisão de apresentação,
#     0,01 cm² em As de 10 cm²): fica ao menos 3 ordens de grandeza acima do maior ruído observado e 6 abaixo de
#     qualquer diferença de modelo ou de arredondamento, que portanto não passaria despercebida. Vale também
#     para o equilíbrio do estado final do memorial (Nível 2), refeito à mão no estado em que a rotina resolveu
#     N(t) = Nd por busca de raiz com xtol = 1e-12 no parâmetro de domínio (TOLERANCIA_PARAMETRO do núcleo): o
#     resíduo de força normal dessa busca é |dN/dt| 1e-12, da ordem de 1e-12 relativo (observado: 9e-14).
#   TOLERANCIA_FIBRAS = 1e-4, relativa (≥ 99,99 %): erro de quadratura da regra do ponto médio dos modelos de
#     fibras independentes ([F06], [F39]). O limite teórico desse erro ([F50]) é de ordem 2 na espessura da
#     fibra, (h/n)², e, nos estados testados com n = NUMERO_FIBRAS = 4000, vale no máximo 9,5e-5 do valor
#     esperado (estado de momento pequeno, M = 61 kN·cm); 1e-4 é esse limite arredondado para cima. A ordem 2 e
#     o limite são conferidos por testes próprios ([N07]).
#   TOLERANCIA_RETANGULAR = 0,05, relativa (≥ 95 %): não é erro numérico. Compara dois diagramas tensão-
#     deformação diferentes, ambos admitidos pela NBR 6118:2023 (17.2.2): para fck <= 50 MPa, o retangular
#     simplificado tem resultante 1,2 % menor (0,8x contra 0,8095x, [F03] e [F04]) aplicada 0,016x mais perto
#     da borda comprimida (0,400x contra 0,416x); o efeito no momento depende do braço de alavanca e é
#     compensado em parte pela armadura. 5 % (cerca de 4 vezes a diferença da resultante) é um limite de
#     plausibilidade entre os dois modelos; o maior desvio observado é de 1,4 %.
#   Variáveis: TOLERANCIA_EXATA, TOLERANCIA_FIBRAS, TOLERANCIA_RETANGULAR.
#   Usada em: nível do módulo (l. 1196, 1199, 1204);
#       nivel_1 (l. 1395, 1398, 1400, 1402, 1412, 1414, 1431, 1449, 1452, 1473, 1475, 1479);
#       nivel_1_flexao_obliqua (l. 1532, 1564, 1578, 1580, 1591, 1593, 1599, 1603, 1632, 1665);
#       nivel_2 (l. 1865, 1867, 1872, 1879, 1882, 1886); nivel_2_bastos (l. 2012, 2017);
#       conferir_mn1r_independente (l. 2097, 2100, 2104, 2117)
TOLERANCIA_EXATA = 1e-9
TOLERANCIA_FIBRAS = 1e-4
TOLERANCIA_RETANGULAR = 0.05

# [N02] Discretização: número de fibras do modelo independente, número de segmentos do pilar no método
#   geral, pontos da curva momento-curvatura do método geral e iterações máximas da linha elástica.
#   Variáveis: NUMERO_FIBRAS, SEGMENTOS_PILAR, PONTOS_CURVA_GERAL, ITERACOES_METODO_GERAL, PRECISAO_FLECHA.
#   Usada em: nível do módulo (l. 937, 938); nivel_1 (l. 1441, 1445, 1461, 1464);
#       nivel_1_metodo_geral_elastico (l. 1678); linha_elastica (l. 1744, 1751, 1757, 1763); metodo_geral (l. 1780);
#       teste_elastico (l. 1840); momento_fibras_com_forca.residuo_forca (l. 2067); momento_fibras_com_forca (l. 2073)
#   Usada também em: validacao_literatura.py: metodo_geral_constante (l. 1179)
NUMERO_FIBRAS = 4000
SEGMENTOS_PILAR = 60
PONTOS_CURVA_GERAL = 600
ITERACOES_METODO_GERAL = 2000
PRECISAO_FLECHA = 1e-9

# [R01] Pilares do memorial de cálculo (Nível 2, verificação do equilíbrio do estado final "à mão"):
#   (identificador, b_w, h, fck, lambda, nu, e1/b_w), um por método de 2ª ordem e faixa de esbeltez, todos
#   viáveis na matriz: V1 curvatura aproximada (lambda 60), V2 curvatura no limite (lambda 90), V3 e V4
#   pilar-padrão acoplado com fluência (lambda 110 e 140).
#   Variáveis: CASOS_MEMORIAL.  Fonte: Seção 3.7 do PP (registro dos valores intermediários).
#   Usada em: nivel_2 (l. 1860); conferir_mn1r_independente (l. 2079)
CASOS_MEMORIAL = (("V1", 30.0, 50.0, 30.0, 60.0, 0.60, 0.30),
                  ("V2", 20.0, 30.0, 25.0, 90.0, 0.40, 0.15),
                  ("V3", 30.0, 50.0, 30.0, 110.0, 0.40, 0.15),
                  ("V4", 40.0, 80.0, 35.0, 140.0, 0.20, 0.15))

# [F01] Força normal resistente da seção na reta b (deformação uniforme eps_c2):
#   N = sigma_pico A_c + A_s min(E_s eps_c2, fyd).  Variáveis: tensao_pico, area_concreto, area_aco, eps_c2, fyd.
#   Fonte: NBR 6118:2023, 17.2.2 (domínio 5, reta b) e 8.3.6.
#   Usada em: nivel_1 (l. 1397)
def forca_compressao_centrada(tensao_pico: float, area_concreto: float, area_aco: float, eps_c2: float,
                              fyd: float) -> float:
    return tensao_pico * area_concreto + area_aco * min(nucleo.MODULO_ACO * eps_c2, fyd)


# [F20] Força normal resistente na reta b com seção líquida (área das barras descontada do concreto, [H05]
#   do núcleo): N = sigma_pico (A_c - A_s) + A_s min(E_s eps_c2, fyd).
#   Variáveis: tensao_pico, area_concreto, area_aco, eps_c2, fyd.  Fonte: NBR 6118:2023, 17.2.2 e 8.3.6.
#   Usada em: nivel_1 (l. 1393)
def forca_compressao_centrada_secao_liquida(tensao_pico: float, area_concreto: float, area_aco: float,
                                            eps_c2: float, fyd: float) -> float:
    return tensao_pico * (area_concreto - area_aco) + area_aco * min(nucleo.MODULO_ACO * eps_c2, fyd)


# [F02] Força normal resistente na reta a (tração uniforme de 10 ‰): N = -A_s fyd.
#   Variáveis: area_aco, fyd.  Fonte: NBR 6118:2023, 17.2.2.
#   Usada em: nivel_1 (l. 1400, 1402)
def forca_tracao_pura(area_aco: float, fyd: float) -> float:
    return -area_aco * fyd


# [F03] Bloco parábola-retângulo (n = 2) com eps_topo = eps_cu e linha neutra x: resultante
#   R_c = beta sigma_pico b x, com beta = 1 - eps_c2 / (3 eps_cu), aplicada a a = (a/x) x do topo, com
#   a/x = 1 - (eps_cu²/2 - eps_c2²/12) / [eps_cu (eps_cu - eps_c2/3)].
#   Variáveis: eps_c2, eps_cu, tensao_pico, largura, linha_neutra.  Fonte: integração de 8.2.10.1 (n = 2).
#   Usada em: resultante_bloco_parabolico (l. 156, 157); nivel_1 (l. 1408); memorial_pilar (l. 1698)
def fator_beta_parabola(eps_c2: float, eps_cu: float) -> float:
    return 1.0 - eps_c2 / (3.0 * eps_cu)


def posicao_relativa_resultante(eps_c2: float, eps_cu: float) -> float:
    return 1.0 - (eps_cu ** 2 / 2.0 - eps_c2 ** 2 / 12.0) / (eps_cu * (eps_cu - eps_c2 / 3.0))


def resultante_bloco_parabolico(eps_c2: float, eps_cu: float, tensao_pico: float, largura: float,
                                linha_neutra: float) -> tuple[float, float]:
    return (fator_beta_parabola(eps_c2, eps_cu) * tensao_pico * largura * linha_neutra,
            posicao_relativa_resultante(eps_c2, eps_cu) * linha_neutra)


# [F04] Diagrama retangular simplificado (fck <= 50 MPa, largura constante): tensão alfa_c eta_c fcd, com
#   alfa_c = 0,85, até y = 0,8 x; R_c = 0,85 eta_c fcd b (0,8 x), aplicada a 0,4 x do topo.
#   Variáveis: fcd, eta_c, largura, linha_neutra.  Fonte: NBR 6118:2023, 17.2.2 e).
#   Usada em: memorial_pilar (l. 1731)
def resultante_bloco_retangular(fcd: float, eta_c: float, largura: float, linha_neutra: float) -> tuple[float, float]:
    return 0.85 * eta_c * fcd * largura * 0.8 * linha_neutra, 0.4 * linha_neutra


# [F05] Equilíbrio "à mão" de uma seção com camadas de armadura e resultante de concreto conhecida:
#   eps_si = eps_cu (x - y_i)/x;  N_R = R_c + soma[A_si (sigma_si - sigma_c,i)];
#   M_R = R_c (h/2 - a) + soma[A_si (sigma_si - sigma_c,i) (h/2 - y_i)], em que sigma_c,i é a tensão do
#   concreto deslocado pela barra i (seção líquida, [F26] e [F27]; zero na seção bruta).
#   Variáveis: resultante_concreto, posicao_resultante (a), linha_neutra, eps_cu, altura, profundidades,
#   areas_camadas, fyd, tensoes_concreto_barras.  Fonte: NBR 6118:2023, 17.2.2; [H05] do núcleo.
#   Usada em: memorial_pilar (l. 1705, 1736)
def equilibrio_manual(resultante_concreto: float, posicao_resultante: float, linha_neutra: float, eps_cu: float,
                      altura: float, profundidades: np.ndarray, areas_camadas: np.ndarray,
                      fyd: float, tensoes_concreto_barras: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    deformacoes = deformacao_barras_estado_final(eps_cu, linha_neutra, profundidades)
    tensoes = np.clip(nucleo.MODULO_ACO * deformacoes, -fyd, fyd)
    forcas = areas_camadas * (tensoes - tensoes_concreto_barras)
    forca_normal = resultante_concreto + forcas.sum()
    momento = resultante_concreto * (altura / 2.0 - posicao_resultante) + (forcas * (altura / 2.0 - profundidades)).sum()
    return forca_normal, momento, deformacoes, tensoes


# [F29] Deformação das barras no estado final com eps_topo = eps_cu (domínios 2 a 4): eps_si = eps_cu (x - y_i)/x.
#   Variáveis: eps_cu, linha_neutra (x), profundidades (y_i).  Fonte: NBR 6118:2023, 17.2.2 a).
#   Usada em: equilibrio_manual (l. 178); memorial_pilar (l. 1701)
def deformacao_barras_estado_final(eps_cu: float, linha_neutra: float, profundidades: np.ndarray) -> np.ndarray:
    return eps_cu * (linha_neutra - profundidades) / linha_neutra


# [F26] Tensão do concreto na posição de cada barra, diagrama parábola-retângulo (8.2.10.1), para o desconto do
#   concreto deslocado: sigma_c = sigma_pico [1 - (1 - eps/eps_c2)^n] para 0 <= eps <= eps_c2; sigma_pico para
#   eps > eps_c2; zero na tração.  Variáveis: deformacoes, tensao_pico, eps_c2, expoente.
#   Fonte: NBR 6118:2023, 8.2.10.1; [H05] do núcleo.
#   Usada em: memorial_pilar (l. 1702)
def tensao_concreto_barras_parabola(deformacoes: np.ndarray, tensao_pico: float, eps_c2: float,
                                    expoente: float) -> np.ndarray:
    limitada = np.clip(deformacoes, 0.0, eps_c2)
    return tensao_pico * (1.0 - (1.0 - limitada / eps_c2) ** expoente)


# [F27] Tensão do concreto na posição de cada barra, diagrama retangular simplificado: 0,85 eta_c fcd até
#   y = 0,8 x e zero abaixo.  Variáveis: fcd, eta_c, linha_neutra, profundidades.  Fonte: NBR 6118:2023, 17.2.2 e).
#   Usada em: memorial_pilar (l. 1733)
def tensao_concreto_barras_retangular(fcd: float, eta_c: float, linha_neutra: float,
                                      profundidades: np.ndarray) -> np.ndarray:
    return np.where(profundidades <= 0.8 * linha_neutra, 0.85 * eta_c * fcd, 0.0)


# [F28] Força do concreto deslocado pelas barras: soma(A_si sigma_c,i).  Variáveis: areas_camadas, tensoes.
#   Usada em: memorial_pilar (l. 1725)
def forca_concreto_deslocado(areas_camadas: np.ndarray, tensoes_concreto_barras: np.ndarray) -> float:
    return float((areas_camadas * tensoes_concreto_barras).sum())


# [F06] Modelo de fibras independente (regra do ponto médio, n fibras de espessura dy = h/n):
#   N = soma[sigma_c(eps_j) b dy] + soma[A_si (sigma_si - d sigma_c(eps_si))];
#   M = soma[sigma_c(eps_j) b dy (h/2 - y_j)] + soma[A_si (sigma_si - d sigma_c(eps_si)) (h/2 - y_i)],
#   com d = 1 na seção líquida (concreto deslocado pelas barras descontado) e d = 0 na seção bruta.
#   Variáveis: deformacao_topo, curvatura, altura, largura, tensao_pico, eps_c2, expoente, profundidades,
#   areas_camadas, fyd, numero_fibras, descontar_area_barras.  Fonte: NBR 6118:2023, 17.2.2; [H05] do núcleo.
#   Usada em: nivel_1 (l. 1439, 1457); momento_fibras_com_forca.residuo_forca (l. 2065);
#       momento_fibras_com_forca (l. 2072)
def esforcos_por_fibras(deformacao_topo: float, curvatura: float, altura: float, largura: float, tensao_pico: float,
                        eps_c2: float, expoente: float, profundidades: np.ndarray, areas_camadas: np.ndarray,
                        fyd: float, numero_fibras: int, descontar_area_barras: bool) -> tuple[float, float]:
    espessura = altura / numero_fibras
    centros = (np.arange(numero_fibras) + 0.5) * espessura
    deformacoes = deformacao_topo - curvatura * centros
    parabola = np.clip(deformacoes, 0.0, eps_c2)
    tensoes = tensao_pico * (1.0 - (1.0 - parabola / eps_c2) ** expoente)
    forcas_concreto = tensoes * largura * espessura
    deformacoes_aco = deformacao_topo - curvatura * profundidades
    tensoes_aco = np.clip(nucleo.MODULO_ACO * deformacoes_aco, -fyd, fyd)
    if descontar_area_barras:
        parabola_aco = np.clip(deformacoes_aco, 0.0, eps_c2)
        tensoes_aco = tensoes_aco - tensao_pico * (1.0 - (1.0 - parabola_aco / eps_c2) ** expoente)
    forcas_aco = areas_camadas * tensoes_aco
    forca = forcas_concreto.sum() + forcas_aco.sum()
    momento = (forcas_concreto * (altura / 2.0 - centros)).sum() + (forcas_aco * (altura / 2.0 - profundidades)).sum()
    return float(forca), float(momento)


# [F07] Rigidez à flexão tangente para deformação uniforme eps_0 (concreto e aço em regime não fissurado):
#   seção bruta: EI = E_t b h³/12 + E_s soma[A_si (y_i - h/2)²];
#   seção líquida: EI = E_t [b h³/12 - soma(A_si (y_i - h/2)²)] + E_s soma[A_si (y_i - h/2)²],
#   com E_t = sigma_pico n/eps_c2 (1 - eps_0/eps_c2)^(n-1).
#   Variáveis: deformacao_uniforme, tensao_pico, eps_c2, expoente, largura, altura, profundidades, areas_camadas,
#   descontar_area_barras.
#   Fonte: derivada do diagrama de 8.2.10.1 e da Lei de Hooke (8.3.6); [H05] do núcleo.
#   Usada em: nivel_1 (l. 1427)
def rigidez_tangente_inicial(deformacao_uniforme: float, tensao_pico: float, eps_c2: float, expoente: float,
                             largura: float, altura: float, profundidades: np.ndarray,
                             areas_camadas: np.ndarray, descontar_area_barras: bool) -> float:
    modulo_tangente = tensao_pico * expoente / eps_c2 * (1.0 - deformacao_uniforme / eps_c2) ** (expoente - 1.0)
    inercia_barras = (areas_camadas * (profundidades - altura / 2.0) ** 2).sum()
    inercia_concreto = largura * altura ** 3 / 12.0 - (inercia_barras if descontar_area_barras else 0.0)
    return modulo_tangente * inercia_concreto + nucleo.MODULO_ACO * inercia_barras


# [F08] Pequeno incremento de curvatura em torno do centro da seção: eps_topo = eps_0 + kappa h/2.
#   Variáveis: deformacao_uniforme, curvatura, altura.
#   Usada em: nivel_1 (l. 1420, 1422)
def deformacao_topo_centrada(deformacao_uniforme: float, curvatura: float, altura: float) -> float:
    return deformacao_uniforme + curvatura * altura / 2.0


# [F11] Derivada por diferença central: dM/dkappa ≈ [M(+kappa) - M(-kappa)] / (2 kappa), que elimina o termo de
#   2ª ordem do erro de truncamento da diferença simples M/kappa.
#   Variáveis: momento_positivo, momento_negativo, curvatura.
#   Usada em: nivel_1 (l. 1426)
def derivada_central(momento_positivo: float, momento_negativo: float, curvatura: float) -> float:
    return (momento_positivo - momento_negativo) / (2.0 * curvatura)


# [F09] Método da rigidez kappa aproximada (pilar-padrão), solução fechada da equação do 2º grau:
#   a M² + b M + c = 0, a = 5 h; b = h² Nd - Nd l_e²/320 - 5 h alfa_b M1d,A; c = -Nd h² alfa_b M1d,A;
#   M_d,tot = [-b + raiz(b² - 4ac)] / (2a) >= M1d,A.
#   Variáveis: altura, forca_normal_calculo, comprimento_equivalente, alfa_b, momento_primeira_ordem.
#   Fonte: NBR 6118:2023, 15.8.3.3.3.
#   Usada em: nivel_2 (l. 1908); nivel_2_bastos (l. 2000)
def momento_total_rigidez_kappa(altura: float, forca_normal_calculo: float, comprimento_equivalente: float,
                                alfa_b: float, momento_primeira_ordem: float) -> float:
    coeficiente_a = 5.0 * altura
    coeficiente_b = (altura ** 2 * forca_normal_calculo - forca_normal_calculo * comprimento_equivalente ** 2 / 320.0
                     - 5.0 * altura * alfa_b * momento_primeira_ordem)
    coeficiente_c = -forca_normal_calculo * altura ** 2 * alfa_b * momento_primeira_ordem
    raiz = (-coeficiente_b + math.sqrt(coeficiente_b ** 2 - 4.0 * coeficiente_a * coeficiente_c)) / (2.0 * coeficiente_a)
    return max(raiz, momento_primeira_ordem)


# [F10] Método geral (linha elástica por diferenças finitas) com a formulação gama_f3: em cada nó i,
#   M_ext,i = M1d,A + Nd e_i; kappa_i = kappa_def(M_ext,i / gama_f3) pela curva de deformação;
#   e_(i-1) - 2 e_i + e_(i+1) = -kappa_i dx², e_0 = e_n = 0; M_d,tot = M1d,A + Nd e_max.
#   Variáveis: momento_primeira_ordem, forca_normal_calculo, flechas, segmentos, comprimento.
#   Fonte: NBR 6118:2023, 15.8.3.2 e 15.3.1.
#   Usada em: linha_elastica (l. 1748, 1749, 1752, 1756, 1757, 1760, 1762);
#       metodo_geral.curvatura_inelastica (l. 1784)
#   Usada também em: validacao_literatura.py: metodo_geral_constante.curvatura_inelastica (l. 1185)
def momentos_externos(momento_primeira_ordem: float, forca_normal_calculo: float, flechas: np.ndarray) -> np.ndarray:
    return momento_primeira_ordem + forca_normal_calculo * flechas


def momentos_reduzidos(momentos: np.ndarray) -> np.ndarray:
    return momentos / nucleo.GAMA_F3


def passo_segmento(comprimento: float, segmentos: int) -> float:
    return comprimento / segmentos


def matriz_banda_segunda_derivada(nos_internos: int) -> np.ndarray:
    banda = np.zeros((3, nos_internos))
    banda[0, 1:] = 1.0
    banda[1, :] = -2.0
    banda[2, :-1] = 1.0
    return banda


def termo_independente(curvaturas: np.ndarray, passo: float) -> np.ndarray:
    return -curvaturas * passo ** 2


def diferenca_maxima(valores_novos: np.ndarray, valores_anteriores: np.ndarray) -> float:
    return float(np.max(np.abs(valores_novos - valores_anteriores)))


# [R02] Situações do método geral ([F10]): "convergiu" (maior variação das flechas entre duas iterações menor
#   que PRECISAO_FLECHA), "instável" (momento reduzido acima do pico da curva M-1/r: não há configuração de
#   equilíbrio) e "não convergiu" (ITERACOES_METODO_GERAL esgotadas sem atingir a precisão; o resultado
#   não é aceito como valor do método geral).
#   Variáveis: SITUACAO_CONVERGIU, SITUACAO_INSTAVEL, SITUACAO_NAO_CONVERGIU.
#   Usada em: linha_elastica (l. 1754, 1761, 1762); atende_metodo_geral (l. 1809); sem_equilibrio (l. 1828);
#       nivel_2 (l. 1932, 1933, 1939, 1944, 1945); main (l. 2232)
SITUACAO_CONVERGIU = "convergiu"
SITUACAO_INSTAVEL = "instável"
SITUACAO_NAO_CONVERGIU = "não convergiu"

# [R03] Teste elástico do método geral (Nível 1): pilar biapoiado de material elástico linear com momentos
#   iguais M0 nas extremidades (momento de 1ª ordem constante, como na matriz), para três razões N/N_e.
#   Variáveis: RIGIDEZ_TESTE_ELASTICO (EI, kN·cm²), COMPRIMENTO_TESTE_ELASTICO (cm),
#   MOMENTO_TESTE_ELASTICO (M0, kN·cm), RAZOES_CARGA_CRITICA (N/N_e), TOLERANCIA_SECANTE.
#   TOLERANCIA_SECANTE = 1e-3, relativa (concordância mínima 99,9 %): erro de discretização das diferenças
#     finitas centrais com SEGMENTOS_PILAR = 60 segmentos de comprimento Δ = ℓ/60. O operador discreto tem carga
#     crítica (πΔ/ℓ)²/12 = 2,3e-4 menor que a de Euler, e a fórmula da secante amplifica esse erro pelo fator
#     θ tg θ / 2, θ = (π/2) raiz(N/Ne) ([F51]): estimativas de 6,8e-5, 2,6e-4 e 9,6e-4 para N/Ne = 0,2, 0,5
#     e 0,8; 1e-3 é a estimativa do caso mais severo arredondada para cima (observados: 1,4e-5, 1,3e-4 e
#     7,7e-4). A ordem 2 é conferida com o dobro de segmentos ([N07]).
#   Usada em: nível do módulo (l. 1202); nivel_1_metodo_geral_elastico (l. 1670, 1675);
#       teste_elastico (l. 1842, 1845, 1846, 1849, 1850, 1851, 1854, 1855, 1856)
RIGIDEZ_TESTE_ELASTICO = 1.0e7
COMPRIMENTO_TESTE_ELASTICO = 400.0
MOMENTO_TESTE_ELASTICO = 1000.0
RAZOES_CARGA_CRITICA = (0.2, 0.5, 0.8)
TOLERANCIA_SECANTE = 1e-3


# [F21] Curvatura de material elástico linear: 1/r = M / EI.  Variáveis: momentos, rigidez (EI).
#   Usada em: teste_elastico (l. 1846)
def curvatura_elastica(momentos: np.ndarray, rigidez: float) -> np.ndarray:
    return momentos / rigidez


# [F22] Força normal crítica de Euler do pilar biapoiado: N_e = pi² EI / l².
#   Variáveis: rigidez (EI), comprimento (l).  Fonte: teoria da estabilidade elástica (Euler).
#   Usada em: teste_elastico (l. 1842)
def carga_critica_euler(rigidez: float, comprimento: float) -> float:
    return math.pi ** 2 * rigidez / comprimento ** 2


# [F23] Força normal do teste a partir da razão N/N_e: N = (N/N_e) N_e.  Variáveis: razao, carga_critica.
#   Usada em: teste_elastico (l. 1843)
def forca_da_razao(razao: float, carga_critica: float) -> float:
    return razao * carga_critica


# [F24] Fórmula da secante (solução exata da linha elástica do pilar biapoiado elástico com momentos iguais M0
#   nas extremidades): M_max = M0 / cos[(pi/2) raiz(N/N_e)].
#   Variáveis: momento_extremidade (M0), forca_normal (N), carga_critica (N_e).
#   Fonte: teoria da flexo-compressão elástica (fórmula da secante).
#   Usada em: teste_elastico (l. 1851)
def momento_maximo_secante(momento_extremidade: float, forca_normal: float, carga_critica: float) -> float:
    return momento_extremidade / math.cos(math.pi / 2.0 * math.sqrt(forca_normal / carga_critica))


# [F25] Pilar-padrão com material elástico (deformada senoidal da NBR 6118:2023, 15.8.3.3.2:
#   e2 = (l²/10) 1/r, com 1/r = M_tot / EI): M_tot = M0 / [1 - N l² / (10 EI)].
#   Variáveis: momento_extremidade (M0), forca_normal (N), comprimento (l), rigidez (EI).
#   Usada em: teste_elastico (l. 1854)
def momento_total_pilar_padrao_elastico(momento_extremidade: float, forca_normal: float, comprimento: float,
                                        rigidez: float) -> float:
    return momento_extremidade / (1.0 - forca_normal * comprimento ** 2 / (10.0 * rigidez))


# [F12] Profundidade da linha neutra a partir do estado de deformação: x = eps_topo / k.
#   Variáveis: deformacao_topo, curvatura.  Fonte: hipótese das seções planas (NBR 6118:2023, 17.2.2).
#   Usada em: memorial_pilar (l. 1697)
def profundidade_linha_neutra(deformacao_topo: float, curvatura: float) -> float:
    return deformacao_topo / curvatura


# [F13] Curvatura com o concreto em eps_cu no topo e linha neutra x: k = eps_cu / x.
#   Variáveis: eps_cu, linha_neutra.  Fonte: domínios 3 a 4a (NBR 6118:2023, 17.2.2).
#   Usada em: nivel_1 (l. 1404)
def curvatura_por_linha_neutra(eps_cu: float, linha_neutra: float) -> float:
    return eps_cu / linha_neutra


# [F14] Linhas neutras dos testes do bloco de compressão: x = f h, com f em FRACOES_LINHA_NEUTRA.
#   Variáveis: altura, FRACOES_LINHA_NEUTRA.
#   Usada em: linhas_neutras_teste (l. 418); nivel_1 (l. 1403)
FRACOES_LINHA_NEUTRA = (0.35, 0.70, 1.00)


def linhas_neutras_teste(altura: float) -> tuple[float, ...]:
    return tuple(fracao * altura for fracao in FRACOES_LINHA_NEUTRA)


# [F15] Profundidades de duas camadas simétricas de armadura (duas barras por camada): d' e h - d'.
#   Variáveis: altura, distancia (d').
#   Usada em: secao_duas_camadas (l. 1362)
def profundidades_duas_camadas(altura: float, distancia: float) -> np.ndarray:
    return np.array([distancia, distancia, altura - distancia, altura - distancia])


# [F16] Momento, em relação ao centro da seção, de uma resultante R aplicada à profundidade a: R (h/2 - a).
#   Variáveis: resultante, altura, posicao.
#   Usada em: nivel_1 (l. 1414)
def momento_resultante_centro(resultante: float, altura: float, posicao: float) -> float:
    return resultante * (altura / 2.0 - posicao)


# [F17] Curvatura de sinal oposto (diferença central de [F11]): -k.  Variáveis: curvatura.
#   Usada em: nivel_1 (l. 1423)
def curvatura_oposta(curvatura: float) -> float:
    return -curvatura


# [F18] Limite de decréscimo admitido em N_R(t) por arredondamento: -tol max|N_R|.
#   Variáveis: forcas, tolerancia.
#   Usada em: nivel_1 (l. 1479); nivel_1_flexao_obliqua (l. 1564)
def limite_decrescimo(forcas, tolerancia: float) -> float:
    return -tolerancia * float(np.abs(np.asarray(forcas)).max())


# [F19] Fator de tolerância do critério de aprovação: 1 + tol (Delta% de referência entre 1 + tol e 1).
#   Variáveis: tolerancia.
#   Usada em: registrar (l. 1351)
def fator_tolerancia(tolerancia: float) -> float:
    return 1.0 + tolerancia


# [N03] Semente do gerador pseudoaleatório das amostras de domínio (reprodutível): fck x h (inteiro).
#   Variáveis: fck, altura.
#   Usada em: nivel_1 (l. 1433)
def semente_aleatoria(fck: float, altura: float) -> int:
    return int(fck * altura)

# [N04] Engenharia reversa dos exemplos de Bastos (2017) e comparação da leitura do ábaco (Nível 2).
#   Regras do passo a passo do autor, identificadas nos itens 15.2.1 a 16.2.3: λ = 3,46 ℓe/h (RAIZ_12_BASTOS no
#   lugar de raiz(12) = 3,4641...); cada valor é impresso com poucas casas (λ com uma, 48,44 -> 48,4; ν e μ com
#   duas, 0,777 -> 0,78; 1/r com 5 algarismos significativos nos itens 15.2.x e 4 nos itens 16.2.x) e é o valor
#   IMPRESSO que entra na etapa seguinte (λ impresso na Eq. 41; ν impresso em 1/r; 1/r impresso em Md,tot; ν e μ
#   impressos na entrada do ábaco). Um valor impresso é reproduzido pela cadeia do autor ([F49]) quando ela,
#   arredondada às casas impressas com a meia unidade para cima, em módulo (o mesmo critério de [N01] de
#   validacao_literatura.py), dá o número impresso: |impresso| - m (1 + FOLGA_PONTO_FLUTUANTE) <= |cadeia| <
#   |impresso| + m (1 - FOLGA_PONTO_FLUTUANTE), com o mesmo sinal, em que m = FRACAO_RESOLUCAO x resolução (meia
#   unidade). A meia unidade exata acima do impresso não reproduz (é truncamento); a exata abaixo reproduz.
#   FOLGA_PONTO_FLUTUANTE (relativa) absorve o erro de representação binária dos valores decimais.
#   SECAO_LIQUIDA_ABACO = False: os ábacos de Venturini e Rodrigues (1987) são calculados com a seção bruta: nas
#   equações de equilíbrio da apostila (eq. 4.6, p. 14; p. 16 do arquivo), N_d = b_w ∫ σc dy + Σ A_si σ_si, com o
#   concreto integrado em toda a largura, sem descontar a área das barras. A comparação do ω usa, portanto, a seção
#   bruta, a mesma hipótese do ábaco (como o arranjo, [R04]); o ω com a seção líquida, adotada na rotina ([H05] do
#   núcleo), é informado ao lado.
#   ω lido no ábaco (critério "leitura"): comparado só como informação, pela diferença |ω_autor - ω_calculado|, sem
#   limite de reprodução (item 1 do arquivo "Pessoal 2" do autor: nenhum limite de aceitação sem estudo que o
#   justifique). Como escala, informa-se o espaçamento das curvas de ω dos ábacos A-4, A-27 e A-29, Δω = 0,10
#   (ESPACAMENTO_CURVAS_ABACO, conferido nas páginas dos ábacos, p. 41, 64 e 66 do arquivo). A comparação é feita no
#   próprio ponto (ν, μ) em que o autor entrou no ábaco, com a seção no arranjo e na hipótese do ábaco, de modo que
#   só resta a leitura gráfica; o próprio autor lê ω = 0,30 em (ν = 0,77; μ = 0,12), item 15.2.1, e ω = 0,33 em
#   (ν = 0,78; μ = 0,12), item 16.2.1, pontos em que o ω calculado varia só cerca de 0,01.
#   Variáveis: RAIZ_12_BASTOS, FRACAO_RESOLUCAO, FOLGA_PONTO_FLUTUANTE, SECAO_LIQUIDA_ABACO, ESPACAMENTO_CURVAS_ABACO.
#   Fonte: Bastos (2017, p. 33-54); Venturini e Rodrigues (1987), eq. 4.6 (p. 14) e ábacos A-4, A-27 e A-29.
#   Usada em: limite_reproducao (l. 1060); limite_inferior_reproducao (l. 1064); limite_superior_reproducao (l. 1068);
#       esbeltez_bastos (l. 1096); nivel_2_bastos (l. 2039)
RAIZ_12_BASTOS = 3.46
FRACAO_RESOLUCAO = 0.5
FOLGA_PONTO_FLUTUANTE = 1e-9
SECAO_LIQUIDA_ABACO = False
ESPACAMENTO_CURVAS_ABACO = 0.10

# [R04] Exemplos resolvidos de Bastos (2017, itens 15.2.1, 15.2.2, 16.2.1 e 16.2.3, p. 33-54): pilares
#   biapoiados de nós fixos, C20, CA-50, γc = γf = 1,4, γs = 1,15, ℓe = 280 cm e d' = 4,0 cm. Cada linha é uma
#   direção de flexão: largura (dimensão perpendicular à flexão), altura (na direção da flexão), M1d,A aplicado
#   (kN·cm; 0 no pilar intermediário) e, em "impresso", cada valor do passo a passo COMO IMPRESSO no texto
#   (ponto como separador decimal; o número de casas impressas define a resolução de [F49]). Nd é o dado de
#   partida: o autor usa Nd (1.100, 1.500, 1.554 e 700 kN) em todas as etapas e imprime Nk = Nd/1,4 arredondado
#   (785,7 = 1.100/1,4; 1,4 x 785,7 = 1.099,98). eq41_b e eq41_c são os coeficientes impressos da Eq. 41 (rigidez
#   κ aproximada, coeficiente de Md² = 19200). A armadura é comparada pela taxa mecânica ω, com a seção montada
#   no arranjo do ábaco que o autor usou ("abaco", "arranjo"): nos itens 15.2.1, 15.2.2 e 16.2.1, ábaco A-4
#   (d'/h = 0,20, duas camadas iguais nas faces perpendiculares à flexão, barras ao longo do lado maior, como
#   Bastos descreve na p. 36); no item 16.2.3, ábacos A-29 (direção de 20 cm, d'/h = 0,20) e A-27 (direção de
#   40 cm, d'/h = 0,10), ambos com 8 barras iguais, 3 por face nas quatro faces (cantos e meio de cada face):
#   um mesmo arranjo físico no perímetro serve às duas direções ([F54]). O item 16.2.3 tem a mesma configuração da
#   matriz deste trabalho: e1 = 10 cm na direção de 40 cm (2ª ordem dispensada) e só o momento mínimo, com 2ª
#   ordem, na direção de 20 cm; ν impresso (0,61) vale para as duas direções.
#   (Versão anterior desta nota: dizia que os ábacos A-27 e A-29 não tinham o arranjo descrito no texto, e o ω do
#   item 16.2.3 não era comparado; conferidas as páginas dos ábacos, são ábacos de perímetro, e o texto não descreve
#   outro arranjo.)
#   Variáveis: DISTANCIA_BASTOS, FCK_BASTOS, COMPRIMENTO_BASTOS, ARRANJO_DUAS_CAMADAS_BASTOS,
#   ARRANJO_TRES_POR_FACE_BASTOS, EXEMPLOS_BASTOS.
#   Fonte: Bastos (2017), páginas indicadas em cada linha.
#   Usada em: nível do módulo (l. 525, 531, 537, 544, 548); fcd_bastos (l. 1088);
#       cadeia_exata_bastos (l. 1124, 1125, 1129); cadeia_autor_bastos (l. 1143, 1151);
#       secao_exemplo_bastos (l. 1973, 1975, 1976); nivel_2_bastos (l. 1987, 1988, 1996, 2000)
DISTANCIA_BASTOS = 4.0
FCK_BASTOS = 20.0
COMPRIMENTO_BASTOS = 280.0
ARRANJO_DUAS_CAMADAS_BASTOS = "duas camadas (A-1 a A-5)"
ARRANJO_TRES_POR_FACE_BASTOS = "8 barras, 3 por face nas quatro faces (A-26 a A-29)"
EXEMPLOS_BASTOS = (
    {"exemplo": "15.2.1", "paginas": "p. 33-37", "largura": 50.0, "altura": 20.0, "M1d_aplicado": 0.0,
     "metodo": nucleo.METODO_CURVATURA, "abaco": "A-4", "arranjo": ARRANJO_DUAS_CAMADAS_BASTOS,
     "impresso": {"Nd": "1100", "Nk": "785.7", "lambda": "48.4", "M1d_min": "2310", "lambda1": "35", "alfa_b": "1.0",
                  "nu": "0.77", "curvatura": "1.9685e-4", "Md_curvatura": "4008", "eq41_b": "-11408320",
                  "eq41_c": "-1.951488e11", "Md_kappa": "3500", "mu_curvatura": "0.14", "mu_kappa": "0.12",
                  "omega_curvatura": "0.38", "omega_kappa": "0.30"}},
    {"exemplo": "15.2.2", "paginas": "p. 37-40", "largura": 50.0, "altura": 20.0, "M1d_aplicado": 0.0,
     "metodo": nucleo.METODO_CURVATURA, "abaco": "A-4", "arranjo": ARRANJO_DUAS_CAMADAS_BASTOS,
     "impresso": {"Nd": "1500", "Nk": "1071", "lambda": "48.4", "M1d_min": "3150", "lambda1": "35", "alfa_b": "1.0",
                  "nu": "1.05", "curvatura": "1.6129e-4", "Md_curvatura": "5047", "eq41_b": "-15556800",
                  "eq41_c": "-3.6288e11", "Md_kappa": "4771", "mu_curvatura": "0.18", "mu_kappa": "0.17",
                  "omega_curvatura": "0.78", "omega_kappa": "0.76"}},
    {"exemplo": "16.2.1", "paginas": "p. 41-46", "largura": 70.0, "altura": 20.0, "M1d_aplicado": 2170.0,
     "metodo": nucleo.METODO_CURVATURA, "abaco": "A-4", "arranjo": ARRANJO_DUAS_CAMADAS_BASTOS,
     "impresso": {"Nd": "1554", "Nk": "1110", "lambda": "48.4", "M1d_min": "3263.4", "lambda1": "35", "alfa_b": "1.0",
                  "nu": "0.78", "curvatura": "1.953e-4", "Md_curvatura": "5642.8", "eq41_b": "-16116845",
                  "eq41_c": "-389477652480", "Md_kappa": "4943.1", "mu_curvatura": "0.14", "mu_kappa": "0.12",
                  "omega_curvatura": "0.40", "omega_kappa": "0.33"}},
    {"exemplo": "16.2.3 (direção de 40 cm)", "paginas": "p. 51-54", "largura": 20.0, "altura": 40.0,
     "M1d_aplicado": 7000.0, "metodo": nucleo.METODO_DISPENSADO, "abaco": "A-27",
     "arranjo": ARRANJO_TRES_POR_FACE_BASTOS,
     "impresso": {"Nd": "700", "Nk": "500", "lambda": "24.2", "M1d_min": "1890.0", "lambda1": "35", "alfa_b": "1.0",
                  "nu": "0.61", "Md_curvatura": "7000.0", "mu_curvatura": "0.15", "omega_curvatura": "0.28"}},
    {"exemplo": "16.2.3 (direção de 20 cm)", "paginas": "p. 51-54", "largura": 40.0, "altura": 20.0,
     "M1d_aplicado": 0.0, "metodo": nucleo.METODO_CURVATURA, "abaco": "A-29", "arranjo": ARRANJO_TRES_POR_FACE_BASTOS,
     "impresso": {"Nd": "700", "Nk": "500", "lambda": "48.4", "M1d_min": "1470.0", "lambda1": "35", "alfa_b": "1.0",
                  "nu": "0.61", "curvatura": "0.0002252", "Md_curvatura": "2705.9", "mu_curvatura": "0.12",
                  "omega_curvatura": "0.20"}},
)

# [R08] Etapas da engenharia reversa de Bastos, na ordem do texto: (chave de [R04], rótulo, unidade, critério
#   de reprodução: "impressão" = arredondamento às casas impressas ([N04]), "leitura" = ω lido no ábaco, sem
#   critério de reprodução, só a diferença) e
#   observações sobre valores específicos (exemplo, chave).
#   ETAPAS_ROTINA_BASTOS: etapas que a rotina calcula e que são testadas contra a cadeia exata; PARES_OMEGA_BASTOS:
#   (ω, μ e Md,tot correspondentes) da curvatura aproximada e da rigidez κ.
#   Variáveis: ETAPAS_BASTOS, ETAPAS_ROTINA_BASTOS, PARES_OMEGA_BASTOS, OBSERVACOES_BASTOS.
#   Usada em: linha_engenharia_reversa (l. 1177); nivel_2_bastos (l. 2008, 2009, 2025, 2050)
ETAPAS_ROTINA_BASTOS = ("lambda", "M1d_min", "lambda1", "alfa_b", "nu", "curvatura", "Md_curvatura")
PARES_OMEGA_BASTOS = (("omega_curvatura", "mu_curvatura", "Md_curvatura"), ("omega_kappa", "mu_kappa", "Md_kappa"))
ETAPAS_BASTOS = (("Nk", "Nk = Nd/(γn γf)", "kN", "impressão"), ("lambda", "λ", "", "impressão"),
                 ("M1d_min", "M1d,mín", "kN·cm", "impressão"), ("lambda1", "λ1", "", "impressão"),
                 ("alfa_b", "αb", "", "impressão"), ("nu", "ν", "", "impressão"),
                 ("curvatura", "1/r (curvatura aproximada)", "1/cm", "impressão"),
                 ("Md_curvatura", "Md,tot (curvatura aproximada ou 1ª ordem)", "kN·cm", "impressão"),
                 ("eq41_b", "Eq. 41: coeficiente de Md", "", "impressão"),
                 ("eq41_c", "Eq. 41: termo independente", "", "impressão"),
                 ("Md_kappa", "Md,tot (rigidez κ aproximada)", "kN·cm", "impressão"),
                 ("mu_curvatura", "μ com Md,tot da curvatura", "", "impressão"),
                 ("mu_kappa", "μ com Md,tot da rigidez κ", "", "impressão"),
                 ("omega_curvatura", "ω do ábaco com Md,tot da curvatura (ou de 1ª ordem)", "", "leitura"),
                 ("omega_kappa", "ω do ábaco com Md,tot da rigidez κ", "", "leitura"))
OBSERVACOES_BASTOS = {
    ("15.2.1", "Nk"): "O exemplo parte de Nd = 1.100 kN (usado em todas as etapas); Nk impresso = 1.100/1,4 "
                      "arredondado. A conta impressa 1,4 x 785,7 = 1.100 dá 1.099,98.",
    ("15.2.2", "Nk"): "O exemplo parte de Nd = 1.500 kN (usado em todas as etapas); Nk impresso = 1.500/1,4 "
                      "arredondado. A conta impressa 1,4 x 1.071 = 1.500 dá 1.499,4.",
    ("16.2.1", "omega_kappa"): "Leitura do autor incoerente com a do item 15.2.1: ω = 0,33 em (ν = 0,78; μ = 0,12) "
                               "e ω = 0,30 em (ν = 0,77; μ = 0,12), pontos em que o ω calculado difere só cerca de "
                               "0,01 ([N04]).",
    ("15.2.1", "Md_kappa"): "A raiz positiva da equação impressa pelo autor (19200 Md² - 11.408.320 Md - "
                            "1,951488e11 = 0, ou Md² - 594,2 Md - 10.164.000 = 0) é 3.499,0; o texto imprime 3.500, "
                            "arredondamento do resultado final a 3 algarismos significativos (diferença de "
                            "1,0 kN·cm, 0,028 %).",
}


# [F30] Coeficiente adicional γn de pilares com menor dimensão b entre 14 cm e 19 cm: γn = 1,95 - 0,05 b; para
#   b >= 19 cm, γn = 1,00 (Bastos aplica Nd = γn γf Nk; nos exemplos, b = 20 cm e γn = 1,00).
#   Variáveis: menor_dimensao (cm).  Fonte: NBR 6118:2023, 13.2.3, Tabela 13.1.
#   Usada em: forca_caracteristica_bastos (l. 603)
def coeficiente_gama_n(menor_dimensao: float) -> float:
    return 1.95 - 0.05 * min(menor_dimensao, 19.0)


# [F31] Força normal característica impressa por Bastos a partir do Nd de partida: Nk = Nd / (γn γf).
#   Variáveis: forca_normal_calculo (Nd), menor_dimensao, gama_f.  Fonte: NBR 6118:2023, 11.7.1 e 13.2.3.
#   Usada em: cadeia_exata_bastos (l. 1133)
def forca_caracteristica_bastos(forca_normal_calculo: float, menor_dimensao: float, gama_f: float) -> float:
    return forca_normal_calculo / (coeficiente_gama_n(menor_dimensao) * gama_f)


# [F32] Taxa mecânica de armadura: ω = As fyd / (Ac fcd) (grandeza lida nos ábacos de Venturini).
#   Variáveis: area_aco, fyd, area_concreto, fcd.  Fonte: Bastos (2017, Eq. 56).
#   Usada em: nivel_2_bastos (l. 2037)
def taxa_mecanica(area_aco: float, fyd: float, area_concreto: float, fcd: float) -> float:
    return area_aco * fyd / (area_concreto * fcd)


# [F33] Conferência independente do pilar-padrão acoplado a diagramas M, N, 1/r (V3 e V4 do memorial):
#   tensão de pico da curva de deformação 1,10 fcd = 1,10 fck/γc; força normal dessa curva Nd/γf3;
#   momento externo M_ext(1/r) = αb M1d,A + Nd (ℓe²/10)(1/r); resíduo R(1/r) = γf3 M_def(1/r) - M_ext(1/r),
#   que deve ser nulo na curvatura de equilíbrio e negativo em toda curvatura menor (equilíbrio estável
#   mais próximo da configuração inicial).
#   Variáveis: fck (kN/cm²), gama_c, forca_normal_calculo, gama_f3, alfa_b, momento_primeira_ordem,
#   comprimento_equivalente, curvatura, momento_deformacao.  Fonte: NBR 6118:2023, 15.3.1 (Figura 15.1) e
#   15.8.3.3.4.
#   Usada em: tensao_pico_deformacao (l. 630); forca_curva_deformacao (l. 634); momento_deformacao_majorado (l. 643);
#       residuo_mn1r (l. 647); curvaturas_varredura (l. 703); nivel_2 (l. 1884);
#       conferir_mn1r_independente (l. 2101, 2108, 2110, 2117, 2119)
GAMA_F3_INDEPENDENTE = 1.1
FATOR_PICO_DEFORMACAO = 1.10
FRACAO_VARREDURA_ESTABILIDADE = np.linspace(0.02, 0.98, 49)


def tensao_pico_deformacao(fck_kn_cm2: float, gama_c: float) -> float:
    return FATOR_PICO_DEFORMACAO * fck_kn_cm2 / gama_c


def forca_curva_deformacao(forca_normal_calculo: float) -> float:
    return forca_normal_calculo / GAMA_F3_INDEPENDENTE


def momento_externo_pilar_padrao(alfa_b: float, momento_primeira_ordem: float, forca_normal_calculo: float,
                                 comprimento_equivalente: float, curvatura: float) -> float:
    return alfa_b * momento_primeira_ordem + forca_normal_calculo * comprimento_equivalente ** 2 / 10.0 * curvatura


def momento_deformacao_majorado(momento_deformacao: float) -> float:
    return GAMA_F3_INDEPENDENTE * momento_deformacao


def residuo_mn1r(momento_deformacao: float, momento_externo: float) -> float:
    return momento_deformacao_majorado(momento_deformacao) - momento_externo


# [F34] Fluência recalculada de forma independente (V3 e V4): θ1 = 1/(100 raiz(ℓ)), ℓ em m, limitado a
#   [1/300; 1/200]; ea = θ1 ℓ/2; Nk = Nd/γf; Ngk = fg Nk; Nqk = (1 - fg) Nk; Nsg = Ngk + ψ2 Nqk; Msg = Nsg e1;
#   Eci = αE 5600 raiz(fck) (MPa); Ic = b h³/12; Ne = 10 Eci Ic/ℓe²;
#   ecc = (Msg/Nsg + ea)[e^(φ Nsg/(Ne - Nsg)) - 1]; M1d,A,fluência = M1d,A + Nd ecc.
#   Variáveis: comprimento (cm), forca_normal_calculo, gama_f, fracao_permanente, psi2, excentricidade_1,
#   fck_mpa, alfa_e, largura, altura, coeficiente_fluencia.  Fonte: NBR 6118:2023, 11.3.3.4.2, 8.2.8 e
#   15.8.4 (a norma escreve a base dos logaritmos naturais arredondada, 2,718; com ela, ecc seria 0,014 %
#   menor nos casos V3 e V4).
#   Usada em: conferir_mn1r_independente (l. 2088, 2100)
def fluencia_independente(comprimento: float, forca_normal_calculo: float, gama_f: float, fracao_permanente: float,
                          psi2: float, excentricidade_1: float, fck_mpa: float, alfa_e: float, largura: float,
                          altura: float, coeficiente_fluencia: float) -> dict:
    desaprumo = min(1.0 / 200.0, max(1.0 / 300.0, 1.0 / (100.0 * math.sqrt(comprimento / 100.0))))
    excentricidade_a = desaprumo * comprimento / 2.0
    forca_k = forca_normal_calculo / gama_f
    forca_sg = fracao_permanente * forca_k + psi2 * (1.0 - fracao_permanente) * forca_k
    momento_sg = forca_sg * excentricidade_1
    modulo = alfa_e * 5600.0 * math.sqrt(fck_mpa) / 10.0
    forca_e = 10.0 * modulo * largura * altura ** 3 / 12.0 / comprimento ** 2
    excentricidade_cc = (momento_sg / forca_sg + excentricidade_a) * (
        math.exp(coeficiente_fluencia * forca_sg / (forca_e - forca_sg)) - 1.0)
    return {"ea": excentricidade_a, "Nsg": forca_sg, "Msg": momento_sg, "Ne": forca_e, "ecc": excentricidade_cc}


def momento_primeira_ordem_fluencia(momento_primeira_ordem: float, forca_normal_calculo: float,
                                    excentricidade_cc: float) -> float:
    return momento_primeira_ordem + forca_normal_calculo * excentricidade_cc

# [F35] Excentricidade de 1ª ordem a partir do momento aplicado: e1 = M1d,A / Nd.
#   Variáveis: momento, forca_normal.  Fonte: definição (Bastos, 2017, p. 42).
#   Usada em: cadeia_exata_bastos (l. 1125); nivel_2_bastos (l. 1997)
def excentricidade_do_momento(momento: float, forca_normal: float) -> float:
    return momento / forca_normal


# [F36] Momento de 1ª ordem adotado antes da fluência: M1d,A = max(Nd e1; M1d,mín).
#   Variáveis: momento_aplicado, momento_minimo.  Fonte: NBR 6118:2023, 11.3.3.4.3.
#   Usada em: conferir_mn1r_independente (l. 2098)
def momento_primeira_ordem_base(momento_aplicado: float, momento_minimo: float) -> float:
    return max(momento_aplicado, momento_minimo)


# [F37] Curvaturas da varredura de estabilidade ([F33]): frações de 2 % a 98 % da curvatura de equilíbrio,
#   e intervalo da busca da deformação do topo no modelo de fibras: de -12 ‰ (toda a seção tracionada além
#   do alongamento último de 10 ‰) a eps_c2 + k h + 1 ‰ (toda a seção comprimida além de eps_c2).
#   Variáveis: curvatura_equilibrio, curvatura, altura, eps_c2.
#   Usada em: deformacao_superior_busca (l. 707); momento_fibras_com_forca (l. 2070, 2071);
#       conferir_mn1r_independente (l. 2121)
DEFORMACAO_INFERIOR_BUSCA = -0.012
FOLGA_SUPERIOR_BUSCA = 0.001


def curvaturas_varredura(curvatura_equilibrio: float) -> np.ndarray:
    return FRACAO_VARREDURA_ESTABILIDADE * curvatura_equilibrio


def deformacao_superior_busca(curvatura: float, altura: float, eps_c2: float) -> float:
    return eps_c2 + curvatura * altura + FOLGA_SUPERIOR_BUSCA


# [N05] Critério de aprovação dos testes: relativo, Δ% = |calculado - esperado| / |esperado| x 100 <= 100 tol
#   (concordância = 100 % - Δ% >= 100 % - 100 tol, [F48]), quando o valor esperado não é nulo; absoluto,
#   |calculado - esperado| <= tol, quando o esperado é nulo, na taxa mecânica lida no ábaco ([N04]), na ordem de
#   convergência e na razão erro/limite teórico ([N07]); lógico (tol = 0, verdadeiro = 1) nas verificações de
#   sim ou não.
#   Variáveis: calculado, esperado.
#   Usada em: registrar (l. 1349, 1354)
def diferenca_absoluta(calculado: float, esperado: float) -> float:
    return abs(calculado - esperado)


# [F38] Etapas do roteiro normativo recalculadas de forma independente para os pilares do memorial (direção de
#   b_w; h da norma = altura na direção considerada, [H07] do núcleo, aqui b_w):
#   λ = raiz(12) ℓe/h; M1d,mín = Nd (1,5 + 0,03 h), h em cm; M1d,A = max(Nd e1; M1d,mín);
#   αb = 0,60 + 0,40 MB/MA >= 0,40 com MB = MA (curvatura simples), ou 1,0 se os momentos forem menores que o
#   mínimo; λ1 = (25 + 12,5 e1/h)/αb, limitado a [35; 90]; ν = Nd/(Ac fcd), fcd = fck/1,4;
#   1/r = 0,005/[h (ν + 0,5)] <= 0,005/h (curvatura aproximada).
#   Variáveis: comprimento_equivalente, altura, largura, forca_normal_calculo, excentricidade_1, fck_mpa, nu.
#   Fonte: NBR 6118:2023, 15.8.2, 11.3.3.4.3, 15.8.3.3.2 e 12.3.3.
#   Usada em: etapas_roteiro_independentes (l. 743); cadeia_exata_bastos (l. 1124); cadeia_autor_bastos (l. 1148);
#       nivel_1_flexao_obliqua (l. 1625); nivel_2 (l. 1873)
def curvatura_aproximada_independente(altura: float, nu: float) -> float:
    return min(0.005 / (altura * (nu + 0.5)), 0.005 / altura)


def etapas_roteiro_independentes(comprimento_equivalente: float, altura: float, largura: float,
                                 forca_normal_calculo: float, excentricidade_1: float, fck_mpa: float) -> dict:
    esbeltez = math.sqrt(12.0) * comprimento_equivalente / altura
    momento_minimo = forca_normal_calculo * (1.5 + 0.03 * altura)
    momento_aplicado = forca_normal_calculo * excentricidade_1
    alfa_b = 1.0 if momento_aplicado < momento_minimo else max(0.40, 0.60 + 0.40 * 1.0)
    esbeltez_limite = min(90.0, max(35.0, (25.0 + 12.5 * excentricidade_1 / altura) / alfa_b))
    nu = forca_normal_calculo / (largura * altura * fck_mpa / 10.0 / 1.4)
    curvatura = curvatura_aproximada_independente(altura, nu)
    return {"lambda": esbeltez, "M1d_min": momento_minimo, "M1d_A": max(momento_aplicado, momento_minimo),
            "alfa_b": alfa_b, "lambda1": esbeltez_limite, "nu": nu, "curvatura": curvatura}

# [F39] Modelo de fibras oblíquo independente (regra do ponto médio numa grade de n x n fibras de b_w/n x h/n):
#   eps_ij = eps_topo - k (cos alfa x_i + sen alfa y_j), a partir do vértice (0, 0) (alfa medido a partir do eixo
#   de b_w, como em [F63] do núcleo);
#   sigma_c = sigma_pico [1 - (1 - eps/eps_c2)^n] para 0 <= eps <= eps_c2, sigma_pico além e 0 na tração;
#   sigma_s = E_s eps com |sigma_s| <= fyd; F_s = A_s (sigma_s - d sigma_c(eps_s)), d = 1 na seção líquida;
#   N = soma sigma_c dA + soma F_s;  M_h = soma sigma_c dA (h/2 - y) + soma F_s (h/2 - y_s);
#   M_b = soma sigma_c dA (b_w/2 - x) + soma F_s (b_w/2 - x_s).
#   Variáveis: deformacao_topo, curvatura, angulo, largura, altura, tensao_pico, eps_c2, expoente, x_barras,
#   y_barras, areas_barras, fyd, numero_fibras, descontar_area_barras.
#   Fonte: NBR 6118:2023, 17.2.2 e 8.2.10.1; [H05] do núcleo (escrito sem usar as funções do núcleo).
#   Usada em: nivel_1_flexao_obliqua (l. 1525); nivel_1_flexao_obliqua.fibras_no_parametro (l. 1650)
def esforcos_obliquos_por_fibras(deformacao_topo: float, curvatura: float, angulo: float, largura: float,
                                 altura: float, tensao_pico: float, eps_c2: float, expoente: float,
                                 x_barras: np.ndarray, y_barras: np.ndarray, areas_barras: np.ndarray, fyd: float,
                                 numero_fibras: int, descontar_area_barras: bool) -> tuple[float, float, float]:
    x = (np.arange(numero_fibras) + 0.5) * largura / numero_fibras
    y = (np.arange(numero_fibras) + 0.5) * altura / numero_fibras
    area_fibra = largura * altura / numero_fibras ** 2
    deformacoes = deformacao_topo - curvatura * (math.cos(angulo) * x[None, :] + math.sin(angulo) * y[:, None])
    parabola = np.clip(deformacoes, 0.0, eps_c2)
    forcas_concreto = tensao_pico * (1.0 - (1.0 - parabola / eps_c2) ** expoente) * area_fibra
    deformacoes_aco = deformacao_topo - curvatura * (math.cos(angulo) * x_barras + math.sin(angulo) * y_barras)
    tensoes_aco = np.clip(nucleo.MODULO_ACO * deformacoes_aco, -fyd, fyd)
    if descontar_area_barras:
        parabola_aco = np.clip(deformacoes_aco, 0.0, eps_c2)
        tensoes_aco = tensoes_aco - tensao_pico * (1.0 - (1.0 - parabola_aco / eps_c2) ** expoente)
    forcas_aco = areas_barras * tensoes_aco
    forca = forcas_concreto.sum() + forcas_aco.sum()
    momento_h = (forcas_concreto * (altura / 2.0 - y[:, None])).sum() + (forcas_aco * (altura / 2.0 - y_barras)).sum()
    momento_b = (forcas_concreto * (largura / 2.0 - x[None, :])).sum() + (forcas_aco * (largura / 2.0 - x_barras)).sum()
    return float(forca), float(momento_h), float(momento_b)


# [F40] Polígono regular de raio R com vértices nos ângulos theta_k = k Delta (k = 0 ... m, Delta = 90°/m): no
#   vértice, o raio é R; na bissetriz entre dois vértices, R cos(Delta/2) (apótema).
#   Variáveis: raio, numero_lados (m), angulos.
#   Usada em: nivel_1_flexao_obliqua (l. 1581, 1584, 1589)
def vertices_poligono(raio: float, numero_lados: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angulos = np.linspace(0.0, math.pi / 2.0, numero_lados + 1)
    return raio * np.cos(angulos), raio * np.sin(angulos), angulos


def apotema(raio: float, numero_lados: int) -> float:
    return raio * math.cos(math.pi / 2.0 / numero_lados / 2.0)


def bissetrizes(angulos: np.ndarray) -> np.ndarray:
    return (angulos[1:] + angulos[:-1]) / 2.0


# [F41] Equação da elipse no ponto de raio r e direção psi (psi = 0 na direção de b_w):
#   (r cos psi / a)² + (r sen psi / c)² (= 1 sobre a elipse).
#   Variáveis: raio, direcao, semieixo_b (a), semieixo_h (c).  Fonte: NBR 6118:2023, Figura 15.2.
#   Usada em: nivel_1_flexao_obliqua (l. 1602)
def equacao_elipse(raio: float, direcao: float, semieixo_b: float, semieixo_h: float) -> float:
    return (raio * math.cos(direcao) / semieixo_b) ** 2 + (raio * math.sin(direcao) / semieixo_h) ** 2


# [F42] Momento total do pilar-padrão com curvatura aproximada, escrito de forma independente:
#   M_d,tot = alfa_b M1d,A + Nd (l_e² / 10) (1/r), com 1/r de [F38].
#   Variáveis: alfa_b, momento_primeira_ordem, forca_normal_calculo, comprimento_equivalente, curvatura.
#   Fonte: NBR 6118:2023, 15.8.3.3.2.
#   Usada em: cadeia_exata_bastos (l. 1128); cadeia_autor_bastos (l. 1150); nivel_1_flexao_obliqua (l. 1630)
def momento_total_curvatura_independente(alfa_b: float, momento_primeira_ordem: float, forca_normal_calculo: float,
                                         comprimento_equivalente: float, curvatura: float) -> float:
    return alfa_b * momento_primeira_ordem + forca_normal_calculo * comprimento_equivalente ** 2 / 10.0 * curvatura


# [F43] Diferença entre o fator da envoltória com a discretização de [N04] e com os passos pela metade:
#   |eta_refinado - eta|.  Variáveis: fator_refinado, fator.
#   Usada em: nivel_1_flexao_obliqua (l. 1618)
def diferenca_refino(fator_refinado: float, fator: float) -> float:
    return abs(fator_refinado - fator)


# [N06] Testes da flexão oblíqua: fibras por lado da grade de [F39]; inclinações e parâmetros de domínio testados;
#   inclinação dos testes de limite (alfa -> 0 e alfa -> 90°); lados do polígono de [F40].
#   TOLERANCIA_LIMITE = 1e-6 (relativa, concordância mínima 99,9999 %, e absoluta na razão |M transversal|/|M|):
#     o estado com alfa = ANGULO_LIMITE = 1e-7 rad difere do de alfa = 0 por um termo de primeira ordem, c alfa,
#     cujo coeficiente c depende da forma da seção e da posição das barras (até 5,1 nos testes); a tolerância é
#     10 alfa, folga de 2 sobre o maior coeficiente observado, e continua 1000 vezes abaixo de qualquer erro de
#     sinal ou de eixo, que daria diferenças da ordem de 1.
#   TOLERANCIA_REFINO = 1e-4, absoluta no fator η de [M08] do núcleo: η é apresentado com quatro casas decimais
#     (ex.: 0,9931); com os passos da discretização de [N04] do núcleo pela metade, η não pode mudar mais que uma
#     unidade da última casa apresentada (maior mudança observada: 4,8e-6).
#   TOLERANCIA_RAIZ_FIBRAS = 1e-12: tolerância (xtol) da busca independente do parâmetro de domínio t em
#     [0; PARAMETRO_DOMINIO_MAXIMO] que equilibra Nd no modelo de fibras em grade (a mesma ordem da
#     TOLERANCIA_PARAMETRO do núcleo); o erro de força normal que ela deixa, |dN/dt| 1e-12, é desprezível diante da
#     TOLERANCIA_FIBRAS com que o ponto é comparado. AMOSTRAS_N_CRESCENTE e AMOSTRAS_VETORIAL: número de valores
#     de t, igualmente espaçados em [0; PARAMETRO_DOMINIO_MAXIMO], dos testes de crescimento de N_R na flexão
#     normal e na oblíqua (passo 0,01) e da forma vetorial de [F18] do núcleo (passo 0,001).
#   Variáveis: FIBRAS_POR_LADO, ANGULOS_OBLIQUOS_TESTE (graus), PARAMETROS_OBLIQUOS_TESTE, ANGULO_LIMITE (rad),
#   TOLERANCIA_LIMITE, LADOS_POLIGONO, TOLERANCIA_REFINO, TOLERANCIA_RAIZ_FIBRAS, AMOSTRAS_N_CRESCENTE,
#   AMOSTRAS_VETORIAL.
#   Usada em: nível do módulo (l. 878, 1207, 1209, 1211); nivel_1 (l. 1477);
#       nivel_1_flexao_obliqua (l. 1515, 1520, 1527, 1539, 1548, 1551, 1555, 1557, 1567, 1569, 1581, 1589, 1604, 1608, 1620, 1659);
#       nivel_1_flexao_obliqua.fibras_no_parametro (l. 1655)
FIBRAS_POR_LADO = 1200
ANGULOS_OBLIQUOS_TESTE = (20.0, 45.0, 70.0)
PARAMETROS_OBLIQUOS_TESTE = (0.9, 1.5, 2.5)
ANGULO_LIMITE = 1e-7
TOLERANCIA_LIMITE = 1e-6
LADOS_POLIGONO = 12
TOLERANCIA_REFINO = 1e-4
TOLERANCIA_RAIZ_FIBRAS = 1e-12
AMOSTRAS_N_CRESCENTE = 301
AMOSTRAS_VETORIAL = 3001

# [R05] Seções da validação da flexão oblíqua (b_w, h, arranjo, nível de [R02] do núcleo, d' em cm, As em cm²) e
#   pilares do teste de refinamento da envoltória e da recomputação dos semieixos (curvatura aproximada nas duas
#   direções: M02, M05 e M08 do teste do momento mínimo; P051, o menor eta da matriz).
#   Pontos da envoltória resistente refeitos ponta a ponta com o modelo de fibras [F39] (equilíbrio N_R = Nd
#   resolvido à parte) nos pilares PILARES_PONTO_ENVOLTORIA e inclinações ANGULOS_PONTO_ENVOLTORIA (graus).
#   Variáveis: SECOES_OBLIQUAS_TESTE, PILARES_ENVOLTORIA_TESTE, PILARES_PONTO_ENVOLTORIA, ANGULOS_PONTO_ENVOLTORIA.
#   Usada em: nivel_1_flexao_obliqua (l. 1509, 1565, 1612, 1633, 1639)
SECOES_OBLIQUAS_TESTE = ((20.0, 30.0, nucleo.ARRANJO_PERIMETRO, 2, 3.5, 12.0),
                         (40.0, 80.0, nucleo.ARRANJO_DUAS_FACES, 3, 3.75, 60.0),
                         (30.0, 30.0, nucleo.ARRANJO_PERIMETRO, 2, 3.5, 16.0))
PILARES_ENVOLTORIA_TESTE = ("M02", "M05", "M08", "P051")
PILARES_PONTO_ENVOLTORIA = ("M02", "P051")
ANGULOS_PONTO_ENVOLTORIA = (30.0, 60.0)

# [F44] Ângulos auxiliares dos testes da flexão oblíqua: complementar (90° - alfa, simetria da seção quadrada);
#   inclinações-limite (alfa = ANGULO_LIMITE e 90° - ANGULO_LIMITE, em rad); grades de [N04] do núcleo com passo
#   dado (teste de refinamento): alfa = passo, 2 passo, ... < 90° e psi = 0 a 90° com o passo dado.
#   Variáveis: graus, passo (graus).
#   Usada em: discretizacao_envoltoria (l. 1497, 1498); nivel_1_flexao_obliqua (l. 1537, 1569, 1615)
def angulo_complementar(graus: float) -> float:
    return 90.0 - graus


ANGULOS_LIMITE = {nucleo.DIRECAO_B: ANGULO_LIMITE, nucleo.DIRECAO_H: math.pi / 2.0 - ANGULO_LIMITE}


def grade_linha_neutra(passo: float) -> np.ndarray:
    return np.radians(np.arange(passo, 90.0, passo))


def grade_direcoes(passo: float) -> np.ndarray:
    return np.radians(np.linspace(0.0, 90.0, round(90.0 / passo) + 1))


def metade(passo: float) -> float:
    return passo / 2.0


# [F45] Razão entre o momento transversal e o momento da direção no teste de limite: |M_transversal| / |M_direção|.
#   Variáveis: momento_transversal, momento_direcao.
#   Usada em: nivel_1_flexao_obliqua (l. 1553)
def razao_transversal(momento_transversal: float, momento_direcao: float) -> float:
    return abs(momento_transversal) / abs(momento_direcao)

# [F46] Projeção de um ponto (x, y) na normal à linha neutra de inclinação alfa, escrita de forma independente:
#   u = x cos alfa + y sen alfa (alfa a partir do eixo de b_w; no vértice oposto, a altura projetada H; na barra
#   mais afastada, d).  Variáveis: x, y, angulo (rad).
#   Usada em: nivel_1_flexao_obliqua (l. 1646, 1647)
def projecao_obliqua(x, y, angulo: float):
    return np.asarray(x, dtype=float) * math.cos(angulo) + np.asarray(y, dtype=float) * math.sin(angulo)


# [F47] Resíduo do equilíbrio de forças do modelo de fibras: N_fibras(t) - Nd.  Variáveis: forca, forca_alvo.
#   Usada em: nivel_1_flexao_obliqua (l. 1657)
def residuo_forca_fibras(forca: float, forca_alvo: float) -> float:
    return forca - forca_alvo

# [N07] Testes de convergência e parâmetros dos testes de fibras e da rigidez tangente (Nível 1).
#   FATOR_REFINO_CONVERGENCIA = 2: razão entre as discretizações comparadas (NUMERO_FIBRAS/2 e NUMERO_FIBRAS
#   fibras; SEGMENTOS_PILAR e 2 SEGMENTOS_PILAR segmentos). ORDEM_TEORICA = 2 (regra do ponto médio e diferença
#   central). TOLERANCIA_ORDEM = 0,01, absoluta na ordem observada p = log2(e_grossa/e_fina) ([F52]): a razão
#   entre os erros é 4 (1 + O(h²)), e o termo O(h²) é da ordem de 1e-3 nas discretizações usadas (maior |p - 2|
#   observado: 1,1e-3); 0,01 separa a ordem 2 de qualquer outra (ordem 1 daria p = 1).
#   PARAMETRO_SEM_LINHA_NEUTRA = 2,5: estado de domínio 5 com toda a seção comprimida (ε da borda menos
#   comprimida = 1 ‰), sem o "bico" do diagrama em ε = 0, em que a razão 4 vale sem perturbação.
#   RAZAO_LIMITE_MAXIMA = 1: o erro das fibras não pode exceder o limite teórico de [F50].
#   AMOSTRAS_FIBRAS: estados de deformação sorteados por seção ([N03]), com o parâmetro de domínio em
#   [0; PARAMETRO_DOMINIO_MAXIMO]. MOMENTO_MINIMO_RELATIVO (kN·cm): o critério relativo do momento só é aplicado
#   quando |M| > 1 kN·cm (com M -> 0 o desvio relativo perde o sentido; o limite absoluto de [F50] cobre todos os
#   estados).
#   DEFORMACAO_UNIFORME_RIGIDEZ e CURVATURA_DIFERENCA_CENTRAL: estado do teste da rigidez tangente ([F07], [F11]):
#   ε0 = 0,5 ‰ < εc2 e aço elástico (Es ε0 = 10,5 kN/cm² < fyd); κ = 1e-8/cm. Com n = 2 nessa faixa, M(κ) é
#   linear em κ (o termo de 2º grau se anula pela simetria da seção), e a diferença central é exata a menos do
#   arredondamento, estimado em ε_máquina h/δ ≈ 1e-11 (δ = deslocamento da resultante, da ordem de 1e-3 cm):
#   aplica-se o critério das expressões exatas (TOLERANCIA_EXATA).
#   Variáveis: FATOR_REFINO_CONVERGENCIA, NUMERO_FIBRAS_GROSSO, SEGMENTOS_REFINADOS, ORDEM_TEORICA,
#   TOLERANCIA_ORDEM, PARAMETRO_SEM_LINHA_NEUTRA, RAZAO_LIMITE_MAXIMA, AMOSTRAS_FIBRAS, PARAMETRO_DOMINIO_MAXIMO,
#   MOMENTO_MINIMO_RELATIVO, DEFORMACAO_UNIFORME_RIGIDEZ, CURVATURA_DIFERENCA_CENTRAL.
#   Usada em: nível do módulo (l. 937, 938, 1213, 1215);
#       nivel_1 (l. 1416, 1417, 1435, 1436, 1450, 1454, 1455, 1461, 1464, 1465, 1466, 1467, 1471, 1477);
#       nivel_1_flexao_obliqua (l. 1557, 1604, 1658); nivel_1_metodo_geral_elastico (l. 1676, 1678, 1680, 1681)
FATOR_REFINO_CONVERGENCIA = 2
NUMERO_FIBRAS_GROSSO = NUMERO_FIBRAS // FATOR_REFINO_CONVERGENCIA
SEGMENTOS_REFINADOS = SEGMENTOS_PILAR * FATOR_REFINO_CONVERGENCIA
ORDEM_TEORICA = 2.0
TOLERANCIA_ORDEM = 0.01
PARAMETRO_SEM_LINHA_NEUTRA = 2.5
RAZAO_LIMITE_MAXIMA = 1.0
AMOSTRAS_FIBRAS = 12
PARAMETRO_DOMINIO_MAXIMO = 3.0
MOMENTO_MINIMO_RELATIVO = 1.0
DEFORMACAO_UNIFORME_RIGIDEZ = 0.0005
CURVATURA_DIFERENCA_CENTRAL = 1e-8


# [F50] Limite teórico do erro de quadratura do modelo de fibras independente ([F06], regra do ponto médio com
#   n fibras de espessura dy = h/n), para o diagrama parábola-retângulo com expoente n_p >= 2 (fck <= 50 MPa,
#   n_p = 2, os casos testados). O integrando da força é f(y) = b σc(ε0 - κ y):
#   (a) na fibra que contém a linha neutra (ε = 0), f tem um "bico" (a derivada salta de 0 para
#       c = b n_p σ_pico |κ| / εc2); o erro do ponto médio nessa fibra é no máximo c dy²/8 (máximo da função
#       u²/2 ou (1 - u)²/2 da posição relativa u do bico na fibra);
#   (b) nas demais, o erro é no máximo (dy²/24) h máx|f''| (núcleo de Peano da regra do ponto médio), com
#       |f''| <= b n_p (n_p - 1) σ_pico κ² / εc2² (em εc2 a derivada é contínua);
#   e para o momento, cujo integrando é g(y) = f(y) (h/2 - y): bico de c (h/2 - y0) <= c h/2 e
#   |g''| <= (h/2)|f''| + 2|f'|, |f'| <= c. As barras (e o concreto que elas deslocam) são pontuais nos dois
#   modelos e não acrescentam erro. Limites:
#   E_N = c dy²/8 + (h/24) dy² máx|f''|;  E_M = c (h/2) dy²/8 + (h/24) dy² [(h/2) máx|f''| + 2 c].
#   Variáveis: largura, altura, tensao_pico, eps_c2, expoente, curvatura, numero_fibras.
#   Fonte: análise de erro da regra do ponto médio (núcleo de Peano); 8.2.10.1 da NBR 6118:2023.
#   Usada em: nivel_1 (l. 1443)
def limite_erro_fibras(largura: float, altura: float, tensao_pico: float, eps_c2: float, expoente: float,
                       curvatura: float, numero_fibras: int) -> tuple[float, float]:
    espessura = altura / numero_fibras
    salto = largura * expoente * tensao_pico * abs(curvatura) / eps_c2
    segunda_derivada = largura * expoente * (expoente - 1.0) * tensao_pico * curvatura ** 2 / eps_c2 ** 2
    limite_forca = salto * espessura ** 2 / 8.0 + altura / 24.0 * espessura ** 2 * segunda_derivada
    limite_momento = (salto * altura / 2.0 * espessura ** 2 / 8.0
                      + altura / 24.0 * espessura ** 2 * (altura / 2.0 * segunda_derivada + 2.0 * salto))
    return limite_forca, limite_momento


# [F51] Estimativa do erro relativo do método geral elástico (diferenças finitas centrais com m segmentos) em
#   relação à fórmula da secante: a carga crítica do operador discreto é menor que a de Euler na razão
#   (π/m)²/12, e M_max = M0/cos θ, θ = (π/2) raiz(N/Ne), amplifica um erro relativo δ em Ne por θ tg θ / 2
#   (d ln M_max / d ln Ne = -θ tg θ / 2): erro ≈ (θ tg θ / 2) (π/m)²/12.
#   Variáveis: razao (N/Ne), segmentos (m).  Fonte: análise do operador de diferenças centrais.
#   Usada em: teste_elastico (l. 1848)
def estimativa_erro_secante(razao: float, segmentos: int) -> float:
    angulo = math.pi / 2.0 * math.sqrt(razao)
    return angulo * math.tan(angulo) / 2.0 * (math.pi / segmentos) ** 2 / 12.0


# [F52] Ordem de convergência observada entre duas discretizações (a mais fina com passo 1/fator da outra):
#   p = ln(|e_grossa| / |e_fina|) / ln(fator), com e = valor numérico - valor exato.
#   Variáveis: calculado_grosso, calculado_fino, exato, fator.
#   Usada em: nivel_1 (l. 1466); nivel_1_metodo_geral_elastico (l. 1679)
def ordem_observada(calculado_grosso: float, calculado_fino: float, exato: float, fator: float) -> float:
    return math.log(abs(calculado_grosso - exato) / abs(calculado_fino - exato)) / math.log(fator)


# [F53] Razão entre o erro observado e o limite teórico de [F50]: |calculado - exato| / limite.
#   Variáveis: calculado, exato, limite.
#   Usada em: nivel_1 (l. 1446, 1447)
def razao_erro_limite(calculado: float, exato: float, limite: float) -> float:
    return abs(calculado - exato) / limite


# [F48] Concordância entre calculado e esperado (apresentação dos testes de critério relativo, [N05]):
#   concordância = 100 % - Δ%; o texto é escrito em aritmética decimal exata, com Δ% arredondado a dois algarismos
#   significativos (Δ% = 7,2e-14 -> 99,999999999999928 %); concordância mínima = 100 % - 100 tol
#   (tol = 1e-9 -> 99,9999999 %; tol = 0 -> 100 %, verificação lógica).
#   Variáveis: desvio_percentual, tolerancia.
#   Usada em: registrar (l. 1356, 1357); resumo_tolerancias (l. 2133, 2135)
def texto_concordancia(desvio_percentual: float) -> str:
    if desvio_percentual == 0.0:
        return "100"
    return format((Decimal(100) - Decimal("%.1e" % desvio_percentual)).normalize(), "f")


def texto_concordancia_minima(tolerancia: float) -> str:
    return format((Decimal(100) - Decimal(100) * Decimal(repr(tolerancia))).normalize(), "f")


# [F49] Cadeias da engenharia reversa de Bastos (2017) ([N04], [R04], [R08]).
#   Valor impresso: número do texto de [R04]; resolução = unidade do último algarismo impresso (10 elevado ao
#   expoente decimal do texto: "5642.8" -> 0,1; "1.9685e-4" -> 1e-8; "4008" -> 1); fcd = fck/γc = 2,0/1,4 kN/cm².
#   Cadeia do autor: o passo a passo de Bastos, cada etapa calculada com 3,46 no lugar de raiz(12) e com os
#   valores IMPRESSOS das etapas anteriores:
#     Nk = Nd/(γn γf) ([F31]); λ = 3,46 ℓe/h; M1d,mín, αb, λ1 e ν como em [F38] (o autor só arredonda ao imprimir);
#     1/r = 0,005/[h (ν impresso + 0,5)] <= 0,005/h; método: 2ª ordem dispensada se λ impresso < λ1 impresso;
#     M1d,A = max(M1d,A aplicado; M1d,mín impresso); Md,tot = αb M1d,A + Nd (ℓe²/10)(1/r impresso) ([F42]), ou
#     Md,tot = M1d,A com a 2ª ordem dispensada;
#     Eq. 41: a Md² + b Md + c = 0, a = 19200, b = 3840 h Nd - λ² h Nd - 19200 αb M1d,A,
#     c = -3840 αb h Nd M1d,A, com λ impresso; Md,tot = raiz positiva [-b + raiz(b² - 4ac)]/(2a) dos coeficientes
#     impressos; μ = Md,tot impresso/(h Ac fcd).
#   Cadeia exata: o mesmo passo a passo com raiz(12) e sem nenhum arredondamento (λ, M1d,A, αb, λ1, ν e 1/r de
#   [F38]; Md,tot de [F42]; Eq. 41 com λ exato, idêntica à forma da NBR de [F09]; μ com o Md,tot exato).
#   Decomposição da diferença entre a rotina e o valor impresso (em % do impresso):
#     total (rotina - impresso) = arredondamento desta etapa na impressão (autor - impresso)
#                               + efeito de 3,46 e dos arredondamentos das etapas anteriores (exata - autor)
#                               + (rotina - exata), só ponto flutuante (conferido pelos testes);
#   reprodução (critério "impressão", [N04]): meia unidade m = FRACAO_RESOLUCAO x resolução (gravada em
#   "limite_reproducao"); faixa, em módulo: limite inferior |impresso| - m (1 + FOLGA_PONTO_FLUTUANTE) (incluído) e
#   limite superior |impresso| + m (1 - FOLGA_PONTO_FLUTUANTE) (excluído); sinais compatíveis: mesmo sinal, ou
#   impresso nulo; reproduz quando os sinais são compatíveis e limite inferior <= |autor| < limite superior, isto é,
#   quando o valor da cadeia do autor, arredondado às casas impressas com a meia unidade para cima, em módulo, dá o
#   número impresso; no critério "leitura" (ω lido no ábaco) não há critério de reprodução (limite None; "reproduz"
#   None), só a diferença.
#   Variáveis: texto, exemplo, impresso, exata, forca (Nd), comprimento_equivalente, altura, esbeltez,
#   esbeltez_limite, alfa_b, momento_primeira_ordem, momento, mu, area_concreto, fcd, valor, referencia, criterio.
#   Fonte: Bastos (2017, Eq. 33, 41 e 54, p. 33-54); NBR 6118:2023, 15.8.3.3.2 e 15.8.3.3.3.
#   Usada em: limite_reproducao (l. 1060); reproduz (l. 1078, 1079, 1080);
#       cadeia_exata_bastos (l. 1126, 1130, 1131, 1136, 1137);
#       cadeia_autor_bastos (l. 1143, 1146, 1153, 1155, 1157, 1158, 1160, 1161);
#       linha_engenharia_reversa (l. 1167, 1168, 1170, 1173, 1174, 1175);
#       nivel_2_bastos (l. 1993, 1998, 1999, 2005, 2007, 2033, 2052); texto_reproduz (l. 2156)
def valor_impresso(texto: str) -> float:
    return float(Decimal(texto))


def resolucao_impressa(texto: str) -> float:
    return float(Decimal(1).scaleb(Decimal(texto).as_tuple().exponent))


def limite_reproducao(texto: str, criterio: str) -> float | None:
    return None if criterio == "leitura" else FRACAO_RESOLUCAO * resolucao_impressa(texto)


def limite_inferior_reproducao(impresso: float, meia_unidade: float) -> float:
    return abs(impresso) - meia_unidade * (1.0 + FOLGA_PONTO_FLUTUANTE)


def limite_superior_reproducao(impresso: float, meia_unidade: float) -> float:
    return abs(impresso) + meia_unidade * (1.0 - FOLGA_PONTO_FLUTUANTE)


def sinais_compativeis(autor: float, impresso: float) -> bool:
    return impresso == 0.0 or (autor > 0.0) == (impresso > 0.0)


def reproduz(autor: float, impresso: float, meia_unidade: float | None) -> bool | None:
    if meia_unidade is None:
        return None
    return (sinais_compativeis(autor, impresso)
            and limite_inferior_reproducao(impresso, meia_unidade) <= abs(autor)
            < limite_superior_reproducao(impresso, meia_unidade))


def desvio_assinado(valor: float, referencia: float) -> float:
    return (valor - referencia) / referencia * 100.0


def fcd_bastos() -> float:
    return FCK_BASTOS / 10.0 / 1.4


def area_bastos(exemplo: dict) -> float:
    return exemplo["largura"] * exemplo["altura"]


def esbeltez_bastos(comprimento_equivalente: float, altura: float) -> float:
    return RAIZ_12_BASTOS * comprimento_equivalente / altura


def metodo_bastos(esbeltez: float, esbeltez_limite: float) -> str:
    return nucleo.METODO_DISPENSADO if esbeltez < esbeltez_limite else nucleo.METODO_CURVATURA


def coeficientes_equacao_41(altura: float, forca: float, esbeltez: float, alfa_b: float,
                            momento_primeira_ordem: float) -> tuple[float, float, float]:
    return (19200.0,
            3840.0 * altura * forca - esbeltez ** 2 * altura * forca - 19200.0 * alfa_b * momento_primeira_ordem,
            -3840.0 * alfa_b * altura * forca * momento_primeira_ordem)


def raiz_positiva(coeficiente_a: float, coeficiente_b: float, coeficiente_c: float) -> float:
    return (-coeficiente_b + math.sqrt(coeficiente_b ** 2 - 4.0 * coeficiente_a * coeficiente_c)) / (2.0 * coeficiente_a)


def momento_adimensional_bastos(momento: float, altura: float, area_concreto: float, fcd: float) -> float:
    return momento / (altura * area_concreto * fcd)


def momento_do_mu(mu: float, altura: float, area_concreto: float, fcd: float) -> float:
    return mu * altura * area_concreto * fcd


def cadeia_exata_bastos(exemplo: dict, forca: float) -> dict:
    altura = exemplo["altura"]
    etapas = etapas_roteiro_independentes(COMPRIMENTO_BASTOS, altura, exemplo["largura"], forca,
                                          excentricidade_do_momento(exemplo["M1d_aplicado"], forca), FCK_BASTOS)
    metodo = metodo_bastos(etapas["lambda"], etapas["lambda1"])
    momento_curvatura = (etapas["M1d_A"] if metodo == nucleo.METODO_DISPENSADO else
                         momento_total_curvatura_independente(etapas["alfa_b"], etapas["M1d_A"], forca,
                                                              COMPRIMENTO_BASTOS, etapas["curvatura"]))
    coeficientes = coeficientes_equacao_41(altura, forca, etapas["lambda"], etapas["alfa_b"], etapas["M1d_A"])
    momento_kappa = raiz_positiva(*coeficientes)
    return {**etapas, "metodo": metodo,
            "Nk": forca_caracteristica_bastos(forca, min(exemplo["largura"], altura), nucleo.GAMA_F),
            "Md_curvatura": momento_curvatura, "eq41_b": coeficientes[1], "eq41_c": coeficientes[2],
            "Md_kappa": momento_kappa,
            "mu_curvatura": momento_adimensional_bastos(momento_curvatura, altura, area_bastos(exemplo), fcd_bastos()),
            "mu_kappa": momento_adimensional_bastos(momento_kappa, altura, area_bastos(exemplo), fcd_bastos())}


def cadeia_autor_bastos(exemplo: dict, impresso: dict, exata: dict) -> dict:
    altura = exemplo["altura"]
    forca = impresso["Nd"]
    autor = {"Nk": exata["Nk"], "lambda": esbeltez_bastos(COMPRIMENTO_BASTOS, altura), "M1d_min": exata["M1d_min"],
             "lambda1": exata["lambda1"], "alfa_b": exata["alfa_b"], "nu": exata["nu"]}
    momento_base = max(exemplo["M1d_aplicado"], impresso["M1d_min"])
    autor["metodo"] = metodo_bastos(impresso["lambda"], impresso["lambda1"])
    if "nu" in impresso:
        autor["curvatura"] = curvatura_aproximada_independente(altura, impresso["nu"])
    autor["Md_curvatura"] = (momento_base if autor["metodo"] == nucleo.METODO_DISPENSADO else
                             momento_total_curvatura_independente(impresso["alfa_b"], momento_base, forca,
                                                                  COMPRIMENTO_BASTOS, impresso["curvatura"]))
    if "eq41_b" in impresso:
        coeficientes = coeficientes_equacao_41(altura, forca, impresso["lambda"], impresso["alfa_b"], momento_base)
        autor["eq41_b"], autor["eq41_c"] = coeficientes[1], coeficientes[2]
        autor["Md_kappa"] = raiz_positiva(coeficientes[0], impresso["eq41_b"], impresso["eq41_c"])
    if "mu_curvatura" in impresso:
        autor["mu_curvatura"] = momento_adimensional_bastos(impresso["Md_curvatura"], altura, area_bastos(exemplo),
                                                            fcd_bastos())
    if "mu_kappa" in impresso:
        autor["mu_kappa"] = momento_adimensional_bastos(impresso["Md_kappa"], altura, area_bastos(exemplo),
                                                        fcd_bastos())
    return autor


def linha_engenharia_reversa(exemplo: dict, chave: str, rotulo: str, unidade: str, criterio: str, texto: str,
                             autor: float, exata: float, rotina) -> dict:
    impresso = valor_impresso(texto)
    limite = limite_reproducao(texto, criterio)
    return {"exemplo": exemplo["exemplo"], "paginas": exemplo["paginas"], "etapa": rotulo, "chave": chave,
            "unidade": unidade, "impresso": texto, "resolucao": resolucao_impressa(texto), "criterio": criterio,
            "limite_reproducao": limite, "autor": autor, "exata": exata, "rotina": rotina,
            "diferenca_autor_impresso": abs(autor - impresso),
            "reproduz": reproduz(autor, impresso, limite),
            "desvio_total_percentual": None if rotina is None else desvio_assinado(rotina, impresso),
            "parcela_impressao_percentual": desvio_assinado(autor, impresso),
            "parcela_propagada_percentual": (exata - autor) / impresso * 100.0,
            "abaco": exemplo["abaco"], "observacao": OBSERVACOES_BASTOS.get((exemplo["exemplo"], chave), "")}


# [F54] Profundidades das barras (a partir da face comprimida) no arranjo dos ábacos A-26 a A-29 de Venturini
#   (1987): 8 barras iguais, 3 por face nas quatro faces (cantos e meio de cada face), isto é, 3 barras em d',
#   2 em h/2 e 3 em h - d' (camadas com 3/8, 2/8 e 3/8 de As), nas duas direções de flexão.
#   Variáveis: altura, distancia (d').  Fonte: Venturini (1987), desenho da seção dos ábacos A-26 a A-29.
#   Usada em: secao_exemplo_bastos (l. 1975)
def profundidades_tres_por_face(altura: float, distancia: float) -> np.ndarray:
    return np.array([distancia] * 3 + [altura / 2.0] * 2 + [altura - distancia] * 3)


# [R06] Limites internos de alarme (constante, valor, critério e origem do desvio; detalhes em [N01], [R03], [N04],
#   [N06] e [N07]). Não são critério de aceitação dos resultados (item 1 do arquivo "Pessoal 2" do autor): servem
#   só para acusar erro de programação quando a rotina é reexecutada (teste com alarme = desvio acima do limite, ou
#   verificação lógica falsa). A análise dos desvios é feita pela sua origem, descrita no último campo.
#   Variáveis: TOLERANCIAS.
#   Usada em: nome_tolerancia (l. 1335); resumo_tolerancias (l. 2129)
TOLERANCIAS = (
    ("TOLERANCIA_EXATA", TOLERANCIA_EXATA, "relativo",
     "Expressões exatas (e equilíbrios resolvidos por busca de raiz com xtol = 1e-12): só o arredondamento do "
     "ponto flutuante de 64 bits (ruído de 1e-15 a 1e-12)."),
    ("TOLERANCIA_FIBRAS", TOLERANCIA_FIBRAS, "relativo",
     "Erro de quadratura das fibras (ponto médio, ordem 2): limite teórico [F50] de até 9,5e-5 nos estados "
     "testados."),
    ("TOLERANCIA_SECANTE", TOLERANCIA_SECANTE, "relativo",
     "Diferenças finitas com 60 segmentos: erro estimado [F51] de até 9,6e-4 (N/Ne = 0,8)."),
    ("TOLERANCIA_RETANGULAR", TOLERANCIA_RETANGULAR, "relativo",
     "Diferença entre os diagramas parábola-retângulo e retangular, ambos admitidos pela NBR (resultante 1,2 % "
     "menor); não é erro numérico."),
    ("TOLERANCIA_LIMITE", TOLERANCIA_LIMITE, "relativo",
     "Limites α -> 0 e α -> 90°: termo de 1ª ordem em α = 1e-7 rad."),
    ("TOLERANCIA_LIMITE", TOLERANCIA_LIMITE, "absoluto",
     "Limites α -> 0 e α -> 90°: momento transversal relativo de 1ª ordem em α = 1e-7 rad."),
    ("TOLERANCIA_REFINO", TOLERANCIA_REFINO, "absoluto",
     "Refinamento da envoltória: variação de η com os passos da discretização pela metade."),
    ("TOLERANCIA_ORDEM", TOLERANCIA_ORDEM, "absoluto",
     "Ordem de convergência observada: razão de erros 4 (1 + O(h²)) com a discretização dobrada."),
    ("RAZAO_LIMITE_MAXIMA", RAZAO_LIMITE_MAXIMA, "absoluto",
     "Erro das fibras dividido pelo seu limite teórico [F50]."),
)

# [R07] Explicação, por nível, de por que calculado e esperado diferem (texto do relatório e do auditor).
#   Variáveis: EXPLICACAO_NIVEIS.
#   Usada em: main (l. 2181, 2186)
EXPLICACAO_NIVEIS = {
    "1": ("Nível 1 (validação matemática): o esperado é uma solução exata (forma fechada, identidade algébrica, "
          "simetria) ou um modelo numérico independente de erro conhecido. Nos testes exatos, a única diferença é o "
          "arredondamento da aritmética de ponto flutuante (da ordem de 1e-15 relativo; até 1e-12 na derivada "
          "numérica da rigidez tangente). Nos modelos de fibras, a "
          "diferença é o erro de quadratura da regra do ponto médio, de ordem 2 na espessura da fibra e limitado "
          "teoricamente ([F50]); no método geral elástico, o erro de "
          "discretização das diferenças finitas, de ordem 2 no comprimento do segmento e amplificado perto da carga "
          "crítica ([F51]), e o término de todas as análises do método geral nos pilares com λ > 90 por convergência ou "
          "instabilidade; nos limites α -> 0 e α -> 90°, o termo de primeira ordem em α = 1e-7 rad. As ordens "
          "de convergência são conferidas com discretizações dobradas."),
    "2": ("Nível 2 (validação normativa): (a) no memorial V1 a V4, as etapas refeitas à mão e por expressões "
          "independentes coincidem até o ponto flutuante; o diagrama retangular simplificado é outro modelo "
          "admitido pela norma, com resultante 1,2 % menor, e é comparado como plausibilidade. (b) Em Bastos (2017), "
          "a diferença entre o valor impresso e a rotina vem dos arredondamentos do autor (3,46 no lugar de "
          "raiz(12); λ impresso com uma casa, ν e μ com duas, 1/r com 4 ou 5 algarismos, e o valor impresso usado "
          "na etapa seguinte) e da leitura visual do ábaco. A engenharia reversa refaz o passo a passo com essas "
          "regras e reproduz os valores impressos; o mesmo passo a passo com raiz(12) e sem arredondamentos coincide "
          "com a rotina até o arredondamento do ponto flutuante. O ω é comparado, como "
          "informação, no próprio ponto (ν, μ) em que o autor entrou no ábaco, com a seção no arranjo e na hipótese "
          "do ábaco (seção bruta, Venturini e Rodrigues, 1987, eq. 4.6): resta a leitura gráfica, em ábacos com "
          "curvas espaçadas de 0,10. As comparações método geral x pilar-padrão acoplado e curvatura x rigidez κ, "
          "gravadas em 'comparacoes', comparam métodos diferentes e não fazem parte da validação (no TCC, Seções 4.2 e "
          "4.4)."),
}

# [F55] Rigidez secante do diagrama M, N, 1/r (NBR 6118:2023, 15.3.1, Figura 15.1), na forma do CAD/Pilar do TQS
#   (Kimura, [2005?]) e de Franco (2010): reta AB da origem ao ponto B da curva de deformação (1,10 fcd e Nd/γf3) em
#   que M = MRd/γf3; (EI)sec = (MRd/γf3)/(1/r)_B; κsec = (EI)sec/(Ac h² fcd); Md,tot = αb M1d,A / [1 - λ² ν/(120 κsec)]
#   (expressão da rigidez κ da Seção 15.8.3.3.3 com κsec). Usada só na comparação complementar da Seção 4.4 do TCC
#   (parcela esperada da divergência com o TQS nos pilares com λ > 90), sem efeito no dimensionamento.
#   Variáveis: momento_resistente, curvatura_ponto_b, rigidez, area_concreto, altura, fcd, alfa_b,
#   momento_primeira_ordem, esbeltez, nu, kappa_sec, momento_curva, alvo.
#   Fonte: NBR 6118:2023, 15.3.1 e 15.8.3.3.3; Kimura ([2005?]), item C.3; Franco (2010, p. 34).
#   Usada em: rigidez_secante (l. 1265); comparacao_secante (l. 1958, 1960, 1961, 1962, 1967)
#   Usada também em: validacao_literatura.py: curvatura_do_momento_deformacao (l. 1169);
#       validacao_literatura.py: executar_kimura (l. 1314, 1315, 1316, 1317, 1333, 1360, 1363);
#       validacao_literatura.py: executar_franco (l. 1895, 1911, 1914, 1927, 1934)
def momento_ponto_b(momento_resistente: float) -> float:
    return momento_resistente / nucleo.GAMA_F3


def rigidez_secante(momento_resistente: float, curvatura_ponto_b: float) -> float:
    return momento_ponto_b(momento_resistente) / curvatura_ponto_b


def kappa_secante(rigidez: float, area_concreto: float, altura: float, fcd: float) -> float:
    return rigidez / (area_concreto * altura ** 2 * fcd)


def momento_total_kappa_secante(alfa_b: float, momento_primeira_ordem: float, esbeltez: float, nu: float,
                                kappa_sec: float) -> float:
    return alfa_b * momento_primeira_ordem / (1.0 - esbeltez ** 2 * nu / (120.0 * kappa_sec))


def diferenca_momento_curva(momento_curva: float, alvo: float) -> float:
    return momento_curva - alvo


# [R09] Área necessária e fator de carga pelo método geral (comparação complementar da Seção 4.2 do TCC, sem efeito
#   no dimensionamento). Com as mesmas posições das barras da armadura adotada, uma área A_s atende ao método geral
#   quando a análise de [F10] converge ([R02]) e M_Rd(A_s) >= M_d,tot do método geral (verificação da seção com
#   fcd, como no pilar-padrão acoplado). A área necessária é a menor área que atende, buscada por bisseção entre
#   zero e o limite de busca do núcleo (8 % de A_c, [F39] do núcleo), e é marcada quando passa de A_s,máx da rotina
#   (4 % de A_c, [F38] do núcleo); se o limite de busca não atende, não há área necessária abaixo dele. O fator
#   de carga é o maior f para o qual o método geral encontra equilíbrio com a armadura adotada, f Nd e f M1d,A
#   (inclusive na força normal da curva M, N, 1/r), buscado por bisseção entre FATOR_CARGA_INFERIOR e
#   FATOR_CARGA_SUPERIOR; só é calculado quando não há equilíbrio com f = 1. As duas buscas param quando a largura
#   relativa do intervalo ([F56]) fica abaixo de TOLERANCIA_BISSECAO; o critério é monótono (mais área ou menos
#   carga não tiram o equilíbrio), condição da bisseção.
#   Variáveis: TOLERANCIA_BISSECAO, FATOR_CARGA_INFERIOR, FATOR_CARGA_SUPERIOR.
#   Fonte: NBR 6118:2023, 15.8.3.2 e 15.3.1.
#   Usada em: metodo_geral (l. 1767); intervalo_bissecao (l. 1796); fator_carga_equilibrio (l. 1833, 1834, 1835, 1836)
TOLERANCIA_BISSECAO = 1e-7
FATOR_CARGA_INFERIOR = 0.5
FATOR_CARGA_SUPERIOR = 1.0


# [F56] Bisseção de [R09]: ponto médio (inferior + superior)/2 e largura relativa do intervalo
#   (superior - inferior)/superior.
#   Variáveis: inferior, superior.
#   Usada em: intervalo_bissecao (l. 1796, 1797)
def ponto_medio(inferior: float, superior: float) -> float:
    return 0.5 * (inferior + superior)


def largura_relativa(inferior: float, superior: float) -> float:
    return (superior - inferior) / superior


# [F57] Carga multiplicada pelo fator de carga de [R09]: f Nd ou f M1d,A (f = 1 reproduz a carga de cálculo).
#   Variáveis: valor, fator.
#   Usada em: metodo_geral (l. 1777, 1789)
def carga_multiplicada(valor: float, fator: float) -> float:
    return valor * fator


# [F58] Razões da comparação complementar de [R09]: A_s,nec,MG / A_s,adot (área pelo método geral em relação à
#   adotada) e M_d,tot,MG / M_Rd(A_s,adot) (momento do método geral em relação ao resistente da armadura adotada).
#   Variáveis: numerador, denominador.
#   Usada em: nivel_2 (l. 1931, 1936)
def razao_comparacao(numerador: float, denominador: float) -> float:
    return numerador / denominador

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


def nome_tolerancia(tolerancia: float, criterio: str) -> str:
    """Constante de [R06] que corresponde ao valor e ao critério da tolerância de um teste."""
    if tolerancia == 0.0:
        return "verificação lógica"
    for constante, valor, criterio_quadro, _ in TOLERANCIAS:
        if valor == tolerancia and criterio_quadro == criterio:
            return constante
    return ""


def registrar(relatorio: list[dict], nivel: str, teste: str, calculado: float, esperado: float, tolerancia: float,
              unidade: str = "", absoluta: bool = False) -> None:
    """Registra um teste. Critério ([N05]): relativo (Δ% <= 100 tol, concordância >= 100 % - 100 tol, [F48])
    quando o valor esperado não é nulo; absoluto (|calculado - esperado| <= tol) quando o esperado é nulo ou quando
    absoluta=True (leitura de ábaco, ordem de convergência, razão erro/limite); lógico quando tol = 0."""
    desvio = nucleo.desvio_percentual(calculado, esperado) if esperado != 0.0 else math.nan
    criterio = "absoluto" if absoluta or esperado == 0.0 else "relativo"
    if criterio == "absoluto":
        aprovado = bool(diferenca_absoluta(calculado, esperado) <= tolerancia)
    else:
        aprovado = bool(desvio <= nucleo.desvio_percentual(fator_tolerancia(tolerancia), 1.0))
    relativo_numerico = criterio == "relativo" and tolerancia > 0.0
    relatorio.append({"nivel": nivel, "teste": teste, "calculado": calculado, "esperado": esperado,
                      "desvio_percentual": desvio, "diferenca_absoluta": diferenca_absoluta(calculado, esperado),
                      "tolerancia": tolerancia, "criterio": criterio, "constante": nome_tolerancia(tolerancia, criterio),
                      "concordancia": texto_concordancia(desvio) if relativo_numerico else None,
                      "concordancia_minima": texto_concordancia_minima(tolerancia) if relativo_numerico else None,
                      "unidade": unidade, "aprovado": aprovado})


def secao_duas_camadas(altura: float, largura: float, distancia: float) -> nucleo.SecaoDirecao:
    return nucleo.secao_por_camadas(altura, largura, profundidades_duas_camadas(altura, distancia))


@contextlib.contextmanager
def hipotese_secao(descontar_area_barras: bool):
    """Fixa temporariamente a hipótese [H05] do núcleo (True: seção líquida; False: seção bruta) e restaura o
    valor original."""
    valor_original = nucleo.DESCONTAR_AREA_BARRAS
    nucleo.DESCONTAR_AREA_BARRAS = descontar_area_barras
    try:
        yield
    finally:
        nucleo.DESCONTAR_AREA_BARRAS = valor_original


def nivel_1(relatorio: list[dict]) -> None:
    for fck in (25.0, 35.0):
        material = nucleo.Material(fck)
        for largura, altura, area_aco in ((30.0, 20.0, 10.0), (80.0, 40.0, 60.0)):
            secao = secao_duas_camadas(altura, largura, 4.0)
            area_concreto = nucleo.area_secao_retangular(largura, altura)
            rotulo = "fck %.0f, %.0fx%.0f (b_w x h, flexão na direção de b_w), As %.0f" % (fck, altura, largura,
                                                                                        area_aco)
            with hipotese_secao(False):
                forca_bruta = nucleo.forca_no_parametro(secao, material, area_aco, 3.0)
                forca_tracao_bruta = nucleo.forca_no_parametro(secao, material, area_aco, 0.0)
            with hipotese_secao(True):
                forca_liquida = nucleo.forca_no_parametro(secao, material, area_aco, 3.0)
                forca_tracao_liquida = nucleo.forca_no_parametro(secao, material, area_aco, 0.0)
            registrar(relatorio, "1", "compressão centrada (reta b), seção líquida (adotada) — " + rotulo,
                      forca_liquida,
                      forca_compressao_centrada_secao_liquida(material.tensao_pico_resistencia, area_concreto,
                                                              area_aco, material.eps_c2, material.fyd),
                      TOLERANCIA_EXATA, "kN")
            registrar(relatorio, "1", "compressão centrada (reta b), seção bruta — " + rotulo, forca_bruta,
                      forca_compressao_centrada(material.tensao_pico_resistencia, area_concreto, area_aco,
                                                material.eps_c2, material.fyd), TOLERANCIA_EXATA, "kN")
            registrar(relatorio, "1", "tração pura (reta a), seção líquida (sem desconto na tração) — " + rotulo,
                      forca_tracao_liquida, forca_tracao_pura(area_aco, material.fyd), TOLERANCIA_EXATA, "kN")
            registrar(relatorio, "1", "tração pura (reta a), seção bruta — " + rotulo, forca_tracao_bruta,
                      forca_tracao_pura(area_aco, material.fyd), TOLERANCIA_EXATA, "kN")
            for linha_neutra in linhas_neutras_teste(altura):
                curvatura = curvatura_por_linha_neutra(material.eps_cu, linha_neutra)
                forca, momento = nucleo.esforcos_concreto_retangular(material.eps_cu, curvatura, altura, largura,
                                                                     material.tensao_pico_resistencia, material.eps_c2,
                                                                     material.expoente)
                resultante, posicao = resultante_bloco_parabolico(material.eps_c2, material.eps_cu,
                                                                  material.tensao_pico_resistencia, largura,
                                                                  linha_neutra)
                registrar(relatorio, "1", "resultante do bloco parábola-retângulo, x = %.1f cm — %s" % (
                    linha_neutra, rotulo), float(forca), resultante, TOLERANCIA_EXATA, "kN")
                registrar(relatorio, "1", "momento do bloco parábola-retângulo, x = %.1f cm — %s" % (
                    linha_neutra, rotulo), float(momento), momento_resultante_centro(resultante, altura, posicao), TOLERANCIA_EXATA,
                    "kN·cm")
            deformacao_uniforme = DEFORMACAO_UNIFORME_RIGIDEZ
            curvatura = CURVATURA_DIFERENCA_CENTRAL
            for descontar, nome_hipotese in ((True, "seção líquida (adotada)"), (False, "seção bruta")):
                with hipotese_secao(descontar):
                    _, momento_positivo = nucleo.esforcos_resistentes(secao, material, area_aco, deformacao_topo_centrada(
                        deformacao_uniforme, curvatura, altura), curvatura)
                    _, momento_negativo = nucleo.esforcos_resistentes(secao, material, area_aco, deformacao_topo_centrada(
                        deformacao_uniforme, curvatura_oposta(curvatura), altura), curvatura_oposta(curvatura))
                registrar(relatorio, "1", "rigidez tangente inicial (dM/dκ, diferença central), %s — %s" % (
                    nome_hipotese, rotulo),
                          derivada_central(float(momento_positivo), float(momento_negativo), curvatura),
                          rigidez_tangente_inicial(deformacao_uniforme, material.tensao_pico_resistencia,
                                                   material.eps_c2, material.expoente, largura, altura,
                                                   secao.profundidades,
                                                   nucleo.areas_barras_secao(area_aco, secao.fracoes_area), descontar),
                          TOLERANCIA_EXATA, "kN·cm²")
            areas_camadas = nucleo.areas_barras_secao(area_aco, secao.fracoes_area)
            gerador = np.random.default_rng(semente_aleatoria(fck, altura))
            razoes_limite = []
            for amostra in range(AMOSTRAS_FIBRAS):
                parametro = gerador.uniform(0.0, PARAMETRO_DOMINIO_MAXIMO)
                deformacao_topo, curvatura = nucleo.configuracao_no_parametro(secao, material, parametro)
                forca, momento = nucleo.esforcos_resistentes(secao, material, area_aco, deformacao_topo, curvatura)
                forca_fibras, momento_fibras = esforcos_por_fibras(
                    deformacao_topo, curvatura, altura, largura, material.tensao_pico_resistencia, material.eps_c2,
                    material.expoente, secao.profundidades, areas_camadas, material.fyd, NUMERO_FIBRAS,
                    nucleo.DESCONTAR_AREA_BARRAS)
                limite_forca, limite_momento = limite_erro_fibras(largura, altura, material.tensao_pico_resistencia,
                                                                  material.eps_c2, material.expoente, curvatura,
                                                                  NUMERO_FIBRAS)
                razoes_limite.extend((razao_erro_limite(forca_fibras, float(forca), limite_forca),
                                      razao_erro_limite(momento_fibras, float(momento), limite_momento)))
                registrar(relatorio, "1", "fibras x forma fechada, N, t = %.3f — %s" % (parametro, rotulo),
                          float(forca), forca_fibras, TOLERANCIA_FIBRAS, "kN")
                if abs(momento_fibras) > MOMENTO_MINIMO_RELATIVO:
                    registrar(relatorio, "1", "fibras x forma fechada, M, t = %.3f — %s" % (parametro, rotulo),
                              float(momento), momento_fibras, TOLERANCIA_FIBRAS, "kN·cm")
            registrar(relatorio, "1", "fibras: maior |erro| / limite teórico [F50] nas %d comparações de N e M — %s" % (
                len(razoes_limite), rotulo), max(razoes_limite), 0.0, RAZAO_LIMITE_MAXIMA, absoluta=True)
            deformacao_topo, curvatura = nucleo.configuracao_no_parametro(secao, material, PARAMETRO_SEM_LINHA_NEUTRA)
            exatos = nucleo.esforcos_resistentes(secao, material, area_aco, deformacao_topo, curvatura)
            grossos, finos = (esforcos_por_fibras(deformacao_topo, curvatura, altura, largura,
                                                  material.tensao_pico_resistencia, material.eps_c2, material.expoente,
                                                  secao.profundidades, areas_camadas, material.fyd, fibras,
                                                  nucleo.DESCONTAR_AREA_BARRAS)
                              for fibras in (NUMERO_FIBRAS_GROSSO, NUMERO_FIBRAS))
            for nome, grosso, fino, exato in zip(("N", "M"), grossos, finos, exatos):
                registrar(relatorio, "1", "fibras: ordem de convergência de %s (%d e %d fibras, t = %.1f, sem linha "
                          "neutra na seção) — %s" % (nome, NUMERO_FIBRAS_GROSSO, NUMERO_FIBRAS,
                                                     PARAMETRO_SEM_LINHA_NEUTRA, rotulo),
                          ordem_observada(grosso, fino, float(exato), FATOR_REFINO_CONVERGENCIA), ORDEM_TEORICA,
                          TOLERANCIA_ORDEM, absoluta=True)
            for parametro_limite in (1.0, 2.0):
                antes = nucleo.configuracao_no_parametro(secao, material, np.nextafter(parametro_limite, 0.0))
                depois = nucleo.configuracao_no_parametro(secao, material,
                                                          np.nextafter(parametro_limite, PARAMETRO_DOMINIO_MAXIMO))
                registrar(relatorio, "1", "continuidade dos domínios em t = %.0f (ε topo) — %s" % (
                    parametro_limite, rotulo), depois[0], antes[0], TOLERANCIA_EXATA)
                registrar(relatorio, "1", "continuidade dos domínios em t = %.0f (1/r) — %s" % (
                    parametro_limite, rotulo), depois[1], antes[1], TOLERANCIA_EXATA)
            forcas = [nucleo.forca_no_parametro(secao, material, area_aco, valor) for valor in
                      np.linspace(0.0, PARAMETRO_DOMINIO_MAXIMO, AMOSTRAS_N_CRESCENTE)]
            registrar(relatorio, "1", "N_R não decrescente com o parâmetro de domínio — " + rotulo,
                      float(np.all(np.diff(forcas) >= limite_decrescimo(forcas, TOLERANCIA_EXATA))), 1.0, 0.0)


@contextlib.contextmanager
def discretizacao_angulos(angulos: np.ndarray):
    """Fixa temporariamente as inclinações da linha neutra de [N04] do núcleo e restaura as originais."""
    originais = nucleo.ANGULOS_LINHA_NEUTRA
    nucleo.ANGULOS_LINHA_NEUTRA = angulos
    try:
        yield
    finally:
        nucleo.ANGULOS_LINHA_NEUTRA = originais


@contextlib.contextmanager
def discretizacao_envoltoria(passo_linha_neutra: float, passo_direcoes: float):
    """Fixa temporariamente as grades de [N04] do núcleo e restaura as originais."""
    originais = nucleo.ANGULOS_LINHA_NEUTRA, nucleo.DIRECOES_ENVOLTORIA
    nucleo.ANGULOS_LINHA_NEUTRA = grade_linha_neutra(passo_linha_neutra)
    nucleo.DIRECOES_ENVOLTORIA = grade_direcoes(passo_direcoes)
    try:
        yield
    finally:
        nucleo.ANGULOS_LINHA_NEUTRA, nucleo.DIRECOES_ENVOLTORIA = originais


def nivel_1_flexao_obliqua(relatorio: list[dict]) -> None:
    """Envoltória resistente na flexão oblíqua e verificação da envoltória mínima ([F63] a [F72], [R12], [M08])."""
    material = nucleo.Material(30.0)
    geometrias = {}
    for largura, altura, arranjo, nivel, distancia, area in SECOES_OBLIQUAS_TESTE:
        x, y, _ = nucleo.posicoes_barras(largura, altura, distancia, arranjo, nivel,
                                         nucleo.espacamento_maximo_eixos(min(largura, altura)))
        geometrias[(largura, altura)] = (x, y, area)
        rotulo = "%.0fx%.0f, %d barras, As %.0f" % (largura, altura, x.size, area)
        areas = nucleo.areas_barras_secao(area, nucleo.fracao_area_barras_iguais(x.size))
        for graus in ANGULOS_OBLIQUOS_TESTE:
            angulo = math.radians(graus)
            n_b, n_h = nucleo.direcao_normal_linha_neutra(angulo)
            altura_projetada = nucleo.profundidade_obliqua(largura, altura, n_b, n_h)
            profundidade_util = nucleo.profundidade_obliqua(x, y, n_b, n_h).max()
            for parametro in PARAMETROS_OBLIQUOS_TESTE:
                deformacao_topo, curvatura = nucleo.configuracao_estado_limite(
                    parametro, altura_projetada, profundidade_util, material.eps_c2, material.eps_cu)
                rotina = nucleo.esforcos_resistentes_obliquos(largura, altura, x, y, area, material, deformacao_topo,
                                                              curvatura, angulo)
                fibras = esforcos_obliquos_por_fibras(deformacao_topo, curvatura, angulo, largura, altura,
                                                      material.tensao_pico_resistencia, material.eps_c2,
                                                      material.expoente, x, y, areas, material.fyd, FIBRAS_POR_LADO,
                                                      nucleo.DESCONTAR_AREA_BARRAS)
                for nome, calculado, esperado, unidade in zip(("N", "M_h", "M_b"), rotina, fibras,
                                                              ("kN", "kN·cm", "kN·cm")):
                    registrar(relatorio, "1", "flexão oblíqua, faixas de Gauss x fibras, %s, α = %.0f°, t = %.1f — %s"
                              % (nome, graus, parametro, rotulo), float(calculado), esperado, TOLERANCIA_FIBRAS,
                              unidade)
        for direcao, nome_direcao, indice_direcao, indice_transversal in (
                (nucleo.DIRECAO_B, "α → 0 (direção de b_w)", 2, 1), (nucleo.DIRECAO_H, "α → 90° (direção de h)", 1, 2)):
            secao = nucleo.secao_na_direcao(largura, altura, x, y, direcao)
            angulo = ANGULOS_LIMITE[direcao]
            n_b, n_h = nucleo.direcao_normal_linha_neutra(angulo)
            for parametro in PARAMETROS_OBLIQUOS_TESTE:
                reta = nucleo.esforcos_resistentes(secao, material, area,
                                                   *nucleo.configuracao_no_parametro(secao, material, parametro))
                obliqua = nucleo.esforcos_resistentes_obliquos(
                    largura, altura, x, y, area, material,
                    *nucleo.configuracao_estado_limite(parametro, nucleo.profundidade_obliqua(largura, altura, n_b, n_h),
                                                       nucleo.profundidade_obliqua(x, y, n_b, n_h).max(),
                                                       material.eps_c2, material.eps_cu), angulo)
                registrar(relatorio, "1", "flexão oblíqua no limite %s, N, t = %.1f — %s" % (
                    nome_direcao, parametro, rotulo), float(obliqua[0]), float(reta[0]), TOLERANCIA_LIMITE, "kN")
                registrar(relatorio, "1", "flexão oblíqua no limite %s, M, t = %.1f — %s" % (
                    nome_direcao, parametro, rotulo), float(obliqua[indice_direcao]), float(reta[1]),
                          TOLERANCIA_LIMITE, "kN·cm")
                registrar(relatorio, "1", "flexão oblíqua no limite %s, |M transversal|/|M|, t = %.1f — %s" % (
                    nome_direcao, parametro, rotulo), razao_transversal(float(obliqua[indice_transversal]),
                                                                        float(obliqua[indice_direcao])), 0.0,
                          TOLERANCIA_LIMITE, absoluta=True)
        n_b, n_h = nucleo.direcao_normal_linha_neutra(math.radians(45.0))
        parametros = np.linspace(0.0, PARAMETRO_DOMINIO_MAXIMO, AMOSTRAS_N_CRESCENTE)
        forcas = nucleo.esforcos_resistentes_obliquos(
            largura, altura, x, y, area, material,
            *nucleo.configuracao_estado_limite(parametros, nucleo.profundidade_obliqua(largura, altura, n_b, n_h),
                                               nucleo.profundidade_obliqua(x, y, n_b, n_h).max(), material.eps_c2,
                                               material.eps_cu), math.radians(45.0))[0]
        registrar(relatorio, "1", "flexão oblíqua: N_R não decrescente com o parâmetro de domínio, α = 45° — " + rotulo,
                  float(np.all(np.diff(forcas) >= limite_decrescimo(forcas, TOLERANCIA_EXATA))), 1.0, 0.0)
    largura, altura = SECOES_OBLIQUAS_TESTE[2][:2]
    x, y, area = geometrias[(largura, altura)]
    for parametro in PARAMETROS_OBLIQUOS_TESTE:
        resultados = []
        for graus in (ANGULOS_OBLIQUOS_TESTE[0], angulo_complementar(ANGULOS_OBLIQUOS_TESTE[0])):
            angulo = math.radians(graus)
            n_b, n_h = nucleo.direcao_normal_linha_neutra(angulo)
            resultados.append(nucleo.esforcos_resistentes_obliquos(
                largura, altura, x, y, area, material,
                *nucleo.configuracao_estado_limite(parametro, nucleo.profundidade_obliqua(largura, altura, n_b, n_h),
                                                   nucleo.profundidade_obliqua(x, y, n_b, n_h).max(),
                                                   material.eps_c2, material.eps_cu), angulo))
        registrar(relatorio, "1", "simetria da seção quadrada: N(α) = N(90° - α), t = %.1f" % parametro,
                  float(resultados[0][0]), float(resultados[1][0]), TOLERANCIA_EXATA, "kN")
        registrar(relatorio, "1", "simetria da seção quadrada: M_h(α) = M_b(90° - α), t = %.1f" % parametro,
                  float(resultados[0][1]), float(resultados[1][2]), TOLERANCIA_EXATA, "kN·cm")
    momentos_b, momentos_h, angulos = vertices_poligono(100.0, LADOS_POLIGONO)
    inicio, fim = nucleo.segmento_do_contorno(angulos, angulos)
    raios = nucleo.raio_contorno(momentos_b[inicio], momentos_h[inicio], momentos_b[fim], momentos_h[fim], angulos)
    medios = bissetrizes(angulos)
    inicio, fim = nucleo.segmento_do_contorno(angulos, medios)
    raios_medios = nucleo.raio_contorno(momentos_b[inicio], momentos_h[inicio], momentos_b[fim], momentos_h[fim],
                                        medios)
    for nome, valores, esperado in (("nos vértices (raio R)", raios, 100.0),
                                    ("nas bissetrizes (apótema)", raios_medios, apotema(100.0, LADOS_POLIGONO))):
        registrar(relatorio, "1", "raio do contorno [F71] em polígono regular, máximo %s" % nome, float(valores.max()),
                  esperado, TOLERANCIA_EXATA)
        registrar(relatorio, "1", "raio do contorno [F71] em polígono regular, mínimo %s" % nome, float(valores.min()),
                  esperado, TOLERANCIA_EXATA)
    registrar(relatorio, "1", "[M08] reconhece contorno ordenado e rejeita contorno invertido",
              float(nucleo.contorno_ordenado(angulos) and not nucleo.contorno_ordenado(angulos[::-1])), 1.0, 0.0)
    semieixo_b, semieixo_h = 300.0, 500.0
    for nome, direcao, esperado in (("ψ = 0", 0.0, semieixo_b), ("ψ = 90°", math.radians(90.0), semieixo_h)):
        registrar(relatorio, "1", "raio da elipse [F69] em %s (semieixo)" % nome,
                  float(nucleo.raio_envoltoria_minima(direcao, semieixo_b, semieixo_h)), esperado, TOLERANCIA_EXATA)
    for direcao in (0.3, 0.7, 1.2):
        registrar(relatorio, "1", "ponto da elipse [F69] satisfaz a equação [F41], ψ = %.1f rad" % direcao,
                  equacao_elipse(float(nucleo.raio_envoltoria_minima(direcao, semieixo_b, semieixo_h)), direcao,
                                 semieixo_b, semieixo_h), 1.0, TOLERANCIA_EXATA)
    parametros = np.linspace(0.0, PARAMETRO_DOMINIO_MAXIMO, AMOSTRAS_VETORIAL)
    vetor = nucleo.configuracao_estado_limite(parametros, 30.0, 26.5, material.eps_c2, material.eps_cu)
    escalares = np.array([nucleo.configuracao_estado_limite(float(valor), 30.0, 26.5, material.eps_c2,
                                                            material.eps_cu) for valor in parametros])
    registrar(relatorio, "1", "[F18] vetorial idêntico ao escalar em %d parâmetros de domínio" % AMOSTRAS_VETORIAL,
              float(np.array_equal(vetor[0], escalares[:, 0]) and np.array_equal(vetor[1], escalares[:, 1])), 1.0, 0.0)
    casos = {caso.identificador: caso for caso in [*processar_matriz.montar_teste_minimo(),
                                                   *processar_matriz.montar_matriz()]}
    for identificador in PILARES_ENVOLTORIA_TESTE:
        pilar = casos[identificador].pilar
        resultado = nucleo.dimensionar_pilar(pilar)
        with discretizacao_envoltoria(metade(nucleo.PASSO_ANGULO_LINHA_NEUTRA), metade(nucleo.PASSO_DIRECAO_ENVOLTORIA)):
            refinado = nucleo.dimensionar_pilar(pilar)
        registrar(relatorio, "1", "envoltória mínima: |η com passos pela metade - η| [N04] — %s (η = %.4f)" % (
            identificador, resultado.envoltoria.fator), diferenca_refino(refinado.envoltoria.fator,
                                                                          resultado.envoltoria.fator), 0.0,
                  TOLERANCIA_REFINO, absoluta=True)
        for direcao, rotulo, altura, largura in ((resultado.envoltoria.direcao_b, "b_w", pilar.largura_bw, pilar.altura_h),
                                                 (resultado.envoltoria.direcao_h, "h", pilar.altura_h, pilar.largura_bw)):
            if direcao.metodo != nucleo.METODO_CURVATURA:
                continue
            etapas = etapas_roteiro_independentes(pilar.comprimento_equivalente, altura, largura,
                                                  pilar.forca_normal_calculo, nucleo.EXCENTRICIDADE_ENVOLTORIA_MINIMA,
                                                  pilar.fck_mpa)
            registrar(relatorio, "1", "semieixo da elipse Md,tot,mín na direção de %s, recalculado (curvatura "
                      "aproximada, só M1d,mín, αb = 1) — %s" % (rotulo, identificador), direcao.momento_total,
                      momento_total_curvatura_independente(etapas["alfa_b"], etapas["M1d_A"],
                                                           pilar.forca_normal_calculo, pilar.comprimento_equivalente,
                                                           etapas["curvatura"]), TOLERANCIA_EXATA, "kN·cm")
    for identificador in PILARES_PONTO_ENVOLTORIA:
        pilar = casos[identificador].pilar
        resultado = nucleo.dimensionar_pilar(pilar)
        candidato = resultado.candidato
        material_pilar = nucleo.Material(pilar.fck_mpa, pilar.fyk_mpa)
        areas = nucleo.areas_barras_secao(candidato.area, nucleo.fracao_area_barras_iguais(candidato.x.size))
        for graus in ANGULOS_PONTO_ENVOLTORIA:
            angulo = math.radians(graus)
            with discretizacao_angulos(np.array([angulo])):
                momentos_b, momentos_h = nucleo.envoltoria_resistente(
                    pilar.largura_bw, pilar.altura_h, candidato.x, candidato.y, candidato.area, material_pilar,
                    pilar.forca_normal_calculo, resultado.momentos_resistentes[nucleo.DIRECAO_B],
                    resultado.momentos_resistentes[nucleo.DIRECAO_H])
            altura_projetada = projecao_obliqua(pilar.largura_bw, pilar.altura_h, angulo)
            profundidade_util = float(projecao_obliqua(candidato.x, candidato.y, angulo).max())

            def fibras_no_parametro(parametro: float) -> tuple[float, float, float]:
                return esforcos_obliquos_por_fibras(
                    *nucleo.configuracao_estado_limite(parametro, altura_projetada, profundidade_util,
                                                       material_pilar.eps_c2, material_pilar.eps_cu),
                    angulo, pilar.largura_bw, pilar.altura_h, material_pilar.tensao_pico_resistencia,
                    material_pilar.eps_c2, material_pilar.expoente, candidato.x, candidato.y, areas,
                    material_pilar.fyd, FIBRAS_POR_LADO, nucleo.DESCONTAR_AREA_BARRAS)

            parametro = brentq(lambda valor: residuo_forca_fibras(fibras_no_parametro(valor)[0],
                                                                  pilar.forca_normal_calculo), 0.0, PARAMETRO_DOMINIO_MAXIMO,
                               xtol=TOLERANCIA_RAIZ_FIBRAS)
            _, momento_h_fibras, momento_b_fibras = fibras_no_parametro(parametro)
            for nome, calculado, esperado in (("M_b", momentos_b[1], momento_b_fibras),
                                              ("M_h", momentos_h[1], momento_h_fibras)):
                registrar(relatorio, "1", "ponto da envoltória resistente a Nd, %s, α = %.0f°, refeito com fibras e "
                          "equilíbrio próprio — %s" % (nome, graus, identificador), float(calculado), esperado,
                          TOLERANCIA_FIBRAS, "kN·cm")


def nivel_1_metodo_geral_elastico(relatorio: list[dict], comparacoes: list[dict]) -> None:
    """Verificação do solver do método geral contra a solução exata elástica (fórmula da secante)."""
    for razao in RAZOES_CARGA_CRITICA:
        teste = teste_elastico(razao)
        comparacoes.append(teste)
        registrar(relatorio, "1", "método geral elástico x fórmula da secante, N/Ne = %.1f (%s em %d iterações)" % (
            razao, teste["situacao_metodo_geral"], teste["iteracoes_metodo_geral"]),
                  teste["Mmax_metodo_geral_kNcm"], teste["Mmax_secante_kNcm"], TOLERANCIA_SECANTE, "kN·cm")
        refinado = teste_elastico(razao, SEGMENTOS_REFINADOS)
        registrar(relatorio, "1", "método geral elástico: ordem de convergência com %d e %d segmentos, N/Ne = %.1f" % (
            SEGMENTOS_PILAR, SEGMENTOS_REFINADOS, razao),
                  ordem_observada(teste["Mmax_metodo_geral_kNcm"], refinado["Mmax_metodo_geral_kNcm"],
                                  teste["Mmax_secante_kNcm"], FATOR_REFINO_CONVERGENCIA), ORDEM_TEORICA,
                  TOLERANCIA_ORDEM, absoluta=True)


def memorial_pilar(pilar: nucleo.PilarTeste) -> dict:
    """Refaz à mão as etapas do roteiro normativo e verifica o equilíbrio do estado final (direção de b_w, a de e1;
    na seção vista nessa direção, a altura é b_w e a largura é h)."""
    resultado = nucleo.dimensionar_pilar(pilar)
    pilar = resultado.pilar
    material = nucleo.Material(pilar.fck_mpa, pilar.fyk_mpa)
    direcao = resultado.direcao_b
    candidato = resultado.candidato
    secao = nucleo.secao_na_direcao(pilar.largura_bw, pilar.altura_h, candidato.x, candidato.y, nucleo.DIRECAO_B)
    area_concreto = nucleo.area_secao_retangular(pilar.largura_bw, pilar.altura_h)
    area_calculada = direcao.area_necessaria
    parametro = nucleo.parametro_ultimo_para_forca(secao, material, area_calculada, resultado.forca_normal)
    deformacao_topo, curvatura = nucleo.configuracao_no_parametro(secao, material, parametro)
    linha_neutra = profundidade_linha_neutra(deformacao_topo, curvatura)
    resultante, posicao = resultante_bloco_parabolico(material.eps_c2, material.eps_cu,
                                                      material.tensao_pico_resistencia, pilar.altura_h, linha_neutra)
    areas_camadas = nucleo.areas_barras_secao(area_calculada, secao.fracoes_area)
    deformacoes_barras = deformacao_barras_estado_final(material.eps_cu, linha_neutra, secao.profundidades)
    tensoes_deslocado = (tensao_concreto_barras_parabola(deformacoes_barras, material.tensao_pico_resistencia,
                                                         material.eps_c2, material.expoente)
                         if nucleo.DESCONTAR_AREA_BARRAS else np.zeros_like(deformacoes_barras))
    forca_manual, momento_manual, deformacoes_aco, tensoes_aco = equilibrio_manual(
        resultante, posicao, linha_neutra, material.eps_cu, pilar.largura_bw, secao.profundidades, areas_camadas,
        material.fyd, tensoes_deslocado)
    memorial = {
        "pilar": pilar.identificador, "secao": "%.0f x %.0f cm" % (pilar.largura_bw, pilar.altura_h),
        "fck_MPa": pilar.fck_mpa, "lambda": resultado.esbeltez_b, "nu": resultado.nu, "arranjo": pilar.arranjo,
        "e1_cm": pilar.excentricidade_primeira_ordem, "M1d_aplicado_kNcm": direcao.momento_aplicado,
        "fcd_kN_cm2": material.fcd, "fyd_kN_cm2": material.fyd, "Ac_cm2": area_concreto,
        "Nd_kN": resultado.forca_normal, "le_cm": resultado.comprimento_equivalente,
        "M1d_min_kNcm": direcao.momento_minimo, "alfa_b": direcao.alfa_b, "lambda1": direcao.esbeltez_limite,
        "metodo": direcao.metodo,
        "ea_cm": direcao.excentricidade_acidental, "Nsg_kN": direcao.forca_quase_permanente,
        "Ne_kN": direcao.forca_euler, "ecc_cm": direcao.excentricidade_fluencia,
        "M1d_A_kNcm": direcao.momento_primeira_ordem, "curvatura_1_cm": direcao.curvatura,
        "Md_tot_kNcm": direcao.momento_total, "As_nec_cm2": area_calculada,
        "As_min_cm2": resultado.area_minima, "As_calc_cm2": resultado.area_calculada,
        "barras": "%d phi %.1f mm" % (candidato.numero_barras, processar_matriz.cm_para_mm(candidato.diametro)),
        "As_adot_cm2": candidato.area, "d_linha_cm": candidato.distancia_eixo,
        "parametro_t": parametro, "eps_topo": deformacao_topo, "x_cm": linha_neutra,
        "Rc_kN": resultante, "a_cm": posicao, "camadas_y_cm": secao.profundidades.tolist(),
        "Rc_deslocado_kN": forca_concreto_deslocado(areas_camadas, tensoes_deslocado),
        "secao_liquida": bool(nucleo.DESCONTAR_AREA_BARRAS),
        "eps_camadas": deformacoes_aco.tolist(), "sigma_camadas_kN_cm2": tensoes_aco.tolist(),
        "N_manual_kN": forca_manual, "M_manual_kNcm": momento_manual,
        "MRd_As_nec_kNcm": nucleo.momento_resistente(secao, material, area_calculada, resultado.forca_normal),
    }
    resultante_ret, posicao_ret = resultante_bloco_retangular(material.fcd, material.eta_c, pilar.altura_h,
                                                              linha_neutra)
    tensoes_deslocado_ret = (tensao_concreto_barras_retangular(material.fcd, material.eta_c, linha_neutra,
                                                               secao.profundidades)
                             if nucleo.DESCONTAR_AREA_BARRAS else np.zeros_like(deformacoes_barras))
    forca_ret, momento_ret, _, _ = equilibrio_manual(resultante_ret, posicao_ret, linha_neutra, material.eps_cu,
                                                     pilar.largura_bw, secao.profundidades, areas_camadas, material.fyd,
                                                     tensoes_deslocado_ret)
    memorial.update({"N_retangular_kN": forca_ret, "M_retangular_kNcm": momento_ret})
    return memorial


def linha_elastica(momento_primeira_ordem: float, forca_normal: float, comprimento: float,
                   curvatura_do_momento, segmentos: int = SEGMENTOS_PILAR) -> tuple[float, str, int]:
    """Linha elástica por diferenças finitas ([F10]) com iteração de ponto fixo sobre as flechas.
    curvatura_do_momento(M_ext) devolve as curvaturas nos nós, ou None se não houver equilíbrio.
    Devolve (M_max, situação de [R02], iterações)."""
    passo = passo_segmento(comprimento, segmentos)
    banda = matriz_banda_segunda_derivada(segmentos - 1)
    flechas = np.zeros(segmentos + 1)
    for iteracao in range(1, ITERACOES_METODO_GERAL + 1):
        curvaturas_nos = curvatura_do_momento(momentos_externos(momento_primeira_ordem, forca_normal, flechas))
        if curvaturas_nos is None:
            return math.inf, SITUACAO_INSTAVEL, iteracao
        novas = np.zeros_like(flechas)
        novas[1:-1] = solve_banded((1, 1), banda, termo_independente(curvaturas_nos[1:-1], passo))
        convergiu = diferenca_maxima(novas, flechas) < PRECISAO_FLECHA
        flechas = novas
        if convergiu:
            return (float(momentos_externos(momento_primeira_ordem, forca_normal, flechas).max()),
                    SITUACAO_CONVERGIU, iteracao)
    return (float(momentos_externos(momento_primeira_ordem, forca_normal, flechas).max()), SITUACAO_NAO_CONVERGIU,
            ITERACOES_METODO_GERAL)


def metodo_geral(pilar: nucleo.PilarTeste, resultado: nucleo.ResultadoPilar, area: float | None = None,
                 fator_carga: float = FATOR_CARGA_SUPERIOR) -> tuple[float, str, int]:
    """M_d,tot pelo método geral com a curva M-1/r de cálculo (1,10 fcd e Nd/gama_f3), nas posições das barras da
    armadura adotada. Sem argumentos opcionais: área adotada e carga de cálculo. Com 'area' e 'fator_carga': área
    e fator de [R09] (comparação complementar)."""
    material = nucleo.Material(pilar.fck_mpa)
    direcao = resultado.direcao_b
    secao = nucleo.secao_na_direcao(pilar.largura_bw, pilar.altura_h, resultado.candidato.x, resultado.candidato.y,
                                    nucleo.DIRECAO_B)
    if area is None:
        area = resultado.candidato.area
    forca_normal = carga_multiplicada(resultado.forca_normal, fator_carga)
    forca_deformacao = nucleo.forca_normal_curva_deformacao(forca_normal)
    curvaturas, momentos = nucleo.curva_momento_curvatura(secao, material, area, forca_deformacao, True,
                                                          PONTOS_CURVA_GERAL)
    pico = int(np.argmax(momentos))

    def curvatura_inelastica(momentos_nos: np.ndarray):
        reduzidos = momentos_reduzidos(momentos_nos)
        if reduzidos.max() > momentos[pico]:
            return None
        return np.interp(reduzidos, momentos[:pico + 1], curvaturas[:pico + 1])

    return linha_elastica(carga_multiplicada(direcao.momento_primeira_ordem, fator_carga), forca_normal,
                          resultado.comprimento_equivalente, curvatura_inelastica)


def intervalo_bissecao(criterio, inferior: float, superior: float) -> tuple[float, float]:
    """Bisseção de [R09] para um critério monótono, falso em 'inferior' e verdadeiro em 'superior': devolve o
    intervalo final (inferior, superior), com largura relativa ([F56]) abaixo de TOLERANCIA_BISSECAO."""
    while largura_relativa(inferior, superior) > TOLERANCIA_BISSECAO:
        meio = ponto_medio(inferior, superior)
        if criterio(meio):
            superior = meio
        else:
            inferior = meio
    return inferior, superior


def atende_metodo_geral(pilar: nucleo.PilarTeste, resultado: nucleo.ResultadoPilar, secao: nucleo.SecaoDirecao,
                        area: float) -> bool:
    """Critério de [R09]: o método geral converge com a área dada e M_Rd(área) >= M_d,tot do método geral."""
    momento_geral, situacao, _ = metodo_geral(pilar, resultado, area)
    if situacao != SITUACAO_CONVERGIU:
        return False
    return nucleo.momento_resistente(secao, nucleo.Material(pilar.fck_mpa), area, resultado.forca_normal) \
        >= momento_geral


def area_necessaria_metodo_geral(pilar: nucleo.PilarTeste, resultado: nucleo.ResultadoPilar,
                                 secao: nucleo.SecaoDirecao) -> float:
    """Menor área que atende ao método geral ([R09]), nas posições das barras adotadas; math.inf acima do limite de
    busca do núcleo (8 % de A_c)."""
    limite = nucleo.armadura_limite_busca(resultado.area_concreto)
    if not atende_metodo_geral(pilar, resultado, secao, limite):
        return math.inf
    _, superior = intervalo_bissecao(lambda area: atende_metodo_geral(pilar, resultado, secao, area), 0.0, limite)
    return superior


def sem_equilibrio(pilar: nucleo.PilarTeste, resultado: nucleo.ResultadoPilar, fator_carga: float) -> bool:
    """Critério do fator de carga de [R09]: o método geral não encontra equilíbrio com a armadura adotada."""
    return metodo_geral(pilar, resultado, None, fator_carga)[1] != SITUACAO_CONVERGIU


def fator_carga_equilibrio(pilar: nucleo.PilarTeste, resultado: nucleo.ResultadoPilar) -> float:
    """Maior fator de carga com equilíbrio pelo método geral e a armadura adotada ([R09])."""
    if sem_equilibrio(pilar, resultado, FATOR_CARGA_INFERIOR):
        return FATOR_CARGA_INFERIOR
    inferior, _ = intervalo_bissecao(lambda fator: sem_equilibrio(pilar, resultado, fator), FATOR_CARGA_INFERIOR,
                                     FATOR_CARGA_SUPERIOR)
    return inferior


def teste_elastico(razao: float, segmentos: int = SEGMENTOS_PILAR) -> dict:
    """Método geral com material elástico linear x fórmula da secante x pilar-padrão elástico (l²/10)."""
    carga_critica = carga_critica_euler(RIGIDEZ_TESTE_ELASTICO, COMPRIMENTO_TESTE_ELASTICO)
    forca = forca_da_razao(razao, carga_critica)
    momento_geral, situacao, iteracoes = linha_elastica(
        MOMENTO_TESTE_ELASTICO, forca, COMPRIMENTO_TESTE_ELASTICO,
        lambda momentos_nos: curvatura_elastica(momentos_nos, RIGIDEZ_TESTE_ELASTICO), segmentos)
    return {"tipo": "elastico: metodo geral x secante x pilar-padrao", "N/Ne": razao, "Ne_kN": carga_critica,
            "segmentos": segmentos, "estimativa_erro_relativo": estimativa_erro_secante(razao, segmentos),
            "N_kN": forca, "M0_kNcm": MOMENTO_TESTE_ELASTICO, "EI_kNcm2": RIGIDEZ_TESTE_ELASTICO,
            "l_cm": COMPRIMENTO_TESTE_ELASTICO,
            "Mmax_secante_kNcm": momento_maximo_secante(MOMENTO_TESTE_ELASTICO, forca, carga_critica),
            "Mmax_metodo_geral_kNcm": momento_geral, "situacao_metodo_geral": situacao,
            "iteracoes_metodo_geral": iteracoes,
            "Mtot_pilar_padrao_kNcm": momento_total_pilar_padrao_elastico(MOMENTO_TESTE_ELASTICO, forca,
                                                                          COMPRIMENTO_TESTE_ELASTICO,
                                                                          RIGIDEZ_TESTE_ELASTICO)}


def nivel_2(relatorio: list[dict], memoriais: list[dict], comparacoes: list[dict]) -> None:
    casos = tuple(processar_matriz.criar_pilar(*dados).pilar for dados in CASOS_MEMORIAL)
    for pilar in casos:
        memorial = memorial_pilar(pilar)
        memoriais.append(memorial)
        registrar(relatorio, "2", "%s: N do estado final à mão (bloco parábola-retângulo) = Nd" % pilar.identificador,
                  memorial["N_manual_kN"], memorial["Nd_kN"], TOLERANCIA_EXATA, "kN")
        registrar(relatorio, "2", "%s: M do estado final à mão (bloco parábola-retângulo) = M_Rd(As,nec)" %
                  pilar.identificador, memorial["M_manual_kNcm"], memorial["MRd_As_nec_kNcm"], TOLERANCIA_EXATA,
                  "kN·cm")
        registrar(relatorio, "2", "%s: M_Rd(As,nec) >= Md,tot (folga não negativa)" % pilar.identificador,
                  float(memorial["MRd_As_nec_kNcm"] >= memorial["Md_tot_kNcm"]), 1.0, 0.0)
        registrar(relatorio, "2", "%s: M com diagrama retangular simplificado ≈ Md,tot" % pilar.identificador,
                  memorial["M_retangular_kNcm"], memorial["Md_tot_kNcm"], TOLERANCIA_RETANGULAR, "kN·cm")
        etapas = etapas_roteiro_independentes(memorial["le_cm"], pilar.largura_bw, pilar.altura_h, memorial["Nd_kN"],
                                              memorial["e1_cm"], memorial["fck_MPa"])
        for nome, chave_memorial, chave_etapa in (("λ", "lambda", "lambda"), ("M1d,mín", "M1d_min_kNcm", "M1d_min"),
                                                  ("αb", "alfa_b", "alfa_b"), ("λ1", "lambda1", "lambda1"),
                                                  ("ν", "nu", "nu")):
            registrar(relatorio, "2", "%s: %s recalculado de forma independente" % (pilar.identificador, nome),
                      memorial[chave_memorial], etapas[chave_etapa], TOLERANCIA_EXATA)
        if memorial["metodo"] == nucleo.METODO_CURVATURA:
            registrar(relatorio, "2", "%s: 1/r da curvatura aproximada recalculado" % pilar.identificador,
                      memorial["curvatura_1_cm"], etapas["curvatura"], TOLERANCIA_EXATA, "1/cm")
            registrar(relatorio, "2", "%s: Md,tot = αb M1d,A + Nd (ℓe²/10)(1/r) recalculado" % pilar.identificador,
                      memorial["Md_tot_kNcm"], momento_externo_pilar_padrao(
                          etapas["alfa_b"], etapas["M1d_A"], memorial["Nd_kN"], memorial["le_cm"],
                          etapas["curvatura"]), TOLERANCIA_EXATA, "kN·cm")
    for caso in processar_matriz.montar_matriz():
        pilar = caso.pilar
        if caso.esbeltez <= nucleo.LAMBDA_LIMITE_CURVATURA:
            if caso.esbeltez > nucleo.LAMBDA1_MIN:
                material = nucleo.Material(pilar.fck_mpa, pilar.fyk_mpa)
                area_concreto = nucleo.area_secao_retangular(pilar.largura_bw, pilar.altura_h)
                forca = pilar.forca_normal_calculo
                comprimento = pilar.comprimento_equivalente
                momento_aplicado = nucleo.momento_excentricidade(forca, pilar.excentricidade_primeira_ordem)
                momento_minimo = nucleo.momento_minimo_primeira_ordem(forca, pilar.largura_bw)
                momento_a = nucleo.momento_primeira_ordem_adotado(momento_aplicado, momento_minimo)
                alfa_b = nucleo.coeficiente_alfa_b(
                    momento_aplicado, nucleo.momento_extremidade_b(momento_aplicado, nucleo.RAZAO_MOMENTOS_EXTREMIDADE),
                    nucleo.momentos_menores_que_minimo(momento_aplicado, momento_minimo))
                curvatura = nucleo.curvatura_aproximada(nucleo.forca_normal_adimensional(forca, area_concreto,
                                                                                          material.fcd), pilar.largura_bw)
                comparacoes.append({
                    "pilar": pilar.identificador, "tipo": "curvatura x kappa", "lambda": caso.esbeltez,
                    "nu": caso.nu, "e1/b_w": caso.excentricidade_relativa, "b_w": pilar.largura_bw,
                    "Md_curvatura_kNcm": nucleo.momento_total_pilar_padrao(alfa_b, momento_a, forca, comprimento,
                                                                           curvatura),
                    "Md_kappa_kNcm": momento_total_rigidez_kappa(pilar.largura_bw, forca, comprimento, alfa_b,
                                                                 momento_a)})
            continue
        resultado = nucleo.dimensionar_pilar(pilar)
        if resultado.candidato is None or resultado.direcao_b.metodo != nucleo.METODO_MN1R:
            continue
        momento_geral, situacao, iteracoes = metodo_geral(pilar, resultado)
        secao = nucleo.secao_na_direcao(pilar.largura_bw, pilar.altura_h, resultado.candidato.x,
                                        resultado.candidato.y, nucleo.DIRECAO_B)
        equilibrio = nucleo.equilibrio_pilar_padrao_acoplado(
            secao, nucleo.Material(pilar.fck_mpa), resultado.candidato.area, resultado.forca_normal,
            resultado.direcao_b.momento_primeira_ordem, resultado.direcao_b.alfa_b, resultado.comprimento_equivalente)
        momento_resistente_adotada = resultado.momentos_resistentes[nucleo.DIRECAO_B]
        area_geral = area_necessaria_metodo_geral(pilar, resultado, secao)
        comparacoes.append({"pilar": pilar.identificador, "tipo": "pilar-padrao x metodo geral",
                            "lambda": caso.esbeltez, "nu": caso.nu, "e1/b_w": caso.excentricidade_relativa,
                            "b_w": pilar.largura_bw, "h": pilar.altura_h, "fck_MPa": pilar.fck_mpa,
                            "Md_pilar_padrao_kNcm": equilibrio[1] if equilibrio else math.inf,
                            "Md_metodo_geral_kNcm": momento_geral, "situacao_metodo_geral": situacao,
                            "iteracoes_metodo_geral": iteracoes,
                            "As_calc_cm2": resultado.area_calculada, "As_min_cm2": resultado.area_minima,
                            "As_adot_cm2": resultado.candidato.area, "As_max_cm2": resultado.area_maxima,
                            "MRd_As_adot_kNcm": momento_resistente_adotada,
                            "Md_metodo_geral/MRd": (razao_comparacao(momento_geral, momento_resistente_adotada)
                                                    if situacao == SITUACAO_CONVERGIU else None),
                            "atende_metodo_geral_As_adot": (situacao == SITUACAO_CONVERGIU
                                                            and momento_resistente_adotada >= momento_geral),
                            "As_nec_metodo_geral_cm2": area_geral,
                            "As_nec_metodo_geral/As_adot": razao_comparacao(area_geral, resultado.candidato.area),
                            "As_nec_metodo_geral_acima_As_max": area_geral > resultado.area_maxima,
                            "fator_carga_equilibrio": (fator_carga_equilibrio(pilar, resultado)
                                                       if situacao == SITUACAO_INSTAVEL else None)})
        comparacoes.append(comparacao_secante(caso, resultado, secao, equilibrio))
    situacoes_geral = [comparacao["situacao_metodo_geral"] for comparacao in comparacoes
                       if comparacao["tipo"] == "pilar-padrao x metodo geral"]
    registrar(relatorio, "1", "método geral: nenhum pilar terminou sem convergir (%d convergiram, %d instáveis)" % (
        situacoes_geral.count(SITUACAO_CONVERGIU), situacoes_geral.count(SITUACAO_INSTAVEL)),
              float(situacoes_geral.count(SITUACAO_NAO_CONVERGIU)), 0.0, 0.0)


def comparacao_secante(caso, resultado: nucleo.ResultadoPilar, secao: nucleo.SecaoDirecao, equilibrio) -> dict:
    """Pilar-padrão acoplado pela curva de deformação (rotina) x pela rigidez secante AB ([F55], forma do CAD/Pilar),
    com a armadura adotada e o mesmo M1d,A com fluência."""
    pilar = resultado.pilar
    material = nucleo.Material(pilar.fck_mpa, pilar.fyk_mpa)
    area = resultado.candidato.area
    direcao = resultado.direcao_b
    momento_resistente = resultado.momentos_resistentes[nucleo.DIRECAO_B]
    forca_deformacao = nucleo.forca_normal_curva_deformacao(resultado.forca_normal)
    limite = nucleo.curvatura_ultima(secao, material, area, forca_deformacao, True)
    curvatura_b = brentq(lambda curvatura: diferenca_momento_curva(
        nucleo.momento_curva_deformacao(secao, material, area, forca_deformacao, curvatura),
        momento_ponto_b(momento_resistente)), nucleo.CURVATURA_NULA, limite, xtol=1e-14)
    rigidez = rigidez_secante(momento_resistente, curvatura_b)
    kappa_sec = kappa_secante(rigidez, resultado.area_concreto, pilar.largura_bw, material.fcd)
    return {"pilar": pilar.identificador, "tipo": "curva x secante", "lambda": caso.esbeltez, "nu": caso.nu,
            "e1/b_w": caso.excentricidade_relativa, "b_w": pilar.largura_bw, "MRd_kNcm": momento_resistente,
            "curvatura_B_1_cm": curvatura_b, "EI_sec_kNcm2": rigidez, "kappa_sec": kappa_sec,
            "Md_curva_kNcm": equilibrio[1] if equilibrio else math.inf,
            "Md_secante_kNcm": momento_total_kappa_secante(direcao.alfa_b, direcao.momento_primeira_ordem,
                                                           resultado.esbeltez_b, resultado.nu, kappa_sec)}


def secao_exemplo_bastos(exemplo: dict) -> nucleo.SecaoDirecao:
    """Seção do exemplo de Bastos no arranjo do ábaco que o autor usou ([R04])."""
    if exemplo["arranjo"] == ARRANJO_TRES_POR_FACE_BASTOS:
        return nucleo.secao_por_camadas(exemplo["altura"], exemplo["largura"],
                                        profundidades_tres_por_face(exemplo["altura"], DISTANCIA_BASTOS))
    return secao_duas_camadas(exemplo["altura"], exemplo["largura"], DISTANCIA_BASTOS)


def nivel_2_bastos(relatorio: list[dict], comparacoes: list[dict], engenharia: list[dict]) -> None:
    """Exemplos resolvidos de Bastos (2017) ([R04]) em dupla verificação ([F49]):
    1) testes: o passo a passo do autor refeito com valores exatos (raiz(12), ν e 1/r sem arredondamento) coincide
       com a rotina etapa por etapa;
    2) engenharia reversa (informativa): o mesmo passo a passo com 3,46 e com os valores impressos reproduz cada
       valor publicado, e a diferença rotina x publicado é decomposta em arredondamento da etapa e efeito propagado;
       a taxa mecânica ω da seção da rotina, no ponto (ν, μ) em que o autor entrou no ábaco (A-4, A-27 ou A-29),
       com a seção no arranjo e na hipótese do ábaco (seção bruta), é comparada com o ω lido por ele ([N04])."""
    material = nucleo.Material(FCK_BASTOS)
    for exemplo in EXEMPLOS_BASTOS:
        rotulo = "Bastos (2017) %s, %s" % (exemplo["exemplo"], exemplo["paginas"])
        largura, altura = exemplo["largura"], exemplo["altura"]
        textos = exemplo["impresso"]
        area_concreto = nucleo.area_secao_retangular(largura, altura)
        impresso = {chave: valor_impresso(texto) for chave, texto in textos.items()}
        forca = impresso["Nd"]
        secao = secao_exemplo_bastos(exemplo)
        direcao = nucleo.analisar_direcao(secao, material, forca, COMPRIMENTO_BASTOS, nucleo.DIRECAO_B,
                                          excentricidade_do_momento(exemplo["M1d_aplicado"], forca))
        exata = cadeia_exata_bastos(exemplo, forca)
        autor = cadeia_autor_bastos(exemplo, impresso, exata)
        momento_kappa = momento_total_rigidez_kappa(altura, forca, COMPRIMENTO_BASTOS, direcao.alfa_b,
                                                    direcao.momento_primeira_ordem)
        rotina = {"lambda": direcao.esbeltez, "M1d_min": direcao.momento_minimo, "lambda1": direcao.esbeltez_limite,
                  "alfa_b": direcao.alfa_b, "nu": nucleo.forca_normal_adimensional(forca, area_concreto, material.fcd),
                  "curvatura": direcao.curvatura, "Md_curvatura": direcao.momento_total, "Md_kappa": momento_kappa,
                  "mu_curvatura": momento_adimensional_bastos(direcao.momento_total, altura, area_concreto,
                                                              material.fcd),
                  "mu_kappa": momento_adimensional_bastos(momento_kappa, altura, area_concreto, material.fcd)}
        for chave, nome, unidade, _ in ETAPAS_BASTOS:
            if chave not in ETAPAS_ROTINA_BASTOS or chave not in textos:
                continue
            registrar(relatorio, "2", "%s: %s — passo a passo do autor com valores exatos = rotina" % (rotulo, nome),
                      rotina[chave], exata[chave], TOLERANCIA_EXATA, unidade)
        registrar(relatorio, "2", "%s: método de 2ª ordem — rotina = passo a passo exato = publicado (%s)" % (
            rotulo, exemplo["metodo"]), float(direcao.metodo == exata["metodo"] == exemplo["metodo"]), 1.0, 0.0)
        if "Md_kappa" in textos:
            registrar(relatorio, "2", "%s: Md,tot (rigidez κ) — Eq. 41 do autor com λ exato = forma da NBR [F09]" % (
                rotulo), rotina["Md_kappa"], exata["Md_kappa"], TOLERANCIA_EXATA, "kN·cm")
        complementos = {}
        if "omega_curvatura" in textos:
            taxas = {}
            comparacao = {"pilar": "Bastos %s" % exemplo["exemplo"], "tipo": "Bastos (ábaco %s)" % exemplo["abaco"],
                          "paginas": exemplo["paginas"], "abaco": exemplo["abaco"], "arranjo": exemplo["arranjo"],
                          "Md_curvatura_kNcm": direcao.momento_total,
                          "Md_kappa_kNcm": momento_kappa if "Md_kappa" in textos else None}
            for chave_omega, chave_mu, chave_momento in PARES_OMEGA_BASTOS:
                if chave_omega not in textos:
                    continue
                for descontar in (True, False):
                    with hipotese_secao(descontar):
                        areas = (nucleo.armadura_flexo_compressao(
                                     secao, material, nucleo.forca_normal_de_nu(impresso["nu"], area_concreto,
                                                                                material.fcd),
                                     momento_do_mu(impresso[chave_mu], altura, area_concreto, material.fcd)),
                                 nucleo.armadura_flexo_compressao(secao, material, forca, exata[chave_momento]),
                                 nucleo.armadura_flexo_compressao(secao, material, forca, rotina[chave_momento]))
                    taxas[(chave_omega, descontar)] = tuple(
                        taxa_mecanica(area, material.fyd, area_concreto, material.fcd) for area in areas)
                autor[chave_omega], exata[chave_omega], rotina[chave_omega] = taxas[(chave_omega,
                                                                                    SECAO_LIQUIDA_ABACO)]
                complementos[chave_omega] = ("Seção bruta, como no ábaco (Venturini e Rodrigues, 1987, eq. 4.6); com a "
                                             "seção líquida da rotina: ω = %s no ponto do autor e %s no ponto exato."
                                             % (("%.4f" % taxas[(chave_omega, True)][0]).replace(".", ","),
                                                ("%.4f" % taxas[(chave_omega, True)][2]).replace(".", ",")))
                comparacao.update({chave_omega + "_publicado": impresso[chave_omega],
                                   chave_omega + "_liquida": taxas[(chave_omega, True)][2],
                                   chave_omega + "_bruta": taxas[(chave_omega, False)][2],
                                   chave_omega + "_ponto_autor_liquida": taxas[(chave_omega, True)][0],
                                   chave_omega + "_ponto_autor_bruta": taxas[(chave_omega, False)][0]})
            comparacoes.append(comparacao)
        for chave, nome, unidade, criterio in ETAPAS_BASTOS:
            if chave in textos:
                linha = linha_engenharia_reversa(exemplo, chave, nome, unidade, criterio, textos[chave], autor[chave],
                                                 exata[chave], rotina.get(chave))
                if chave in complementos:
                    linha["observacao"] = " ".join(texto for texto in (linha["observacao"], complementos[chave]) if texto)
                engenharia.append(linha)


def momento_fibras_com_forca(curvatura: float, forca_alvo: float, altura: float, largura: float,
                             tensao_pico: float, material: nucleo.Material, profundidades: np.ndarray,
                             areas_camadas: np.ndarray) -> float:
    """Momento do modelo de fibras independente ([F06]) na curvatura dada, com a deformação do topo que
    equilibra a força normal alvo (busca de Brent)."""
    def residuo_forca(deformacao_topo: float) -> float:
        return nucleo.residuo(esforcos_por_fibras(deformacao_topo, curvatura, altura, largura, tensao_pico,
                                                  material.eps_c2, material.expoente, profundidades, areas_camadas,
                                                  material.fyd, NUMERO_FIBRAS, nucleo.DESCONTAR_AREA_BARRAS)[0],
                              forca_alvo)

    deformacao_topo = brentq(residuo_forca, DEFORMACAO_INFERIOR_BUSCA,
                             deformacao_superior_busca(curvatura, altura, material.eps_c2), xtol=1e-14)
    return esforcos_por_fibras(deformacao_topo, curvatura, altura, largura, tensao_pico, material.eps_c2,
                               material.expoente, profundidades, areas_camadas, material.fyd, NUMERO_FIBRAS,
                               nucleo.DESCONTAR_AREA_BARRAS)[1]


def conferir_mn1r_independente(relatorio: list[dict]) -> None:
    """V3 e V4: fluência e equilíbrio do pilar-padrão acoplado a M, N, 1/r refeitos sem o motor ([F33], [F34])."""
    for dados in CASOS_MEMORIAL:
        pilar = processar_matriz.criar_pilar(*dados).pilar
        resultado = nucleo.dimensionar_pilar(pilar)
        direcao = resultado.direcao_b
        if direcao.metodo != nucleo.METODO_MN1R or resultado.candidato is None:
            continue
        pilar = resultado.pilar
        rotulo = "%s (M, N, 1/r, λ = %.0f)" % (pilar.identificador, resultado.esbeltez_b)
        forca = resultado.forca_normal
        fluencia = fluencia_independente(resultado.comprimento_equivalente, forca, nucleo.GAMA_F,
                                         nucleo.FRACAO_PERMANENTE, nucleo.PSI2, pilar.excentricidade_primeira_ordem,
                                         pilar.fck_mpa, nucleo.ALFA_E_AGREGADO, pilar.altura_h, pilar.largura_bw,
                                         nucleo.COEFICIENTE_FLUENCIA)
        pares = (("ea", direcao.excentricidade_acidental, "cm"), ("Nsg", direcao.forca_quase_permanente, "kN"),
                 ("Msg", direcao.momento_quase_permanente, "kN·cm"), ("Ne", direcao.forca_euler, "kN"),
                 ("ecc", direcao.excentricidade_fluencia, "cm"))
        for chave, calculado, unidade in pares:
            registrar(relatorio, "2", "%s: %s recalculado (15.8.4)" % (rotulo, chave), calculado, fluencia[chave],
                      TOLERANCIA_EXATA, unidade)
        momento_base = momento_primeira_ordem_base(direcao.momento_aplicado, direcao.momento_minimo)
        registrar(relatorio, "2", "%s: M1d,A com fluência = M1d,A + Nd ecc" % rotulo, direcao.momento_primeira_ordem,
                  momento_primeira_ordem_fluencia(momento_base, forca, fluencia["ecc"]), TOLERANCIA_EXATA, "kN·cm")
        momento_externo = momento_externo_pilar_padrao(direcao.alfa_b, direcao.momento_primeira_ordem, forca,
                                                       resultado.comprimento_equivalente, direcao.curvatura)
        registrar(relatorio, "2", "%s: Md,tot = αb M1d,A + Nd (ℓe²/10)(1/r) na curvatura de equilíbrio" % rotulo,
                  direcao.momento_total, momento_externo, TOLERANCIA_EXATA, "kN·cm")
        secao = nucleo.secao_na_direcao(pilar.largura_bw, pilar.altura_h, resultado.candidato.x,
                                        resultado.candidato.y, nucleo.DIRECAO_B)
        material = nucleo.Material(pilar.fck_mpa, pilar.fyk_mpa)
        tensao_pico = tensao_pico_deformacao(nucleo.mpa_para_kn_cm2(pilar.fck_mpa), nucleo.GAMA_C)
        areas_camadas = nucleo.areas_barras_secao(direcao.area_necessaria, secao.fracoes_area)
        forca_deformacao = forca_curva_deformacao(forca)

        def momento_deformacao(curvatura: float) -> float:
            return momento_fibras_com_forca(curvatura, forca_deformacao, pilar.largura_bw, pilar.altura_h,
                                            tensao_pico, material, secao.profundidades, areas_camadas)

        registrar(relatorio, "2", "%s: γf3 M_def(1/r) por fibras (1,10 fcd; Nd/γf3) = M_ext(1/r)" % rotulo,
                  momento_deformacao_majorado(momento_deformacao(direcao.curvatura)), momento_externo, TOLERANCIA_FIBRAS,
                  "kN·cm")
        residuos = [residuo_mn1r(momento_deformacao(curvatura), momento_externo_pilar_padrao(
            direcao.alfa_b, direcao.momento_primeira_ordem, forca, resultado.comprimento_equivalente, curvatura))
            for curvatura in curvaturas_varredura(direcao.curvatura)]
        registrar(relatorio, "2", "%s: equilíbrio estável mais próximo (R < 0 em 49 curvaturas menores)" % rotulo,
                  float(max(residuos) < 0.0), 1.0, 0.0)


def resumo_tolerancias(relatorio: list[dict]) -> list[dict]:
    """Quadro de [R06] com o número de testes e o maior desvio observado de cada limite interno de alarme."""
    quadro = []
    for constante, valor, criterio, origem in TOLERANCIAS:
        testes = [item for item in relatorio if item["constante"] == constante and item["criterio"] == criterio]
        if criterio == "relativo":
            maior = max((item["desvio_percentual"] for item in testes), default=math.nan)
            observado = "Δ%% máx. = %.2g (concordância mín. observada %s %%)" % (maior, texto_concordancia(maior)) \
                if testes else "–"
            exigido = "alarme se concordância < %s %%" % texto_concordancia_minima(valor)
        else:
            maior = max((item["diferenca_absoluta"] for item in testes), default=math.nan)
            observado = "|Δ| máx. = %.2g" % maior if testes else "–"
            exigido = "alarme se |Δ| > %g" % valor
        quadro.append({"constante": constante, "valor": valor, "criterio": criterio, "exigido": exigido,
                       "testes": len(testes), "observado": observado, "maior_desvio": maior, "origem": origem})
    logicos = [item for item in relatorio if item["constante"] == "verificação lógica"]
    quadro.append({"constante": "verificação lógica", "valor": 0.0, "criterio": "lógico",
                   "exigido": "alarme se falsa", "testes": len(logicos),
                   "observado": "%d verdadeiros" % sum(item["aprovado"] for item in logicos), "maior_desvio": 0.0,
                   "origem": "Verificações de sim ou não (método, monotonicidade, convergência, contorno)."})
    return quadro


def texto_valor(valor) -> str:
    return "–" if valor is None else "%.10g" % valor


def texto_reproduz(reproduz) -> str:
    """Coluna "Reproduz": sim/NÃO nos valores impressos; "–" nas leituras de ω (sem critério de reprodução)."""
    return "–" if reproduz is None else ("sim" if reproduz else "NÃO")


def texto_percentual(valor) -> str:
    """Percentual com sinal e 4 casas; o zero negativo do arredondamento (ex.: -1e-15) é exibido como +0.0000."""
    return "–" if valor is None else ("%+.4f" % valor).replace("-0.0000", "+0.0000")


def main() -> int:
    pasta_saida = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "saida"
    pasta_saida.mkdir(parents=True, exist_ok=True)
    relatorio, memoriais, comparacoes, engenharia = [], [], [], []
    nivel_1(relatorio)
    nivel_1_flexao_obliqua(relatorio)
    nivel_1_metodo_geral_elastico(relatorio, comparacoes)
    nivel_2(relatorio, memoriais, comparacoes)
    nivel_2_bastos(relatorio, comparacoes, engenharia)
    conferir_mn1r_independente(relatorio)
    alarmes = [item for item in relatorio if not item["aprovado"]]
    tolerancias = resumo_tolerancias(relatorio)
    impressos = [linha for linha in engenharia if linha["reproduz"] is not None]
    leituras = [linha for linha in engenharia if linha["reproduz"] is None]
    reproduzidos = sum(linha["reproduz"] for linha in impressos)
    (pasta_saida / "validacao.json").write_text(json.dumps({
        "testes": relatorio, "memoriais": memoriais, "comparacoes": comparacoes, "tolerancias": tolerancias,
        "explicacao_niveis": EXPLICACAO_NIVEIS, "engenharia_reversa_bastos": engenharia,
        "resumo_engenharia_reversa": {"valores": len(impressos), "reproduzidos": reproduzidos,
                                      "leituras": len(leituras)}},
        indent=1, ensure_ascii=False, default=float), encoding="utf8")
    linhas = ["VALIDAÇÃO DA ROTINA (validacao.py) — %d testes; alarmes internos: %d" % (len(relatorio), len(alarmes)),
              "", "POR QUE CALCULADO E ESPERADO DIFEREM, POR NÍVEL", *(EXPLICACAO_NIVEIS[nivel] for nivel in ("1", "2")),
              "", "LIMITES INTERNOS DE ALARME ([R06]; não são critério de aceitação, servem para acusar erro de "
              "programação ao reexecutar a rotina): limite, número de testes, maior desvio observado e origem do desvio"]
    for item in tolerancias:
        linhas.append("  %-22s %-9s %-32s %4d testes | %s | %s" % (item["constante"], item["criterio"],
                                                                    item["exigido"], item["testes"], item["observado"],
                                                                    item["origem"]))
    linhas += ["", "TESTES", "%-4s %-120s %14s %14s %-9s %10s %-22s %-14s %s" % (
        "Nív.", "Teste", "Calculado", "Esperado", "Critério", "Δ% ou |Δ|", "Concordância (%)", "Alarme abaixo de",
        "Alarme")]
    for item in relatorio:
        desvio = item["desvio_percentual"] if item["criterio"] == "relativo" else item["diferenca_absoluta"]
        if item["concordancia"] is None:
            concordancia, minima = "–", ("se falsa" if item["tolerancia"] == 0.0 else "|Δ| > %g" % item["tolerancia"])
        else:
            concordancia, minima = item["concordancia"], item["concordancia_minima"]
        linhas.append("%-4s %-120s %14.6g %14.6g %-9s %10.3g %-22s %-14s %s" % (
            item["nivel"], item["teste"][:120], item["calculado"], item["esperado"], item["criterio"], desvio,
            concordancia, minima, "não" if item["aprovado"] else "SIM"))
    linhas += ["", "ENGENHARIA REVERSA DE BASTOS (2017) — %d de %d valores impressos nas etapas de cálculo reproduzidos "
               "pela cadeia do autor (arredondada às casas impressas, com a meia unidade para cima, em módulo); %d "
               "leituras de ω no ábaco comparadas só pela diferença" % (reproduzidos, len(impressos), len(leituras)),
               "  autor = passo a passo com 3,46 e valores impressos; exata = mesmo passo a passo com raiz(12) e sem "
               "arredondamentos; total = (rotina - impresso)/impresso = impressão (autor - impresso) + propagado "
               "(exata - autor), em %",
               "%-28s %-42s %14s %16s %16s %16s %-8s %10s %10s %10s  %s" % (
                   "Exemplo", "Etapa", "Impresso", "Autor", "Exata", "Rotina", "Reproduz", "Total %", "Impressão %",
                   "Propagado %", "Observação")]
    for linha in engenharia:
        linhas.append("%-28s %-42s %14s %16s %16s %16s %-8s %10s %10s %10s  %s" % (
            linha["exemplo"], linha["etapa"][:42], linha["impresso"], texto_valor(linha["autor"]),
            texto_valor(linha["exata"]), texto_valor(linha["rotina"]), texto_reproduz(linha["reproduz"]),
            texto_percentual(linha["desvio_total_percentual"]), texto_percentual(linha["parcela_impressao_percentual"]),
            texto_percentual(linha["parcela_propagada_percentual"]), linha["observacao"]))
    linhas += ["", "COMPARAÇÃO COMPLEMENTAR (fora da validação): PILAR-PADRÃO ACOPLADO x MÉTODO GERAL ([F10], [R02], "
               "[R09]), mesma curva M, N, 1/r, mesmas posições das barras e mesmo M1d,A com fluência",
               "%-6s %8s %6s %5s %6s %-10s %6s %12s %12s %10s %12s %12s %9s %9s" % (
                   "Pilar", "Seção", "λ", "ν", "e1/bw", "Situação", "Iter.", "Md,pp kN·cm", "Md,MG kN·cm",
                   "MG/MRd", "As,adot cm²", "As,MG cm²", "As razão", "Fator")]
    for comparacao in comparacoes:
        if comparacao["tipo"] != "pilar-padrao x metodo geral":
            continue
        linhas.append("%-6s %8s %6.0f %5.1f %6.2f %-10s %6d %12.1f %12s %10s %12.3f %12.3f %9.4f %9s" % (
            comparacao["pilar"], "%.0fx%.0f" % (comparacao["b_w"], comparacao["h"]), comparacao["lambda"],
            comparacao["nu"], comparacao["e1/b_w"], comparacao["situacao_metodo_geral"],
            comparacao["iteracoes_metodo_geral"], comparacao["Md_pilar_padrao_kNcm"],
            texto_valor(comparacao["Md_metodo_geral_kNcm"] if comparacao["situacao_metodo_geral"] == SITUACAO_CONVERGIU
                        else None), texto_valor(comparacao["Md_metodo_geral/MRd"]),
            comparacao["As_adot_cm2"], comparacao["As_nec_metodo_geral_cm2"], comparacao["As_nec_metodo_geral/As_adot"],
            texto_valor(comparacao["fator_carga_equilibrio"])))
    (pasta_saida / "validacao.txt").write_text("\n".join(linhas) + "\n", encoding="utf8")
    print("testes: %d | alarmes internos: %d | memoriais: %d | comparações: %d | engenharia reversa: %d de %d "
          "impressos reproduzidos" % (len(relatorio), len(alarmes), len(memoriais), len(comparacoes), reproduzidos,
                                      len(impressos)))
    for item in alarmes:
        print("ALARME:", item["teste"], item["calculado"], item["esperado"])
    return 1 if alarmes else 0


if __name__ == "__main__":
    sys.exit(main())
