# -*- coding: utf-8 -*-
"""
figuras_tcc.py
==============
Gera, em vetor (SVG) e em imagem de reserva (PNG, 300 dpi), as figuras do TCC calculadas pela rotina:
  1) dominios.svg     : domínios de deformação no estado-limite último (adaptado da ABNT NBR 6118:2023,
                        Figura 17.1), desenho esquemático com as deformações em escala linear (Capítulo 2);
  2) interacao.svg    : diagrama de interação M-N de uma seção da matriz, calculado por pilares_nbr6118.py,
                        com o ponto de cálculo de um pilar-teste (Capítulo 2);
  3) viabilidade.svg  : mapa de viabilidade da matriz de pilares-teste (λ × ν, por nível de e1/b_w), lido de
                        saida/resultados_matriz.csv (Capítulo 4).
Grava também os dados de cada figura em CSV (interacao_dados.csv e viabilidade_dados.csv).

Uso:  python3 figuras_tcc.py [pasta_de_saida]     (padrão: saida/figuras)
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import pilares_nbr6118 as nucleo  # noqa: E402
import processar_matriz  # noqa: E402

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — INÍCIO
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [R01] Formato das figuras: fonte com as mesmas métricas da Times New Roman do texto, corpo 10 pt na largura
#   final (11 pt nos rótulos dos domínios e 9 pt nos textos secundários), linhas finas em preto e cinza,
#   resolução da imagem de reserva.
#   Variáveis: FONTE, CORPO_PT, CORPO_ROTULO_PT, CORPO_SECUNDARIO_PT, CORPO_CELULA_PT, LARGURA_CM,
#   ALTURA_DOMINIOS_CM, ALTURA_INTERACAO_CM, ALTURA_VIABILIDADE_CM, RESOLUCAO_DPI, ESPESSURA_LINHA,
#   CINZA_GRADE, CM_POR_POLEGADA.
#   Usada em: cm_para_polegada (l. 146); configurar_estilo (l. 567); salvar (l. 573);
#       figura_dominios (l. 582, 592, 596, 615, 618, 621, 622, 630); figura_interacao (l. 663, 668, 674, 678);
#       figura_viabilidade (l. 715, 725, 727, 737, 745); formatar_eixos (l. 801);
#       grafico_diagramas (l. 838, 873, 874, 877)
#   Usada também em: figuras_complementares.py: figura_secao (l. 365, 376, 383, 385, 388, 394, 397, 404, 406, 412, 415, 419, 423, 426, 432, 438, 441, 444);
#       figuras_complementares.py: painel_mn1r (l. 493, 503, 509, 519, 521, 525, 530);
#       figuras_complementares.py: figura_mn1r (l. 541, 549, 554, 555, 558);
#       figuras_complementares.py: figura_taxa (l. 578, 611, 613, 619, 621, 623, 624, 636);
#       figuras_complementares.py: figura_auditor (l. 699);
#       figuras_complementares.py: figura_fusco (l. 718, 728, 739, 743, 755)
FONTE = "Liberation Serif"
CORPO_PT = 10
CORPO_ROTULO_PT = 11
CORPO_SECUNDARIO_PT = 9
CORPO_CELULA_PT = 7.5
LARGURA_CM = 15.0
ALTURA_DOMINIOS_CM = 9.0
ALTURA_INTERACAO_CM = 9.5
ALTURA_VIABILIDADE_CM = 9.0
RESOLUCAO_DPI = 300
ESPESSURA_LINHA = 1.0
CINZA_GRADE = "#c8c8c8"
CM_POR_POLEGADA = 2.54

# [R02] Domínios (ABNT NBR 6118:2023, Figura 17.1), concreto do grupo I (fck <= 50 MPa): deformações em ‰
#   (alongamento negativo, encurtamento positivo) e posições relativas na altura (y/h, a partir da face mais
#   comprimida): d/h ilustrativo, aço CA-50 (eps_yd = fyd/Es).
#   Variáveis: EPS_SU, EPS_CU, EPS_C2, RELACAO_D_H, FYD_KN_CM2.
#   Usada em: nível do módulo (l. 115, 119, 123);
#       figura_dominios (l. 579, 580, 581, 584, 585, 586, 587, 588, 589, 590, 594, 596, 598, 599, 600, 622, 623, 631, 633, 634, 635)
EPS_SU = -10.0
EPS_CU = 3.5
EPS_C2 = 2.0
RELACAO_D_H = 0.88
FYD_KN_CM2 = nucleo.resistencia_calculo_aco(nucleo.mpa_para_kn_cm2(nucleo.RESISTENCIA_ACO_MPA))


# [F01] Deformação de escoamento de cálculo do aço na tração, em ‰ e com o sinal do alongamento (negativo):
#   eps_yd = -(fyd / Es) x 1000.
#   Variáveis: fyd (kN/cm²), modulo (kN/cm²).  Fonte: NBR 6118:2023, 8.3.6.
#   Usada em: figura_dominios (l. 579)
def deformacao_escoamento_tracao_por_mil(fyd: float, modulo: float) -> float:
    return -fyd / modulo * 1000.0


# [F02] Profundidade relativa do pivô C: y_C/h = (eps_cu - eps_c2)/eps_cu.  Variáveis: eps_cu, eps_c2.
#   Fonte: NBR 6118:2023, 17.2.2, Figura 17.1.
#   Usada em: figura_dominios (l. 581)
def profundidade_relativa_pivo_c(eps_cu: float, eps_c2: float) -> float:
    return (eps_cu - eps_c2) / eps_cu


# [F08] Abscissa (deformação) de um perfil retilíneo de deformações numa altura y: interpolação linear entre os
#   pontos (eps_1, y_1) e (eps_2, y_2): eps(y) = eps_1 + (eps_2 - eps_1) (y - y_1)/(y_2 - y_1).
#   Variáveis: linha ((eps_1, y_1), (eps_2, y_2)), altura (y relativo).
#   Usada em: figura_dominios (l. 603, 605, 607, 609, 611, 616)
def x_na_altura(linha, altura: float) -> float:
    (x_1, y_1), (x_2, y_2) = linha
    return x_1 + (x_2 - x_1) * (altura - y_1) / (y_2 - y_1)


# [R04] Posições dos textos da figura dos domínios (abscissa em ‰, ordenada em y/h): alturas em que os rótulos
#   dos domínios são centrados entre as duas fronteiras de cada domínio, rótulo do domínio 4a (fora da faixa
#   estreita, ligado a ela por uma linha), afastamento horizontal dos rótulos dos pivôs, dos textos "reta a",
#   ε_yd e das cotas d, h e y_C, setas de alongamento e encurtamento e limites dos eixos.
#   Variáveis: NIVEIS_ROTULOS, POSICAO_ROTULO_4A, AFASTAMENTO_PIVO, X_ROTULO_RETA_A, Y_ROTULO_RETA_B,
#   Y_ROTULOS_DEFORMACAO, AFASTAMENTO_EPS_YD, X_COTAS, Y_SETAS, Y_TEXTO_SETAS, INICIO_SETA, LIMITES_X_DOMINIOS,
#   LIMITES_Y_DOMINIOS.
#   Usada em: figura_dominios (l. 598, 599, 600, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 616, 617, 618, 620, 622, 624, 625, 627, 628, 629, 631, 633, 634, 635, 636, 637)
NIVEIS_ROTULOS = {"1": 0.55, "2": 0.22, "3": 0.40, "4": 0.55, "4a": 0.80, "5": 0.66}
POSICAO_ROTULO_4A = (-1.0, 1.03)
AFASTAMENTO_PIVO = 0.3
X_ROTULO_RETA_A = EPS_SU - 0.35
Y_ROTULO_RETA_B = 1.03
Y_ROTULOS_DEFORMACAO = -0.035
AFASTAMENTO_EPS_YD = (-0.15, 0.02)
X_COTAS = EPS_C2 + 0.3
Y_SETAS = 1.16
Y_TEXTO_SETAS = 1.19
INICIO_SETA = 0.4
LIMITES_X_DOMINIOS = (EPS_SU - 1.0, EPS_CU + 2.6)
LIMITES_Y_DOMINIOS = (1.27, -0.1)


# [F09] Deslocamento de uma coordenada de texto: x + dx (rótulos ao lado de um ponto).  Variáveis: x, dx.
#   Usada em: figura_dominios (l. 602, 625); obstaculos_do_painel (l. 922); posicionar_rotulo (l. 934)
def deslocar(x: float, deslocamento: float) -> float:
    return x + deslocamento


# [F10] Sentido oposto de um afastamento (setas e rótulos à esquerda): -dx.  Variáveis: deslocamento.
#   Usada em: figura_dominios (l. 598, 600, 631); formatar_eixos (l. 796); grafico_diagramas (l. 842, 847, 867)
def oposto(deslocamento: float) -> float:
    return -deslocamento


# [F03] Conversão de centímetros para polegadas (tamanho das figuras do matplotlib).  Variáveis: valor_cm.
#   Usada em: figura_dominios (l. 582); figura_interacao (l. 663); figura_viabilidade (l. 715);
#       grafico_diagramas (l. 838, 839)
#   Usada também em: figuras_complementares.py: figura_secao (l. 365);
#       figuras_complementares.py: figura_mn1r (l. 541, 542); figuras_complementares.py: figura_taxa (l. 578, 579);
#       figuras_complementares.py: figura_fusco (l. 718, 719)
def cm_para_polegada(valor_cm: float) -> float:
    return valor_cm / CM_POR_POLEGADA


# [F04] Espelhamento do diagrama de interação para momentos negativos (seção e armadura simétricas):
#   (M, N) -> (-M, N).  Variáveis: momentos.
#   Usada em: figura_interacao (l. 667); diagrama_pilar (l. 772)
def espelhar(momentos: np.ndarray) -> np.ndarray:
    return -momentos


# [F05] Ponto médio entre dois valores (posição dos rótulos dos domínios).  Variáveis: a, b.
#   Usada em: centros_faixas (l. 282); figura_dominios (l. 603, 605, 607, 609, 611, 616, 620, 634, 635)
def ponto_medio(a: float, b: float) -> float:
    return (a + b) / 2.0


# [R03] Diagrama de interação: seção de 30 x 50 cm (b_w x h), na direção de b_w, fck = 30 MPa, com a geometria de
#   barras adotada pela rotina para o pilar P160 (lambda = 90, nu = 0,40, e1/b_w = 0,30); armaduras: mínima geométrica
#   (0,4 % Ac), a calculada do P160 e a máxima fora da emenda (4 % Ac).
#   Variáveis: PILAR_EXEMPLO, TAXAS_COMPARACAO.
#   Usada em: pilar_exemplo (l. 643); figura_interacao (l. 655, 656, 657, 659, 660, 661, 662, 671, 683)
PILAR_EXEMPLO = "P160"
TAXAS_COMPARACAO = (nucleo.TAXA_MINIMA_GEOMETRICA, nucleo.TAXA_MAXIMA_FORA_EMENDA)


# [F06] Área de aço a partir da taxa geométrica: As = rho Ac.  Variáveis: taxa, area_concreto.
#   Usada em: figura_interacao (l. 655, 656, 660, 661)
def area_da_taxa(taxa: float, area_concreto: float) -> float:
    return taxa * area_concreto


# [R05] Números das figuras (eixos, legendas e rótulos), pela regra do SI adotada no texto do TCC: vírgula decimal;
#   sinal de menos U+2212; parte inteira com até quatro algarismos sem separador (1285,7; 4000); com cinco ou mais
#   (MINIMO_ALGARISMOS_AGRUPAMENTO), grupos de três (ALGARISMOS_POR_GRUPO) a partir da vírgula, separados por espaço
#   não separável U+00A0 (SEPARADOR_MILHAR): 10 000; 12 500. Nunca ponto. Número dos eixos: "%g", com essas regras.
#   Variáveis: texto (com vírgula decimal), valor; SEPARADOR_MILHAR, MINIMO_ALGARISMOS_AGRUPAMENTO,
#   ALGARISMOS_POR_GRUPO, SINAL_MENOS.
#   Usada em: agrupar_milhares (l. 197, 199, 200, 202, 203, 204, 206); texto_eixo (l. 210); texto_fixo (l. 293);
#       texto_eixo_absoluto (l. 503); texto_eixo_milhar (l. 507); figura_interacao (l. 679);
#       figura_viabilidade (l. 733, 734)
#   Usada também em: figuras_complementares.py: numero (l. 317);
#       figuras_complementares.py: figura_secao (l. 448, 449); figuras_complementares.py: painel_mn1r (l. 527);
#       figuras_complementares.py: figura_mn1r (l. 556, 557); figuras_complementares.py: figura_taxa (l. 608);
#       figuras_complementares.py: figura_fusco (l. 745)
SEPARADOR_MILHAR = " "
MINIMO_ALGARISMOS_AGRUPAMENTO = 5
ALGARISMOS_POR_GRUPO = 3
SINAL_MENOS = "−"


def agrupar_milhares(texto: str) -> str:
    sinal = texto[:1] if texto[:1] in ("-", SINAL_MENOS) else ""
    inteira, virgula, decimal = texto[len(sinal):].partition(",")
    if len(inteira) < MINIMO_ALGARISMOS_AGRUPAMENTO or not inteira.isdigit():
        return texto.replace("-", SINAL_MENOS)
    grupos = []
    while len(inteira) > ALGARISMOS_POR_GRUPO:
        grupos.insert(0, inteira[-ALGARISMOS_POR_GRUPO:])
        inteira = inteira[:-ALGARISMOS_POR_GRUPO]
    grupos.insert(0, inteira)
    return (sinal + SEPARADOR_MILHAR.join(grupos) + virgula + decimal).replace("-", SINAL_MENOS)


def texto_eixo(valor: float, _posicao=None) -> str:
    return agrupar_milhares(("%g" % valor).replace(".", ","))


# [F07] Taxa em porcentagem para o rótulo: 100 rho.  Variáveis: taxa.
#   Usada em: figura_interacao (l. 657, 662)
#   Usada também em: figuras_complementares.py: painel_mn1r (l. 490)
def porcentagem(taxa: float) -> float:
    return taxa * 100.0


# [R06] Mapa de viabilidade: arquivo de resultados, níveis da matriz (processar_matriz, [R01]) e códigos das
#   situações de inviabilidade (núcleo, constantes STATUS_*): A = As,calc > As,máx (4 % Ac); E = As,calc > 8 % Ac
#   ou sem equilíbrio estável; D = nenhum arranjo com As,adot <= As,máx atende ao detalhamento. Pilares por
#   célula = seções x classes de concreto.
#   Variáveis: ARQUIVO_RESULTADOS, CODIGOS_SITUACAO, PILARES_POR_CELULA.
#   Usada em: contagens_viabilidade (l. 694, 701); figura_viabilidade (l. 719, 729); linhas_viaveis (l. 752)
ARQUIVO_RESULTADOS = Path(__file__).resolve().parent / "saida" / "resultados_matriz.csv"
CODIGOS_SITUACAO = {nucleo.STATUS_ACIMA_MAXIMA: "A", nucleo.STATUS_SECAO_INSUFICIENTE: "E",
                    nucleo.STATUS_SEM_ARRANJO: "D"}
PILARES_POR_CELULA = len(processar_matriz.SECOES) * len(processar_matriz.RESISTENCIAS_CONCRETO)

# [R07] Cores do mapa: rampa sequencial de um só matiz, do mais claro (nenhum pilar viável) ao mais escuro
#   (todos viáveis), com luminosidade decrescente (legível também impressa em tons de cinza); texto em tinta
#   escura nas células claras e em branco nas escuras (a partir de LIMIAR_TEXTO_CLARO pilares viáveis).
#   Variáveis: RAMPA_VIABILIDADE, TINTA_TEXTO, TINTA_TEXTO_CLARO, LIMIAR_TEXTO_CLARO, COR_FRONTEIRA.
#   Usada em: figura_viabilidade (l. 713, 723, 732, 745)
#   Usada também em: figuras_complementares.py: figura_fusco (l. 729)
RAMPA_VIABILIDADE = ("#eef4fc", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")
TINTA_TEXTO = "#0b0b0b"
TINTA_TEXTO_CLARO = "#ffffff"
LIMIAR_TEXTO_CLARO = 5
COR_FRONTEIRA = "#0b0b0b"


# [F11] Fronteiras entre os métodos de 2ª ordem no mapa (posição entre duas linhas da grade, em índices):
#   depois do último λ <= λ1 mínimo (2ª ordem dispensada; λ1 = 35 em todos os pilares da matriz) e depois
#   do último λ <= 90 (curvatura aproximada); acima, M, N, 1/r com fluência.
#   Variáveis: esbeltezes (níveis em ordem crescente), limites (λ das fronteiras).
#   Usada em: figura_viabilidade (l. 731, 742)
LIMITES_METODO = (nucleo.LAMBDA1_MIN, nucleo.LAMBDA_LIMITE_CURVATURA)


def fronteiras_metodo(esbeltezes, limites) -> list[float]:
    return [sum(1 for valor in esbeltezes if valor <= limite) - 0.5 for limite in limites]


# [F12] Posição vertical do texto secundário de uma célula (causas de inviabilidade), abaixo do centro:
#   linha + 0,28 (índices da grade, eixo invertido); e do número de pilares viáveis, acima do centro.
#   Variáveis: linha.
#   Usada em: posicao_causas (l. 265); posicao_contagem (l. 269); figura_viabilidade (l. 724, 726)
DESLOCAMENTO_CAUSAS = 0.28
DESLOCAMENTO_CONTAGEM = -0.10


def posicao_causas(linha: int) -> float:
    return linha + DESLOCAMENTO_CAUSAS


def posicao_contagem(linha: int) -> float:
    return linha + DESLOCAMENTO_CONTAGEM


# [F13] Faixas dos métodos de 2ª ordem no mapa: centro de cada faixa entre as fronteiras de [F11] (índices
#   da grade), para os rótulos à direita do último painel, e textos dos rótulos.
#   Variáveis: fronteiras, numero_linhas.
#   Usada em: figura_viabilidade (l. 742, 743, 744)
ROTULOS_METODO = ("2ª ordem\ndispensada", "curvatura\naproximada", "$M$, $N$, $1/r$\ncom fluência")
POSICAO_ROTULOS_METODO = 1.04


def centros_faixas(fronteiras: list[float], numero_linhas: int) -> list[float]:
    limites = [-0.5, *fronteiras, numero_linhas - 0.5]
    return [ponto_medio(inferior, superior) for inferior, superior in zip(limites[:-1], limites[1:])]


# [R08] Texto de um número com casas fixas (títulos dos painéis: 0,15 e 0,30; N_d das legendas: 1285,7), com
#   vírgula decimal e o separador de milhar de [R05].  Variáveis: valor, casas.
#   Usada em: texto_lado (l. 396, 397); figura_viabilidade (l. 737); legenda_diagramas (l. 823)
#   Usada também em: figuras_complementares.py: figura_taxa (l. 618, 620)
CASAS_NIVEL_EXCENTRICIDADE = 2


def texto_fixo(valor: float, casas: int) -> str:
    return agrupar_milhares(("%.*f" % (casas, valor)).replace(".", ","))


# [R09] Diagramas de interação dos pilares viáveis (item 2 do arquivo "Pessoal 2" do autor): um gráfico por seção e
#   por nível de nu, em cada direção de flexão (9 gráficos por direção), com os diagramas de todos os pilares viáveis
#   daquela seção e daquele nu (armadura e barras adotadas, bloco parábola-retângulo, seção líquida,
#   pilares_nbr6118.diagrama_interacao) em kN e kN·m, para que o par (N_d, M_d,tot) de cada pilar seja lido nos
#   eixos. Cada gráfico tem dois painéis com o mesmo eixo de momentos: em cima, os diagramas inteiros (trecho de
#   tração cortado em FRACAO_TRACAO_VISIVEL da maior força de compressão); embaixo, a faixa ampliada em torno das
#   linhas de N_d, em que cada ponto de cálculo é ligado por um traço ao ponto em que o seu próprio diagrama corta a
#   linha de N_d (M_Rd da rotina, marcado por um traço vertical). O diagrama é simétrico (armadura simétrica): a
#   metade esquerda mostra uma família e a direita outra ([R11]); o eixo mostra |M| nos dois lados. Classe do
#   concreto: cor e marcador (rampa categorial de três posições, conferida no validador de paletas para todos os
#   pares, com o marcador e o rótulo como codificação secundária); índice de esbeltez: rótulo acima do ponto ([R11]).
#   O texto do SVG é gravado em curvas (TEXTO_SVG_DIAGRAMAS), para que a figura apareça no documento exatamente
#   como foi desenhada. Tamanhos dos marcadores em pontos tipográficos (TAMANHO_PONTO: ponto de cálculo;
#   TAMANHO_MARCA: traço vertical do M_Rd).
#   Variáveis: CORES_CLASSE, MARCADORES_CLASSE, ALTURA_DIAGRAMAS_CM, PROPORCAO_PAINEIS, FOLGA_MOMENTO,
#   FRACAO_TRACAO_VISIVEL, FOLGA_COMPRESSAO, MEIA_FAIXA_ZOOM, FRACAO_FAIXA_UNICA, DIVISOES_MOMENTO,
#   DIVISOES_NORMAL, SUBDIVISOES_GRADE, PASSOS_NORMALIZADOS, PONTOS_DIAGRAMA, TAMANHO_PONTO, TAMANHO_MARCA,
#   ESPESSURA_CURVA, ESPESSURA_PAR, CORPO_ROTULO_PONTO_PT, NIVEIS_E1_DIAGRAMAS, COLUNAS_DIRECAO,
#   PREFIXO_DIAGRAMAS, ARQUIVO_DADOS_DIAGRAMAS, TEXTO_SVG_DIAGRAMAS.
#   Usada em: lado_esquerdo (l. 392); texto_lado (l. 396); passo_redondo (l. 408); subpasso (l. 412);
#       limite_momento (l. 422); limites_forca (l. 426); distancia_linhas (l. 431); limites_faixa (l. 436);
#       diagrama_pilar (l. 766, 768); formatar_eixos (l. 804); eixo_forca (l. 809);
#       legenda_diagramas (l. 821, 822, 825, 826, 829, 830, 832);
#       grafico_diagramas (l. 839, 840, 842, 854, 855, 862, 863, 872, 879, 882); obstaculos_do_painel (l. 915);
#       posicionar_rotulo (l. 937); desenhar_ampliado (l. 988, 992, 994, 995, 996, 997); main (l. 1039)
#   Usada também em: figuras_complementares.py: figura_taxa (l. 594, 599, 625);
#       figuras_complementares.py: main (l. 766)
CORES_CLASSE = {25.0: "#2a78d6", 30.0: "#eb6834", 35.0: "#1baf7a"}
MARCADORES_CLASSE = {25.0: "o", 30.0: "s", 35.0: "^"}
ALTURA_DIAGRAMAS_CM = 10.2
PROPORCAO_PAINEIS = (1.7, 1.5)
FOLGA_MOMENTO = 1.06
FRACAO_TRACAO_VISIVEL = 0.25
FOLGA_COMPRESSAO = 1.03
MEIA_FAIXA_ZOOM = 0.75
FRACAO_FAIXA_UNICA = 0.15
DIVISOES_MOMENTO = 10
DIVISOES_NORMAL = 6
SUBDIVISOES_GRADE = 5
PASSOS_NORMALIZADOS = (1.0, 2.0, 2.5, 5.0, 10.0)
PONTOS_DIAGRAMA = 401
TAMANHO_PONTO = 4.3
TAMANHO_MARCA = 7.0
ESPESSURA_CURVA = 0.6
ESPESSURA_PAR = 1.2
CORPO_ROTULO_PONTO_PT = 6.3
NIVEIS_E1_DIAGRAMAS = (0.15, 0.30)
COLUNAS_DIRECAO = {nucleo.DIRECAO_B: ("Md,tot b_w (kN·m)", "MRd b_w (kN·m)", "Método b_w"),
                   nucleo.DIRECAO_H: ("Md,tot h (kN·m)", "MRd h (kN·m)", "Método h")}
PREFIXO_DIAGRAMAS = {nucleo.DIRECAO_B: "diagramas_bw", nucleo.DIRECAO_H: "diagramas_h"}
ARQUIVO_DADOS_DIAGRAMAS = "diagramas_dados.csv"
TEXTO_SVG_DIAGRAMAS = "path"


# [R10] Faixas de leitura da parte ampliada: os pares (ponto de cálculo, M_Rd) de uma mesma linha de N_d e de uma
#   mesma metade do gráfico que se sobreporiam (intervalos de |M| com folga de FOLGA_PAR_RELATIVA da largura do eixo
#   para o marcador e o rótulo) são desenhados em faixas paralelas à linha, na ordem FAIXAS_LEITURA (0 = sobre a
#   linha, 1 acima, -1 abaixo, ...), afastadas de PASSO_FAIXA vezes a distância entre linhas de N_d; a força de todos
#   é N_d (as faixas servem só para a leitura).  Variáveis: FAIXAS_LEITURA, PASSO_FAIXA, FOLGA_PAR_RELATIVA.
#   Usada em: ordenada_faixa (l. 457); intervalo_par (l. 461); distribuir_faixas (l. 788)
FAIXAS_LEITURA = (0, 1, -1, 2, -2, 3, -3)
PASSO_FAIXA = 0.17
FOLGA_PAR_RELATIVA = 0.03


# [R11] Metades e ligação ponto-M_Rd: à esquerda os pilares com e1/b_w = 0,15, à direita os com 0,30. Na direção de
#   b_w (M_Rd perto do M_d,tot), o par é ligado por um traço, em faixas de leitura ([R10]); na de h (M_Rd de várias
#   vezes o M_d,tot), ponto e marca do M_Rd ficam sobre a linha de N_d, com o lambda também abaixo da marca. O lambda
#   do ponto fica acima dele, no primeiro nível (NIVEIS_ROTULO_PONTO acima, NIVEIS_ROTULO_MARCA abaixo; o primeiro a
#   AFASTAMENTO_ROTULO_PT da linha, os demais a PASSO_NIVEL_PT, maior que a altura do texto) e deslocamento lateral
#   (DESLOCAMENTOS_LATERAIS, frações da largura do rótulo) em que a caixa do rótulo, ampliada das folgas horizontal
#   e vertical FOLGA_ROTULOS_PT, fica no painel sem tocar outro rótulo, marcador, traço ou o eixo central, e a
#   linha-guia (ESPESSURA_GUIA_PT, nos níveis além do primeiro) não toca rótulo nem marcador (pode cruzar traço); a
#   faixa do primeiro nível acima de cada ponto (abaixo de cada marca, na direção de h) fica reservada ao rótulo
#   dele, para que nenhum rótulo pareça ser de outro ponto; sem posição livre, fica a de menor sobreposição.
#   Distâncias em pontos tipográficos, independentes da escala do eixo. Fundo dos rótulos: ESTILO_CAIXA_ROTULO.
#   Variáveis: linha (pilar), direcao; LIGAR_POR_TRACO, NIVEIS_ROTULO_PONTO, NIVEIS_ROTULO_MARCA,
#   AFASTAMENTO_ROTULO_PT, PASSO_NIVEL_PT, FOLGA_ROTULOS_PT, ESPESSURA_GUIA_PT, DESLOCAMENTOS_LATERAIS,
#   ESTILO_CAIXA_ROTULO.
#   Usada em: deslocamento_nivel (l. 470); coincidentes (l. 482); alinhamento_rotulo (l. 486);
#       diagrama_pilar (l. 771); legenda_diagramas (l. 824); grafico_diagramas (l. 846, 868, 871, 877);
#       posicionar_rotulo (l. 938, 944, 950, 954, 967); desenhar_ampliado (l. 1002, 1003, 1009)
LIGAR_POR_TRACO = {nucleo.DIRECAO_B: True, nucleo.DIRECAO_H: False}
NIVEIS_ROTULO_PONTO = (1, 2, 3, 4)
NIVEIS_ROTULO_MARCA = (-1, -2, -3, -4)
AFASTAMENTO_ROTULO_PT = 4.0
PASSO_NIVEL_PT = 6.6
FOLGA_ROTULOS_PT = (1.2, 0.4)
ESPESSURA_GUIA_PT = 0.4
DESLOCAMENTOS_LATERAIS = (0.0, 0.4, -0.4)
ESTILO_CAIXA_ROTULO = {"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}
FRACAO_COINCIDENCIA = 0.01
FRACAO_EIXO_CENTRAL = 0.03


def lado_esquerdo(linha: dict, _direcao: str) -> bool:
    return float(linha["e1/b_w"]) == NIVEIS_E1_DIAGRAMAS[0]


def texto_lado(esquerda: bool, _direcao: str) -> str:
    return "$e_1/b_w$ = %s" % texto_fixo(NIVEIS_E1_DIAGRAMAS[0] if esquerda else NIVEIS_E1_DIAGRAMAS[1],
                                        CASAS_NIVEL_EXCENTRICIDADE)


# [F14] Passo "redondo" dos eixos: o menor valor da forma k x 10^n (k em PASSOS_NORMALIZADOS) que divide o intervalo
#   em no máximo o número de divisões dado; subdivisão da grade fina: passo / SUBDIVISOES_GRADE.
#   Variáveis: intervalo, divisoes, passo.
#   Fórmula: passo = mín{k · 10^{n} ≥ intervalo/divisões},  n = ⌊log_{10}(intervalo/divisões)⌋
#   Usada em: formatar_eixos (l. 798); eixo_forca (l. 809, 812); grafico_diagramas (l. 842)
def passo_redondo(intervalo: float, divisoes: int) -> float:
    bruto = intervalo / divisoes
    ordem = 10.0 ** np.floor(np.log10(bruto))
    return next(fator * ordem for fator in PASSOS_NORMALIZADOS if fator * ordem >= bruto)


def subpasso(passo: float) -> float:
    return passo / SUBDIVISOES_GRADE


# [F15] Limites dos eixos: momentos de -FOLGA_MOMENTO x |M|máx a +FOLGA_MOMENTO x |M|máx; forças de
#   máx(N_mín; -FRACAO_TRACAO_VISIVEL x N_máx) a FOLGA_COMPRESSAO x N_máx no painel de cima; na faixa ampliada,
#   de N_d,mín - MEIA_FAIXA_ZOOM x Δ a N_d,máx + MEIA_FAIXA_ZOOM x Δ, com Δ a distância entre linhas de N_d
#   consecutivas (ou FRACAO_FAIXA_UNICA x N_d, com uma só linha); intervalo total de um eixo: fim - início.
#   Variáveis: momento_maximo, forca_minima, forca_maxima, forcas_calculo, inicio, fim.
#   Usada em: eixo_forca (l. 809); grafico_diagramas (l. 841, 842, 844, 845, 847, 858, 867)
def limite_momento(momento_maximo: float) -> float:
    return FOLGA_MOMENTO * momento_maximo


def limites_forca(forca_minima: float, forca_maxima: float) -> tuple[float, float]:
    return max(forca_minima, -FRACAO_TRACAO_VISIVEL * forca_maxima), FOLGA_COMPRESSAO * forca_maxima


def distancia_linhas(forcas_calculo: list[float]) -> float:
    if len(forcas_calculo) == 1:
        return FRACAO_FAIXA_UNICA * forcas_calculo[0]
    return min(posterior - anterior for anterior, posterior in zip(forcas_calculo[:-1], forcas_calculo[1:]))


def limites_faixa(forcas_calculo: list[float], distancia: float) -> tuple[float, float]:
    return forcas_calculo[0] - MEIA_FAIXA_ZOOM * distancia, forcas_calculo[-1] + MEIA_FAIXA_ZOOM * distancia


def intervalo_eixo(inicio: float, fim: float) -> float:
    return fim - inicio


# [F16] Faixa de leitura ([R10]): ordenada do par = N_d + faixa x PASSO_FAIXA x Δ; intervalo ocupado por um par:
#   de mín(|M_d|, |M_Rd|) - folga a máx(|M_d|, |M_Rd|) + folga, com folga = FOLGA_PAR_RELATIVA x largura do eixo.
#   Rótulos ([R11]): deslocamento vertical do nível n, em pontos, sinal(n) x [AFASTAMENTO_ROTULO_PT + (|n| - 1) x
#   PASSO_NIVEL_PT], e em relação ao primeiro nível; deslocamento lateral = fração x largura da caixa do rótulo.
#   Marcas com |a - b| < FRACAO_COINCIDENCIA x largura do eixo (1 %, da ordem do marcador: mesmas barras e mesmo
#   M_Rd, ou pontos que se confundem no desenho) recebem um rótulo único, com os lambdas separados por barra
#   ("30/60/90"). Rótulos a menos de FRACAO_EIXO_CENTRAL da largura do eixo do centro são alinhados para fora.
#   Variáveis: forca_calculo, faixa, nivel, primeiro, distancia, momento_calculo, momento_resistente, largura_eixo,
#   posicao_a, posicao_b, fracao, caixa; FRACAO_COINCIDENCIA, FRACAO_EIXO_CENTRAL.
#   Fórmula: Δy_{n} = sinal(n) · [a + (|n| − 1) · p]
#   Usada em: deslocamento_relativo (l. 474); distribuir_faixas (l. 786, 789); obstaculos_do_painel (l. 922);
#       posicionar_rotulo (l. 934, 936, 939, 945, 946, 961, 965); agrupar_coincidentes (l. 976);
#       desenhar_ampliado (l. 989)
def ordenada_faixa(forca_calculo: float, faixa: int, distancia: float) -> float:
    return forca_calculo + faixa * PASSO_FAIXA * distancia


def intervalo_par(momento_calculo: float, momento_resistente: float, largura_eixo: float) -> tuple[float, float]:
    folga = FOLGA_PAR_RELATIVA * largura_eixo
    return min(momento_calculo, momento_resistente) - folga, max(momento_calculo, momento_resistente) + folga


def sobrepoe(intervalo_a: tuple[float, float], intervalo_b: tuple[float, float]) -> bool:
    return intervalo_a[0] < intervalo_b[1] and intervalo_b[0] < intervalo_a[1]


def deslocamento_nivel(nivel: int) -> float:
    return float(np.copysign(AFASTAMENTO_ROTULO_PT + (abs(nivel) - 1) * PASSO_NIVEL_PT, nivel))


def deslocamento_relativo(nivel: int, primeiro: int) -> float:
    return deslocamento_nivel(nivel) - deslocamento_nivel(primeiro)


def deslocamento_lateral(fracao: float, caixa: tuple) -> float:
    return fracao * (caixa[2] - caixa[0])


def coincidentes(posicao_a: float, posicao_b: float, largura_eixo: float) -> bool:
    return abs(posicao_a - posicao_b) < FRACAO_COINCIDENCIA * largura_eixo


def alinhamento_rotulo(posicao: float, largura_eixo: float) -> str:
    if abs(posicao) >= FRACAO_EIXO_CENTRAL * largura_eixo:
        return "center"
    return "right" if posicao < 0.0 else "left"


# [F17] Nível de nu em décimos, para o nome do arquivo de cada gráfico (0,2 -> 02): round(10 nu).
#   Variáveis: nivel.
#   Usada em: grafico_diagramas (l. 879)
def decimos(nivel: float) -> int:
    return round(nivel * 10.0)


# [R12] Textos dos números dos eixos dos diagramas: |M| com valor absoluto (os dois lados do gráfico são |M|) e N
#   em kN inteiros; nos dois, a vírgula decimal, o sinal de menos e o separador de milhar do SI de [R05] (o mesmo
#   formatador nos dois eixos e nas legendas: 1000 e 1500 em |M|; 2500, 10 000 e 12 500 em N).  Variáveis: valor.
#   Usada em: formatar_eixos (l. 799, 800)
def texto_eixo_absoluto(valor: float, _posicao=None) -> str:
    return agrupar_milhares(("%g" % abs(valor)).replace(".", ","))


def texto_eixo_milhar(valor: float, _posicao=None) -> str:
    return agrupar_milhares("%.0f" % valor)


# [F18] Caixas (x0, y0, x1, y1) em pixels de tela, para os rótulos ([R11]): pontos em pixels, p_px = p_pt x dpi/72,
#   e o inverso; caixa de largura w e altura a centrada em (x, y): (x - w/2, y - a/2, x + w/2, y + a/2); caixa de um
#   trecho entre (x_a, y_a) e (x_b, y_b) com espessura e (linha-guia, traço, eixo central); caixa deslocada (dx, dy)
#   e ampliada das folgas (f_x, f_y); área da sobreposição, máx(0; mín(x1) - máx(x0)) x máx(0; mín(y1) - máx(y0)); caixa
#   contida noutra (painel); abscissa dentro da caixa (a guia vertical encontra o rótulo).
#   Variáveis: pontos, pixels, dpi, x, y, largura, altura, x_a, y_a, x_b, y_b, espessura, caixa, dx, dy, folga,
#   folga_x, folga_y, caixa_a, caixa_b, limites; PONTOS_POR_POLEGADA.
#   Fórmula: p_{px} = p_{pt} · dpi/72;  A = máx(0; mín(x_{1}) − máx(x_{0})) · máx(0; mín(y_{1}) − máx(y_{0}))
#   Usada em: pontos_para_pixels (l. 525); caixa_do_trecho (l. 534); pixels_para_pontos (l. 546);
#       obstaculos_do_painel (l. 914, 916, 917, 918, 922, 923, 925);
#       posicionar_rotulo (l. 934, 946, 947, 948, 950, 951, 953, 954, 955, 958, 961)
PONTOS_POR_POLEGADA = 72.0


def pontos_para_pixels(pontos: float, dpi: float) -> float:
    return pontos * dpi / PONTOS_POR_POLEGADA


def caixa_em_torno(x: float, y: float, largura: float, altura: float) -> tuple:
    return x - largura / 2.0, y - altura / 2.0, x + largura / 2.0, y + altura / 2.0


def caixa_do_trecho(x_a: float, y_a: float, x_b: float, y_b: float, espessura: float) -> tuple:
    meia_espessura = espessura / 2.0
    return ampliar_caixa((min(x_a, x_b), min(y_a, y_b), max(x_a, x_b), max(y_a, y_b)), meia_espessura, meia_espessura)


def deslocar_caixa(caixa: tuple, dx: float, dy: float) -> tuple:
    return caixa[0] + dx, caixa[1] + dy, caixa[2] + dx, caixa[3] + dy


def contem_abscissa(caixa: tuple, x: float) -> bool:
    return caixa[0] <= x <= caixa[2]


def pixels_para_pontos(pixels: float, dpi: float) -> float:
    return pixels * PONTOS_POR_POLEGADA / dpi


def ampliar_caixa(caixa: tuple, folga_x: float, folga_y: float) -> tuple:
    return caixa[0] - folga_x, caixa[1] - folga_y, caixa[2] + folga_x, caixa[3] + folga_y


def area_sobreposta(caixa_a: tuple, caixa_b: tuple) -> float:
    largura = min(caixa_a[2], caixa_b[2]) - max(caixa_a[0], caixa_b[0])
    altura = min(caixa_a[3], caixa_b[3]) - max(caixa_a[1], caixa_b[1])
    return max(0.0, largura) * max(0.0, altura)


def caixa_contida(caixa: tuple, limites: tuple) -> bool:
    return limites[0] <= caixa[0] and limites[1] <= caixa[1] and caixa[2] <= limites[2] and caixa[3] <= limites[3]
# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


def configurar_estilo() -> None:
    plt.rcParams.update({"font.family": FONTE, "font.size": CORPO_PT, "mathtext.fontset": "stix",
                         "axes.linewidth": 0.6, "svg.fonttype": "none"})


def salvar(figura, pasta: Path, nome: str) -> None:
    figura.savefig(pasta.joinpath(nome + ".svg"), format="svg", bbox_inches="tight", pad_inches=0.04)
    figura.savefig(pasta.joinpath(nome + ".png"), format="png", dpi=RESOLUCAO_DPI, bbox_inches="tight",
                   pad_inches=0.04)
    plt.close(figura)


def figura_dominios(pasta: Path) -> None:
    eps_yd = deformacao_escoamento_tracao_por_mil(FYD_KN_CM2, nucleo.MODULO_ACO)
    y_d = RELACAO_D_H
    y_c = profundidade_relativa_pivo_c(EPS_CU, EPS_C2)
    figura, eixo = plt.subplots(figsize=(cm_para_polegada(LARGURA_CM), cm_para_polegada(ALTURA_DOMINIOS_CM)))
    # perfis de deformação nas fronteiras dos domínios: ((eps, y) no topo, (eps, y) no outro extremo)
    linha_a = ((EPS_SU, 0.0), (EPS_SU, y_d))
    linha_12 = ((0.0, 0.0), (EPS_SU, y_d))
    linha_23 = ((EPS_CU, 0.0), (EPS_SU, y_d))
    linha_34 = ((EPS_CU, 0.0), (eps_yd, y_d))
    linha_44a = ((EPS_CU, 0.0), (0.0, y_d))
    linha_4a5 = ((EPS_CU, 0.0), (0.0, 1.0))
    linha_b = ((EPS_C2, 0.0), (EPS_C2, 1.0))
    for inicio, fim in (linha_a, linha_12, linha_23, linha_34, linha_44a, linha_4a5, linha_b):
        eixo.plot([inicio[0], fim[0]], [inicio[1], fim[1]], color="black", linewidth=ESPESSURA_LINHA)
    eixo.plot([0.0, 0.0], [0.0, 1.0], color="black", linewidth=0.6, linestyle=(0, (4, 3)))
    eixo.plot([EPS_SU, EPS_CU], [0.0, 0.0], color="black", linewidth=0.6)
    for y in (y_d, 1.0, y_c):
        eixo.plot([EPS_SU if y == y_d else 0.0, EPS_C2], [y, y], color=CINZA_GRADE, linewidth=0.6,
                  linestyle=(0, (2, 2)))
    for (x, y), nome, afastamento, alinhamento in (((EPS_SU, y_d), "A", oposto(AFASTAMENTO_PIVO), "right"),
                                                   ((EPS_CU, 0.0), "B", AFASTAMENTO_PIVO, "left"),
                                                   ((EPS_C2, y_c), "C", oposto(AFASTAMENTO_PIVO), "right")):
        eixo.plot(x, y, marker="o", markersize=4.5, color="black")
        eixo.text(deslocar(x, afastamento), y, nome, ha=alinhamento, va="center", fontweight="bold")
    rotulos = (("1", ponto_medio(x_na_altura(linha_a, NIVEIS_ROTULOS["1"]), x_na_altura(linha_12, NIVEIS_ROTULOS["1"])),
                NIVEIS_ROTULOS["1"]),
               ("2", ponto_medio(x_na_altura(linha_12, NIVEIS_ROTULOS["2"]), x_na_altura(linha_23, NIVEIS_ROTULOS["2"])),
                NIVEIS_ROTULOS["2"]),
               ("3", ponto_medio(x_na_altura(linha_23, NIVEIS_ROTULOS["3"]), x_na_altura(linha_34, NIVEIS_ROTULOS["3"])),
                NIVEIS_ROTULOS["3"]),
               ("4", ponto_medio(x_na_altura(linha_34, NIVEIS_ROTULOS["4"]), x_na_altura(linha_44a, NIVEIS_ROTULOS["4"])),
                NIVEIS_ROTULOS["4"]),
               ("5", ponto_medio(x_na_altura(linha_4a5, NIVEIS_ROTULOS["5"]), x_na_altura(linha_b, NIVEIS_ROTULOS["5"])),
                NIVEIS_ROTULOS["5"]))
    estilo_circulo = dict(boxstyle="circle,pad=0.18", facecolor="white", edgecolor="black", linewidth=0.5)
    for texto, x, y in rotulos:
        eixo.text(x, y, texto, ha="center", va="center", fontsize=CORPO_ROTULO_PT, bbox=estilo_circulo)
    alvo_4a = (ponto_medio(x_na_altura(linha_44a, NIVEIS_ROTULOS["4a"]), x_na_altura(linha_4a5, NIVEIS_ROTULOS["4a"])),
               NIVEIS_ROTULOS["4a"])
    eixo.annotate("4a", xy=alvo_4a, xytext=POSICAO_ROTULO_4A, ha="center", va="center", fontsize=CORPO_ROTULO_PT,
                  bbox=estilo_circulo, arrowprops=dict(arrowstyle="-", linewidth=0.5))
    eixo.text(X_ROTULO_RETA_A, ponto_medio(0.0, y_d), "reta a", ha="center", va="center", rotation=90,
              fontsize=CORPO_SECUNDARIO_PT)
    eixo.text(EPS_C2, Y_ROTULO_RETA_B, "reta b", ha="center", va="top", fontsize=CORPO_SECUNDARIO_PT)
    for x, texto in ((EPS_SU, "10 ‰"), (0.0, "0"), (EPS_C2, "$\\varepsilon_{c2}$"), (EPS_CU, "$\\varepsilon_{cu}$")):
        eixo.text(x, Y_ROTULOS_DEFORMACAO, texto, ha="center", va="bottom")
    eixo.text(deslocar(eps_yd, AFASTAMENTO_EPS_YD[0]), deslocar(y_d, AFASTAMENTO_EPS_YD[1]), "$\\varepsilon_{yd}$",
              ha="right", va="top")
    eixo.text(X_COTAS, y_d, "$d$", ha="left", va="center")
    eixo.text(X_COTAS, 1.0, "$h$", ha="left", va="center")
    eixo.text(X_COTAS, y_c, "$\\dfrac{\\varepsilon_{cu} - \\varepsilon_{c2}}{\\varepsilon_{cu}}\\,h$",
              ha="left", va="center", fontsize=CORPO_SECUNDARIO_PT)
    eixo.annotate("", xy=(EPS_SU, Y_SETAS), xytext=(oposto(INICIO_SETA), Y_SETAS),
                  arrowprops=dict(arrowstyle="->", linewidth=0.6))
    eixo.annotate("", xy=(EPS_CU, Y_SETAS), xytext=(INICIO_SETA, Y_SETAS), arrowprops=dict(arrowstyle="->", linewidth=0.6))
    eixo.text(ponto_medio(EPS_SU, 0.0), Y_TEXTO_SETAS, "alongamento", ha="center", va="top")
    eixo.text(ponto_medio(0.0, EPS_CU), Y_TEXTO_SETAS, "encurtamento", ha="center", va="top")
    eixo.set_xlim(*LIMITES_X_DOMINIOS)
    eixo.set_ylim(*LIMITES_Y_DOMINIOS)
    eixo.axis("off")
    salvar(figura, pasta, "dominios")


def pilar_exemplo():
    caso = next(caso for caso in processar_matriz.montar_matriz() if caso.identificador == PILAR_EXEMPLO)
    resultado = nucleo.dimensionar_pilar(caso.pilar)
    return caso, resultado


def figura_interacao(pasta: Path) -> dict:
    caso, resultado = pilar_exemplo()
    pilar = resultado.pilar
    material = nucleo.Material(pilar.fck_mpa, pilar.fyk_mpa)
    secao = nucleo.secao_na_direcao(pilar.largura_bw, pilar.altura_h, resultado.candidato.x, resultado.candidato.y,
                                    nucleo.DIRECAO_B)
    area_concreto = nucleo.area_secao_retangular(pilar.largura_bw, pilar.altura_h)
    curvas = [(area_da_taxa(TAXAS_COMPARACAO[0], area_concreto), (0, (5, 3)),
               "$A_s$ = %.1f cm² ($\\rho$ = %.1f %%)" % (area_da_taxa(TAXAS_COMPARACAO[0], area_concreto),
                                                          porcentagem(TAXAS_COMPARACAO[0]))),
              (resultado.area_calculada, "-", "$A_{s,calc}$ = %.2f cm² (%s)" % (resultado.area_calculada,
                                                                                 PILAR_EXEMPLO)),
              (area_da_taxa(TAXAS_COMPARACAO[1], area_concreto), (0, (1, 1.5)),
               "$A_s$ = %.1f cm² ($\\rho$ = %.1f %%)" % (area_da_taxa(TAXAS_COMPARACAO[1], area_concreto),
                                                          porcentagem(TAXAS_COMPARACAO[1])))]
    figura, eixo = plt.subplots(figsize=(cm_para_polegada(LARGURA_CM), cm_para_polegada(ALTURA_INTERACAO_CM)))
    for area, estilo, rotulo in curvas:
        forcas, momentos = nucleo.diagrama_interacao(secao, material, area)
        momentos_m = nucleo.kn_cm_para_kn_m(momentos)
        eixo.plot(np.concatenate([espelhar(momentos_m[::-1]), momentos_m]), np.concatenate([forcas[::-1], forcas]),
                  color="black", linewidth=ESPESSURA_LINHA, linestyle=estilo, label=rotulo.replace(".", ","))
    momento_total = nucleo.kn_cm_para_kn_m(resultado.direcao_b.momento_total)
    eixo.plot(momento_total, resultado.forca_normal, marker="o", markersize=5, color="black", linestyle="none",
              label="Ponto de cálculo do %s ($N_d$, $M_{d,tot}$)" % PILAR_EXEMPLO)
    eixo.axhline(0.0, color="black", linewidth=0.5)
    eixo.axvline(0.0, color="black", linewidth=0.5)
    eixo.grid(True, color=CINZA_GRADE, linewidth=0.4)
    eixo.set_axisbelow(True)
    eixo.set_xlabel("$M_{Rd}$ (kN·m)")
    eixo.set_ylabel("$N_{Rd}$ (kN), compressão positiva")
    eixo.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=CORPO_SECUNDARIO_PT, ncol=2)
    formatador = matplotlib.ticker.FuncFormatter(texto_eixo)
    eixo.xaxis.set_major_formatter(formatador)
    eixo.yaxis.set_major_formatter(formatador)
    salvar(figura, pasta, "interacao")
    return {"pilar": PILAR_EXEMPLO, "Nd_kN": resultado.forca_normal, "Md_tot_kNm": momento_total,
            "As_calc_cm2": resultado.area_calculada, "barras": resultado.candidato.numero_barras,
            "diametro_mm": processar_matriz.cm_para_mm(resultado.candidato.diametro),
            "d_linha_cm": resultado.candidato.distancia_eixo, "lambda": caso.esbeltez, "nu": caso.nu,
            "e1_bw": caso.excentricidade_relativa}


def contagens_viabilidade() -> tuple[dict, dict]:
    """Por célula (e1/b_w, λ, ν): número de pilares viáveis e contagem das causas de inviabilidade."""
    viaveis: Counter = Counter()
    causas: dict = {}
    with ARQUIVO_RESULTADOS.open(encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo, delimiter=";"):
            celula = (float(linha["e1/b_w"]), float(linha["λ (dir. b_w)"]), float(linha["ν"]))
            causas.setdefault(celula, Counter())
            if linha["Situação"] == nucleo.STATUS_OK:
                viaveis[celula] += 1
            else:
                causas[celula][CODIGOS_SITUACAO[linha["Situação"]]] += 1
    return viaveis, causas


def texto_causas(contagem: Counter) -> str:
    return "  ".join("%s %d" % (codigo, contagem[codigo]) for codigo in ("A", "E", "D") if contagem[codigo])


def figura_viabilidade(pasta: Path) -> list[dict]:
    viaveis, causas = contagens_viabilidade()
    esbeltezes = processar_matriz.ESBELTEZES
    forcas = processar_matriz.FORCAS_ADIMENSIONAIS
    mapa_cores = LinearSegmentedColormap.from_list("viabilidade", RAMPA_VIABILIDADE)
    figura, eixos = plt.subplots(1, len(processar_matriz.EXCENTRICIDADES_RELATIVAS), sharey=True,
                                 figsize=(cm_para_polegada(LARGURA_CM), cm_para_polegada(ALTURA_VIABILIDADE_CM)))
    dados = []
    for eixo, relativa in zip(eixos, processar_matriz.EXCENTRICIDADES_RELATIVAS):
        grade = np.array([[viaveis[(relativa, esbeltez, forca)] for forca in forcas] for esbeltez in esbeltezes])
        eixo.imshow(grade, cmap=mapa_cores, vmin=0, vmax=PILARES_POR_CELULA, aspect="auto")
        for linha, esbeltez in enumerate(esbeltezes):
            for coluna, forca in enumerate(forcas):
                quantidade = int(grade[linha, coluna])
                tinta = TINTA_TEXTO_CLARO if quantidade >= LIMIAR_TEXTO_CLARO else TINTA_TEXTO
                eixo.text(coluna, posicao_contagem(linha), "%d" % quantidade, ha="center", va="center",
                          color=tinta, fontsize=CORPO_PT, fontweight="bold")
                eixo.text(coluna, posicao_causas(linha), texto_causas(causas[(relativa, esbeltez, forca)]),
                          ha="center", va="center", color=tinta, fontsize=CORPO_CELULA_PT)
                dados.append({"e1/b_w": relativa, "λ": esbeltez, "ν": forca, "viáveis": quantidade,
                              "pilares": PILARES_POR_CELULA,
                              **{codigo: causas[(relativa, esbeltez, forca)][codigo] for codigo in ("A", "E", "D")}})
        for fronteira in fronteiras_metodo(esbeltezes, LIMITES_METODO):
            eixo.axhline(fronteira, color=COR_FRONTEIRA, linewidth=1.2)
        eixo.set_xticks(range(len(forcas)), [texto_eixo(forca) for forca in forcas])
        eixo.set_yticks(range(len(esbeltezes)), [texto_eixo(esbeltez) for esbeltez in esbeltezes])
        eixo.tick_params(length=0)
        eixo.set_xlabel("$\\nu$")
        eixo.set_title("$e_1/b_w$ = %s" % texto_fixo(relativa, CASAS_NIVEL_EXCENTRICIDADE), fontsize=CORPO_PT)
        for borda in eixo.spines.values():
            borda.set_linewidth(0.6)
    eixos[0].set_ylabel("$\\lambda$ (direção de $b_w$)")
    ultimo = eixos[-1]
    for centro, rotulo in zip(centros_faixas(fronteiras_metodo(esbeltezes, LIMITES_METODO), len(esbeltezes)),
                              ROTULOS_METODO):
        ultimo.text(POSICAO_ROTULOS_METODO, centro, rotulo, transform=ultimo.get_yaxis_transform(), ha="left",
                    va="center", fontsize=CORPO_SECUNDARIO_PT, color=TINTA_TEXTO)
    salvar(figura, pasta, "viabilidade")
    return dados


# ----------------------------------------------------------------------------- diagramas por seção e nu ([R09])
def linhas_viaveis() -> list[dict]:
    with ARQUIVO_RESULTADOS.open(encoding="utf-8") as arquivo:
        return [linha for linha in csv.DictReader(arquivo, delimiter=";") if linha["Situação"] == nucleo.STATUS_OK]


def vetor_barras(texto: str) -> np.ndarray:
    return np.array([float(valor) for valor in texto.split()])


def diagrama_pilar(linha: dict, direcao: str) -> dict:
    """Diagrama de interação do pilar (armadura e barras adotadas), em kN e kN·m, e o seu ponto de cálculo; na
    metade esquerda do gráfico ([R11]), os momentos são espelhados ([F04])."""
    material = nucleo.Material(float(linha["fck (MPa)"]), float(linha["fyk (MPa)"]))
    secao = nucleo.secao_na_direcao(float(linha["b_w (cm)"]), float(linha["h (cm)"]),
                                    vetor_barras(linha["x barras (cm)"]), vetor_barras(linha["y barras (cm)"]), direcao)
    forcas, momentos = nucleo.diagrama_interacao(secao, material, float(linha["As,adot (cm²)"]), PONTOS_DIAGRAMA)
    momentos = nucleo.kn_cm_para_kn_m(momentos)
    chave_momento, chave_resistente, chave_metodo = COLUNAS_DIRECAO[direcao]
    forca_calculo = float(linha["Nd (kN)"])
    momento_calculo, momento_resistente = float(linha[chave_momento]), float(linha[chave_resistente])
    esquerda = lado_esquerdo(linha, direcao)
    sinal = espelhar if esquerda else np.asarray
    return {"linha": linha, "classe": float(linha["fck (MPa)"]), "esbeltez": float(linha["λ (dir. b_w)"]),
            "esquerda": esquerda, "forcas": forcas, "momentos": sinal(momentos), "forca_calculo": forca_calculo,
            "momento_calculo": momento_calculo, "momento_resistente": momento_resistente,
            "momento_lido": float(np.interp(forca_calculo, forcas, momentos)),
            "desenho_calculo": float(sinal(momento_calculo)), "desenho_resistente": float(sinal(momento_resistente)),
            "metodo": linha[chave_metodo]}


def distribuir_faixas(diagramas: list[dict], largura_eixo: float) -> None:
    """Faixa de leitura de cada par ([R10]): a primeira de FAIXAS_LEITURA sem sobreposição na mesma linha de N_d
    e na mesma metade do gráfico."""
    ocupadas = {}
    for diagrama in sorted(diagramas, key=lambda item: abs(item["desenho_calculo"])):
        intervalo = intervalo_par(diagrama["desenho_calculo"], diagrama["desenho_resistente"], largura_eixo)
        chave = (diagrama["forca_calculo"], diagrama["esquerda"])
        faixa = next(faixa for faixa in FAIXAS_LEITURA
                     if not any(sobrepoe(intervalo, outro) for outro in ocupadas.get((chave, faixa), [])))
        ocupadas.setdefault((chave, faixa), []).append(intervalo)
        diagrama["faixa"] = faixa


def formatar_eixos(eixo, limite: float, passo_momento: float) -> None:
    eixo.axvline(0.0, color="#0b0b0b", linewidth=0.7, zorder=3)
    eixo.set_xlim(oposto(limite), limite)
    eixo.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(passo_momento))
    eixo.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(subpasso(passo_momento)))
    eixo.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(texto_eixo_absoluto))
    eixo.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(texto_eixo_milhar))
    eixo.grid(True, which="major", color=CINZA_GRADE, linewidth=0.45)
    eixo.grid(True, which="minor", color="#ececec", linewidth=0.3)
    eixo.set_axisbelow(True)
    eixo.tick_params(labelsize=CORPO_ROTULO_PONTO_PT, length=2.5, width=0.5)
    eixo.tick_params(which="minor", length=0)


def eixo_forca(eixo, inicio: float, fim: float) -> None:
    passo = passo_redondo(intervalo_eixo(inicio, fim), DIVISOES_NORMAL)
    eixo.set_ylim(inicio, fim)
    eixo.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(passo))
    eixo.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(subpasso(passo)))


def legenda_diagramas(figura, diagramas: list[dict], direcao: str) -> None:
    alcas = []
    for classe in processar_matriz.RESISTENCIAS_CONCRETO:
        forcas = sorted({diagrama["forca_calculo"] for diagrama in diagramas if diagrama["classe"] == classe})
        if forcas:
            alcas.append(matplotlib.lines.Line2D(
                [], [], color=CORES_CLASSE[classe], marker=MARCADORES_CLASSE[classe], markerfacecolor=CORES_CLASSE[classe],
                markeredgecolor="white", linewidth=1.0, markersize=TAMANHO_PONTO,
                label="C%.0f: $N_d$ = %s kN" % (classe, texto_fixo(forcas[0], 1))))
    if LIGAR_POR_TRACO[direcao]:
        alcas.append(matplotlib.lines.Line2D([], [], color="#52514e", marker="|", markersize=TAMANHO_MARCA,
                                             markeredgewidth=ESPESSURA_PAR, linewidth=ESPESSURA_PAR,
                                             label="ponto de cálculo ($M_{d,tot}$) até o $M_{Rd}$ do próprio diagrama"))
    else:
        alcas.append(matplotlib.lines.Line2D([], [], color="#52514e", marker="|", markersize=TAMANHO_MARCA,
                                             markeredgewidth=ESPESSURA_PAR, linewidth=0.0,
                                             label="$M_{Rd}$ do próprio diagrama ($\\lambda$ abaixo; ponto: $\\lambda$ acima)"))
    figura.legend(handles=alcas, loc="lower center", ncol=len(alcas), fontsize=CORPO_ROTULO_PONTO_PT, frameon=False,
                  bbox_to_anchor=(0.5, 0.0), handlelength=2.0, columnspacing=1.2)


def grafico_diagramas(pasta: Path, direcao: str, largura: float, altura: float, nivel_nu: float,
                      diagramas: list[dict]) -> str:
    figura, (eixo, ampliado) = plt.subplots(2, 1, sharex=True, figsize=(cm_para_polegada(LARGURA_CM),
                                                                         cm_para_polegada(ALTURA_DIAGRAMAS_CM)),
                                            gridspec_kw={"height_ratios": PROPORCAO_PAINEIS, "hspace": 0.08})
    limite = limite_momento(max(float(np.max(np.abs(diagrama["momentos"]))) for diagrama in diagramas))
    passo_momento = passo_redondo(intervalo_eixo(oposto(limite), limite), DIVISOES_MOMENTO)
    forcas_calculo = sorted({diagrama["forca_calculo"] for diagrama in diagramas})
    distancia = distancia_linhas(forcas_calculo)
    inicio_faixa, fim_faixa = limites_faixa(forcas_calculo, distancia)
    if LIGAR_POR_TRACO[direcao]:
        distribuir_faixas(diagramas, intervalo_eixo(oposto(limite), limite))
    else:
        for diagrama in diagramas:
            diagrama["faixa"] = 0
    for painel in (eixo, ampliado):
        formatar_eixos(painel, limite, passo_momento)
        for diagrama in sorted(diagramas, key=lambda item: item["classe"]):
            painel.plot(diagrama["momentos"], diagrama["forcas"], color=CORES_CLASSE[diagrama["classe"]],
                        linewidth=ESPESSURA_CURVA, alpha=0.9, zorder=2)
        for forca in forcas_calculo:
            painel.axhline(forca, color="#7a7a7a", linewidth=0.5, linestyle=(0, (3, 2)), zorder=1)
    eixo_forca(eixo, *limites_forca(min(float(diagrama["forcas"].min()) for diagrama in diagramas),
                                    max(float(diagrama["forcas"].max()) for diagrama in diagramas)))
    eixo.axhspan(inicio_faixa, fim_faixa, color="#f1f1ee", zorder=0)
    for diagrama in diagramas:
        eixo.plot(diagrama["desenho_calculo"], diagrama["forca_calculo"], marker=MARCADORES_CLASSE[diagrama["classe"]],
                  markersize=3.2, markerfacecolor=CORES_CLASSE[diagrama["classe"]], markeredgecolor="white",
                  markeredgewidth=0.4, linestyle="none", zorder=6)
    eixo_forca(ampliado, inicio_faixa, fim_faixa)
    figura.subplots_adjust(left=0.105, right=0.99, top=0.985, bottom=0.175)
    sobrepostos = desenhar_ampliado(ampliado, diagramas, distancia, intervalo_eixo(oposto(limite), limite),
                                    LIGAR_POR_TRACO[direcao])
    for esquerda, posicao, alinhamento in ((True, 0.01, "left"), (False, 0.99, "right")):
        quantidade = sum(diagrama["esquerda"] == esquerda for diagrama in diagramas)
        eixo.text(posicao, 0.97, "%s: %s" % (texto_lado(esquerda, direcao), texto_quantidade(quantidade)),
                  transform=eixo.transAxes, ha=alinhamento, va="top", fontsize=CORPO_ROTULO_PONTO_PT, color="#52514e")
    eixo.set_ylabel("$N$ (kN)", fontsize=CORPO_SECUNDARIO_PT)
    ampliado.set_ylabel("$N$ (kN), faixa\nde $N_d$ ampliada", fontsize=CORPO_SECUNDARIO_PT)
    texto_dimensao = "$b_w$" if direcao == nucleo.DIRECAO_B else "$h$"
    ampliado.set_xlabel("$|M|$ (kN·m), na direção de %s:  à esquerda, %s;  à direita, %s" % (
        texto_dimensao, texto_lado(True, direcao), texto_lado(False, direcao)), fontsize=CORPO_SECUNDARIO_PT)
    legenda_diagramas(figura, diagramas, direcao)
    nome = "%s_%.0fx%.0f_nu%02d" % (PREFIXO_DIAGRAMAS[direcao], largura, altura, decimos(nivel_nu))
    if sobrepostos:
        print("%s: %d rótulo(s) sem posição livre de sobreposição" % (nome, sobrepostos))
    with matplotlib.rc_context({"svg.fonttype": TEXTO_SVG_DIAGRAMAS}):
        salvar(figura, pasta, nome)
    return nome


def figuras_diagramas(pasta: Path) -> list[dict]:
    """Os 9 gráficos de cada direção ([R09]) e a tabela com os valores de cada par (ponto de cálculo, M_Rd)."""
    linhas = linhas_viaveis()
    tabela = []
    for direcao in (nucleo.DIRECAO_B, nucleo.DIRECAO_H):
        for largura, altura in processar_matriz.SECOES:
            for nivel_nu in processar_matriz.FORCAS_ADIMENSIONAIS:
                grupo = [diagrama_pilar(linha, direcao) for linha in linhas
                         if float(linha["b_w (cm)"]) == largura and float(linha["h (cm)"]) == altura
                         and float(linha["ν"]) == nivel_nu]
                nome = grafico_diagramas(pasta, direcao, largura, altura, nivel_nu, grupo)
                tabela += [{"gráfico": nome, "ID": diagrama["linha"]["ID"], "direção": direcao,
                            "b_w (cm)": largura, "h (cm)": altura, "fck (MPa)": diagrama["classe"],
                            "λ (dir. b_w)": diagrama["esbeltez"], "ν": nivel_nu, "e1/b_w": diagrama["linha"]["e1/b_w"],
                            "metade": "esquerda" if diagrama["esquerda"] else "direita", "método": diagrama["metodo"],
                            "Governa": diagrama["linha"]["Governa"], "Barras": diagrama["linha"]["Barras"],
                            "As,adot (cm²)": diagrama["linha"]["As,adot (cm²)"], "Nd (kN)": diagrama["forca_calculo"],
                            "Md,tot (kN·m)": diagrama["momento_calculo"], "MRd (kN·m)": diagrama["momento_resistente"],
                            "M lido no diagrama em Nd (kN·m)": diagrama["momento_lido"],
                            "faixa de leitura": diagrama["faixa"]} for diagrama in grupo]
    return tabela


# ----------------------------------------------------------------------------- rótulos do painel ampliado ([R11])
def obstaculos_do_painel(eixo, renderizador, pontos: list, marcas: list, tracos: list, colunas: list) -> list[tuple]:
    """Caixas de tela ([F18]) de pontos, marcas, traços (que a linha-guia pode cruzar), eixo central e colunas do
    primeiro nível de cada ponto rotulado, reservadas ao rótulo do próprio ponto ([R11]): (caixa, cruzável, dono)."""
    ponto, marca, traco = (pontos_para_pixels(tamanho, eixo.figure.dpi)
                           for tamanho in (TAMANHO_PONTO, TAMANHO_MARCA, ESPESSURA_PAR))
    caixas = [(caixa_em_torno(*eixo.transData.transform(centro), ponto, ponto), False, None) for centro in pontos]
    caixas += [(caixa_em_torno(*eixo.transData.transform(centro), traco, marca), False, None) for centro in marcas]
    caixas += [(caixa_do_trecho(*eixo.transData.transform(inicio), *eixo.transData.transform(fim), traco), True, None)
               for inicio, fim in tracos]
    for dono, niveis in colunas:
        x_tela, y_tela = eixo.transData.transform(dono)
        topo = deslocar(y_tela, pontos_para_pixels(deslocamento_nivel(niveis[1]), eixo.figure.dpi))
        caixas.append((caixa_do_trecho(x_tela, y_tela, x_tela, topo, ponto), False, dono))
    limites, centro = eixo.get_window_extent(renderizador).extents, eixo.transData.transform((0.0, 0.0))[0]
    caixas.append((caixa_do_trecho(centro, limites[1], centro, limites[3], traco), False, None))
    return caixas


def posicionar_rotulo(eixo, renderizador, posicao: float, ordenada: float, texto: str, niveis: tuple,
                      obstaculos: list[tuple], largura_eixo: float, proprios: tuple) -> bool:
    """Rótulo na primeira posição livre de [R11]; acrescenta as caixas dele aos obstáculos e devolve se ficou livre."""
    dpi = eixo.figure.dpi
    x_tela, y_tela = eixo.transData.transform((posicao, ordenada))
    inicio_guia = deslocar(y_tela, pontos_para_pixels(deslocamento_nivel(niveis[0]), dpi))
    limites = tuple(eixo.get_window_extent(renderizador).extents)
    opcoes = dict(textcoords="offset points", ha=alinhamento_rotulo(posicao, largura_eixo), zorder=7,
                  va="bottom" if niveis[0] > 0 else "top", fontsize=CORPO_ROTULO_PONTO_PT, color="#0b0b0b",
                  bbox=ESTILO_CAIXA_ROTULO)
    medida = eixo.annotate(texto, xy=(posicao, ordenada), xytext=(0.0, deslocamento_nivel(niveis[0])), **opcoes)
    caixa_base = tuple(medida.get_window_extent(renderizador).extents)
    medida.remove()
    tentativas = []
    for nivel in niveis:
        for fracao in DESLOCAMENTOS_LATERAIS:
            lateral = deslocamento_lateral(fracao, caixa_base)
            vertical = pontos_para_pixels(deslocamento_relativo(nivel, niveis[0]), dpi)
            caixa = deslocar_caixa(caixa_base, lateral, vertical)
            if not contem_abscissa(caixa, x_tela):
                continue
            folgas = (pontos_para_pixels(folga, dpi) for folga in FOLGA_ROTULOS_PT)
            partes = [(ampliar_caixa(caixa, *folgas), False)]
            if nivel != niveis[0]:
                partes.append((caixa_do_trecho(x_tela, inicio_guia, x_tela, caixa[1] if nivel > 0 else caixa[3],
                                               pontos_para_pixels(ESPESSURA_GUIA_PT, dpi)), True))
            sobreposicao = sum(area_sobreposta(parte, outra) for parte, guia in partes
                               for outra, cruzavel, dono in obstaculos
                               if not (guia and cruzavel) and dono not in proprios)
            tentativas.append((not caixa_contida(caixa, limites), sobreposicao, nivel, lateral, partes))
    livres = [tentativa for tentativa in tentativas if not tentativa[0] and tentativa[1] == 0.0]
    fora, sobreposicao, nivel, lateral, partes = livres[0] if livres else min(tentativas, key=lambda item: item[:2])
    eixo.annotate(texto, xy=(posicao, ordenada), xytext=(pixels_para_pontos(lateral, dpi), deslocamento_nivel(nivel)),
                  **opcoes)
    obstaculos.extend((parte, False, None) for parte, _ in partes)
    if nivel != niveis[0]:
        eixo.annotate("", xy=(posicao, ordenada), xytext=(0.0, deslocamento_nivel(nivel)), zorder=4,
                      textcoords="offset points", arrowprops={"arrowstyle": "-", "color": "#9a9a9a", "shrinkA": 0.0,
                                                              "linewidth": ESPESSURA_GUIA_PT, "shrinkB": 0.0})
    return not fora and sobreposicao == 0.0


def agrupar_coincidentes(posicoes: list[tuple[float, float]], largura_eixo: float) -> list[tuple]:
    """Junta as marcas coincidentes ([F16]) num rótulo único com os lambdas em ordem crescente: (posição, texto,
    posições do grupo)."""
    grupos = []
    for posicao, esbeltez in sorted(posicoes, key=lambda item: abs(item[0])):
        if grupos and coincidentes(posicao, grupos[-1][0], largura_eixo):
            grupos[-1][1].append(esbeltez)
            grupos[-1][2].append(posicao)
        else:
            grupos.append((posicao, [esbeltez], [posicao]))
    return [(posicao, "/".join("%.0f" % valor for valor in sorted(set(esbeltezes))), membros)
            for posicao, esbeltezes, membros in grupos]


def desenhar_ampliado(eixo, diagramas: list[dict], distancia: float, largura_eixo: float, ligar: bool) -> int:
    """Painel ampliado ([R10], [R11]): pares, marcas e rótulos; devolve o número de rótulos com sobreposição."""
    for diagrama in diagramas:
        cor = CORES_CLASSE[diagrama["classe"]]
        diagrama["ordenada"] = ordenada_faixa(diagrama["forca_calculo"], diagrama["faixa"], distancia)
        if ligar:
            eixo.plot([diagrama["desenho_calculo"], diagrama["desenho_resistente"]],
                      [diagrama["ordenada"], diagrama["ordenada"]], color=cor, linewidth=ESPESSURA_PAR,
                      solid_capstyle="butt", zorder=4)
        eixo.plot(diagrama["desenho_resistente"], diagrama["ordenada"], marker="|", markersize=TAMANHO_MARCA,
                  markeredgewidth=ESPESSURA_PAR, color=cor, zorder=5)
        eixo.plot(diagrama["desenho_calculo"], diagrama["ordenada"], marker=MARCADORES_CLASSE[diagrama["classe"]],
                  markersize=TAMANHO_PONTO, markerfacecolor=cor, markeredgecolor="white", markeredgewidth=0.6,
                  linestyle="none", zorder=6)
    renderizador = eixo.figure.canvas.get_renderer()
    pontos = [(diagrama["desenho_calculo"], diagrama["ordenada"]) for diagrama in diagramas]
    marcas = [(diagrama["desenho_resistente"], diagrama["ordenada"]) for diagrama in diagramas]
    colunas = [(centro, NIVEIS_ROTULO_PONTO) for centro in pontos]
    colunas += [] if ligar else [(centro, NIVEIS_ROTULO_MARCA) for centro in marcas]
    obstaculos = obstaculos_do_painel(eixo, renderizador, pontos, marcas, list(zip(pontos, marcas)) if ligar else [],
                                      colunas)
    grupos = {}
    for diagrama in diagramas:
        grupos.setdefault((diagrama["ordenada"], diagrama["esquerda"]), []).append(diagrama)
    chaves = (("desenho_calculo", NIVEIS_ROTULO_PONTO),) + ((("desenho_resistente", NIVEIS_ROTULO_MARCA),)
                                                            if not ligar else ())
    livres = [posicionar_rotulo(eixo, renderizador, posicao, ordenada, texto, niveis, obstaculos, largura_eixo,
                                tuple((membro, ordenada) for membro in membros))
              for (ordenada, _), grupo in sorted(grupos.items()) for chave, niveis in chaves
              for posicao, texto, membros in agrupar_coincidentes([(item[chave], item["esbeltez"]) for item in grupo],
                                                                  largura_eixo)]
    return livres.count(False)


def texto_quantidade(quantidade: int) -> str:
    if quantidade == 0:
        return "nenhum pilar"
    return "1 pilar" if quantidade == 1 else "%d pilares" % quantidade


def gravar_csv(caminho: Path, linhas: list[dict]) -> None:
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0].keys()), delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas)


def main() -> None:
    pasta = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "saida" / "figuras"
    pasta.mkdir(parents=True, exist_ok=True)
    configurar_estilo()
    figura_dominios(pasta)
    gravar_csv(pasta.joinpath("interacao_dados.csv"), [figura_interacao(pasta)])
    gravar_csv(pasta.joinpath("viabilidade_dados.csv"), figura_viabilidade(pasta))
    gravar_csv(pasta.joinpath(ARQUIVO_DADOS_DIAGRAMAS), figuras_diagramas(pasta))
    print("figuras gravadas em", pasta)


if __name__ == "__main__":
    main()
