# -*- coding: utf-8 -*-
"""
processar_matriz.py
===================
Monta a matriz fechada de pilares-teste definida na Seção 3.6 do Projeto de Pesquisa (PP), assinado em
01/07/2026 (Quadro 1 do TCC, Tabela 3.1 do PP): 54 combinações-base (3 seções x 3 fck x 6 lambda), cada uma
com três níveis de força normal adimensional nu e dois níveis de excentricidade de primeira ordem e1/b_w
(pequena e moderada), totalizando 54 x 3 x 2 = 324 pilares. Seção b_w x h, com b_w = menor dimensão, na
direção de e1 ([H07] do núcleo). O arranjo da armadura NÃO é fator da matriz: segue a
regra do PP (duas faces opostas; perímetro só quando duas camadas forem inviáveis), [R07] do núcleo.

Cada pilar é convertido nas entradas do Módulo 1 da Figura 3 do TCC (Figura 3.1 do PP): b_w, h, fck, fyk, Nd,
e1, l_e e vinculação; é dimensionado com o módulo pilares_nbr6118.py e exportado para planilha eletrônica
(resultados_matriz.xlsx) e para texto separado por ponto e vírgula (resultados_matriz.csv).

Abas da planilha: Matriz_entradas (tabela completa dos casos), Resultados, Entrada_softwares (dados a
lançar no TQS e resultados a preencher, com Delta% por fórmula idêntica a [F50] do núcleo; o Eberick saiu em 03/11/2026),
Teste_momento_minimo (conjunto complementar de 9 pilares só com o momento mínimo, [R13]; também em
resultados_teste_minimo.csv), Inviaveis (critério que falhou, com a prova, cenário da força na maior dimensão [R17] e
sensibilidade com 8 % [R18]; também em resultados_inviaveis.csv), Resumo e Hipoteses. O cenário [R17] só justifica a
direção da força (Seção 4.2 do TCC); os pilares viáveis nele não são lançados no TQS (revisão de escopo de
23/10/2026: a aba Entrada_softwares_opcional foi retirada).

Uso:  python3 processar_matriz.py [pasta_de_saida]
"""
from __future__ import annotations

import csv
import itertools
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import pilares_nbr6118 as nucleo

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — INÍCIO
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [R01] Níveis dos fatores da matriz fechada (Quadro 1 do TCC, Tabela 3.1 do PP):
#   SECOES — as três seções do PP (20 x 30, 30 x 50 e 40 x 80 cm) dadas como (b_w, h), com b_w = MENOR
#     dimensão: é a direção da excentricidade e1, da flexão de 1ª ordem e do lambda da matriz (direção de
#     menor inércia, [H07] do núcleo). Se o lambda da matriz fosse medido na maior dimensão (h), o da direção de
#     b_w seria 1,5 / 1,67 / 2,0 vezes maior e passaria de 140 a partir de lambda_h = 93 / 84 / 70, fora do
#     escopo lambda <= 140 do PP;
#   RESISTENCIAS_CONCRETO — fck (MPa);  ESBELTEZES — lambda na direção de b_w;
#   FORCAS_ADIMENSIONAIS — nu = Nd/(Ac fcd), níveis baixo, médio e alto;
#   EXCENTRICIDADES_RELATIVAS — e1/b_w, níveis "pequena" e "moderada" (critérios em [R02]); b_w faz o papel
#     do h da norma na direção de e1 ([H07] do núcleo).
#   Total: 3 x 3 x 6 x 3 x 2 = 324 pilares.
#   Variáveis: SECOES, RESISTENCIAS_CONCRETO, ESBELTEZES, FORCAS_ADIMENSIONAIS, EXCENTRICIDADES_RELATIVAS,
#   ROTULOS_EXCENTRICIDADE.  Fonte: PP (01/07/2026), Seção 3.6 e Tabela 3.1; Seção 3.6 e Quadro 1 do TCC.
#   Usada em: nível do módulo (l. 111); montar_matriz (l. 391, 392); montar_teste_minimo (l. 399);
#       escrever_resumo (l. 855)
#   Usada também em: verificar_matriz.py: conferir_resultados (l. 510); figuras_tcc.py: nível do módulo (l. 229);
#       figuras_tcc.py: figura_viabilidade (l. 711, 712, 714, 717); figuras_tcc.py: legenda_diagramas (l. 817);
#       figuras_tcc.py: figuras_diagramas (l. 892, 893); gerar_auditor.py: main (l. 449, 450);
#       figuras_complementares.py: figura_taxa (l. 575, 576, 587, 607, 627, 630)
SECOES = ((20.0, 30.0), (30.0, 50.0), (40.0, 80.0))
RESISTENCIAS_CONCRETO = (25.0, 30.0, 35.0)
ESBELTEZES = (30.0, 60.0, 90.0, 110.0, 130.0, 140.0)
FORCAS_ADIMENSIONAIS = (0.20, 0.40, 0.60)
EXCENTRICIDADES_RELATIVAS = (0.15, 0.30)
ROTULOS_EXCENTRICIDADE = {0.15: "pequena", 0.30: "moderada"}

# [R02] Critérios dos níveis de e1/b_w (verificados em verificar_matriz.py para todos os pilares):
#   (a) e1 >= e_a (Seção 3.6 do PP), e_a de [F32] do núcleo com l = l_e;
#   (b) e1 > e1,min = M1d,min / Nd = 1,5 + 0,03 b_w (cm), para que o nível não seja absorvido pelo momento
#       mínimo ([F21] e [F55] do núcleo) — garante "flexão representativa" (Quadro 1 do TCC, Tabela 3.1 do PP).
#   O maior e1,min/b_w da matriz é 0,105 (b_w = 20 cm); 0,15 e 0,30 atendem (a) e (b) nas três seções e
#   envolvem o valor e1/h ≈ 0,20 dos pilares de extremidade P5 e P6 de BASTOS (2017, cap. 21), em que h é a
#   dimensão na direção do momento (a menor da seção, como b_w aqui).
#   Variáveis: MARGEM_EXCENTRICIDADE_MINIMA (razão mínima e1 / e1,min exigida, > 1).
#   Usada em: (sem uso neste script)
#   Usada também em: verificar_matriz.py: conferir_entradas (l. 329)
MARGEM_EXCENTRICIDADE_MINIMA = 1.0

# [R13] Conjunto complementar do teste do momento mínimo (fora da matriz do PP, que fica inalterada): pilares só
#   com o momento mínimo de 1ª ordem (e1 = 0 nas duas direções, situação de projeto de compressão centrada do
#   pilar intermediário de BASTOS, 2017, item 10.1), para testar como o TQS aplica (i) M1d,mín em cada
#   direção (NBR 6118:2023, 11.3.3.4.3), (ii) os efeitos locais de 2ª ordem somados ao mínimo, com alfa_b = 1
#   (15.8.2 d)), e (iii) a envoltória mínima com 2ª ordem (15.3.2): duas análises isoladas ou a elipse completa
#   ([H06] e [R12] do núcleo). Justificativa: na matriz, o momento mínimo nunca define a armadura — por [R02] b),
#   Nd e1 > M1d,mín na direção de b_w, e a direção de h, a única só com o mínimo, não governa As,calc em nenhum
#   dos 324 pilares —, de modo que diferenças entre a rotina e o TQS no tratamento do mínimo (por exemplo, dispensar
#   o mínimo na direção de h quando o momento de b_w já o supera, ou verificar a elipse) não aparecem no As da
#   matriz; neste conjunto o mínimo governa, e cada parcela da divergência (M1d,mín, 2ª ordem, elipse) pode ser
#   atribuída à sua causa. Níveis: as três seções de [R01]; lambda = 30 (2ª ordem dispensada nas duas
#   direções: envoltória de 1ª ordem), 90 (curvatura aproximada, no limite da faixa) e 110 (primeiro nível da
#   faixa do M, N, 1/r com fluência; com lambda = 140 e nu = 0,40, os três pilares seriam inviáveis); fck = 30 MPa
#   e nu = 0,40 (níveis centrais da matriz). Total: 3 x 3 = 9 pilares, M01 a M09, na ordem seção -> lambda.
#   As entradas lançáveis seguem [R10].
#   Variáveis: ESBELTEZES_TESTE_MINIMO, FCK_TESTE_MINIMO, NU_TESTE_MINIMO, EXCENTRICIDADE_TESTE_MINIMO,
#   ROTULO_TESTE_MINIMO.  Fonte: pedido do autor (22/10/2026), após a análise dos dois níveis de e1/b_w.
#   Usada em: nível do módulo (l. 111); montar_teste_minimo (l. 399, 400, 401); linha_entrada (l. 418)
#   Usada também em: verificar_matriz.py: conferir_resultados (l. 511, 512, 513); gerar_auditor.py: main (l. 451)
ESBELTEZES_TESTE_MINIMO = (30.0, 90.0, 110.0)
FCK_TESTE_MINIMO = 30.0
NU_TESTE_MINIMO = 0.40
EXCENTRICIDADE_TESTE_MINIMO = 0.0
ROTULO_TESTE_MINIMO = "só mínimo (teste)"
ROTULOS_NIVEL_E1 = {**ROTULOS_EXCENTRICIDADE, EXCENTRICIDADE_TESTE_MINIMO: ROTULO_TESTE_MINIMO}

# [R03] Identificação dos pilares: "P" seguido do número de ordem com três dígitos (P001 a P324), na
#   ordem seção -> fck -> lambda -> nu -> e1/b_w.  Variáveis: numero_ordem.
#   Usada em: montar_matriz (l. 393)
def identificador_pilar(numero_ordem: int) -> str:
    return "P%03d" % numero_ordem


# [R14] Identificação dos pilares do teste do momento mínimo ([R13]): "M" seguido do número de ordem com dois
#   dígitos (M01 a M09).  Variáveis: numero_ordem.
#   Usada em: montar_teste_minimo (l. 400)
def identificador_teste_minimo(numero_ordem: int) -> str:
    return "M%02d" % numero_ordem


# [F01] Excentricidade de 1ª ordem a partir do nível relativo da matriz: e1 = (e1/b_w) b_w (b_w = h da norma na
#   direção de e1, [H07] do núcleo).  Variáveis: excentricidade_relativa (e1/b_w), largura (b_w, cm).
#   Fonte: Quadro 1 do TCC (Tabela 3.1 do PP).
#   Usada em: criar_pilar (l. 384)
def excentricidade_de_relativa(excentricidade_relativa: float, largura: float) -> float:
    return excentricidade_relativa * largura


# [R10] Entradas exatas dos níveis (decisão do autor de 22/10/2026: a rotina segue a norma com os valores exatos; se
#   o TQS não aceitar o valor exato, a limitação é dele e deve ser relatada, não compensada na rotina):
#   (a) comprimento: l_e = lambda b_w / raiz(12) ([F20] do núcleo), sem arredondamento, de modo que o lambda
#       calculado na direção de b_w ([F19]) é o próprio nível (a menos do erro de ponto flutuante, [N05] do núcleo);
#   (b) força: Nd = nu Ac fcd ([F26] do núcleo), sem arredondamento; Nk = Nd / gama_f, Ngk = 0,70 Nk e
#       Nqk = 0,30 Nk ([F35] do núcleo).
#   A aba Entrada_softwares mostra esses valores com as casas de [R11] (a célula guarda o valor completo) e tem
#   colunas para registrar o comprimento e a força que o TQS efetivamente aceitou, com o Delta% em relação
#   ao valor exato; a diferença é atribuída à limitação de entrada do programa (Seção 3.9 do PP, análise das
#   divergências). Até 22/10/2026 (noite), l_e era arredondado para baixo ao cm e Nk a 0,1 kN, por suposição (não
#   conferida) de que os programas só aceitariam esses valores.
#   Variáveis: CRITERIO_ENTRADAS (descrição, para relatórios).  Fonte: PP (01/07/2026), Seções 1.4.2 e) e 3.6;
#   NBR 6118:2023, 11.7.1 (Tabela 11.1: gama_g = gama_q = 1,4); decisão do autor (22/10/2026).
#   Usada em: escrever_hipoteses (l. 837)
CRITERIO_ENTRADAS = ("valores exatos dos níveis: ℓe = λ b_w/√12 e Nd = ν Ac fcd, sem arredondamento; o valor que o "
                     "TQS aceitou é registrado na aba Entrada_softwares, e a diferença é limitação de entrada do "
                     "programa, não compensada na rotina")


# [F02] Conversões de unidade para a planilha: cm -> mm (bitolas), cm -> m (comprimentos) e
#   1/cm -> 1/m (curvaturas).  Variáveis: valor_cm, curvatura_cm.
#   Usada em: linha_resultado (l. 447, 464, 466, 467, 481); entradas_lancaveis (l. 528); texto_barras (l. 681);
#       escrever_hipoteses (l. 828, 830)
#   Usada também em: validacao.py: memorial_pilar (l. 1721);
#       sensibilidade_secao_liquida.py: descricao_barras (l. 384);
#       conferir_busca_exaustiva.py: escolha_exaustiva (l. 98); figuras_tcc.py: figura_interacao (l. 685);
#       figuras_complementares.py: figura_secao (l. 403)
def cm_para_mm(valor_cm: float) -> float:
    return valor_cm * 10.0


def cm_para_m(valor_cm: float) -> float:
    return valor_cm / 100.0


def curvatura_por_metro(curvatura_cm: float) -> float:
    return curvatura_cm * 100.0


# [R12] Texto de um número com vírgula decimal e sem zeros à direita (10,0 -> "10"; 12,5 -> "12,5"; 6,3 -> "6,3"),
#   para os textos da aba Hipoteses.  Variáveis: valor.
#   Usada em: escrever_hipoteses (l. 788, 828, 830)
def texto_decimal(valor: float) -> str:
    return ("%.2f" % valor).rstrip("0").rstrip(".").replace(".", ",")


# [R04] Parcela que governa a armadura calculada As,calc = max(As,b ; As,h ; As,min) ([R08] do núcleo):
#   "As,mín" quando a armadura mínima é a maior (empate a favor de As,mín, pois nesse caso o equilíbrio
#   não define a armadura); senão a direção (b_w ou h) de maior As necessária (empate a favor de b_w).
#   Variáveis: area_b, area_h, area_minima.  Fonte: Seção 3.9 do PP (análise das divergências).
#   Usada em: linha_resultado (l. 460); linha_inviavel (l. 718)
#   Usada também em: sensibilidade_secao_liquida.py: governa (l. 388)
def parcela_governante(area_b: float, area_h: float, area_minima: float) -> str:
    if area_minima >= max(area_b, area_h):
        return "As,mín"
    return "direção b_w" if area_b >= area_h else "direção h"


# [R05] Inclusão na modelagem do TQS: somente os pilares viáveis na rotina Python ("ok"); os
#   inviáveis (As,calc > As,max, seção insuficiente ou sem detalhamento) são casos de baixa
#   representatividade para a comparação e ficam só na rotina Python, onde a inviabilidade é resultado.
#   Variáveis: situacao.  Fonte: Tabela 3.1 do PP, Quadro 1 do TCC ("Casos com baixa representatividade podem ser
#   excluídos"); análise do orientador de 22/06/2026, item 7 ("selecionar casos representativos e justificar").
#   Usada em: linha_resultado (l. 483); main (l. 899)
def modelar_nos_softwares(situacao: str) -> bool:
    return situacao == nucleo.STATUS_OK


# [R06] Fórmula da planilha para o desvio percentual (mesma expressão de [F50] do núcleo):
#   Delta% = |X_Python - X_software| / X_software x 100, vazia enquanto o resultado do software não
#   for preenchido.  Variáveis: celula_python, celula_software (endereços A1).  Fonte: Seção 3.9 do PP.
#   TCC: Eq. (2)
#   Usada em: escrever_bloco_softwares (l. 570)
def formula_desvio_percentual(celula_python: str, celula_software: str) -> str:
    return '=IF(OR(%s="",%s=0),"",ABS(%s-%s)/%s*100)' % (celula_software, celula_software, celula_python,
                                                         celula_software, celula_software)


# [R07] Fórmulas da aba Resumo: contagem de pilares por critério e por situação.
#   Variáveis: coluna_criterio, criterio, coluna_situacao, situacao, total_linhas.
#   Usada em: formula_contagem (l. 222, 223); escrever_resumo (l. 863, 865, 869)
def intervalo_coluna(letra: str, total_linhas: int) -> str:
    return "Resultados!$%s$2:$%s$%d" % (letra, letra, ultima_linha_dados(total_linhas))


def formula_contagem(letra_criterio: str, criterio: str, letra_situacao: str, situacao: str,
                     total_linhas: int) -> str:
    return '=COUNTIFS(%s,%s,%s,"%s")' % (intervalo_coluna(letra_criterio, total_linhas), criterio,
                                          intervalo_coluna(letra_situacao, total_linhas), situacao)


# [R09] Numeração de linhas e colunas da planilha: última linha de dados (cabeçalho na linha 1),
#   linha de um dado abaixo de um cabeçalho e número de colunas de um bloco.
#   Variáveis: total_linhas, linha_cabecalho, deslocamento, colunas_fixas, colunas_variaveis.
#   Usada em: intervalo_coluna (l. 217); estilizar_cabecalho (l. 492); escrever_tabela (l. 514);
#       escrever_bloco_softwares (l. 554); escrever_inviaveis (l. 776, 781); escrever_resumo (l. 870)
def ultima_linha_dados(total_linhas: int) -> int:
    return total_linhas + 1


def linha_abaixo(linha_cabecalho: int, deslocamento: int) -> int:
    return linha_cabecalho + deslocamento


def total_colunas(colunas_fixas: int, colunas_variaveis: int) -> int:
    return colunas_fixas + colunas_variaveis


# [F03] Tempo decorrido do processamento (s).  Variáveis: inicio, fim (time.time()).
#   Usada em: main (l. 906)
def tempo_decorrido(inicio: float, fim: float) -> float:
    return fim - inicio


# [R08] Formatação da planilha: fonte, cores (amarelo = células a preencher pelo usuário) e larguras.
#   Variáveis: FONTE_PADRAO, PREENCHIMENTO_ENTRADA, PREENCHIMENTO_CABECALHO, LARGURA_COLUNA.
#   Usada em: estilizar_cabecalho (l. 494, 495); escrever_tabela (l. 512, 516);
#       escrever_bloco_softwares (l. 545, 557, 565, 566, 572, 575); escrever_inviaveis (l. 766, 774, 778);
#       escrever_hipoteses (l. 845); escrever_resumo (l. 873)
#   Usada também em: verificar_matriz.py: escrever (l. 615, 616, 620)
FONTE_PADRAO = "Arial"
PREENCHIMENTO_ENTRADA = PatternFill("solid", fgColor="FFFF00")
PREENCHIMENTO_CABECALHO = PatternFill("solid", fgColor="D9E1F2")
LARGURA_COLUNA = 13

# [R11] Formato de exibição das entradas na aba Entrada_softwares ([R10]): os valores exatos são irracionais ou têm
#   muitas casas, e a célula guarda o valor completo; o formato mostra casas suficientes para a digitação com a
#   precisão que o software aceitar (l com 4 casas em cm e 6 em m, isto é, 1e-4 cm; forças com 4 casas em kN; momentos
#   com 6 casas em kN·m), e o valor efetivamente aceito é registrado nas colunas "digitado" ([R15]), com 6 casas
#   (FORMATO_DIGITADO), para que o valor anotado não apareça arredondado na planilha.
#   Variáveis: FORMATOS_ENTRADA_SOFTWARES (coluna -> formato), FORMATO_PADRAO_SOFTWARES, FORMATO_DIGITADO.
#   Usada em: escrever_bloco_softwares (l. 559, 567)
FORMATOS_ENTRADA_SOFTWARES = {"ℓ = ℓe (cm)": "0.0000", "ℓ (m)": "0.000000", "Nk (kN)": "0.0000", "Ngk (kN)": "0.0000",
                              "Nqk (kN)": "0.0000", "e1 (cm)": "0.0", "Mgk topo = base (kN·m)": "0.000000",
                              "Mqk topo = base (kN·m)": "0.000000", "Mk topo = base (kN·m)": "0.000000",
                              "ν": "0.0000", "λ (dir. b_w)": "0", "λ calculado (rotina)": "0.000000",
                              "λ h calc": "0.000000"}
FORMATO_PADRAO_SOFTWARES = "0.00"
FORMATO_DIGITADO = "0.000000"

# [R15] Programa comparado (Seção 3.8 do TCC; só o TQS desde 03/11/2026, decisão do autor), colunas das entradas ([R10]) nas abas Entrada_softwares e
#   Teste_momento_minimo, na ordem de lançamento, e colunas para registrar o valor de
#   comprimento e de força que o programa aceitou ([R10]), com o Delta% em relação ao valor exato (mesma fórmula
#   de [R06]).  Variáveis: SOFTWARES_COMPARADOS, COLUNAS_LANCAVEIS, COLUNAS_DIGITADAS, DESVIOS_DIGITADOS.
#   Usada em: escrever_bloco_softwares (l. 547, 562); escrever_entrada_softwares (l. 609, 616, 621, 622);
#       escrever_teste_minimo (l. 654, 663, 671, 673)
#   Usada também em: verificar_matriz.py: conferir_resultados (l. 458, 537)
SOFTWARES_COMPARADOS = ("TQS",)
COLUNAS_LANCAVEIS = ("ℓ = ℓe (cm)", "ℓ (m)", "Nk (kN)", "Ngk (kN)", "Nqk (kN)", "e1 (cm)", "Mgk topo = base (kN·m)",
                     "Mqk topo = base (kN·m)", "Mk topo = base (kN·m)")
COLUNAS_DIGITADAS = ("ℓ digitado {} (m)", "Nk digitado {} (kN)")
DESVIOS_DIGITADOS = (("Δ% ℓ digitado {}", "ℓ (m)", "ℓ digitado {} (m)"),
                     ("Δ% Nk digitado {}", "Nk (kN)", "Nk digitado {} (kN)"))


# [R16] Texto da verificação da envoltória mínima com 2ª ordem ([R12] do núcleo): "englobada" se a envoltória
#   resistente da armadura adotada engloba a elipse (eta >= 1), senão "não englobada (atende pelas análises
#   isoladas)": o critério de aceitação da rotina são as duas análises isoladas à flexão composta normal, forma
#   que o último parágrafo da Seção 15.3.2 da NBR 6118:2023 admite para considerar a envoltória mínima com 2ª
#   ordem; a elipse é verificação complementar, e "Barras elipse" dá o primeiro arranjo que atende também a ela.
#   Variáveis: atende.
#   Usada em: linha_resultado (l. 479); escrever_resumo (l. 856)
def texto_envoltoria(atende: bool) -> str:
    return "englobada" if atende else "não englobada (atende pelas análises isoladas)"

# [R17] Cenário da força na maior dimensão, só para os pilares inviáveis no cenário principal (para justificar a
#   inviabilidade; pedido do autor de 22/10/2026): o mesmo pilar, com os mesmos Nd, l_e, e1 (cm), fck e regra de
#   arranjo [R07], e a excentricidade e1 medida ao longo de h, isto é, o momento de 1ª ordem na direção de maior
#   inércia e só o momento mínimo na direção de b_w. Não é outra modelagem: só muda a direção da força. No núcleo, a
#   dimensão na direção da flexão de 1ª ordem é o campo largura_bw de PilarTeste ([H04] e [H07] do núcleo); por isso o
#   pilar do cenário é o mesmo com os dois campos trocados, e as barras de "duas faces" ficam nas faces perpendiculares
#   ao momento, como num projeto. O cenário principal continua sendo o da matriz (momento na menor dimensão, a favor da
#   segurança e da representatividade). Também a direção de b_w é verificada nesse cenário (só M1d,mín e 2ª ordem): a
#   NBR 6118:2023 exige as duas direções principais (11.3.3.4.2, 11.3.3.4.3 e 15.3.2; Bastos, 2017, item 12.2).
#   Variáveis: pilar (PilarTeste do cenário principal).  Fonte: pedido do autor (22/10/2026); [H04] e [H07] do núcleo.
#   Usada em: main (l. 900)
def pilar_forca_maior_dimensao(pilar: nucleo.PilarTeste) -> nucleo.PilarTeste:
    return replace(pilar, largura_bw=pilar.altura_h, altura_h=pilar.largura_bw)


# [R18] Sensibilidade da armadura máxima nos pilares inviáveis: As,máx = TAXA_SENSIBILIDADE Ac (8 %, limite da NBR
#   6118:2023, 17.3.5.3.2, válido fora da emenda só sem sobreposição de barras, por exemplo com luvas), mantidas as
#   demais regras de detalhamento. O valor adotado continua 4 % fora da emenda ([C09] e [F38] do núcleo; Bastos,
#   2017, p. 81). Com 8 %, a situação "As,calc > As,máx" não ocorre, pois o limite da busca da armadura também é
#   8 % ([F39] do núcleo): acima dele, a seção é insuficiente.
#   Variáveis: TAXA_SENSIBILIDADE.  Fonte: NBR 6118:2023, 17.3.5.3.2; Bastos (2017, p. 81).
#   Usada em: main (l. 902, 905)
TAXA_SENSIBILIDADE = nucleo.TAXA_MAXIMA_COM_EMENDA

# [R19] Código do critério de inviabilidade de cada situação de [R05] do núcleo: A = As,calc > As,máx (4 % Ac);
#   E = As,calc > 8 % Ac ou sem equilíbrio estável (seção insuficiente); D = As,calc <= As,máx na geometria de nível
#   [R11] do núcleo, mas nenhum arranjo com As,adot <= As,máx atende a [R09] (As,adot >= As,calc da própria geometria).
#   Variáveis: CODIGOS_SITUACAO.  Fonte: [R05] e [R09] do núcleo.
#   Usada em: colunas_resumo_arranjo (l. 691); linha_inviavel (l. 717, 729, 732, 744)
#   Usada também em: memorial_pilares.py: escrever_documento (l. 948)
CODIGOS_SITUACAO = {nucleo.STATUS_OK: "viável", nucleo.STATUS_ACIMA_MAXIMA: "A", nucleo.STATUS_SECAO_INSUFICIENTE: "E",
                    nucleo.STATUS_SEM_ARRANJO: "D"}


# [R20] Parcela que governa no cenário [R17] (momento na direção de h): "As,mín" quando a armadura mínima é a maior;
#   senão a direção de maior As necessária (empate a favor da direção do momento). Mesma regra de [R04].
#   Variáveis: area_h (direção do momento), area_b (direção de b_w, só M1d,mín), area_minima.
#   Usada em: linha_inviavel (l. 733)
def parcela_governante_cenario(area_h: float, area_b: float, area_minima: float) -> str:
    if area_minima >= max(area_h, area_b):
        return "As,mín"
    return "direção h (com e1)" if area_h >= area_b else "direção b_w (só M1d,mín)"


# [F04] Excentricidade relativa à maior dimensão no cenário [R17]: e1/h.  Variáveis: excentricidade (cm), altura (h, cm).
#   Usada em: linha_inviavel (l. 714)
def excentricidade_relativa_h(excentricidade: float, altura: float) -> float:
    return excentricidade / altura


# [F05] Taxa geométrica em porcentagem: 100 As / Ac.  Variáveis: area_aco, area_concreto.
#   Usada em: linha_inviavel (l. 724, 739)
def taxa_percentual(area_aco: float, area_concreto: float) -> float:
    return 100.0 * area_aco / area_concreto


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CasoMatriz:
    """Níveis da matriz (Quadro 1 do TCC, Tabela 3.1 do PP) e as entradas do Módulo 1 derivadas deles."""
    identificador: str
    largura_bw: float
    altura_h: float
    fck_mpa: float
    esbeltez: float
    nu: float
    excentricidade_relativa: float
    pilar: nucleo.PilarTeste


def criar_pilar(identificador: str, largura: float, altura: float, fck: float, esbeltez: float, nu: float,
                excentricidade_relativa: float, arranjo: str = nucleo.ARRANJO_REGRA) -> CasoMatriz:
    """Converte os níveis (lambda e e1/b_w na direção de b_w, nu) nas entradas lançáveis do Módulo 1 da Figura 3 do
    TCC (l_e, Nd, e1), [R10]."""
    material = nucleo.Material(fck, nucleo.RESISTENCIA_ACO_MPA)
    area_concreto = nucleo.area_secao_retangular(largura, altura)
    pilar = nucleo.PilarTeste(
        identificador=identificador, largura_bw=largura, altura_h=altura, fck_mpa=fck,
        fyk_mpa=nucleo.RESISTENCIA_ACO_MPA,
        forca_normal_calculo=nucleo.forca_normal_de_nu(nu, area_concreto, material.fcd),
        excentricidade_primeira_ordem=excentricidade_de_relativa(excentricidade_relativa, largura),
        comprimento_equivalente=nucleo.comprimento_para_esbeltez(esbeltez, largura),
        vinculacao=nucleo.VINCULACAO_BIAPOIADO, arranjo=arranjo)
    return CasoMatriz(identificador, largura, altura, fck, esbeltez, nu, excentricidade_relativa, pilar)


def montar_matriz() -> list[CasoMatriz]:
    combinacoes = itertools.product(SECOES, RESISTENCIAS_CONCRETO, ESBELTEZES, FORCAS_ADIMENSIONAIS,
                                    EXCENTRICIDADES_RELATIVAS)
    return [criar_pilar(identificador_pilar(numero_ordem), largura, altura, fck, esbeltez, nu, relativa)
            for numero_ordem, ((largura, altura), fck, esbeltez, nu, relativa) in enumerate(combinacoes, start=1)]


def montar_teste_minimo() -> list[CasoMatriz]:
    """Conjunto complementar do teste do momento mínimo ([R13]): seção -> lambda, e1 = 0."""
    combinacoes = itertools.product(SECOES, ESBELTEZES_TESTE_MINIMO)
    return [criar_pilar(identificador_teste_minimo(numero_ordem), largura, altura, FCK_TESTE_MINIMO, esbeltez,
                        NU_TESTE_MINIMO, EXCENTRICIDADE_TESTE_MINIMO)
            for numero_ordem, ((largura, altura), esbeltez) in enumerate(combinacoes, start=1)]


def valor_ou_vazio(valor):
    """Células sem valor numérico finito (não aplicável ou inviável) ficam vazias na planilha."""
    if valor is None or (isinstance(valor, float) and not math.isfinite(valor)):
        return None
    return valor


def linha_entrada(caso: CasoMatriz) -> dict:
    """Linha da tabela completa dos casos (entradas do Módulo 1 e níveis de origem)."""
    pilar = caso.pilar
    return {
        "ID": caso.identificador, "b_w (cm)": caso.largura_bw, "h (cm)": caso.altura_h, "fck (MPa)": caso.fck_mpa,
        "fyk (MPa)": pilar.fyk_mpa, "λ (dir. b_w)": caso.esbeltez, "ν": caso.nu, "e1/b_w": caso.excentricidade_relativa,
        "Nível e1": ROTULOS_NIVEL_E1[caso.excentricidade_relativa], "Nd (kN)": pilar.forca_normal_calculo,
        "e1 (cm)": pilar.excentricidade_primeira_ordem, "ℓe (cm)": pilar.comprimento_equivalente,
        "Vinculação": pilar.vinculacao, "Cobrimento (cm)": pilar.cobrimento,
    }


def linha_resultado(caso: CasoMatriz, resultado: nucleo.ResultadoPilar) -> dict:
    pilar = resultado.pilar
    material = nucleo.Material(pilar.fck_mpa, pilar.fyk_mpa)
    forca_caracteristica, forca_gk, forca_qk = nucleo.parcelas_forca_caracteristica(resultado.forca_normal,
                                                                                  nucleo.FRACAO_PERMANENTE)
    linha = linha_entrada(caso)
    linha.update({
        "fcd (kN/cm²)": material.fcd, "Nk (kN)": forca_caracteristica, "Ngk (kN)": forca_gk, "Nqk (kN)": forca_qk,
        "ν calc": resultado.nu, "λ b_w calc": resultado.esbeltez_b, "λ h calc": resultado.esbeltez_h,
    })
    for rotulo, direcao in (("b_w", resultado.direcao_b), ("h", resultado.direcao_h)):
        linha.update({
            "λ1 %s" % rotulo: direcao.esbeltez_limite, "αb %s" % rotulo: direcao.alfa_b,
            "Método %s" % rotulo: direcao.metodo,
            "Nd·e1 %s (kN·m)" % rotulo: nucleo.kn_cm_para_kn_m(direcao.momento_aplicado),
            "M1d,mín %s (kN·m)" % rotulo: nucleo.kn_cm_para_kn_m(direcao.momento_minimo),
            "e_a %s (cm)" % rotulo: valor_ou_vazio(direcao.excentricidade_acidental),
            "N_sg %s (kN)" % rotulo: valor_ou_vazio(direcao.forca_quase_permanente),
            "M_sg %s (kN·m)" % rotulo: valor_ou_vazio(nucleo.kn_cm_para_kn_m(direcao.momento_quase_permanente)),
            "N_e %s (kN)" % rotulo: valor_ou_vazio(direcao.forca_euler),
            "e_cc %s (cm)" % rotulo: valor_ou_vazio(direcao.excentricidade_fluencia),
            "M1d,A %s (kN·m)" % rotulo: valor_ou_vazio(nucleo.kn_cm_para_kn_m(direcao.momento_primeira_ordem)),
            "Md,tot %s (kN·m)" % rotulo: valor_ou_vazio(nucleo.kn_cm_para_kn_m(direcao.momento_total)),
            "1/r %s (1/m)" % rotulo: valor_ou_vazio(curvatura_por_metro(direcao.curvatura)),
            "As,nec %s (cm²)" % rotulo: valor_ou_vazio(direcao.area_necessaria),
        })
    candidato = resultado.candidato
    envoltoria = resultado.envoltoria
    elipse = resultado.candidato_elipse
    linha.update({
        "As,mín (cm²)": resultado.area_minima, "As,máx (cm²)": resultado.area_maxima,
        "As,calc (cm²)": valor_ou_vazio(resultado.area_calculada),
        "As,nec b_w ref (cm²)": valor_ou_vazio(resultado.direcao_b_referencia.area_necessaria),
        "Md,tot b_w ref (kN·m)": valor_ou_vazio(nucleo.kn_cm_para_kn_m(resultado.direcao_b_referencia.momento_total)),
        "As,calc ref (cm²)": valor_ou_vazio(resultado.area_referencia),
        "d' ref (cm)": resultado.distancia_eixo_referencia,
        "Governa": parcela_governante(resultado.direcao_b.area_necessaria, resultado.direcao_h.area_necessaria,
                                      resultado.area_minima),
        "Arranjo adotado": pilar.arranjo if candidato else None,
        "Arranjos avaliados": ", ".join(resultado.arranjos_avaliados),
        "Barras": "%d φ %.1f" % (candidato.numero_barras, cm_para_mm(candidato.diametro)) if candidato else None,
        "n barras": candidato.numero_barras if candidato else None,
        "φl (mm)": cm_para_mm(candidato.diametro) if candidato else None,
        "φt (mm)": cm_para_mm(candidato.diametro_estribo) if candidato else None,
        "d' (cm)": candidato.distancia_eixo if candidato else None,
        "As,adot (cm²)": candidato.area if candidato else None,
        "ρ adot": nucleo.taxa_geometrica(candidato.area, resultado.area_concreto) if candidato else None,
        "MRd b_w (kN·m)": nucleo.kn_cm_para_kn_m(resultado.momentos_resistentes[nucleo.DIRECAO_B]) if candidato else None,
        "MRd h (kN·m)": nucleo.kn_cm_para_kn_m(resultado.momentos_resistentes[nucleo.DIRECAO_H]) if candidato else None,
        "Md,tot,mín b_w (kN·m)": valor_ou_vazio(nucleo.kn_cm_para_kn_m(envoltoria.direcao_b.momento_total))
        if envoltoria else None,
        "Md,tot,mín h (kN·m)": valor_ou_vazio(nucleo.kn_cm_para_kn_m(envoltoria.direcao_h.momento_total))
        if envoltoria else None,
        "η envoltória mínima": envoltoria.fator if envoltoria else None,
        "ψ crítico (°)": valor_ou_vazio(envoltoria.direcao_critica) if envoltoria else None,
        "Envoltória mínima (15.3.2)": texto_envoltoria(envoltoria.atende) if envoltoria else None,
        "As,adot elipse (cm²)": elipse.area if elipse else None,
        "Barras elipse": "%d φ %.1f" % (elipse.numero_barras, cm_para_mm(elipse.diametro)) if elipse else None,
        "Situação": resultado.status,
        "Modelar nos softwares": "sim" if modelar_nos_softwares(resultado.status) else "não",
        "x barras (cm)": " ".join("%.4f" % valor for valor in candidato.x) if candidato else None,
        "y barras (cm)": " ".join("%.4f" % valor for valor in candidato.y) if candidato else None,
    })
    return linha


def estilizar_cabecalho(planilha, linha: int, total_colunas: int) -> None:
    borda = Border(bottom=Side(style="thin"))
    for coluna in range(1, total_colunas + 1):
        celula = planilha.cell(row=linha, column=coluna)
        celula.font = Font(name=FONTE_PADRAO, bold=True, size=10)
        celula.fill = PREENCHIMENTO_CABECALHO
        celula.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        celula.border = borda


FORMATOS = {"(cm²)": "0.00", "(kN·m)": "0.00", "(kN)": "0.00", "(cm)": "0.00", "(1/m)": "0.0000",
            "ρ adot": "0.00%", "fcd": "0.000", "λ": "0.00", "αb": "0.00", "e1/b_w": "0.00", "e1/h": "0.00", "ν": "0.0000"}


def escrever_tabela(planilha, linhas: list[dict], congelar: str) -> dict[str, str]:
    colunas = list(linhas[0].keys())
    planilha.append(colunas)
    for linha in linhas:
        planilha.append([linha[coluna] for coluna in colunas])
    estilizar_cabecalho(planilha, 1, len(colunas))
    for indice, coluna in enumerate(colunas, start=1):
        letra = get_column_letter(indice)
        planilha.column_dimensions[letra].width = LARGURA_COLUNA
        formato = next((valor for chave, valor in FORMATOS.items() if chave in coluna), None)
        for numero in range(2, ultima_linha_dados(len(linhas)) + 1):
            celula = planilha.cell(row=numero, column=indice)
            celula.font = Font(name=FONTE_PADRAO, size=10)
            if formato and isinstance(celula.value, (int, float)):
                celula.number_format = formato
    planilha.freeze_panes = congelar
    planilha.auto_filter.ref = planilha.dimensions
    return {coluna: get_column_letter(indice) for indice, coluna in enumerate(colunas, start=1)}


def entradas_lancaveis(linha: dict) -> dict:
    """Valores digitados no TQS ([R10]): comprimento, cargas características e momentos de 1ª ordem."""
    comprimento = nucleo.comprimento_real(linha["ℓe (cm)"], nucleo.FATOR_COMPRIMENTO_BIAPOIADO)
    return {
        "ℓ = ℓe (cm)": comprimento, "ℓ (m)": cm_para_m(comprimento), "Nk (kN)": linha["Nk (kN)"],
        "Ngk (kN)": linha["Ngk (kN)"], "Nqk (kN)": linha["Nqk (kN)"], "e1 (cm)": linha["e1 (cm)"],
        "Mgk topo = base (kN·m)": nucleo.kn_cm_para_kn_m(nucleo.momento_excentricidade(linha["Ngk (kN)"],
                                                                                       linha["e1 (cm)"])),
        "Mqk topo = base (kN·m)": nucleo.kn_cm_para_kn_m(nucleo.momento_excentricidade(linha["Nqk (kN)"],
                                                                                       linha["e1 (cm)"])),
        "Mk topo = base (kN·m)": nucleo.kn_cm_para_kn_m(nucleo.momento_excentricidade(linha["Nk (kN)"],
                                                                                      linha["e1 (cm)"])),
    }


def escrever_bloco_softwares(planilha, legenda: list[str], colunas_dados: list[str], linhas_valores: list[list],
                             colunas_preencher: tuple[str, ...], pares_desvio: tuple[tuple[str, str, str], ...],
                             modelar: list[bool], coluna_congelada: str) -> None:
    """Legenda, cabeçalho, valores do Python e, para o programa comparado, células amarelas a preencher e Delta% ([R06])."""
    for texto in legenda:
        planilha.append([texto])
        planilha.cell(row=planilha.max_row, column=1).font = Font(name=FONTE_PADRAO, size=10, italic=True)
    linha_cabecalho = planilha.max_row + 1
    colunas_software = [nome.format(software) for software in SOFTWARES_COMPARADOS
                        for nome in (*colunas_preencher, *(desvio for desvio, _, _ in pares_desvio))]
    cabecalho = [*colunas_dados, *colunas_software]
    planilha.append(cabecalho)
    estilizar_cabecalho(planilha, linha_cabecalho, len(cabecalho))
    posicao = {coluna: get_column_letter(indice) for indice, coluna in enumerate(cabecalho, start=1)}
    for deslocamento, (valores, incluir) in enumerate(zip(linhas_valores, modelar), start=1):
        numero = linha_abaixo(linha_cabecalho, deslocamento)
        for indice, valor in enumerate(valores, start=1):
            celula = planilha.cell(row=numero, column=indice, value=valor)
            celula.font = Font(name=FONTE_PADRAO, size=10)
            if isinstance(valor, (int, float)) and not isinstance(valor, bool):
                celula.number_format = FORMATOS_ENTRADA_SOFTWARES.get(cabecalho[indice - 1], FORMATO_PADRAO_SOFTWARES)
        if not incluir:
            continue
        for software in SOFTWARES_COMPARADOS:
            for nome in colunas_preencher:
                celula = planilha["%s%d" % (posicao[nome.format(software)], numero)]
                celula.fill = PREENCHIMENTO_ENTRADA
                celula.font = Font(name=FONTE_PADRAO, size=10, color="0000FF")
                celula.number_format = FORMATO_DIGITADO if "digitado" in nome else FORMATO_PADRAO_SOFTWARES
            for nome_desvio, nome_python, nome_software in pares_desvio:
                celula = planilha["%s%d" % (posicao[nome_desvio.format(software)], numero)]
                celula.value = formula_desvio_percentual("%s%d" % (posicao[nome_python], numero),
                                                         "%s%d" % (posicao[nome_software.format(software)], numero))
                celula.font = Font(name=FONTE_PADRAO, size=10)
                celula.number_format = "0.0"
    for indice in range(1, len(cabecalho) + 1):
        planilha.column_dimensions[get_column_letter(indice)].width = LARGURA_COLUNA
    planilha.column_dimensions["A"].width = 9
    planilha.freeze_panes = planilha["%s%d" % (coluna_congelada, linha_cabecalho + 1)]
    planilha["A%d" % linha_cabecalho].comment = Comment("Mesma identificação das abas Matriz_entradas e Resultados.",
                                                        "TCC")


def escrever_entrada_softwares(livro: Workbook, linhas: list[dict]) -> None:
    legenda = [
        "Dados de entrada para o TQS (somente pilares com 'Modelar nos softwares' = sim) e resultados a preencher.",
        "Seção b_w × h, com b_w = MENOR dimensão (20, 30 ou 40 cm) e h = maior (30, 50 ou 80 cm). Pilar biapoiado, "
        "ℓ = ℓe (β = 1); carga axial Nk = Ngk + Nqk (Ngk = 70 % de Nk) aplicada com excentricidade e1 medida ao longo de "
        "b_w (flexão na direção de b_w, a de menor inércia, em torno do eixo paralelo a h), igual no topo e na base "
        "(Mgk = Ngk·e1 e Mqk = Nqk·e1 nos casos de carga permanente e variável; Mk = Nk·e1; curvatura simples); na "
        "direção de h, sem momento aplicado (só o mínimo da norma).",
        "Notação: nas fórmulas da NBR 6118:2023 e nos ábacos, h é a altura da seção na direção do momento; aqui essa "
        "altura é b_w (λ e e1/b_w medidos em b_w). Conferir no software qual dimensão recebe o momento: o momento deve "
        "atuar no plano que contém a menor dimensão.",
        "Os valores são os EXATOS usados na rotina (ℓe = λ b_w/√12 e Nd = ν Ac fcd, sem arredondamento; a célula guarda "
        "o valor completo). Digitar com o máximo de casas que o software aceitar e registrar nas colunas 'digitado' o "
        "valor que ele efetivamente aceitou (conferir no relatório do software, por exemplo o λ calculado): a diferença é "
        "limitação de entrada do software e entra na análise das divergências. Conferir que o software adota comprimento "
        "de flambagem igual a ℓ, sem a redução ℓe = ℓ0 + h da Seção 15.6 da NBR 6118:2023.",
        "Células em AMARELO: preencher com os resultados do TQS (As em cm²; Md,tot em kN·m, nas direções de b_w "
        "e de h).",
        "Md,tot h: momento total na direção de h, onde não há momento aplicado; na rotina, M1d,mín acrescido dos efeitos "
        "locais de 2ª ordem de λh (análise isolada da Seção 15.3.2). Mostra se o software aplica o momento mínimo nessa "
        "direção quando o momento de 1ª ordem da direção de b_w já supera o mínimo. Se o software não informar momento "
        "nessa direção, anotar 0.",
        "As,calc ref Python: As,calc na geometria de referência ([R11] do núcleo: duas faces, φl = 10 mm, nível 1, "
        "d' = 3,5 cm), que não depende das barras escolhidas; As,calc − As,calc ref mede o efeito da posição das barras.",
        "Δ% calculado automaticamente: |X_Python − X_software| / X_software × 100 (Seção 3.9 do PP).",
    ]
    colunas_dados = ["ID", "b_w (cm)", "h (cm)", "fck (MPa)", "λ (dir. b_w)", "λ calculado (rotina)", "ν", "e1/b_w",
                     *COLUNAS_LANCAVEIS, "As,calc Python (cm²)", "As,calc ref Python (cm²)", "As,adot Python (cm²)",
                     "Md,tot b_w Python (kN·m)", "Md,tot h Python (kN·m)", "Governa (Python)", "Modelar nos softwares"]
    linhas_valores = []
    for linha in linhas:
        lancaveis = entradas_lancaveis(linha)
        linhas_valores.append([linha["ID"], linha["b_w (cm)"], linha["h (cm)"], linha["fck (MPa)"],
                               linha["λ (dir. b_w)"], linha["λ b_w calc"], linha["ν"], linha["e1/b_w"],
                               *(lancaveis[coluna] for coluna in COLUNAS_LANCAVEIS), linha["As,calc (cm²)"],
                               linha["As,calc ref (cm²)"], linha["As,adot (cm²)"], linha["Md,tot b_w (kN·m)"],
                               linha["Md,tot h (kN·m)"], linha["Governa"], linha["Modelar nos softwares"]])
    escrever_bloco_softwares(
        livro.create_sheet("Entrada_softwares"), legenda, colunas_dados, linhas_valores,
        (*COLUNAS_DIGITADAS, "As,calc {} (cm²)", "As,adot {} (cm²)", "Md,tot b_w {} (kN·m)", "Md,tot h {} (kN·m)"),
        (*DESVIOS_DIGITADOS, ("Δ% As,calc {}", "As,calc Python (cm²)", "As,calc {} (cm²)"),
         ("Δ% Md,tot b_w {}", "Md,tot b_w Python (kN·m)", "Md,tot b_w {} (kN·m)"),
         ("Δ% Md,tot h {}", "Md,tot h Python (kN·m)", "Md,tot h {} (kN·m)")),
        [linha["Modelar nos softwares"] == "sim" for linha in linhas], "H")


def escrever_teste_minimo(livro: Workbook, linhas: list[dict]) -> None:
    legenda = [
        "Teste do momento mínimo e da envoltória mínima com 2ª ordem ([R13]): 9 pilares fora da matriz do PP, só com o "
        "momento mínimo de 1ª ordem (e1 = 0; nenhum momento aplicado nas duas direções). Lançar no TQS com os "
        "mesmos critérios da aba Entrada_softwares (seção b_w × h, b_w = menor dimensão).",
        "Por que este teste: na matriz, o momento mínimo nunca define a armadura (Nd·e1 > M1d,mín na direção de b_w, e a "
        "direção de h, só com o mínimo, não governa As,calc em nenhum pilar); diferenças entre a rotina e o TQS no "
        "tratamento do mínimo não aparecem no As da matriz. Aqui o mínimo governa, e cada parcela de uma divergência "
        "(M1d,mín, 2ª ordem, elipse) pode ser atribuída à sua causa.",
        "Observar no TQS: (1) o M1d,mín de cada direção, Nd (0,015 + 0,03 h), com h = altura da seção na "
        "direção considerada (b_w na direção de b_w; h na direção de h; NBR 6118:2023, 11.3.3.4.3); (2) se o mínimo é "
        "aplicado nas DUAS direções; (3) se os efeitos locais de 2ª ordem são somados ao mínimo, com αb = 1 (15.8.2 d)), "
        "comparando Md,tot b_w e Md,tot h com os da rotina; (4) se a armadura adotada engloba a envoltória mínima com "
        "2ª ordem (elipse da Figura 15.2): compare As,adot do software com As,adot Python (duas análises isoladas de "
        "flexão composta normal, segundo período de 15.3.2) e com As,adot elipse (primeiro arranjo que engloba a elipse).",
        "η envoltória: menor razão entre o raio da envoltória resistente da armadura adotada pela rotina (a Nd) e o raio "
        "da elipse de semieixos Md,tot,mín b_w e Md,tot,mín h; η < 1: a armadura das duas análises isoladas não engloba a "
        "elipse; ψ crítico: direção do vetor momento (0° = direção de b_w; 90° = direção de h) em que η é mínimo.",
        "Critério de aceitação: as duas análises isoladas à flexão composta normal, que o último parágrafo da Seção "
        "15.3.2 da NBR 6118:2023 admite para considerar a envoltória mínima com 2ª ordem; a elipse é verificação "
        "complementar, e 'Barras elipse' é o primeiro arranjo que atende também a ela (η ≥ 1).",
        "Células em AMARELO: preencher com os resultados do TQS (momentos em kN·m; As em cm²). 'Elipse "
        "verificada?': sim, não ou não informado, conforme o relatório do software.",
        "Δ% calculado automaticamente: |X_Python − X_software| / X_software × 100 (Seção 3.9 do PP).",
    ]
    colunas_dados = ["ID", "b_w (cm)", "h (cm)", "fck (MPa)", "λ (dir. b_w)", "λ calculado (rotina)", "ν",
                     *COLUNAS_LANCAVEIS, "M1d,mín b_w Python (kN·m)", "M1d,mín h Python (kN·m)", "Método b_w",
                     "Método h", "Md,tot b_w Python (kN·m)", "Md,tot h Python (kN·m)", "As,calc Python (cm²)",
                     "As,adot Python (cm²)", "Barras Python", "η envoltória", "ψ crítico (°)", "As,adot elipse (cm²)",
                     "Barras elipse", "Modelar nos softwares"]
    linhas_valores = []
    for linha in linhas:
        lancaveis = entradas_lancaveis(linha)
        linhas_valores.append([linha["ID"], linha["b_w (cm)"], linha["h (cm)"], linha["fck (MPa)"],
                               linha["λ (dir. b_w)"], linha["λ b_w calc"], linha["ν"],
                               *(lancaveis[coluna] for coluna in COLUNAS_LANCAVEIS), linha["M1d,mín b_w (kN·m)"],
                               linha["M1d,mín h (kN·m)"], linha["Método b_w"], linha["Método h"],
                               linha["Md,tot b_w (kN·m)"], linha["Md,tot h (kN·m)"], linha["As,calc (cm²)"],
                               linha["As,adot (cm²)"], linha["Barras"], linha["η envoltória mínima"],
                               linha["ψ crítico (°)"], linha["As,adot elipse (cm²)"], linha["Barras elipse"],
                               linha["Modelar nos softwares"]])
    escrever_bloco_softwares(
        livro.create_sheet("Teste_momento_minimo"), legenda, colunas_dados, linhas_valores,
        (*COLUNAS_DIGITADAS, "M1d,mín b_w {} (kN·m)", "M1d,mín h {} (kN·m)", "Md,tot b_w {} (kN·m)",
         "Md,tot h {} (kN·m)", "As,adot {} (cm²)", "Barras {}", "Elipse verificada? {}"),
        (*DESVIOS_DIGITADOS, ("Δ% M1d,mín b_w {}", "M1d,mín b_w Python (kN·m)", "M1d,mín b_w {} (kN·m)"),
         ("Δ% M1d,mín h {}", "M1d,mín h Python (kN·m)", "M1d,mín h {} (kN·m)"),
         ("Δ% Md,tot b_w {}", "Md,tot b_w Python (kN·m)", "Md,tot b_w {} (kN·m)"),
         ("Δ% Md,tot h {}", "Md,tot h Python (kN·m)", "Md,tot h {} (kN·m)")),
        [linha["Modelar nos softwares"] == "sim" for linha in linhas], "H")


def texto_barras(candidato) -> str | None:
    return "%d φ %.1f" % (candidato.numero_barras, cm_para_mm(candidato.diametro)) if candidato is not None else None


def colunas_resumo_arranjo(prefixo: str, resumo: nucleo.ResumoArranjo | None) -> dict:
    """Prova do detalhamento de um arranjo de [R07]: situação, As,calc, candidatos e o candidato de maior área."""
    if resumo is None:
        return {"%s: situação" % prefixo: None, "%s: As,calc (cm²)" % prefixo: None,
                "%s: candidatos" % prefixo: None, "%s: maior candidato" % prefixo: None,
                "%s: As,adot do maior (cm²)" % prefixo: None, "%s: As,calc do maior (cm²)" % prefixo: None}
    maior = resumo.candidato_maior_area
    return {"%s: situação" % prefixo: CODIGOS_SITUACAO[resumo.status],
            "%s: As,calc (cm²)" % prefixo: valor_ou_vazio(resumo.area_calculada),
            "%s: candidatos" % prefixo: resumo.candidatos,
            "%s: maior candidato" % prefixo: texto_barras(maior),
            "%s: As,adot do maior (cm²)" % prefixo: maior.area if maior is not None else None,
            "%s: As,calc do maior (cm²)" % prefixo: valor_ou_vazio(resumo.area_calculada_maior_area)}


def resumos_por_arranjo(resultado: nucleo.ResultadoPilar) -> dict:
    return {resumo.arranjo: resumo for resumo in resultado.resumos_arranjos}


def linha_inviavel(caso: CasoMatriz, principal: nucleo.ResultadoPilar, principal_sensibilidade: nucleo.ResultadoPilar,
                   cenario: nucleo.ResultadoPilar, cenario_sensibilidade: nucleo.ResultadoPilar) -> dict:
    """Linha da tabela dos inviáveis: cenário principal (momento na direção de b_w), sensibilidade com 8 %, cenário
    [R17] (momento na direção de h; no resultado do núcleo, direcao_b é a direção do momento, isto é, a de h, e
    direcao_h é a de b_w) e a sua sensibilidade."""
    pilar = caso.pilar
    resumos = resumos_por_arranjo(principal)
    linha = {
        "ID": caso.identificador, "b_w (cm)": caso.largura_bw, "h (cm)": caso.altura_h, "fck (MPa)": caso.fck_mpa,
        "λ (dir. b_w)": caso.esbeltez, "λ h calc": principal.esbeltez_h, "ν": caso.nu,
        "e1 (cm)": pilar.excentricidade_primeira_ordem, "e1/b_w": caso.excentricidade_relativa,
        "e1/h": excentricidade_relativa_h(pilar.excentricidade_primeira_ordem, caso.altura_h),
        "Nd (kN)": pilar.forca_normal_calculo, "As,mín (cm²)": principal.area_minima,
        "As,máx 4 % (cm²)": principal.area_maxima, "As,máx 8 % (cm²)": principal_sensibilidade.area_maxima,
        "Situação (momento em b_w)": principal.status, "Critério": CODIGOS_SITUACAO[principal.status],
        "Governa": parcela_governante(principal.direcao_b.area_necessaria, principal.direcao_h.area_necessaria,
                                      principal.area_minima),
        "Método b_w": principal.direcao_b.metodo, "Método h": principal.direcao_h.metodo,
        "As,nec b_w (cm²)": valor_ou_vazio(principal.direcao_b.area_necessaria),
        "As,nec h (cm²)": valor_ou_vazio(principal.direcao_h.area_necessaria),
        "As,calc (cm²)": valor_ou_vazio(principal.area_calculada),
        "As,calc / Ac (%)": valor_ou_vazio(taxa_percentual(principal.area_calculada, principal.area_concreto)),
    }
    linha.update(colunas_resumo_arranjo("Duas faces", resumos.get(nucleo.ARRANJO_DUAS_FACES)))
    linha.update(colunas_resumo_arranjo("Perímetro", resumos.get(nucleo.ARRANJO_PERIMETRO)))
    linha.update({
        "Situação com As,máx = 8 %": CODIGOS_SITUACAO[principal_sensibilidade.status],
        "Barras com 8 %": texto_barras(principal_sensibilidade.candidato),
        "As,adot com 8 % (cm²)": principal_sensibilidade.candidato.area if principal_sensibilidade.candidato else None,
        "Situação (momento em h)": cenario.status, "Critério (momento em h)": CODIGOS_SITUACAO[cenario.status],
        "Governa (momento em h)": parcela_governante_cenario(cenario.direcao_b.area_necessaria,
                                                             cenario.direcao_h.area_necessaria, cenario.area_minima),
        "Método h (momento em h)": cenario.direcao_b.metodo, "Método b_w (momento em h)": cenario.direcao_h.metodo,
        "As,nec h, momento em h (cm²)": valor_ou_vazio(cenario.direcao_b.area_necessaria),
        "As,nec b_w, momento em h (cm²)": valor_ou_vazio(cenario.direcao_h.area_necessaria),
        "As,calc, momento em h (cm²)": valor_ou_vazio(cenario.area_calculada),
        "As,calc / Ac, momento em h (%)": valor_ou_vazio(taxa_percentual(cenario.area_calculada, cenario.area_concreto)),
        "Barras, momento em h": texto_barras(cenario.candidato),
        "As,adot, momento em h (cm²)": cenario.candidato.area if cenario.candidato else None,
        "Md,tot h, momento em h (kN·m)": valor_ou_vazio(nucleo.kn_cm_para_kn_m(cenario.direcao_b.momento_total)),
        "Md,tot b_w, momento em h (kN·m)": valor_ou_vazio(nucleo.kn_cm_para_kn_m(cenario.direcao_h.momento_total)),
        "Situação, momento em h, As,máx = 8 %": CODIGOS_SITUACAO[cenario_sensibilidade.status],
        "Barras, momento em h, 8 %": texto_barras(cenario_sensibilidade.candidato),
    })
    return linha


def escrever_inviaveis(livro: Workbook, linhas: list[dict]) -> None:
    planilha = livro.create_sheet("Inviaveis")
    legenda = [
        "Pilares inviáveis no cenário principal (momento de 1ª ordem na direção de b_w, a menor dimensão) e a prova do "
        "critério que falhou ([R05] e [R09] do núcleo; [R19]): A = As,calc > As,máx = 4 % Ac; E = As,calc > 8 % Ac ou sem "
        "equilíbrio estável (célula de As vazia na direção que falhou); D = As,calc ≤ As,máx, mas nenhum arranjo com "
        "As,adot ≤ As,máx atende (o maior candidato que cabe tem As,adot < As,calc da própria geometria).",
        "Duas faces e Perímetro: os dois arranjos da regra do PP ([R07]), na ordem em que foram tentados; candidatos = "
        "arranjos com As,adot ≤ As,máx que atendem aos espaçamentos (NBR 6118:2023, 18.4.2).",
        "Com As,máx = 8 %: sensibilidade [R18] (8 % Ac fora da emenda só vale sem sobreposição de barras, por exemplo com "
        "luvas); o valor adotado é 4 % (8 % na região de emenda com traspasse de todas as barras; Bastos, 2017, p. 81).",
        "Momento em h: cenário [R17], o mesmo pilar (mesmos Nd, ℓe e e1) com a força excêntrica na direção da maior "
        "dimensão; a direção de b_w continua verificada com M1d,mín e 2ª ordem (NBR 6118:2023, 11.3.3.4.3 e 15.3.2).",
    ]
    for texto in legenda:
        planilha.append([texto])
        planilha.cell(row=planilha.max_row, column=1).font = Font(name=FONTE_PADRAO, size=10, italic=True)
    linha_cabecalho = planilha.max_row + 1
    colunas = list(linhas[0].keys())
    planilha.append(colunas)
    estilizar_cabecalho(planilha, linha_cabecalho, len(colunas))
    for linha in linhas:
        planilha.append([linha[coluna] for coluna in colunas])
    for indice, coluna in enumerate(colunas, start=1):
        planilha.column_dimensions[get_column_letter(indice)].width = LARGURA_COLUNA
        formato = next((valor for chave, valor in FORMATOS.items() if chave in coluna), None)
        for numero in range(linha_abaixo(linha_cabecalho, 1), linha_abaixo(linha_cabecalho, len(linhas)) + 1):
            celula = planilha.cell(row=numero, column=indice)
            celula.font = Font(name=FONTE_PADRAO, size=10)
            if formato and isinstance(celula.value, (int, float)) and not isinstance(celula.value, bool):
                celula.number_format = formato
    planilha.freeze_panes = planilha["B%d" % linha_abaixo(linha_cabecalho, 1)]


def escrever_hipoteses(livro: Workbook) -> None:
    planilha = livro.create_sheet("Hipoteses")
    planilha.append(["Parâmetro", "Valor", "Fonte"])
    itens = [
        ("γc / γs / γf", " / ".join(texto_decimal(valor) for valor in (nucleo.GAMA_C, nucleo.GAMA_S, nucleo.GAMA_F)),
         "NBR 6118:2023, Tabelas 11.1 e 12.1"),
        ("Combinação última", "Nd = γf (Ngk + Nqk), γg = γq = 1,4", "NBR 6118:2023, 11.7.1 (Tabela 11.1) e 11.8.2.4 (Tabela 11.3)"),
        ("Combinação quase permanente", "Nsg = Ngk + ψ2 Nqk; Msg = Nsg·e1", "NBR 6118:2023, 11.8.3.2 (Tabela 11.4) e 15.8.4"),
        ("Fração permanente Ngk/Nk", nucleo.FRACAO_PERMANENTE, "Hipótese do trabalho (Seção 3.6 do TCC)"),
        ("ψ2 (edifício residencial)", nucleo.PSI2, "NBR 6118:2023, Tabela 11.2"),
        ("Coeficiente de fluência φ", nucleo.COEFICIENTE_FLUENCIA,
         "NBR 6118:2023, Tabela 8.1 (C20 a C45, U = 55 %, t0 = 60 dias, 2Ac/u = 20 cm; seção 40 × 80: 2,47 ≈ 2,5)"),
        ("γf3 (relação M, N, 1/r)", nucleo.GAMA_F3, "NBR 6118:2023, 15.3.1"),
        ("Curva de deformação", "1,10 fcd com N = Nd/γf3", "NBR 6118:2023, 15.3.1, Figura 15.1"),
        ("Curva de resistência", "0,85 ηc fcd (parábola-retângulo, εc2 = 2,0 ‰, εcu = 3,5 ‰); aço elastoplástico perfeito",
         "NBR 6118:2023, 8.2.10.1, 8.3.6 e 17.2.2"),
        ("Seção de concreto no equilíbrio", "seção líquida: b_w·h menos o concreto deslocado pelas barras comprimidas",
         "NBR 6118:2023, 17.2.2 (omissa quanto ao desconto); Fusco (1981, p. 226); [H05] do núcleo; efeito em "
         "sensibilidade_secao_liquida.txt"),
        ("Aço", "CA-50, fyk = %.0f MPa, Es = 210 GPa; estribos de 5 mm em CA-60" % nucleo.RESISTENCIA_ACO_MPA,
         "Projeto de pesquisa; NBR 6118:2023, 8.3"),
        ("αE (agregado)", nucleo.ALFA_E_AGREGADO, "NBR 6118:2023, 8.2.8 (granito/gnaisse)"),
        ("Cobrimento nominal (cm)", nucleo.COBRIMENTO_NOMINAL,
         "Projeto de pesquisa; NBR 6118:2023, Tabela 7.2 (classe de agressividade I; classe II reduzida a I pela nota a "
         "da Tabela 6.1, para ambientes internos secos)"),
        ("d_máx do agregado (cm)", nucleo.DIAMETRO_MAXIMO_AGREGADO, "Hipótese do trabalho (brita 1)"),
        ("Vinculação", "biapoiado, β = 1, ℓe = ℓ", "Projeto de pesquisa; NBR 6118:2023, 15.6"),
        ("Notação das dimensões", nucleo.NOTACAO_DIMENSOES,
         "PP (01/07/2026), Tabela 3.1; NBR 6118:2023, 11.3.3.4.3 e 15.8.2; Bastos (2017, p. 83-84); [H07] do núcleo"),
        ("Momentos de 1ª ordem (direção b_w)", "M1d,A = M1d,B = Nd·e1 ≥ M1d,mín; αb = 0,60 + 0,40 MB/MA = 1,0",
         "Projeto de pesquisa; NBR 6118:2023, 11.3.3.4.3 e 15.8.2"),
        ("Direção de h", "momento de 1ª ordem igual a M1d,mín, com os efeitos locais de 2ª ordem de λh; análise "
         "isolada", "NBR 6118:2023, 15.3.2 (segundo período)"),
        ("Envoltória mínima com 2ª ordem", nucleo.LEITURA_ENVOLTORIA_MINIMA + "; semieixos Md,tot,mín b_w e "
         "Md,tot,mín h só com M1d,mín (αb = 1); η e primeiro arranjo que engloba a elipse informados",
         "NBR 6118:2023, 11.3.3.4.3, 15.3.2 e 15.8.2 d); [H06] e [R12] do núcleo"),
        ("Teste do momento mínimo", "9 pilares com e1 = 0 (três seções; λ = 30, 90 e 110; fck = 30 MPa; ν = 0,40), "
         "fora da matriz do PP", "[R13] de processar_matriz; aba Teste_momento_minimo"),
        ("Fluência (λ > 90)", "M1d,A + Nd·ecc", "NBR 6118:2023, 15.8.4; Scadelai e Pinheiro (2007, itens 16.4.5 e 16.6)"),
        ("Arranjo da armadura", "duas faces opostas; perímetro só se duas faces forem inviáveis",
         "Hipótese do trabalho (Seção 3.6 do TCC)"),
        ("As,máx (fora da emenda)", "4 % Ac (8 % na emenda, traspasse de todas as barras)",
         "NBR 6118:2023, 17.3.5.3.2 (8 % Ac, inclusive na emenda); Bastos (2017, p. 81)"),
        ("Bitolas longitudinais (mm)",
         "; ".join(texto_decimal(cm_para_mm(valor)) for valor in nucleo.BITOLAS_LONGITUDINAIS), "ABNT NBR 7480:2024"),
        ("Bitolas de estribo (mm)",
         "; ".join(texto_decimal(cm_para_mm(valor)) for valor in nucleo.BITOLAS_ESTRIBO) + " (5 mm em CA-60)",
         "ABNT NBR 7480:2024"),
        ("Detalhamento", "10 mm ≤ φl ≤ menor dimensão/8; espaçamento livre ≥ máx(20 mm; φl; 1,2 dmáx), também na "
         "emenda; espaçamento entre eixos ≤ mín(2 × menor dimensão; 400 mm); φt ≥ máx(5 mm; φl/4)",
         "NBR 6118:2023, 18.4.2.1, 18.4.2.2 e 18.4.3"),
        ("Escolha das barras", "menor As,adot que atende a As,calc; em empate, menor número de barras e, depois, "
         "menor diâmetro; barras de mesmo diâmetro", "Hipótese do trabalho ([R04] do núcleo)"),
        ("Precisão das entradas", CRITERIO_ENTRADAS,
         "Decisão do autor (22/10/2026); PP (01/07/2026), Seções 1.4.2 e) e 3.6; [R10] de processar_matriz"),
    ]
    for item in itens:
        planilha.append(list(item))
    estilizar_cabecalho(planilha, 1, 3)
    for linha in planilha.iter_rows(min_row=2):
        for celula in linha:
            celula.font = Font(name=FONTE_PADRAO, size=10)
    for letra, largura in (("A", 32), ("B", 58), ("C", 58)):
        planilha.column_dimensions[letra].width = largura


def escrever_resumo(livro: Workbook, total_linhas: int, letras: dict[str, str]) -> None:
    planilha = livro.create_sheet("Resumo")
    situacoes = (nucleo.STATUS_OK, nucleo.STATUS_ACIMA_MAXIMA, nucleo.STATUS_SECAO_INSUFICIENTE,
                 nucleo.STATUS_SEM_ARRANJO)
    planilha.append(["Critério", "Valor", "Pilares", *("Situação: %s" % situacao for situacao in situacoes)])
    blocos = (("λ (dir. b_w)", ESBELTEZES), ("ν", FORCAS_ADIMENSIONAIS), ("e1/b_w", EXCENTRICIDADES_RELATIVAS),
              ("Envoltória mínima (15.3.2)", (texto_envoltoria(True), texto_envoltoria(False))))
    for coluna, niveis in blocos:
        for nivel in niveis:
            numero = planilha.max_row + 1
            planilha.cell(row=numero, column=1, value=coluna)
            planilha.cell(row=numero, column=2, value=nivel)
            planilha.cell(row=numero, column=3, value='=COUNTIF(%s,B%d)' % (
                intervalo_coluna(letras[coluna], total_linhas), numero))
            for deslocamento, situacao in enumerate(situacoes, start=4):
                planilha.cell(row=numero, column=deslocamento, value=formula_contagem(
                    letras[coluna], "B%d" % numero, letras["Situação"], situacao, total_linhas))
    numero = planilha.max_row + 1
    planilha.cell(row=numero, column=1, value="Total")
    planilha.cell(row=numero, column=3, value="=COUNTA(%s)" % intervalo_coluna(letras["ID"], total_linhas))
    estilizar_cabecalho(planilha, 1, total_colunas(3, len(situacoes)))
    for linha in planilha.iter_rows(min_row=2):
        for celula in linha:
            celula.font = Font(name=FONTE_PADRAO, size=10)
    for letra, largura in (("A", 12), ("B", 8), ("C", 9), ("D", 14), ("E", 22), ("F", 22), ("G", 22)):
        planilha.column_dimensions[letra].width = largura


def escrever_csv(caminho: Path, linhas: list[dict]) -> None:
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0].keys()), delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas)


def main() -> None:
    pasta_saida = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "saida"
    pasta_saida.mkdir(parents=True, exist_ok=True)
    casos = montar_matriz()
    import verificar_matriz  # importação tardia: verificar_matriz importa este módulo
    falhas_entradas = verificar_matriz.imprimir(verificar_matriz.conferir_entradas(casos))
    if falhas_entradas:
        sys.exit("processamento interrompido: %d requisito(s) das entradas não atendido(s)" % len(falhas_entradas))
    casos_teste = montar_teste_minimo()
    inicio = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        resultados = list(executor.map(nucleo.dimensionar_pilar, [caso.pilar for caso in casos], chunksize=4))
        resultados_teste = list(executor.map(nucleo.dimensionar_pilar, [caso.pilar for caso in casos_teste]))
        inviaveis = [(caso, resultado) for caso, resultado in zip(casos, resultados)
                     if not modelar_nos_softwares(resultado.status)]
        cenarios = [pilar_forca_maior_dimensao(caso.pilar) for caso, _ in inviaveis]
        principais_8 = list(executor.map(nucleo.dimensionar_pilar, [
            replace(caso.pilar, taxa_armadura_maxima=TAXA_SENSIBILIDADE) for caso, _ in inviaveis], chunksize=2))
        cenarios_4 = list(executor.map(nucleo.dimensionar_pilar, cenarios, chunksize=2))
        cenarios_8 = list(executor.map(nucleo.dimensionar_pilar, [
            replace(pilar, taxa_armadura_maxima=TAXA_SENSIBILIDADE) for pilar in cenarios], chunksize=2))
    duracao = tempo_decorrido(inicio, time.time())
    linhas = [linha_resultado(caso, resultado) for caso, resultado in zip(casos, resultados)]
    linhas_teste = [linha_resultado(caso, resultado) for caso, resultado in zip(casos_teste, resultados_teste)]
    escrever_csv(pasta_saida / "resultados_matriz.csv", linhas)
    escrever_csv(pasta_saida / "resultados_teste_minimo.csv", linhas_teste)
    linhas_inviaveis = [linha_inviavel(caso, resultado, principal_8, cenario, cenario_8)
                        for (caso, resultado), principal_8, cenario, cenario_8
                        in zip(inviaveis, principais_8, cenarios_4, cenarios_8)]
    escrever_csv(pasta_saida / "resultados_inviaveis.csv", linhas_inviaveis)
    livro = Workbook()
    entradas = livro.active
    entradas.title = "Matriz_entradas"
    escrever_tabela(entradas, [linha_entrada(caso) for caso in casos], "B2")
    letras = escrever_tabela(livro.create_sheet("Resultados"), linhas, "B2")
    escrever_entrada_softwares(livro, linhas)
    escrever_teste_minimo(livro, linhas_teste)
    escrever_inviaveis(livro, linhas_inviaveis)
    escrever_resumo(livro, len(linhas), letras)
    escrever_hipoteses(livro)
    livro.save(pasta_saida / "resultados_matriz.xlsx")
    situacoes = {}
    for resultado in resultados:
        situacoes[resultado.status] = situacoes.get(resultado.status, 0) + 1
    print("pilares: %d | tempo: %.1f s | situações: %s" % (len(resultados), duracao, situacoes))
    print("teste do momento mínimo: %s" % "; ".join("%s %s η = %.3f" % (linha["ID"], linha["Situação"],
                                                                        linha["η envoltória mínima"] or 0.0)
                                                    for linha in linhas_teste))
    for chave in ("Critério", "Situação com As,máx = 8 %", "Critério (momento em h)",
                  "Situação, momento em h, As,máx = 8 %"):
        contagem = {}
        for linha in linhas_inviaveis:
            contagem[linha[chave]] = contagem.get(linha[chave], 0) + 1
        print("inviáveis (%d), %s: %s" % (len(linhas_inviaveis), chave, contagem))


if __name__ == "__main__":
    main()
