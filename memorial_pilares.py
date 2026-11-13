# -*- coding: utf-8 -*-
"""
memorial_pilares.py
===================
Memorial de cálculo de cada pilar-teste, do dado de entrada à armadura adotada, para conferência à mão.

Para cada pilar (os 324 da matriz e os 9 do teste do momento mínimo), o memorial apresenta, com cada fórmula, a
substituição dos valores e o resultado:
    1) entradas (Módulo 1 da Figura 3), materiais e geometria;
    2) em cada direção principal (bw e h): momento mínimo, αb, λ1, método de 2ª ordem e, conforme a faixa de λ,
       a curvatura aproximada ou a fluência e o equilíbrio do pilar-padrão acoplado a diagramas M, N, 1/r;
    3) armaduras mínima, máxima, necessária em cada direção e calculada;
    4) detalhamento: arranjo, bitola, número de barras por face, d′, espaçamentos, regras da Seção 18.4, estribo,
       espaçamento dos estribos e barras que exigem estribo suplementar (18.2.4); o candidato anterior na ordem do
       critério de escolha e o motivo de não atender;
    5) verificação da armadura adotada: estado de deformação no ELU com N = Nd (profundidade da linha neutra,
       deformação, tensão e força de cada camada de barras, resultante do concreto e concreto deslocado pelas
       barras), com o fechamento de N e de M pelos valores impressos; no pilar-padrão acoplado, o equilíbrio na
       curva de deformação (1,10 fcd e Nd/γf3);
    6) envoltória mínima com 2ª ordem (fator η) e situação final; nos inviáveis, o critério que falhou.

Todos os números vêm do núcleo (pilares_nbr6118.py), com as mesmas funções usadas no dimensionamento; este
programa apenas organiza e imprime os valores. As grandezas derivadas só para exibição (raio de giração, profundidade
da linha neutra, conversões de unidade, somas de fechamento) estão na zona de auditoria abaixo.

Saídas: saida/memorial/memorial_pilares.docx, saida/memorial/memorial_pilares.pdf (pelo LibreOffice, quando
disponível) e saida/memorial/memorial_pilares.json (todos os valores, sem arredondamento).

Uso:  python3 memorial_pilares.py            (todos os pilares)
      python3 memorial_pilares.py P156 P165  (só os pilares indicados, para conferência rápida)
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

import pilares_nbr6118 as nucleo
import processar_matriz

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — INÍCIO
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [R01] Arquivos: pasta de saída do memorial, nomes dos arquivos e resultados de processar_matriz.py usados para os
#   cenários de sensibilidade dos pilares inviáveis (o memorial não os recalcula).
#   Variáveis: PASTA_MEMORIAL, ARQUIVO_DOCX, ARQUIVO_PDF, ARQUIVO_JSON, ARQUIVO_INVIAVEIS.
#   Usada em: nível do módulo (l. 59, 60, 61, 116); escrever_documento (l. 987, 988); converter_pdf (l. 1070, 1072);
#       ler_inviaveis (l. 1077, 1079); main (l. 1090, 1091, 1099)
PASTA_MEMORIAL = Path(__file__).resolve().parent / "saida" / "memorial"
ARQUIVO_DOCX = PASTA_MEMORIAL / "memorial_pilares.docx"
ARQUIVO_PDF = PASTA_MEMORIAL / "memorial_pilares.pdf"
ARQUIVO_JSON = PASTA_MEMORIAL / "memorial_pilares.json"
ARQUIVO_INVIAVEIS = Path(__file__).resolve().parent / "saida" / "resultados_inviaveis.csv"

# [R02] Aparência do documento: fonte, tamanhos (corpo, tabelas, títulos), margens (cm) e largura das colunas das
#   tabelas de camadas (cm; soma 16,95 cm, dentro dos 17,4 cm do texto): cada uma com o maior valor impresso nos 333
#   memoriais (10 517,5058 kN·cm na última, com o separador de milhar de [F01]) ou a maior palavra do cabeçalho, em
#   8 pt, mais as margens das células, para que nenhum número seja partido em duas linhas.
#   Variáveis: FONTE, TAMANHO_CORPO, TAMANHO_TABELA, TAMANHO_TITULO, TAMANHO_TITULO_DOCUMENTO, TAMANHO_TITULO_SECAO,
#   MARGEM_CM, LARGURAS_CAMADAS_CM.
#   Usada em: adicionar_texto (l. 912);
#       escrever_documento (l. 923, 925, 926, 928, 932, 934, 935, 940, 953, 956, 962, 964, 966, 970, 972, 979, 982, 983)
FONTE = "Arial"
TAMANHO_CORPO = 9
TAMANHO_TABELA = 8
TAMANHO_TITULO = 12
TAMANHO_TITULO_DOCUMENTO = 14
TAMANHO_TITULO_SECAO = 10
MARGEM_CM = 1.8
LARGURAS_CAMADAS_CM = (1.35, 1.45, 1.45, 1.55, 1.75, 2.05, 1.75, 1.75, 1.5, 2.35)

# [R03] Casas decimais impressas por tipo de grandeza. Os valores completos ficam no arquivo JSON. A curvatura é
#   impressa em notação científica com ALGARISMOS_CURVATURA algarismos significativos ([F05]), para que Md,tot possa
#   ser refeito à mão com os números impressos (com 6 casas, 0,000120 cm⁻¹ tem só 3 algarismos e a conta refeita
#   chega a diferir 0,14 % do valor impresso). O comprimento equivalente ℓe, o raio de giração i e o índice de
#   esbeltez λ têm 4 casas (ℓe em metros, 6), para que ℓe = λ bw/√12 e λ = ℓe/i, refeitos na calculadora com os
#   números impressos, deem ao menos 99,999 % do valor impresso (com 2 casas, λ = 259,81/8,66 chega a diferir
#   0,06 %); λ1, que não depende de ℓe, fica com 2 casas. Pelo mesmo critério, as contas que usam ℓe e as que as
#   alimentam (Nd, M1d,A, Md,tot, Nsg, Msg, θ1, ea, Ne, ecc, Mext) têm forças e momentos com 4 casas, 1/r com 7
#   algarismos, Eci com 4 casas, fcd e σcd com 6, θ1 com 9, ea com 6 e ecc com 7 (com 2 casas nas forças e nos
#   momentos, 1/r com 5 algarismos, θ1 com 6 e ea com 2, a conta refeita chegava a diferir 0,2 %). Eci também em MPa,
#   com 3 casas (modulo_mpa), para a conta 5600 √fck ser refeita com fck em MPa.
#   TOLERANCIA_COMPARACAO: folga de ponto flutuante nas comparações de espaçamento impressas (a mesma do núcleo, [R03]).
#   Variáveis: CASAS (tipo -> casas), ALGARISMOS_CURVATURA, SOBRESCRITOS, TOLERANCIA_COMPARACAO.
#   Usada em: numero_tipo (l. 167); atende_maximo (l. 268); atende_minimo (l. 272); numero_cientifico (l. 352);
#       curv (l. 591)
TOLERANCIA_COMPARACAO = 1e-9
CASAS = {"forca": 4, "momento": 4, "comprimento": 2, "area": 3, "tensao": 3, "adimensional": 4, "lambda1": 2,
         "esbeltez": 4, "comprimento_equivalente": 4, "comprimento_equivalente_m": 6, "raio_giracao": 4,
         "deformacao": 4, "modulo": 4, "modulo_mpa": 3, "inercia": 0, "fator": 3, "residuo": 4, "angulo": 3,
         "resistencia_calculo": 6, "desaprumo": 9, "excentricidade_acidental": 6, "excentricidade_fluencia": 7}
ALGARISMOS_CURVATURA = 7
SOBRESCRITOS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

# [R04] Rótulos das direções e dos arranjos, para o texto.
#   Variáveis: ROTULO_DIRECAO, ROTULO_ARRANJO.
#   Usada em: blocos_direcao (l. 645); blocos_verificacao (l. 712); blocos_detalhamento (l. 758, 759, 785)
ROTULO_DIRECAO = {nucleo.DIRECAO_B: "direção de bw (com e1)", nucleo.DIRECAO_H: "direção de h (só o momento mínimo)"}
ROTULO_ARRANJO = {nucleo.ARRANJO_DUAS_FACES: "duas faces opostas (perpendiculares a bw)",
                  nucleo.ARRANJO_PERIMETRO: "distribuída no perímetro"}


# [R05] Conferência automática dos memoriais: arquivo de saída, resultados de processar_matriz.py com que as barras
#   adotadas são comparadas e tolerância relativa da comparação de áreas e diâmetros (aritmética de ponto flutuante).
#   Variáveis: ARQUIVO_CONFERENCIA, ARQUIVOS_RESULTADOS, TOLERANCIA_RELATIVA.
#   Usada em: iguais (l. 330); ler_resultados (l. 995); conferir_memoriais (l. 1063)
ARQUIVO_CONFERENCIA = PASTA_MEMORIAL / "conferencia_memorial.json"
ARQUIVOS_RESULTADOS = (Path(__file__).resolve().parent / "saida" / "resultados_matriz.csv",
                       Path(__file__).resolve().parent / "saida" / "resultados_teste_minimo.csv")
TOLERANCIA_RELATIVA = 1e-9


# [F01] Formatação numérica: valor com n casas, vírgula decimal e separador de milhar do SI (regra do texto do TCC):
#   parte inteira com até quatro algarismos sem separador (5785,7143); com MINIMO_ALGARISMOS_AGRUPAMENTO (cinco) ou
#   mais, grupos de ALGARISMOS_POR_GRUPO (três) algarismos separados por SEPARADOR_MILHAR (espaço não separável,
#   U+00A0): 13 375,8483. Nunca ponto. Infinito e NaN como texto.
#   Variáveis: valor, casas, texto; SEPARADOR_MILHAR, MINIMO_ALGARISMOS_AGRUPAMENTO, ALGARISMOS_POR_GRUPO.
#   Usada em: agrupar_milhares (l. 142, 145, 146, 147, 149); numero (l. 163); numero_tipo (l. 167);
#       numero_cientifico (l. 346); kn (l. 559); kncm (l. 563); cm (l. 567); cm2 (l. 571); cm_le (l. 575);
#       raio (l. 579); esbeltez (l. 583); adim (l. 587); mm (l. 595); blocos_estado (l. 614, 616, 620, 621, 632);
#       blocos_direcao (l. 655, 657, 669, 674, 675, 676, 683, 684, 685, 688, 689, 691, 693);
#       blocos_verificacao (l. 725, 727); blocos_detalhamento (l. 744, 761, 779);
#       blocos_pilar (l. 806, 808, 809, 811, 812, 815, 820, 823, 824, 825, 826, 827, 829, 856, 864);
#       escrever_documento (l. 944, 945, 946, 947)
SEPARADOR_MILHAR = "\u00a0"
MINIMO_ALGARISMOS_AGRUPAMENTO = 5
ALGARISMOS_POR_GRUPO = 3


def agrupar_milhares(texto: str) -> str:
    sinal = texto[:1] if texto[:1] == "-" else ""
    inteira, virgula, decimal = texto[len(sinal):].partition(",")
    if len(inteira) < MINIMO_ALGARISMOS_AGRUPAMENTO or not inteira.isdigit():
        return texto
    grupos = []
    while len(inteira) > ALGARISMOS_POR_GRUPO:
        grupos.insert(0, inteira[-ALGARISMOS_POR_GRUPO:])
        inteira = inteira[:-ALGARISMOS_POR_GRUPO]
    grupos.insert(0, inteira)
    return sinal + SEPARADOR_MILHAR.join(grupos) + virgula + decimal


def numero(valor, casas: int) -> str:
    if valor is None:
        return "–"
    valor = float(valor)
    if math.isnan(valor):
        return "não definido"
    if math.isinf(valor):
        return "∞"
    texto = "%.*f" % (casas, valor)
    if float(texto) == 0.0:
        texto = "%.*f" % (casas, 0.0)
    return agrupar_milhares(texto.replace(".", ","))


def numero_tipo(valor, tipo: str) -> str:
    return numero(valor, CASAS[tipo])


# [F02] Conversões para exibição: kN·cm -> kN·m; cm⁻¹ -> m⁻¹; deformação -> ‰; cm -> mm; taxa (%);
#   cm -> m; MPa -> kN/cm² e kN/cm² -> MPa.
#   Variáveis: valor, area_aco, area_concreto.
#   Usada em: calcular_memorial (l. 543); mm (l. 595); blocos_estado (l. 614, 620); blocos_direcao (l. 674, 684);
#       blocos_detalhamento (l. 761); blocos_pilar (l. 823, 825, 827); conferir_memoriais (l. 1023)
def kncm_para_knm(valor: float) -> float:
    return valor / 100.0


def por_cm_para_por_m(valor: float) -> float:
    return valor * 100.0


def por_mil(valor: float) -> float:
    return valor * 1000.0


def cm_para_mm(valor: float) -> float:
    return valor * 10.0


def taxa_percentual(area_aco: float, area_concreto: float) -> float:
    return area_aco / area_concreto * 100.0


def cm_para_m(valor: float) -> float:
    return valor / 100.0


def mpa_para_kncm2(valor: float) -> float:
    return valor / 10.0


def kncm2_para_mpa(valor: float) -> float:
    return valor * 10.0


# [F03] Grandezas derivadas para exibição: raio de giração i = h/raiz(12); profundidade da linha neutra
#   x = εtopo/(1/r) (infinita com curvatura nula); camadas de barras (barras de mesma profundidade reunidas:
#   profundidade, número de barras e área da camada = n A_barra); força de uma camada F = A σ; soma de parcelas;
#   resíduo de fechamento = soma - alvo; total na seção líquida N (ou M) = concreto bruto - concreto deslocado + barras;
#   comparação com limite máximo ou mínimo com a tolerância TOLERANCIA_COMPARACAO (cm); lista de todos os casos;
#   momento externo do pilar-padrão αb M1d,A + Nd (ℓe²/10)(1/r) (forma da [F28]
#   do núcleo, sem o piso M1d,A); γf3 Mdef.
#   Curvatura aproximada antes do limite 0,005/h: 0,005/[h (ν + 0,5)] e o termo ν + 0,5 ([F27] do núcleo).
#   Variáveis: altura, nu, deformacao_topo, curvatura, profundidades, fracoes, area_aco, area, tensao, parcelas, alvo,
#   alfa_b, momento_primeira_ordem, forca, comprimento, momento. Também: braço em relação ao centro h/2 - y e
#   momento da camada F (h/2 - y); total de barras sem proteção das quatro faces; posição anterior na lista de
#   candidatos e ordem a partir de 1; maior diâmetro longitudinal permitido b/8 ([C10] do núcleo); 12 ϕl ([C14]).
#   Fonte: NBR 6118:2023, 15.3.1, 15.8.3.3.2, 17.2.2 e 18.4.2.1; [F15], [F17], [F28] e [F30] do núcleo.
#   Usada em: estado_ultimo (l. 380, 385, 386, 387, 390, 391, 392, 393, 394, 395, 396, 397, 399);
#       dados_direcao (l. 413, 428, 457, 458); dados_detalhamento (l. 473, 501, 506); blocos_estado (l. 625);
#       blocos_direcao (l. 665); blocos_detalhamento (l. 773, 775, 782, 796);
#       conferir_memoriais (l. 1032, 1033, 1049, 1051, 1052); main (l. 1084)
def raio_giracao(altura: float) -> float:
    return altura / math.sqrt(12.0)


def curvatura_sem_limite(nu: float, altura: float) -> float:
    return 0.005 / (altura * (nu + 0.5))


def nu_mais_meio(nu: float) -> float:
    return nu + 0.5


def profundidade_linha_neutra(deformacao_topo: float, curvatura: float) -> float:
    if curvatura == 0.0:
        return math.inf
    return deformacao_topo / curvatura


def camadas_agrupadas(profundidades: np.ndarray, fracoes: np.ndarray, area_aco: float) -> list[tuple]:
    camadas: dict[float, list] = {}
    for profundidade, fracao in zip(np.round(profundidades, 9), fracoes):
        camada = camadas.setdefault(float(profundidade), [0, 0.0])
        camada[0] += 1
        camada[1] += fracao * area_aco
    return [(profundidade, quantidade, area) for profundidade, (quantidade, area) in sorted(camadas.items())]


def forca_camada(area: float, tensao: float) -> float:
    return area * tensao


def soma(parcelas) -> float:
    return float(sum(parcelas))


def residuo_fechamento(valor: float, alvo: float) -> float:
    return valor - alvo


def total_secao_liquida(parcela_concreto: float, parcela_deslocada: float, parcela_aco: float) -> float:
    return parcela_concreto - parcela_deslocada + parcela_aco


def atende_maximo(valor: float, limite: float) -> bool:
    return valor <= limite + TOLERANCIA_COMPARACAO


def atende_minimo(valor: float, limite: float) -> bool:
    return valor >= limite - TOLERANCIA_COMPARACAO


def momento_externo(alfa_b: float, momento_primeira_ordem: float, forca: float, comprimento: float,
                    curvatura: float) -> float:
    return (alfa_b * momento_primeira_ordem
            + forca * comprimento ** 2 / nucleo.DIVISOR_PILAR_PADRAO * curvatura)


def majorado_gama_f3(momento: float) -> float:
    return nucleo.GAMA_F3 * momento


def braco_centro(altura: float, profundidade: float) -> float:
    return altura / 2.0 - profundidade


def momento_camada(forca: float, braco: float) -> float:
    return forca * braco


def todos_os_casos(matriz: list, teste_minimo: list) -> list:
    return matriz + teste_minimo


def numero_total_barras_sem_protecao(quantidade_face_h: int, quantidade_face_b: int) -> int:
    return 2 * quantidade_face_h + 2 * quantidade_face_b


def anterior(posicao: int) -> int:
    return posicao - 1


def diametro_maximo_permitido(menor_dimensao: float) -> float:
    return nucleo.FRACAO_DIAMETRO_MAXIMO * menor_dimensao


def doze_diametros(diametro: float) -> float:
    return nucleo.FATOR_ESPACAMENTO_ESTRIBO_CA50 * diametro


def ordem_humana(posicao: int) -> int:
    return posicao + 1


# [F04] Conferência dos memoriais: valor absoluto de um resíduo; maior valor absoluto de uma lista (0 se vazia);
#   igualdade com tolerância relativa ([R05]).
#   Variáveis: valor, valores, valor_a, valor_b.
#   Usada em: conferir_memoriais (l. 1023, 1024, 1059, 1060)
def valor_absoluto(valor: float) -> float:
    return abs(valor)


def maior_absoluto(valores) -> float:
    return max((abs(valor) for valor in valores), default=0.0)


def iguais(valor_a: float, valor_b: float) -> bool:
    return math.isclose(valor_a, valor_b, rel_tol=TOLERANCIA_RELATIVA)

# [F05] Notação científica com n algarismos significativos: valor = m x 10^k, k = piso(log10 |valor|) e
#   m = valor/10^k com n - 1 casas (se o arredondamento leva |m| a 10, k + 1); zero, infinito e NaN como em [F01].
#   Variáveis: valor, algarismos.
#   Usada em: numero_cientifico (l. 347, 348, 351); curv (l. 591)
def expoente_decimal(valor: float) -> int:
    return math.floor(math.log10(abs(valor)))


def mantissa(valor: float, expoente: int) -> float:
    return valor / 10.0 ** expoente


def numero_cientifico(valor, algarismos: int) -> str:
    if valor is None or not math.isfinite(float(valor)) or float(valor) == 0.0:
        return numero(valor, 0)
    expoente = expoente_decimal(float(valor))
    texto = "%.*f" % (algarismos - 1, mantissa(float(valor), expoente))
    if abs(float(texto)) >= 10.0:
        expoente = expoente + 1
        texto = "%.*f" % (algarismos - 1, mantissa(float(valor), expoente))
    return "%s × 10%s" % (texto.replace(".", ","), str(expoente).translate(SOBRESCRITOS))


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


# ----------------------------------------------------------------------------- cálculo dos valores do memorial
def estado_ultimo(secao: nucleo.SecaoDirecao, material: nucleo.Material, area: float, forca: float,
                  curva_deformacao: bool = False, curvatura_fixa: float | None = None) -> dict | None:
    """Estado de deformação com N_R = forca: no ELU (curvatura da configuração última, [F18]) ou, com
    curvatura_fixa, na curva de deformação do pilar-padrão acoplado (εtopo que equilibra a força).
    Devolve a decomposição em concreto, concreto deslocado e camadas de barras."""
    if curvatura_fixa is None:
        parametro = nucleo.parametro_ultimo_para_forca(secao, material, area, forca, curva_deformacao)
        if math.isnan(parametro):
            return None
        deformacao_topo, curvatura = nucleo.configuracao_no_parametro(secao, material, parametro)
    else:
        curvatura = curvatura_fixa
        deformacao_topo = float(nucleo.deformacoes_topo_para_forca(secao, material, area, np.array([curvatura]),
                                                                   forca, curva_deformacao)[0])
        parametro = math.nan
    tensao_pico = material.tensao_pico(curva_deformacao)
    forca_concreto, momento_concreto = nucleo.esforcos_concreto_retangular(
        deformacao_topo, curvatura, secao.altura, secao.largura, tensao_pico, material.eps_c2, material.expoente)
    camadas = []
    for profundidade, quantidade, area_camada in camadas_agrupadas(secao.profundidades, secao.fracoes_area, area):
        deformacao = float(nucleo.deformacao_na_profundidade(deformacao_topo, curvatura, profundidade))
        tensao = float(nucleo.tensao_aco(deformacao, material.fyd))
        tensao_concreto = (float(nucleo.tensao_concreto(deformacao, tensao_pico, material.eps_c2, material.expoente))
                           if nucleo.DESCONTAR_AREA_BARRAS else 0.0)
        braco = braco_centro(secao.altura, profundidade)
        forca_aco = forca_camada(area_camada, tensao)
        forca_deslocada = forca_camada(area_camada, tensao_concreto)
        camadas.append({"y_cm": profundidade, "barras": quantidade, "A_cm2": area_camada, "eps": deformacao,
                        "sigma_s_kN_cm2": tensao, "F_s_kN": forca_aco, "sigma_c_kN_cm2": tensao_concreto,
                        "F_desl_kN": forca_deslocada, "braco_cm": braco, "M_s_kNcm": momento_camada(forca_aco, braco),
                        "M_desl_kNcm": momento_camada(forca_deslocada, braco)})
    forca_aco_total = soma(camada["F_s_kN"] for camada in camadas)
    forca_deslocada_total = soma(camada["F_desl_kN"] for camada in camadas)
    momento_aco_total = soma(camada["M_s_kNcm"] for camada in camadas)
    momento_deslocado_total = soma(camada["M_desl_kNcm"] for camada in camadas)
    forca_total = total_secao_liquida(float(forca_concreto), forca_deslocada_total, forca_aco_total)
    momento_total = total_secao_liquida(float(momento_concreto), momento_deslocado_total, momento_aco_total)
    return {"parametro_t": parametro, "eps_topo": deformacao_topo, "curvatura_1_cm": curvatura,
            "x_cm": profundidade_linha_neutra(deformacao_topo, curvatura), "tensao_pico_kN_cm2": tensao_pico,
            "N_concreto_bruta_kN": float(forca_concreto), "M_concreto_bruta_kNcm": float(momento_concreto),
            "N_deslocado_kN": forca_deslocada_total, "M_deslocado_kNcm": momento_deslocado_total,
            "N_aco_kN": forca_aco_total, "M_aco_kNcm": momento_aco_total, "N_total_kN": forca_total,
            "M_total_kNcm": momento_total, "camadas": camadas}


def dados_direcao(pilar: nucleo.PilarTeste, material: nucleo.Material, resultado: nucleo.ResultadoPilar,
                  direcao: str, candidato) -> dict:
    solicitacao = resultado.direcao_b if direcao == nucleo.DIRECAO_B else resultado.direcao_h
    dimensoes = ((pilar.largura_bw, pilar.altura_h) if direcao == nucleo.DIRECAO_B
                 else (pilar.altura_h, pilar.largura_bw))
    forca = resultado.forca_normal
    dados = {"direcao": direcao, "altura_cm": dimensoes[0], "largura_cm": dimensoes[1],
             "lambda": solicitacao.esbeltez, "raio_giracao_cm": raio_giracao(dimensoes[0]),
             "e1_cm": solicitacao.excentricidade_primeira_ordem, "M_aplicado_kNcm": solicitacao.momento_aplicado,
             "M1d_min_kNcm": solicitacao.momento_minimo,
             "momentos_menores_que_minimo": nucleo.momentos_menores_que_minimo(solicitacao.momento_aplicado,
                                                                               solicitacao.momento_minimo),
             "alfa_b": solicitacao.alfa_b,
             "e1_sobre_h": nucleo.excentricidade_relativa(solicitacao.momento_aplicado, forca, dimensoes[0]),
             "lambda1": solicitacao.esbeltez_limite, "metodo": solicitacao.metodo,
             "M1d_A_kNcm": nucleo.momento_primeira_ordem_adotado(solicitacao.momento_aplicado,
                                                                  solicitacao.momento_minimo),
             "M1d_A_final_kNcm": solicitacao.momento_primeira_ordem, "Md_tot_kNcm": solicitacao.momento_total,
             "curvatura_1_cm": solicitacao.curvatura, "As_nec_cm2": solicitacao.area_necessaria}
    if solicitacao.metodo == nucleo.METODO_CURVATURA:
        dados["nu"] = resultado.nu
        dados["curvatura_limite_1_cm"] = nucleo.curvatura_aproximada(0.0, dimensoes[0])
        dados["curvatura_sem_limite_1_cm"] = curvatura_sem_limite(resultado.nu, dimensoes[0])
    if solicitacao.metodo == nucleo.METODO_MN1R:
        forca_caracteristica, forca_permanente, forca_variavel = nucleo.parcelas_forca_caracteristica(
            forca, nucleo.FRACAO_PERMANENTE)
        dados.update({"theta1": nucleo.desaprumo_local(resultado.comprimento_equivalente),
                      "ea_cm": solicitacao.excentricidade_acidental, "Nk_kN": forca_caracteristica,
                      "Ngk_kN": forca_permanente, "Nqk_kN": forca_variavel, "Nsg_kN": solicitacao.forca_quase_permanente,
                      "Msg_kNcm": solicitacao.momento_quase_permanente,
                      "Ic_cm4": nucleo.momento_inercia_retangulo(dimensoes[1], dimensoes[0]),
                      "Eci_kN_cm2": material.modulo_inicial, "Ne_kN": solicitacao.forca_euler,
                      "phi": nucleo.COEFICIENTE_FLUENCIA, "ecc_cm": solicitacao.excentricidade_fluencia,
                      "N_deformacao_kN": nucleo.forca_normal_curva_deformacao(forca)})
    if candidato is None:
        return dados
    secao = nucleo.secao_na_direcao(pilar.largura_bw, pilar.altura_h, candidato.x, candidato.y, direcao)
    dados["MRd_kNcm"] = resultado.momentos_resistentes[direcao]
    dados["estado_ultimo"] = estado_ultimo(secao, material, candidato.area, forca)
    if solicitacao.metodo == nucleo.METODO_MN1R:
        equilibrio = nucleo.equilibrio_pilar_padrao_acoplado(secao, material, candidato.area, forca,
                                                             solicitacao.momento_primeira_ordem, solicitacao.alfa_b,
                                                             resultado.comprimento_equivalente)
        dados["equilibrio_adotada"] = None
        if equilibrio is not None:
            curvatura, momento_total = equilibrio
            momento_deformacao = nucleo.momento_curva_deformacao(secao, material, candidato.area,
                                                                 dados["N_deformacao_kN"], curvatura)
            estado = estado_ultimo(secao, material, candidato.area, dados["N_deformacao_kN"], True, curvatura)
            dados["equilibrio_adotada"] = {
                "curvatura_1_cm": curvatura, "Md_tot_kNcm": momento_total, "M_def_kNcm": momento_deformacao,
                "gama_f3_M_def_kNcm": majorado_gama_f3(momento_deformacao),
                "M_ext_kNcm": momento_externo(solicitacao.alfa_b, solicitacao.momento_primeira_ordem, forca,
                                              resultado.comprimento_equivalente, curvatura),
                "residuo_kNcm": float(nucleo.residuo_equilibrio_pilar_padrao(
                    momento_deformacao, solicitacao.alfa_b, solicitacao.momento_primeira_ordem, forca,
                    resultado.comprimento_equivalente, curvatura)),
                "estado": estado}
    return dados


def dados_detalhamento(pilar: nucleo.PilarTeste, material: nucleo.Material, resultado: nucleo.ResultadoPilar) -> dict:
    candidato = resultado.candidato
    menor_dimensao = min(pilar.largura_bw, pilar.altura_h)
    espacamento_maximo = nucleo.espacamento_maximo_eixos(menor_dimensao)
    candidatos = nucleo.gerar_candidatos(pilar, resultado.area_maxima)
    dados = {"arranjo": pilar.arranjo, "menor_dimensao_cm": menor_dimensao,
             "diametro_maximo_permitido_cm": diametro_maximo_permitido(menor_dimensao),
             "s_max_eixos_cm": espacamento_maximo, "candidatos": len(candidatos)}
    if candidato is None:
        resumo = resultado.resumo_arranjo
        if resumo is not None and resumo.candidato_maior_area is not None:
            maior = resumo.candidato_maior_area
            dados["maior_candidato"] = {"barras": maior.numero_barras, "diametro_cm": maior.diametro,
                                        "As_adot_cm2": maior.area, "As_calc_geometria_cm2":
                                        resumo.area_calculada_maior_area}
        return dados
    _, _, faces = nucleo.posicoes_barras(pilar.largura_bw, pilar.altura_h, candidato.distancia_eixo, pilar.arranjo,
                                         candidato.nivel, espacamento_maximo)
    espacamento_livre_minimo = nucleo.espacamento_livre_minimo(candidato.diametro, pilar.diametro_maximo_agregado)
    faces_dados = []
    for comprimento_face, barras_face in faces:
        espacamento = nucleo.espacamento_eixos_face(comprimento_face, barras_face)
        faces_dados.append({"comprimento_cm": comprimento_face, "barras": barras_face, "s_eixos_cm": espacamento,
                            "a_livre_emenda_cm": nucleo.espacamento_livre_emenda(espacamento, candidato.diametro),
                            "sem_protecao": nucleo.barras_sem_protecao_face(comprimento_face, barras_face,
                                                                            candidato.diametro_estribo)})
    posicao = next(indice for indice, item in enumerate(candidatos) if item.area == candidato.area
                   and item.numero_barras == candidato.numero_barras and item.diametro == candidato.diametro)
    dados.update({"diametro_cm": candidato.diametro, "estribo_cm": candidato.diametro_estribo,
                  "d_linha_cm": candidato.distancia_eixo, "barras": candidato.numero_barras,
                  "As_adot_cm2": candidato.area, "nivel": candidato.nivel, "faces": faces_dados,
                  "a_min_cm": espacamento_livre_minimo,
                  "st_max_cm": nucleo.espacamento_maximo_estribos(menor_dimensao, candidato.diametro),
                  "trecho_protegido_cm": nucleo.trecho_protegido_estribo(candidato.diametro_estribo),
                  "barras_sem_protecao": numero_total_barras_sem_protecao(faces_dados[0]["sem_protecao"],
                                                                          faces_dados[1]["sem_protecao"]),
                  "posicao_na_lista": posicao, "diametro_permitido": nucleo.diametro_longitudinal_permitido(
                      candidato.diametro, menor_dimensao)})
    if posicao > 0:
        previo = candidatos[anterior(posicao)]
        verificacoes = {}
        for direcao in nucleo.ORDEM_DIRECOES:
            solicitacao = resultado.direcao_b if direcao == nucleo.DIRECAO_B else resultado.direcao_h
            secao = nucleo.secao_na_direcao(pilar.largura_bw, pilar.altura_h, previo.x, previo.y, direcao)
            verificacoes[direcao] = nucleo.area_atende_direcao(secao, material, resultado.forca_normal,
                                                               resultado.comprimento_equivalente, solicitacao,
                                                               previo.area)
        dados["candidato_anterior"] = {"barras": previo.numero_barras, "diametro_cm": previo.diametro,
                                       "As_adot_cm2": previo.area, "atende_As_min": nucleo.detalhamento_atende(
                                           previo.area, resultado.area_minima), "atende_direcoes": verificacoes}
    return dados


def calcular_memorial(caso: processar_matriz.CasoMatriz) -> dict:
    resultado = nucleo.dimensionar_pilar(caso.pilar)
    pilar = resultado.pilar
    material = nucleo.Material(pilar.fck_mpa, pilar.fyk_mpa)
    forca_caracteristica, forca_permanente, forca_variavel = nucleo.parcelas_forca_caracteristica(
        resultado.forca_normal, nucleo.FRACAO_PERMANENTE)
    dados = {"ID": caso.identificador, "nivel_lambda": caso.esbeltez, "nivel_nu": caso.nu,
             "nivel_e1_bw": caso.excentricidade_relativa, "bw_cm": pilar.largura_bw, "h_cm": pilar.altura_h,
             "fck_MPa": pilar.fck_mpa, "fyk_MPa": pilar.fyk_mpa, "Nd_kN": resultado.forca_normal,
             "Nk_kN": forca_caracteristica, "Ngk_kN": forca_permanente, "Nqk_kN": forca_variavel,
             "e1_cm": pilar.excentricidade_primeira_ordem, "le_cm": resultado.comprimento_equivalente,
             "cobrimento_cm": pilar.cobrimento, "dmax_cm": pilar.diametro_maximo_agregado,
             "fcd_kN_cm2": material.fcd, "fyd_kN_cm2": material.fyd, "eta_c": material.eta_c,
             "sigma_cd_kN_cm2": material.tensao_pico_resistencia, "eps_c2": material.eps_c2, "eps_cu": material.eps_cu,
             "Ac_cm2": resultado.area_concreto, "nu": resultado.nu, "lambda_b": resultado.esbeltez_b,
             "lambda_h": resultado.esbeltez_h, "As_min_cm2": resultado.area_minima, "As_max_cm2": resultado.area_maxima,
             "As_calc_cm2": resultado.area_calculada, "As_calc_ref_cm2": resultado.area_referencia,
             "status": resultado.status, "arranjos_avaliados": list(resultado.arranjos_avaliados),
             "secao_liquida": bool(nucleo.DESCONTAR_AREA_BARRAS)}
    dados["direcoes"] = {direcao: dados_direcao(pilar, material, resultado, direcao, resultado.candidato)
                         for direcao in nucleo.ORDEM_DIRECOES}
    dados["detalhamento"] = dados_detalhamento(pilar, material, resultado)
    if resultado.candidato is not None:
        dados["taxa_adotada_percentual"] = taxa_percentual(resultado.candidato.area, resultado.area_concreto)
    if resultado.envoltoria is not None:
        envoltoria = resultado.envoltoria
        dados["envoltoria"] = {"semieixo_b_kNcm": envoltoria.direcao_b.momento_total,
                               "semieixo_h_kNcm": envoltoria.direcao_h.momento_total, "eta": envoltoria.fator,
                               "psi_graus": envoltoria.direcao_critica, "atende": envoltoria.atende}
        if resultado.candidato_elipse is not None and resultado.candidato_elipse is not resultado.candidato:
            dados["envoltoria"]["arranjo_elipse"] = {
                "barras": resultado.candidato_elipse.numero_barras,
                "diametro_cm": resultado.candidato_elipse.diametro, "As_cm2": resultado.candidato_elipse.area,
                "eta": resultado.envoltoria_elipse.fator}
    return dados


# ----------------------------------------------------------------------------- texto do memorial
def kn(valor) -> str:
    return numero_tipo(valor, "forca")


def kncm(valor) -> str:
    return numero_tipo(valor, "momento")


def cm(valor) -> str:
    return numero_tipo(valor, "comprimento")


def cm2(valor) -> str:
    return numero_tipo(valor, "area")


def cm_le(valor) -> str:
    return numero_tipo(valor, "comprimento_equivalente")


def raio(valor) -> str:
    return numero_tipo(valor, "raio_giracao")


def esbeltez(valor) -> str:
    return numero_tipo(valor, "esbeltez")


def adim(valor) -> str:
    return numero_tipo(valor, "adimensional")


def curv(valor) -> str:
    return numero_cientifico(valor, ALGARISMOS_CURVATURA)


def mm(valor_cm) -> str:
    return numero(cm_para_mm(valor_cm), 1).replace(",0", "") if valor_cm is not None else "–"


def ok(condicao: bool) -> str:
    return "atende" if condicao else "não atende"


def texto_metodo(metodo: str) -> str:
    return {nucleo.METODO_DISPENSADO: "2ª ordem dispensada (λ < λ1; 15.8.2)",
            nucleo.METODO_CURVATURA: "pilar-padrão com curvatura aproximada (15.8.3.3.2)",
            nucleo.METODO_MN1R: "pilar-padrão acoplado a diagramas M, N, 1/r, com fluência (15.8.3.3.4 e 15.8.4)"
            }.get(metodo, metodo)


def blocos_estado(estado: dict, rotulo: str, forca_alvo: float, momento_alvo: float | None,
                  rotulo_momento: str) -> list:
    if estado is None:
        return [("paragrafo", "%s: a força normal não é resistida pela seção (sem configuração de equilíbrio)." % rotulo)]
    blocos = [("paragrafo", "%s: εtopo = %s ‰; 1/r = %s cm⁻¹; x = εtopo/(1/r) = %s cm; tensão de pico do concreto "
               "%s kN/cm²." % (rotulo, numero_tipo(por_mil(estado["eps_topo"]), "deformacao"),
                               curv(estado["curvatura_1_cm"]), cm(estado["x_cm"]),
                               numero_tipo(estado["tensao_pico_kN_cm2"], "tensao")))]
    linhas = []
    for camada in estado["camadas"]:
        linhas.append([cm(camada["y_cm"]), str(camada["barras"]), cm2(camada["A_cm2"]),
                       numero_tipo(por_mil(camada["eps"]), "deformacao"), numero_tipo(camada["sigma_s_kN_cm2"], "tensao"),
                       kn(camada["F_s_kN"]), numero_tipo(camada["sigma_c_kN_cm2"], "tensao"), kn(camada["F_desl_kN"]),
                       cm(camada["braco_cm"]), kncm(camada["M_s_kNcm"])])
    blocos.append(("tabela", ["y (cm)", "barras", "A (cm²)", "ε (‰)", "σs (kN/cm²)", "Fs = A σs (kN)", "σc (kN/cm²)",
                              "F desl. (kN)", "h/2 − y (cm)", "Fs (h/2 − y) (kN·cm)"], linhas))
    fechamento_n = residuo_fechamento(estado["N_total_kN"], forca_alvo)
    texto = ("Concreto (seção bruta, integral do diagrama): Nc = %s kN; Mc = %s kN·cm. Concreto deslocado pelas barras "
             "comprimidas: %s kN; %s kN·cm. Barras: %s kN; %s kN·cm. Total: N = %s − %s + %s = %s kN (alvo %s kN; "
             "diferença %s kN); M = %s − %s + %s = %s kN·cm" % (
                 kn(estado["N_concreto_bruta_kN"]), kncm(estado["M_concreto_bruta_kNcm"]), kn(estado["N_deslocado_kN"]),
                 kncm(estado["M_deslocado_kNcm"]), kn(estado["N_aco_kN"]), kncm(estado["M_aco_kNcm"]),
                 kn(estado["N_concreto_bruta_kN"]), kn(estado["N_deslocado_kN"]), kn(estado["N_aco_kN"]),
                 kn(estado["N_total_kN"]), kn(forca_alvo), numero_tipo(fechamento_n, "residuo"),
                 kncm(estado["M_concreto_bruta_kNcm"]), kncm(estado["M_deslocado_kNcm"]), kncm(estado["M_aco_kNcm"]),
                 kncm(estado["M_total_kNcm"])))
    if momento_alvo is not None:
        texto += " = %s." % rotulo_momento
    blocos.append(("paragrafo", texto))
    return blocos


def blocos_direcao(dados: dict, direcao: dict, numero_secao: str) -> list:
    altura = direcao["altura_cm"]
    forca = dados["Nd_kN"]
    blocos = [("subtitulo", "Etapa %s: %s; altura da seção na direção considerada (h da norma) = %s cm; largura = %s cm"
               % (numero_secao, ROTULO_DIRECAO[direcao["direcao"]], cm(altura), cm(direcao["largura_cm"])))]
    blocos.append(("item", "i = h/√12 = %s/3,4641 = %s cm; λ = ℓe/i = %s/%s = %s" % (
        cm(altura), raio(direcao["raio_giracao_cm"]), cm_le(dados["le_cm"]), raio(direcao["raio_giracao_cm"]),
        esbeltez(direcao["lambda"]))))
    blocos.append(("item", "M1d,mín = Nd (1,5 + 0,03 h) = %s × (1,5 + 0,03 × %s) = %s kN·cm (11.3.3.4.3)" % (
        kn(forca), cm(altura), kncm(direcao["M1d_min_kNcm"]))))
    blocos.append(("item", "Momento aplicado Nd e1 = %s × %s = %s kN·cm; M1d,A = máx(Nd e1; M1d,mín) = %s kN·cm" % (
        kn(forca), cm(direcao["e1_cm"]), kncm(direcao["M_aplicado_kNcm"]), kncm(direcao["M1d_A_kNcm"]))))
    motivo = ("momentos de 1ª ordem menores que o mínimo, 15.8.2 d)" if direcao["momentos_menores_que_minimo"]
              else "momentos iguais no topo e na base, MB/MA = +1: αb = 0,60 + 0,40 × 1 = 1,0, 15.8.2 a)")
    blocos.append(("item", "αb = %s (%s)" % (numero(direcao["alfa_b"], 2), motivo)))
    blocos.append(("item", "λ1 = (25 + 12,5 e1/h)/αb = (25 + 12,5 × %s)/%s, com 35 ≤ λ1 ≤ 90 → λ1 = %s" % (
        adim(direcao["e1_sobre_h"]), numero(direcao["alfa_b"], 2), numero_tipo(direcao["lambda1"], "lambda1"))))
    blocos.append(("item", "Método: λ = %s → %s" % (esbeltez(direcao["lambda"]),
                                                   texto_metodo(direcao["metodo"]))))
    if direcao["metodo"] == nucleo.METODO_DISPENSADO:
        blocos.append(("item", "Md,tot = M1d,A = %s kN·cm" % kncm(direcao["Md_tot_kNcm"])))
    if direcao["metodo"] == nucleo.METODO_CURVATURA:
        limitada = direcao["curvatura_sem_limite_1_cm"] > direcao["curvatura_limite_1_cm"]
        blocos.append(("item", "1/r = 0,005/[h (ν + 0,5)] = 0,005/[%s × %s] = %s cm⁻¹ %s 0,005/h = %s cm⁻¹ → 1/r = %s cm⁻¹"
                               % (cm(altura), adim(nu_mais_meio(direcao["nu"])), curv(direcao["curvatura_sem_limite_1_cm"]),
                                  ">" if limitada else "≤", curv(direcao["curvatura_limite_1_cm"]),
                                  curv(direcao["curvatura_1_cm"]))))
        blocos.append(("item", "Md,tot = αb M1d,A + Nd (ℓe²/10)(1/r) = %s × %s + %s × %s²/10 × %s = %s kN·cm ≥ M1d,A "
                               "(Eq. 1)" % (numero(direcao["alfa_b"], 2), kncm(direcao["M1d_A_kNcm"]), kn(forca),
                                            cm_le(dados["le_cm"]), curv(direcao["curvatura_1_cm"]),
                                            kncm(direcao["Md_tot_kNcm"]))))
    if direcao["metodo"] == nucleo.METODO_MN1R:
        blocos.append(("item", "Fluência (15.8.4): θ1 = 1/(100 √ℓ) = 1/(100 √%s m), com 1/300 ≤ θ1 ≤ 1/200 → θ1 = %s; "
                               "ea = θ1 ℓ/2 = %s cm" % (numero_tipo(cm_para_m(dados["le_cm"]), "comprimento_equivalente_m"),
                                                        numero_tipo(direcao["theta1"], "desaprumo"),
                                                        numero_tipo(direcao["ea_cm"], "excentricidade_acidental"))))
        blocos.append(("item", "Nk = Nd/γf = %s kN; Ngk = 0,70 Nk = %s kN; Nqk = 0,30 Nk = %s kN; Nsg = Ngk + ψ2 Nqk "
                               "= %s + 0,3 × %s = %s kN; Msg = Nsg e1 = %s kN·cm" % (
                                   kn(direcao["Nk_kN"]), kn(direcao["Ngk_kN"]), kn(direcao["Nqk_kN"]),
                                   kn(direcao["Ngk_kN"]), kn(direcao["Nqk_kN"]), kn(direcao["Nsg_kN"]),
                                   kncm(direcao["Msg_kNcm"]))))
        blocos.append(("item", "Ic = b h³/12 = %s cm⁴; Eci = 5600 √fck = 5600 √%s = %s MPa = %s kN/cm²; Ne = 10 Eci Ic/ℓe²"
                               " = %s kN" % (numero_tipo(direcao["Ic_cm4"], "inercia"), numero(dados["fck_MPa"], 0),
                                             numero_tipo(kncm2_para_mpa(direcao["Eci_kN_cm2"]), "modulo_mpa"),
                                             numero_tipo(direcao["Eci_kN_cm2"], "modulo"), kn(direcao["Ne_kN"]))))
        blocos.append(("item", "ecc = (Msg/Nsg + ea)[e^(φ Nsg/(Ne − Nsg)) − 1] = (%s/%s + %s)[e^(%s × %s/(%s − %s)) − 1]"
                               " = %s cm" % (kncm(direcao["Msg_kNcm"]), kn(direcao["Nsg_kN"]),
                                             numero_tipo(direcao["ea_cm"], "excentricidade_acidental"),
                                             numero(direcao["phi"], 1), kn(direcao["Nsg_kN"]), kn(direcao["Ne_kN"]),
                                             kn(direcao["Nsg_kN"]),
                                             numero_tipo(direcao["ecc_cm"], "excentricidade_fluencia"))))
        blocos.append(("item", "M1d,A com fluência = M1d,A + Nd ecc = %s + %s × %s = %s kN·cm" % (
            kncm(direcao["M1d_A_kNcm"]), kn(forca), numero_tipo(direcao["ecc_cm"], "excentricidade_fluencia"),
            kncm(direcao["M1d_A_final_kNcm"]))))
        blocos.append(("item", "Pilar-padrão acoplado: força da curva de deformação Nd/γf3 = %s/1,1 = %s kN; curva com "
                               "1,10 fcd; equilíbrio na menor curvatura em que R = γf3 Mdef(1/r) − [αb M1d,A + "
                               "Nd (ℓe²/10)(1/r)] = 0 (15.3.1)" % (kn(forca), kn(direcao["N_deformacao_kN"]))))
        blocos.append(("item", "Com a área necessária desta direção na geometria adotada (As,nec = %s cm²): 1/r = %s cm⁻¹; "
                               "Md,tot = %s kN·cm (valor registrado nos resultados da matriz e na planilha de comparação "
                               "com o TQS; com As,adot, o equilíbrio é refeito na etapa 5)"
                               % (cm2(direcao["As_nec_cm2"]), curv(direcao["curvatura_1_cm"]),
                                  kncm(direcao["Md_tot_kNcm"]))))
    blocos.append(("item", "Armadura necessária nesta direção, na geometria das barras adotadas: As,nec = %s cm²%s" % (
        cm2(direcao["As_nec_cm2"]), " (menor As com MRd ≥ Md,tot e equilíbrio estável)"
        if direcao["metodo"] == nucleo.METODO_MN1R else " (menor As com MRd(Nd) ≥ Md,tot)")))
    return blocos


def blocos_verificacao(dados: dict, direcao: dict, numero_secao: str) -> list:
    forca = dados["Nd_kN"]
    blocos = [("subtitulo", "Etapa %s: verificação da armadura adotada na %s" % (numero_secao,
                                                                         ROTULO_DIRECAO[direcao["direcao"]]))]
    momento_total = direcao["Md_tot_kNcm"]
    if direcao["metodo"] == nucleo.METODO_MN1R:
        equilibrio = direcao.get("equilibrio_adotada")
        if equilibrio is None:
            blocos.append(("paragrafo", "Com As,adot não há equilíbrio estável no pilar-padrão acoplado."))
            return blocos
        momento_total = equilibrio["Md_tot_kNcm"]
        blocos.append(("item", "Equilíbrio com As,adot: 1/r = %s cm⁻¹; Mdef(1/r) = %s kN·cm; γf3 Mdef = 1,1 × %s = %s "
                               "kN·cm; Mext = αb M1d,A + Nd (ℓe²/10)(1/r) = %s × %s + %s × %s²/10 × %s = %s kN·cm; "
                               "R = %s kN·cm; Md,tot = %s kN·cm" % (
                                   curv(equilibrio["curvatura_1_cm"]), kncm(equilibrio["M_def_kNcm"]),
                                   kncm(equilibrio["M_def_kNcm"]), kncm(equilibrio["gama_f3_M_def_kNcm"]),
                                   numero(direcao["alfa_b"], 2), kncm(direcao["M1d_A_final_kNcm"]), kn(forca),
                                   cm_le(dados["le_cm"]), curv(equilibrio["curvatura_1_cm"]), kncm(equilibrio["M_ext_kNcm"]),
                                   numero_tipo(equilibrio["residuo_kNcm"], "residuo"),
                                   kncm(equilibrio["Md_tot_kNcm"]))))
        blocos += blocos_estado(equilibrio["estado"], "Estado de deformação no equilíbrio (curva com 1,10 fcd e "
                                                      "Nd/γf3)", direcao["N_deformacao_kN"],
                                equilibrio["M_def_kNcm"], "Mdef")
    blocos.append(("item", "MRd(Nd; As,adot) = %s kN·cm %s Md,tot = %s kN·cm → %s" % (
        kncm(direcao["MRd_kNcm"]), "≥" if direcao["MRd_kNcm"] >= momento_total else "<", kncm(momento_total),
        ok(direcao["MRd_kNcm"] >= momento_total))))
    blocos += blocos_estado(direcao["estado_ultimo"], "Estado-limite último com N = Nd (diagrama parábola-retângulo "
                                                      "com 0,85 ηc fcd)", forca, direcao["MRd_kNcm"], "MRd")
    return blocos


def blocos_detalhamento(dados: dict) -> list:
    detalhamento = dados["detalhamento"]
    blocos = [("titulo2", "Etapa 4: armaduras longitudinais e detalhamento")]
    blocos.append(("item", "As,mín = máx(0,15 Nd/fyd; 0,004 Ac) = máx(0,15 × %s/%s; 0,004 × %s) = %s cm² (17.3.5.3.1)"
                           % (kn(dados["Nd_kN"]), numero(dados["fyd_kN_cm2"], 3), cm(dados["Ac_cm2"]),
                              cm2(dados["As_min_cm2"]))))
    blocos.append(("item", "As,máx = 4 %% Ac = %s cm² (fora da região de emenda; com o traspasse de todas as barras na "
                           "mesma seção, a armadura na emenda chega a 8 %% Ac; 17.3.5.3.2)" % cm2(dados["As_max_cm2"])))
    direcoes = dados["direcoes"]
    blocos.append(("item", "As,calc = máx(As,b; As,h; As,mín) = máx(%s; %s; %s) = %s cm² (geometria das barras %s)" % (
        cm2(direcoes["b"]["As_nec_cm2"]), cm2(direcoes["h"]["As_nec_cm2"]), cm2(dados["As_min_cm2"]),
        cm2(dados["As_calc_cm2"]), "adotadas" if "barras" in detalhamento else "de nível 1 do arranjo")))
    blocos.append(("item", "Geometria de referência (duas camadas de ϕ 10 mm, d′ = 3,5 cm): As,calc,ref = %s cm²" %
                   cm2(dados["As_calc_ref_cm2"])))
    if "barras" not in detalhamento:
        return blocos
    menor = detalhamento["menor_dimensao_cm"]
    blocos.append(("item", "Arranjo: %s (regra [R07]: duas faces, perímetro só se duas faces for inviável; arranjos "
                           "avaliados: %s)" % (ROTULO_ARRANJO[detalhamento["arranjo"]],
                                               ", ".join(ROTULO_ARRANJO[arranjo] for arranjo in dados["arranjos_avaliados"]))))
    blocos.append(("item", "Bitola: ϕl = %s mm; 10 mm ≤ ϕl ≤ bw/8 = %s/8 = %s mm → %s (18.4.2.1)" % (
        mm(detalhamento["diametro_cm"]), mm(menor), numero(cm_para_mm(detalhamento["diametro_maximo_permitido_cm"]), 1),
        ok(detalhamento["diametro_permitido"]))))
    blocos.append(("item", "Estribo: ϕt ≥ máx(5 mm; ϕl/4) → ϕt = %s mm; d′ = c + ϕt + ϕl/2 = %s + %s + %s/2 = %s cm (7.4.7)"
                           % (mm(detalhamento["estribo_cm"]), cm(dados["cobrimento_cm"]), cm(detalhamento["estribo_cm"]),
                              cm(detalhamento["diametro_cm"]), cm(detalhamento["d_linha_cm"]))))
    rotulos_faces = ("faces paralelas a h", "faces paralelas a bw")
    for rotulo, face in zip(rotulos_faces, detalhamento["faces"]):
        blocos.append(("item", "%s: L entre eixos das barras de canto = %s cm; %d barras por face; s = L/(n − 1) = %s cm "
                               "≤ smáx = mín(2 bw; 40 cm) = %s cm → %s; na emenda (traspasse de todas as barras), "
                               "a = s − 2 ϕl = %s cm ≥ amín = máx(20 mm; ϕl; 1,2 dmáx) = %s cm → %s (18.4.2.2)" % (
                                   rotulo.capitalize(), cm(face["comprimento_cm"]), face["barras"], cm(face["s_eixos_cm"]),
                                   cm(detalhamento["s_max_eixos_cm"]),
                                   ok(atende_maximo(face["s_eixos_cm"], detalhamento["s_max_eixos_cm"])),
                                   cm(face["a_livre_emenda_cm"]), cm(detalhamento["a_min_cm"]),
                                   ok(atende_minimo(face["a_livre_emenda_cm"], detalhamento["a_min_cm"])))))
    blocos.append(("item", "As,adot = %d ϕ %s mm = %s cm² ≥ As,calc = %s cm² → %s; ρ = %s %% ≤ 4 %%" % (
        detalhamento["barras"], mm(detalhamento["diametro_cm"]), cm2(detalhamento["As_adot_cm2"]),
        cm2(dados["As_calc_cm2"]), ok(detalhamento["As_adot_cm2"] >= dados["As_calc_cm2"]),
        numero(dados["taxa_adotada_percentual"], 2))))
    blocos.append(("item", "Escolha: menor área adotada que atende; em empate, menor número de barras e menor diâmetro "
                           "([R04]); candidato nº %d de %d na ordem do critério." % (
                               ordem_humana(detalhamento["posicao_na_lista"]), detalhamento["candidatos"])))
    previo = detalhamento.get("candidato_anterior")
    if previo is not None:
        falhas = [ROTULO_DIRECAO[direcao] for direcao, atende in previo["atende_direcoes"].items() if not atende]
        motivo = ("As,adot < As,mín" if not previo["atende_As_min"] else
                  "não resiste na " + " e na ".join(falhas) if falhas else "–")
        blocos.append(("item", "Candidato anterior: %d ϕ %s mm = %s cm² → não atende (%s, na própria geometria)" % (
            previo["barras"], mm(previo["diametro_cm"]), cm2(previo["As_adot_cm2"]), motivo)))
    else:
        blocos.append(("item", "Não há candidato anterior: o adotado é o de menor área entre os arranjos que atendem às "
                               "regras de detalhamento."))
    blocos.append(("item", "Espaçamento dos estribos: st ≤ mín(20 cm; bw = %s cm; 12 ϕl = %s cm) = %s cm (18.4.3); "
                           "trecho protegido por estribo poligonal a partir de cada canto: 20 ϕt = %s cm; barras que "
                           "exigem estribo suplementar: %d (18.2.4)" % (
                               cm(menor), cm(doze_diametros(detalhamento["diametro_cm"])),
                               cm(detalhamento["st_max_cm"]), cm(detalhamento["trecho_protegido_cm"]),
                               detalhamento["barras_sem_protecao"])))
    return blocos


def blocos_pilar(dados: dict, inviaveis: dict) -> list:
    detalhamento = dados["detalhamento"]
    viavel = dados["status"] == nucleo.STATUS_OK
    titulo = "Pilar %s (%s × %s cm, fck = %s MPa" % (dados["ID"], cm(dados["bw_cm"]).replace(",00", ""),
                                                    cm(dados["h_cm"]).replace(",00", ""), numero(dados["fck_MPa"], 0))
    if dados["ID"].startswith("P"):
        titulo += ", λ = %s, ν = %s, e1/bw = %s)" % (numero(dados["nivel_lambda"], 0), numero(dados["nivel_nu"], 2),
                                                      numero(dados["nivel_e1_bw"], 2))
    else:
        titulo += ", λ = %s, ν = %s, e1 = 0; teste do momento mínimo)" % (numero(dados["nivel_lambda"], 0),
                                                                          numero(dados["nivel_nu"], 2))
    situacao = ("viável, com %d ϕ %s mm (As,adot = %s cm²; ρ = %s %%)" % (
        detalhamento["barras"], mm(detalhamento["diametro_cm"]), cm2(detalhamento["As_adot_cm2"]),
        numero(dados["taxa_adotada_percentual"], 2)) if viavel else dados["status"])
    blocos = [("titulo", titulo), ("paragrafo", "Situação: " + situacao)]
    blocos.append(("titulo2", "Etapa 1: entradas e materiais"))
    blocos.append(("item", "bw = %s cm; h = %s cm; Ac = bw h = %s cm²; c = %s cm; dmáx = %s mm; aço CA-50 (fyk = %s MPa)"
                           % (cm(dados["bw_cm"]), cm(dados["h_cm"]), cm(dados["Ac_cm2"]), cm(dados["cobrimento_cm"]),
                              mm(dados["dmax_cm"]), numero(dados["fyk_MPa"], 0))))
    blocos.append(("item", "fcd = fck/γc = %s/1,4 = %s kN/cm²; fyd = fyk/γs = %s/1,15 = %s kN/cm²; σcd = 0,85 ηc fcd = %s "
                           "kN/cm² (ηc = %s); εc2 = %s ‰; εcu = %s ‰; Es = 21 000 kN/cm²" % (
                               numero(mpa_para_kncm2(dados["fck_MPa"]), 1),
                               numero_tipo(dados["fcd_kN_cm2"], "resistencia_calculo"),
                               numero(mpa_para_kncm2(dados["fyk_MPa"]), 1), numero(dados["fyd_kN_cm2"], 3),
                               numero_tipo(dados["sigma_cd_kN_cm2"], "resistencia_calculo"), numero(dados["eta_c"], 2),
                               numero(por_mil(dados["eps_c2"]), 1), numero(por_mil(dados["eps_cu"]), 1))))
    blocos.append(("item", "Nd = ν Ac fcd = %s × %s × %s = %s kN; Nk = Nd/γf = %s kN (Ngk = %s kN; Nqk = %s kN)" % (
        numero(dados["nivel_nu"], 2), cm(dados["Ac_cm2"]), numero_tipo(dados["fcd_kN_cm2"], "resistencia_calculo"),
        kn(dados["Nd_kN"]),
        kn(dados["Nk_kN"]), kn(dados["Ngk_kN"]), kn(dados["Nqk_kN"]))))
    blocos.append(("item", "e1 = %s cm, igual no topo e na base (curvatura simples), na direção de bw; ℓe = λ bw/√12 = "
                           "%s cm (biapoiado, ℓe = ℓ); ν = Nd/(Ac fcd) = %s" % (
                               cm(dados["e1_cm"]), cm_le(dados["le_cm"]), adim(dados["nu"]))))
    blocos.append(("item", "Equilíbrio da seção com a seção %s ([H05])." % (
        "líquida (desconto do concreto ocupado pelas barras comprimidas)" if dados["secao_liquida"] else "bruta")))
    blocos.append(("titulo2", "Etapa 2: esbeltez, λ = %s (direção de bw) e λh = %s" % (esbeltez(dados["lambda_b"]),
                                                                                         esbeltez(dados["lambda_h"]))))
    blocos.append(("titulo2", "Etapa 3: esforços de cálculo em cada direção"))
    blocos += blocos_direcao(dados, dados["direcoes"]["b"], "3.1")
    blocos += blocos_direcao(dados, dados["direcoes"]["h"], "3.2")
    blocos += blocos_detalhamento(dados)
    if viavel:
        blocos.append(("titulo2", "Etapa 5: verificação com a armadura adotada"))
        blocos += blocos_verificacao(dados, dados["direcoes"]["b"], "5.1")
        blocos += blocos_verificacao(dados, dados["direcoes"]["h"], "5.2")
        envoltoria = dados.get("envoltoria")
        if envoltoria is not None:
            acoplado = any(dados["direcoes"][chave]["metodo"] == nucleo.METODO_MN1R for chave in ("b", "h"))
            texto = ("Envoltória mínima com 2ª ordem (15.3.2): semieixos calculados com e1 = 0, só com o momento mínimo "
                     "de 1ª ordem (αb = 1), pelo procedimento de 2ª ordem de cada direção%s: Md,tot,mín,b = %s kN·cm e "
                     "Md,tot,mín,h = %s kN·cm; η = %s (ψ = %s°) → %s" % (
                         "; no pilar-padrão acoplado, com As,adot e ecc com Msg = 0 (sem momento de 1ª ordem aplicado, "
                         "só ea)" if acoplado else "",
                         kncm(envoltoria["semieixo_b_kNcm"]), kncm(envoltoria["semieixo_h_kNcm"]),
                         numero(envoltoria["eta"], 4), numero(envoltoria["psi_graus"], 3),
                         "englobada" if envoltoria["atende"] else
                         "não englobada; a armadura adotada atende à 15.3.2 pelas duas análises isoladas à flexão "
                         "composta normal (último parágrafo da seção), critério de aceitação da rotina, e a elipse é "
                         "verificação complementar"))
            elipse = envoltoria.get("arranjo_elipse")
            if elipse is not None:
                texto += "; primeiro arranjo que atende também à elipse: %d ϕ %s mm (%s cm²; η = %s)" % (
                    elipse["barras"], mm(elipse["diametro_cm"]), cm2(elipse["As_cm2"]), numero(elipse["eta"], 4))
            blocos.append(("titulo2", "Etapa 6: envoltória mínima"))
            blocos.append(("paragrafo", texto))
    else:
        blocos.append(("titulo2", "Etapa 5: inviabilidade"))
        maior = detalhamento.get("maior_candidato")
        texto = "Critério: %s. As,calc = %s cm²; As,máx = %s cm²." % (dados["status"], cm2(dados["As_calc_cm2"]),
                                                                    cm2(dados["As_max_cm2"]))
        if maior is not None:
            texto += (" Candidato de maior área com As,adot ≤ As,máx: %d ϕ %s mm = %s cm², que exige %s cm² na própria "
                      "geometria." % (maior["barras"], mm(maior["diametro_cm"]), cm2(maior["As_adot_cm2"]),
                                      cm2(maior["As_calc_geometria_cm2"])))
        linha = inviaveis.get(dados["ID"])
        if linha is not None:
            texto += (" Sensibilidade (processar_matriz.py): com As,máx = 8 %%: %s; com o momento na direção de h: %s; "
                      "momento em h com 8 %%: %s." % (linha.get("Situação com As,máx = 8 %", "–"),
                                                     linha.get("Critério (momento em h)", "–"),
                                                     linha.get("Situação, momento em h, As,máx = 8 %", "–")))
        blocos.append(("paragrafo", texto))
    return blocos


INTRODUCAO = (
    "Este volume reúne o memorial de cálculo de cada pilar-teste processado pela rotina: os 324 pilares da matriz "
    "(P001 a P324, Seção 3.6 do TCC) e os 9 do teste do momento mínimo (M01 a M09). Cada memorial segue o roteiro "
    "da Seção 3.5 e mostra, para cada fórmula, os valores substituídos e o resultado, de modo que qualquer etapa "
    "possa ser refeita à mão, do dado de entrada à armadura adotada. Os números são os da própria rotina "
    "(pilares_nbr6118.py), gerados pelo programa memorial_pilares.py; os valores completos, sem arredondamento, "
    "estão no arquivo memorial_pilares.json.",
    "Unidades: forças em kN, comprimentos em cm, tensões em kN/cm² (1 kN/cm² = 10 MPa), momentos em kN·cm "
    "(100 kN·cm = 1 kN·m), curvaturas em cm⁻¹ e deformações em ‰. Convenções: compressão positiva; profundidades "
    "y medidas a partir da face mais comprimida; momentos em relação ao centro da seção, positivos quando comprimem "
    "a face superior; h é a altura da seção na direção considerada (bw na direção de bw).",
    "No estado-limite último, a força de cada camada de barras é Fs = A σs, com σs = Es ε limitada a fyd; o "
    "concreto é integrado em forma fechada no diagrama parábola-retângulo (Seção 3.5) e, na seção líquida, desconta-se "
    "a força que o concreto teria na posição de cada barra comprimida (F desl. = A σc). O fechamento de N e de M "
    "com os valores impressos é mostrado em cada estado; as diferenças, quando aparecem, são apenas de arredondamento "
    "da impressão.",
    "Ao gerar este volume, o programa confere cada memorial: fechamento de N e de M em cada estado-limite último, "
    "MRd ≥ Md,tot com a armadura adotada nas duas direções, resíduo do equilíbrio do pilar-padrão acoplado, regras "
    "de detalhamento e igualdade da situação e das barras com os resultados da matriz (processar_matriz.py). O "
    "resultado da conferência fica no arquivo conferencia_memorial.json.",
)


def adicionar_texto(paragrafo, texto: str, tamanho: int, negrito: bool = False) -> None:
    from docx.shared import Pt
    trecho = paragrafo.add_run(texto)
    trecho.font.name = FONTE
    trecho.font.size = Pt(tamanho)
    trecho.bold = negrito


def escrever_documento(memoriais: list[dict], inviaveis: dict) -> None:
    from docx import Document
    from docx.enum.text import WD_BREAK
    from docx.shared import Cm, Pt
    documento = Document()
    for secao in documento.sections:
        secao.left_margin = secao.right_margin = secao.top_margin = secao.bottom_margin = Cm(MARGEM_CM)
    estilo = documento.styles["Normal"]
    estilo.font.name = FONTE
    estilo.font.size = Pt(TAMANHO_CORPO)
    estilo.paragraph_format.space_after = Pt(2)
    adicionar_texto(documento.add_paragraph(), "MEMORIAL DE CÁLCULO DOS PILARES-TESTE", TAMANHO_TITULO_DOCUMENTO, True)
    adicionar_texto(documento.add_paragraph(), "Rotina de dimensionamento de pilares retangulares de concreto armado "
                                               "à flexão normal composta reta, com λ ≤ 140, segundo a ABNT NBR "
                                               "6118:2023, TCC de Pedro Costaneski Neto (UFMT/FAET)",
                    TAMANHO_CORPO)
    for texto in INTRODUCAO:
        adicionar_texto(documento.add_paragraph(), texto, TAMANHO_CORPO)
    adicionar_texto(documento.add_paragraph(), "Relação dos pilares", TAMANHO_TITULO, True)
    tabela = documento.add_table(rows=1, cols=6)
    tabela.style = "Table Grid"
    for celula, texto in zip(tabela.rows[0].cells, ("Pilar", "Seção (cm)", "fck", "λ / ν / e1/bw", "Situação",
                                                    "Barras (As,adot)")):
        adicionar_texto(celula.paragraphs[0], texto, TAMANHO_TABELA, True)
    for dados in memoriais:
        detalhamento = dados["detalhamento"]
        celulas = tabela.add_row().cells
        valores = (dados["ID"], "%s × %s" % (numero(dados["bw_cm"], 0), numero(dados["h_cm"], 0)),
                   numero(dados["fck_MPa"], 0),
                   "%s / %s / %s" % (numero(dados["nivel_lambda"], 0), numero(dados["nivel_nu"], 2),
                                     numero(dados["nivel_e1_bw"], 2)),
                   "viável" if dados["status"] == nucleo.STATUS_OK else processar_matriz.CODIGOS_SITUACAO.get(
                       dados["status"], dados["status"]),
                   "%d ϕ %s (%s cm²)" % (detalhamento["barras"], mm(detalhamento["diametro_cm"]),
                                         cm2(detalhamento["As_adot_cm2"])) if "barras" in detalhamento else "–")
        for celula, texto in zip(celulas, valores):
            adicionar_texto(celula.paragraphs[0], texto, TAMANHO_TABELA)
    adicionar_texto(documento.add_paragraph(), "Critérios de inviabilidade: A = As,calc > As,máx (4 % Ac); E = "
                                               "As,calc > 8 % Ac ou sem equilíbrio estável; D = nenhum arranjo com "
                                               "As,adot ≤ As,máx atende ao detalhamento.", TAMANHO_TABELA)
    for dados in memoriais:
        documento.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        for bloco in blocos_pilar(dados, inviaveis):
            tipo = bloco[0]
            if tipo == "titulo":
                adicionar_texto(documento.add_paragraph(), bloco[1], TAMANHO_TITULO, True)
            elif tipo == "titulo2":
                adicionar_texto(documento.add_paragraph(), bloco[1], TAMANHO_TITULO_SECAO, True)
            elif tipo == "subtitulo":
                adicionar_texto(documento.add_paragraph(), bloco[1], TAMANHO_CORPO, True)
            elif tipo == "item":
                paragrafo = documento.add_paragraph()
                paragrafo.paragraph_format.left_indent = Cm(0.4)
                adicionar_texto(paragrafo, "• " + bloco[1], TAMANHO_CORPO)
            elif tipo == "paragrafo":
                adicionar_texto(documento.add_paragraph(), bloco[1], TAMANHO_CORPO)
            elif tipo == "tabela":
                cabecalho, linhas = bloco[1], bloco[2]
                tabela = documento.add_table(rows=1, cols=len(cabecalho))
                tabela.style = "Table Grid"
                tabela.autofit = False
                for celula, texto in zip(tabela.rows[0].cells, cabecalho):
                    adicionar_texto(celula.paragraphs[0], texto, TAMANHO_TABELA, True)
                for linha in linhas:
                    for celula, texto in zip(tabela.add_row().cells, linha):
                        adicionar_texto(celula.paragraphs[0], texto, TAMANHO_TABELA)
                for coluna, largura in zip(tabela.columns, LARGURAS_CAMADAS_CM):
                    coluna.width = Cm(largura)
                    for celula in coluna.cells:
                        celula.width = Cm(largura)
    PASTA_MEMORIAL.mkdir(parents=True, exist_ok=True)
    documento.save(ARQUIVO_DOCX)


def ler_resultados() -> dict:
    """Linhas de resultados_matriz.csv e resultados_teste_minimo.csv, por identificador."""
    import csv
    linhas = {}
    for arquivo_csv in ARQUIVOS_RESULTADOS:
        with arquivo_csv.open(encoding="utf-8") as arquivo:
            for linha in csv.DictReader(arquivo, delimiter=";"):
                linhas[linha["ID"]] = linha
    return linhas


def conferir_memoriais(memoriais: list[dict]) -> dict:
    """Confere cada memorial: fechamento de N e de M nos estados-limites últimos, MRd >= Md,tot com a armadura adotada
    nas duas direções, resíduo do equilíbrio do pilar-padrão acoplado, regras de detalhamento e igualdade da situação
    e das barras com os resultados de processar_matriz.py."""
    resultados = ler_resultados()
    residuos_n, residuos_m, residuos_acoplado = [], [], []
    direcoes, direcoes_resistem, direcoes_acoplado, falhas = 0, 0, 0, []
    viaveis, iguais_resultados, verificacoes_todas = 0, 0, []
    for dados in memoriais:
        linha = resultados[dados["ID"]]
        detalhamento = dados["detalhamento"]
        mesma_situacao = linha["Situação"] == dados["status"]
        if dados["status"] != nucleo.STATUS_OK:
            if mesma_situacao:
                iguais_resultados += 1
            else:
                falhas.append("%s: situação %s no memorial e %s nos resultados" % (dados["ID"], dados["status"],
                                                                                   linha["Situação"]))
            continue
        viaveis += 1
        mesmas_barras = (mesma_situacao and int(linha["n barras"]) == detalhamento["barras"]
                         and iguais(float(linha["φl (mm)"]), cm_para_mm(detalhamento["diametro_cm"]))
                         and iguais(float(linha["As,adot (cm²)"]), detalhamento["As_adot_cm2"]))
        if mesmas_barras:
            iguais_resultados += 1
        else:
            falhas.append("%s: barras do memorial diferentes das dos resultados" % dados["ID"])
        for direcao in dados["direcoes"].values():
            direcoes += 1
            estado = direcao["estado_ultimo"]
            residuos_n.append(residuo_fechamento(estado["N_total_kN"], dados["Nd_kN"]))
            residuos_m.append(residuo_fechamento(estado["M_total_kNcm"], direcao["MRd_kNcm"]))
            momento_total = direcao["Md_tot_kNcm"]
            if direcao["metodo"] == nucleo.METODO_MN1R:
                direcoes_acoplado += 1
                equilibrio = direcao.get("equilibrio_adotada")
                if equilibrio is None:
                    falhas.append("%s, direção %s: sem equilíbrio com a armadura adotada" % (dados["ID"],
                                                                                          direcao["direcao"]))
                    continue
                residuos_acoplado.append(equilibrio["residuo_kNcm"])
                momento_total = equilibrio["Md_tot_kNcm"]
            if direcao["MRd_kNcm"] >= momento_total:
                direcoes_resistem += 1
            else:
                falhas.append("%s, direção %s: MRd < Md,tot" % (dados["ID"], direcao["direcao"]))
        verificacoes = [detalhamento["diametro_permitido"],
                        atende_minimo(detalhamento["As_adot_cm2"], dados["As_calc_cm2"])]
        for face in detalhamento["faces"]:
            verificacoes += [atende_maximo(face["s_eixos_cm"], detalhamento["s_max_eixos_cm"]),
                             atende_minimo(face["a_livre_emenda_cm"], detalhamento["a_min_cm"])]
        verificacoes_todas.extend(verificacoes)
        if not all(verificacoes):
            falhas.append("%s: regra de detalhamento não atendida" % dados["ID"])
    conferencia = {"memoriais": len(memoriais), "viaveis": viaveis, "iguais_aos_resultados": iguais_resultados,
                   "direcoes_viaveis": direcoes, "direcoes_MRd_maior_ou_igual": direcoes_resistem,
                   "direcoes_pilar_padrao_acoplado": direcoes_acoplado,
                   "maior_residuo_N_kN": maior_absoluto(residuos_n), "maior_residuo_M_kNcm": maior_absoluto(residuos_m),
                   "maior_residuo_acoplado_kNcm": maior_absoluto(residuos_acoplado),
                   "regras_detalhamento": len(verificacoes_todas), "regras_atendidas": sum(verificacoes_todas),
                   "falhas": falhas}
    ARQUIVO_CONFERENCIA.write_text(json.dumps(conferencia, ensure_ascii=False, indent=1), encoding="utf-8")
    return conferencia


def converter_pdf() -> bool:
    if shutil.which("soffice") is None:
        return False
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(PASTA_MEMORIAL), str(ARQUIVO_DOCX)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ARQUIVO_PDF.exists()


def ler_inviaveis() -> dict:
    import csv
    if not ARQUIVO_INVIAVEIS.exists():
        return {}
    with ARQUIVO_INVIAVEIS.open(encoding="utf-8") as arquivo:
        return {linha["ID"]: linha for linha in csv.DictReader(arquivo, delimiter=";")}


def main() -> None:
    casos = todos_os_casos(processar_matriz.montar_matriz(), processar_matriz.montar_teste_minimo())
    escolhidos = set(sys.argv[1:])
    if escolhidos:
        casos = [caso for caso in casos if caso.identificador in escolhidos]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        memoriais = list(executor.map(calcular_memorial, casos, chunksize=2))
    PASTA_MEMORIAL.mkdir(parents=True, exist_ok=True)
    ARQUIVO_JSON.write_text(json.dumps(memoriais, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    conferencia = conferir_memoriais(memoriais)
    print("conferência: %d memoriais, %d viáveis, %d iguais aos resultados, %d de %d direções com MRd >= Md,tot, "
          "%d falhas" % (conferencia["memoriais"], conferencia["viaveis"], conferencia["iguais_aos_resultados"],
                         conferencia["direcoes_MRd_maior_ou_igual"], conferencia["direcoes_viaveis"],
                         len(conferencia["falhas"])))
    escrever_documento(memoriais, ler_inviaveis())
    convertido = converter_pdf()
    print("memoriais: %d | docx: %s | pdf: %s" % (len(memoriais), ARQUIVO_DOCX, ARQUIVO_PDF if convertido else "não gerado"))


if __name__ == "__main__":
    main()
