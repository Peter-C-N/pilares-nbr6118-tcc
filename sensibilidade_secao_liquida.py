# -*- coding: utf-8 -*-
"""
sensibilidade_secao_liquida.py
==============================
Mede o efeito da hipótese [H05] do núcleo (pilares_nbr6118.py): resultante do concreto integrada na seção
líquida, com a área ocupada pelas barras comprimidas descontada (hipótese adotada), ou na seção bruta b h.
O programa dimensiona os 324 pilares-teste da matriz, com as mesmas entradas
(processar_matriz.montar_matriz), nas duas hipóteses e faz duas comparações, sempre da hipótese adotada
(referência) para a alternativa:
  1) efeito isolado da hipótese: para cada pilar viável na hipótese adotada, a armadura calculada é refeita
     na hipótese alternativa com a MESMA geometria de barras adotada (mesmas posições e d'), de modo que a
     diferença vem só do desconto da área das barras;
  2) efeito no resultado final: dimensionamento completo na hipótese alternativa (situação, parcela que
     governa, As,calc e barras adotadas), em que a escolha das barras pode mudar e, com ela, d';
  3) diagrama retangular simplificado da NBR 6118:2023, 17.2.2 e) (profundidade 0,8 x e tensão 0,85 eta_c fcd),
     nas seções líquida e bruta, comparado com o parábola-retângulo adotado, com as mesmas configurações últimas
     de deformação ([F18] do núcleo), o mesmo aço e a geometria de barras adotada: M_Rd da direção de b_w com
     N = Nd em todos os pilares viáveis e As,nec dessa direção nos pilares com lambda <= 90 (M_d,tot não depende
     da armadura). Na seção líquida, o retângulo desconta 0,85 eta_c fcd A_si nas barras dentro da profundidade
     0,8 x, como o parábola-retângulo desconta sigma_c(eps_si) A_si ([F61] do núcleo).

Uso:  python3 sensibilidade_secao_liquida.py [pasta_de_saida]
      (grava sensibilidade_secao_liquida.csv (partes 1 e 2), sensibilidade_secao_liquida_diagramas.csv (parte 3)
      e o resumo sensibilidade_secao_liquida.txt)
"""
from __future__ import annotations

import csv
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scipy.optimize import brentq

import pilares_nbr6118 as nucleo
import processar_matriz

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — INÍCIO
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [R01] Arquivos de saída e número de pilares listados no resumo como maiores variações.
#   Variáveis: ARQUIVO_COMPARACAO (partes 1 e 2), ARQUIVO_DIAGRAMAS (parte 3), ARQUIVO_RESUMO,
#   QUANTIDADE_MAIORES_VARIACOES.
#   Usada em: resumo_diagramas (l. 376); resumo (l. 448); main (l. 498, 506, 511)
ARQUIVO_COMPARACAO = "sensibilidade_secao_liquida.csv"
ARQUIVO_DIAGRAMAS = "sensibilidade_secao_liquida_diagramas.csv"
ARQUIVO_RESUMO = "sensibilidade_secao_liquida.txt"
QUANTIDADE_MAIORES_VARIACOES = 5

# [R02] Situação "dimensionado" (pilar viável), igual à usada pelo núcleo ([R05] do núcleo).
#   Variáveis: SITUACAO_VIAVEL.
#   Usada em: linha_comparacao (l. 405, 406); resumo (l. 442, 443); main (l. 491)
SITUACAO_VIAVEL = nucleo.STATUS_OK


# [R03] Armadura governada pelo cálculo (e não pela armadura mínima): As,calc > As,mín.
#   Variáveis: area_calculada, area_minima.
#   Usada em: resumo_diagramas (l. 336, 338); resumo (l. 445)
def governada_pelo_calculo(area_calculada: float, area_minima: float) -> bool:
    return area_calculada > area_minima


# [R04] Hipótese adotada (referência) e alternativa da comparação, com os nomes usados nas colunas.
#   Variáveis: HIPOTESE_ADOTADA, HIPOTESE_ALTERNATIVA, NOME_HIPOTESE.
#   Usada em: esforcos_modelo (l. 205); nível do módulo (l. 392, 393); main (l. 489, 490, 494, 503)
HIPOTESE_ADOTADA = nucleo.DESCONTAR_AREA_BARRAS
HIPOTESE_ALTERNATIVA = not nucleo.DESCONTAR_AREA_BARRAS
NOME_HIPOTESE = {True: "líquida", False: "bruta"}

# [R05] Modelos da resultante do concreto comparados na parte 3 (todos com as mesmas configurações últimas de
#   deformação [F18] do núcleo, o mesmo aço [F13]/[F17] do núcleo e a geometria de barras adotada):
#   PR-L parábola-retângulo, seção líquida (adotado na rotina: [F16] e [F61] do núcleo; referência);
#   PR-B parábola-retângulo, seção bruta ([F16] do núcleo, sem [F61]);
#   RET-L retângulo de [C01], seção líquida (desconto de [F03]);  RET-B retângulo de [C01], seção bruta.
#   Variáveis: MODELO_REFERENCIA, MODELOS_DIAGRAMA, MODELOS_PARABOLA_RETANGULO, MODELOS_SECAO_LIQUIDA,
#   DESCRICAO_MODELO.
#   Usada em: esforcos_modelo (l. 200, 201, 211); comparacao_diagramas (l. 270); linha_diagramas (l. 309, 314, 315);
#       texto_diagramas_pilar (l. 325);
#       resumo_diagramas (l. 330, 332, 333, 336, 337, 338, 344, 346, 353, 356, 360, 364, 367, 369, 372)
MODELO_REFERENCIA = "PR-L"
MODELOS_DIAGRAMA = ("PR-L", "PR-B", "RET-L", "RET-B")
MODELOS_PARABOLA_RETANGULO = ("PR-L", "PR-B")
MODELOS_SECAO_LIQUIDA = ("PR-L", "RET-L")
DESCRICAO_MODELO = {
    "PR-L": "parábola-retângulo, seção líquida (adotado)",
    "PR-B": "parábola-retângulo, seção bruta",
    "RET-L": "retângulo 0,8 x, seção líquida",
    "RET-B": "retângulo 0,8 x, seção bruta",
}

# [R06] Extremos do parâmetro t da configuração última ([F18] do núcleo): t = 0 é a reta a (tração uniforme,
#   eps = -10 ‰) e t = 3 é a reta b (compressão uniforme, eps = eps_c2). A força normal resistente N_R(t) cresce
#   de t = 0 a t = 3, de modo que N_R(0) <= N <= N_R(3) é a condição de existência de M_Rd para a força N, e
#   N_R(3) é a força normal resistente na compressão uniforme (os mesmos extremos do núcleo,
#   parametro_ultimo_para_forca e limites_busca_armadura).
#   Variáveis: PARAMETRO_RETA_A, PARAMETRO_RETA_B.
#   Usada em: momento_resistente_modelo (l. 227, 228, 232); area_necessaria_modelo (l. 242, 244, 248);
#       comparacao_diagramas (l. 271)
PARAMETRO_RETA_A = 0.0
PARAMETRO_RETA_B = 3.0

# [C01] Diagrama retangular simplificado (NBR 6118:2023, 17.2.2 e)): profundidade y = lambda x com lambda = 0,8
#   (fck <= 50 MPa) e tensão alfa_c eta_c fcd = 0,85 eta_c fcd (largura constante, não diminui da linha neutra para a
#   borda comprimida), a mesma tensão de pico do parábola-retângulo ([F09] do núcleo, Material.tensao_pico_resistencia).
#   Variáveis: FATOR_PROFUNDIDADE_RETANGULO.  Fonte: NBR 6118:2023, 17.2.2 e).
#   Fórmula: y = 0,8 · x;  σ_{c} = 0,85 · η_{c} · f_{cd}
#   Usada em: profundidade_retangulo (l. 134)
FATOR_PROFUNDIDADE_RETANGULO = 0.8


# [F01] Variação percentual com sinal, da hipótese ou modelo de referência para o alternativo:
#   Delta% = (valor_alternativa - valor_adotada) / valor_adotada x 100.
#   Variáveis: valor_adotada (referência), valor_alternativa.
#   Fórmula: Δ\% = \frac{X_{alt} − X_{ref}}{X_{ref}} · 100
#   Usada em: linha_diagramas (l. 317); linha_comparacao (l. 415, 420)
def variacao_percentual(valor_adotada: float, valor_alternativa: float) -> float:
    return (valor_alternativa - valor_adotada) / valor_adotada * 100.0


# [F02] Profundidade do retângulo de tensões: y = mín(0,8 x; h), com x = eps_topo / k a profundidade da linha
#   neutra; sem curvatura e com eps_topo > 0 (compressão uniforme), y = h; com eps_topo <= 0, y = 0.
#   Variáveis: deformacao_topo, curvatura (k = 1/r), altura (h na direção da flexão).  Fonte: NBR 6118:2023, 17.2.2 e).
#   Fórmula: y = mín(0,8 · \frac{ε_{topo}}{k}; h)  (y = h se k = 0 e ε_{topo} > 0; y = 0 se ε_{topo} ≤ 0)
#   Usada em: esforcos_modelo (l. 208)
def profundidade_retangulo(deformacao_topo: float, curvatura: float, altura: float) -> float:
    if deformacao_topo <= 0.0:
        return 0.0
    if curvatura <= 0.0:
        return altura
    return min(FATOR_PROFUNDIDADE_RETANGULO * deformacao_topo / curvatura, altura)


# [F03] Resultantes do concreto no retângulo, em relação ao centro da seção: N_c = sigma b y; M_c = N_c (h/2 - y/2);
#   na seção líquida, descontam-se N_desl = soma(sigma A_si) e M_desl = soma[sigma A_si (h/2 - y_i)] das barras com
#   y_i <= y (dentro do retângulo, o mesmo critério de [F27] de validacao.py).  Variáveis: tensao (0,85 eta_c fcd), largura (b), altura (h), profundidade (y),
#   profundidades_barras (y_i), areas (A_si), descontar (seção líquida).  Fonte: NBR 6118:2023, 17.2.2 e); [H05] do
#   núcleo.
#   Fórmula: N_{c} = σ_{c} · b · y;  M_{c} = N_{c} · (\frac{h}{2} − \frac{y}{2})
#   Fórmula: seção líquida: N_{c} − ∑_{y_{i} ≤ y} σ_{c} · A_{si};  M_{c} − ∑_{y_{i} ≤ y} σ_{c} · A_{si} · (\frac{h}{2} − y_{i})
#   Usada em: esforcos_modelo (l. 209)
def esforcos_concreto_retangulo(tensao: float, largura: float, altura: float, profundidade: float,
                                profundidades_barras, areas, descontar: bool) -> tuple[float, float]:
    forca = tensao * largura * profundidade
    momento = forca * (altura / 2.0 - profundidade / 2.0)
    if descontar:
        dentro = profundidades_barras <= profundidade
        forca = forca - float((tensao * areas[dentro]).sum())
        momento = momento - float((tensao * areas[dentro] * (altura / 2.0 - profundidades_barras[dentro])).sum())
    return forca, momento


# [F04] Diferença absoluta entre dois valores, usada na conferência do modelo de referência (PR-L) com os valores
#   gravados pela rotina (M_Rd e As,nec da direção de b_w): |a - b|.
#   Variáveis: valor_a, valor_b.
#   Fórmula: |a − b|
#   Usada em: resumo_diagramas (l. 330, 333, 369)
def diferenca_absoluta(valor_a: float, valor_b: float) -> float:
    return abs(valor_a - valor_b)

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


def definir_hipotese(descontar_area_barras: bool) -> None:
    """Inicializador de cada processo: fixa a hipótese [H05] antes de qualquer cálculo."""
    nucleo.DESCONTAR_AREA_BARRAS = descontar_area_barras


def executar(funcao, argumentos: list, descontar_area_barras: bool) -> list:
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=definir_hipotese,
                             initargs=(descontar_area_barras,)) as executor:
        return list(executor.map(funcao, argumentos, chunksize=4))


def area_calculada_na_geometria(dados: tuple) -> float:
    """As,calc ([R08] do núcleo) do pilar com a geometria de barras dada, na hipótese [H05] em vigor."""
    pilar, candidato = dados
    material = nucleo.Material(pilar.fck_mpa, pilar.fyk_mpa)
    area_concreto = nucleo.area_secao_retangular(pilar.largura_bw, pilar.altura_h)
    areas = {}
    for direcao, excentricidade in ((nucleo.DIRECAO_B, pilar.excentricidade_primeira_ordem),
                                    (nucleo.DIRECAO_H, 0.0)):
        secao = nucleo.secao_na_direcao(pilar.largura_bw, pilar.altura_h, candidato.x, candidato.y, direcao)
        areas[direcao] = nucleo.analisar_direcao(secao, material, pilar.forca_normal_calculo,
                                                 pilar.comprimento_equivalente, direcao,
                                                 excentricidade).area_necessaria
    return nucleo.area_calculada_pilar(areas[nucleo.DIRECAO_B], areas[nucleo.DIRECAO_H],
                                       nucleo.armadura_minima(pilar.forca_normal_calculo, material.fyd, area_concreto))


# ----------------------------------------------------------------------------- parte 3: diagrama retangular
def esforcos_modelo(modelo: str, secao: nucleo.SecaoDirecao, material: nucleo.Material, area_aco: float,
                    deformacao_topo: float, curvatura: float) -> tuple[float, float]:
    """(N, M) resistentes da seção no modelo [R05] para uma configuração de deformações (eps_topo, 1/r)."""
    if modelo in MODELOS_PARABOLA_RETANGULO:
        nucleo.DESCONTAR_AREA_BARRAS = modelo in MODELOS_SECAO_LIQUIDA
        try:
            forca, momento = nucleo.esforcos_resistentes(secao, material, area_aco, deformacao_topo, curvatura)
        finally:
            nucleo.DESCONTAR_AREA_BARRAS = HIPOTESE_ADOTADA
        return float(forca), float(momento)
    areas = nucleo.areas_barras_secao(area_aco, secao.fracoes_area)
    profundidade = profundidade_retangulo(deformacao_topo, curvatura, secao.altura)
    forca_c, momento_c = esforcos_concreto_retangulo(material.tensao_pico_resistencia, secao.largura, secao.altura,
                                                     profundidade, secao.profundidades, areas,
                                                     modelo in MODELOS_SECAO_LIQUIDA)
    forca_s, momento_s = nucleo.esforcos_armadura(deformacao_topo, curvatura, secao.altura, secao.profundidades,
                                                  areas, material.fyd)
    return float(nucleo.soma_concreto_aco(forca_c, forca_s)), float(nucleo.soma_concreto_aco(momento_c, momento_s))


def forca_modelo(modelo: str, secao: nucleo.SecaoDirecao, material: nucleo.Material, area_aco: float,
                 parametro: float) -> float:
    """N_R(t) do modelo na configuração última de parâmetro t ([F18] do núcleo)."""
    deformacao_topo, curvatura = nucleo.configuracao_no_parametro(secao, material, parametro)
    return esforcos_modelo(modelo, secao, material, area_aco, deformacao_topo, curvatura)[0]


def momento_resistente_modelo(modelo: str, secao: nucleo.SecaoDirecao, material: nucleo.Material, area_aco: float,
                              forca_normal: float) -> float:
    """M_Rd do modelo com N_R = N (busca de t como no núcleo, parametro_ultimo_para_forca); NaN se N não é resistida."""
    if not (forca_modelo(modelo, secao, material, area_aco, PARAMETRO_RETA_A) <= forca_normal
            <= forca_modelo(modelo, secao, material, area_aco, PARAMETRO_RETA_B)):
        return math.nan
    parametro = brentq(lambda valor: nucleo.residuo(forca_modelo(modelo, secao, material, area_aco, valor),
                                                    forca_normal),
                       PARAMETRO_RETA_A, PARAMETRO_RETA_B, xtol=nucleo.TOLERANCIA_PARAMETRO)
    deformacao_topo, curvatura = nucleo.configuracao_no_parametro(secao, material, parametro)
    return esforcos_modelo(modelo, secao, material, area_aco, deformacao_topo, curvatura)[1]


def area_necessaria_modelo(modelo: str, secao: nucleo.SecaoDirecao, material: nucleo.Material, forca_normal: float,
                           momento_solicitante: float) -> float:
    """Menor As com M_Rd(N; As) >= M_d,tot no modelo: mesmos limites e mesma busca do núcleo
    (limites_busca_armadura, armadura_flexo_compressao, [M02], [M04]); inf se nem o limite [F39] atende."""
    area_superior = nucleo.armadura_limite_busca(nucleo.area_secao_retangular(secao.largura, secao.altura))
    if forca_modelo(modelo, secao, material, 0.0, PARAMETRO_RETA_B) >= forca_normal:
        area_inferior = 0.0
    elif forca_modelo(modelo, secao, material, area_superior, PARAMETRO_RETA_B) < forca_normal:
        area_inferior = area_superior
    else:
        area_inferior = brentq(lambda area: nucleo.residuo(forca_modelo(modelo, secao, material, area,
                                                                        PARAMETRO_RETA_B), forca_normal),
                               0.0, area_superior, xtol=nucleo.tolerancia_area_reta_b())

    def folga(area: float) -> float:
        return nucleo.folga_momento(momento_resistente_modelo(modelo, secao, material, area, forca_normal),
                                    momento_solicitante)

    if folga(area_superior) < 0.0:
        return math.inf
    if folga(area_inferior) >= 0.0:
        return area_inferior
    return nucleo.menor_valor_que_atende(folga, area_inferior, area_superior, nucleo.TOLERANCIA_AREA)


def comparacao_diagramas(resultado: nucleo.ResultadoPilar) -> dict:
    """Parte 3, por pilar viável (geometria adotada, direção de b_w): N_R na reta b, M_Rd com N = Nd e, com
    lambda <= 90 (M_d,tot independente da armadura), As,nec para M_d,tot, nos quatro modelos de [R05]."""
    pilar, candidato = resultado.pilar, resultado.candidato
    material = nucleo.Material(pilar.fck_mpa, pilar.fyk_mpa)
    secao = nucleo.secao_na_direcao(pilar.largura_bw, pilar.altura_h, candidato.x, candidato.y, nucleo.DIRECAO_B)
    calcula_area = resultado.direcao_b.metodo != nucleo.METODO_MN1R
    valores = {}
    for modelo in MODELOS_DIAGRAMA:
        valores[("N", modelo)] = forca_modelo(modelo, secao, material, candidato.area, PARAMETRO_RETA_B)
        valores[("MRd", modelo)] = momento_resistente_modelo(modelo, secao, material, candidato.area,
                                                             pilar.forca_normal_calculo)
        valores[("As", modelo)] = (area_necessaria_modelo(modelo, secao, material, pilar.forca_normal_calculo,
                                                          resultado.direcao_b.momento_total)
                                   if calcula_area else math.nan)
    return valores


GRANDEZAS_DIAGRAMA = ("N", "MRd", "As")
NOME_GRANDEZA = {"N": "N_R reta b", "MRd": "M_Rd b_w", "As": "As,nec b_w"}
UNIDADE_GRANDEZA = {"N": "kN", "MRd": "kN·m", "As": "cm²"}


def coluna_modelo(grandeza: str, modelo: str) -> str:
    return "%s %s (%s)" % (NOME_GRANDEZA[grandeza], modelo, UNIDADE_GRANDEZA[grandeza])


def coluna_variacao(grandeza: str, modelo: str) -> str:
    return "Δ%% %s %s" % (NOME_GRANDEZA[grandeza], modelo)


def valor_em_unidade_de_saida(grandeza: str, valor: float) -> float:
    """M_Rd de kN·cm (núcleo) para kN·m (saída); N em kN e As em cm² sem conversão."""
    return nucleo.kn_cm_para_kn_m(valor) if grandeza == "MRd" else valor


def linha_diagramas(caso, resultado: nucleo.ResultadoPilar, valores: dict) -> dict:
    linha = {
        "ID": caso.identificador, "b_w (cm)": caso.largura_bw, "h (cm)": caso.altura_h, "fck (MPa)": caso.fck_mpa,
        "λ": caso.esbeltez, "ν": caso.nu, "e1/b_w": caso.excentricidade_relativa,
        "Método b_w": resultado.direcao_b.metodo, "Barras": descricao_barras(resultado),
        "As,adot (cm²)": resultado.candidato.area, "As,mín (cm²)": resultado.area_minima,
        "Governa": governa(resultado),
        "M_Rd b_w da rotina (kN·m)": nucleo.kn_cm_para_kn_m(resultado.momentos_resistentes[nucleo.DIRECAO_B]),
        "As,nec b_w da rotina (cm²)": resultado.direcao_b.area_necessaria,
    }
    for grandeza in GRANDEZAS_DIAGRAMA:
        for modelo in MODELOS_DIAGRAMA:
            valor = valores[(grandeza, modelo)]
            linha[coluna_modelo(grandeza, modelo)] = "" if math.isnan(valor) else valor_em_unidade_de_saida(grandeza,
                                                                                                           valor)
    for grandeza in GRANDEZAS_DIAGRAMA:
        referencia = valores[(grandeza, MODELO_REFERENCIA)]
        for modelo in MODELOS_DIAGRAMA[1:]:
            comparavel = math.isfinite(referencia) and referencia > 0.0 and math.isfinite(valores[(grandeza, modelo)])
            linha[coluna_variacao(grandeza, modelo)] = (variacao_percentual(referencia, valores[(grandeza, modelo)])
                                                        if comparavel else "")
    return linha


def texto_diagramas_pilar(linha: dict, modelo: str) -> str:
    return "     %s (λ %.0f, ν %.2f, e1/b_w %.2f, %.0f x %.0f cm, fck %.0f, %s): As,nec %.3f -> %.3f cm² (%+.2f %%)" % (
        linha["ID"], linha["λ"], linha["ν"], linha["e1/b_w"], linha["b_w (cm)"], linha["h (cm)"], linha["fck (MPa)"],
        linha["Barras"], linha[coluna_modelo("As", MODELO_REFERENCIA)], linha[coluna_modelo("As", modelo)],
        linha[coluna_variacao("As", modelo)])


def resumo_diagramas(linhas: list[dict]) -> list[str]:
    diferenca_momento = max(diferenca_absoluta(linha[coluna_modelo("MRd", MODELO_REFERENCIA)],
                                               linha["M_Rd b_w da rotina (kN·m)"]) for linha in linhas)
    com_area = [linha for linha in linhas if linha[coluna_modelo("As", MODELO_REFERENCIA)] != ""]
    diferenca_area = max(diferenca_absoluta(linha[coluna_modelo("As", MODELO_REFERENCIA)],
                                            linha["As,nec b_w da rotina (cm²)"]) for linha in com_area)
    pelo_calculo = [linha for linha in com_area
                    if governada_pelo_calculo(linha[coluna_modelo("As", MODELO_REFERENCIA)], linha["As,mín (cm²)"])]
    pela_minima = [linha for linha in com_area if linha[coluna_variacao("As", MODELOS_DIAGRAMA[1])] != ""
                   and not governada_pelo_calculo(linha[coluna_modelo("As", MODELO_REFERENCIA)],
                                                  linha["As,mín (cm²)"])]
    texto = [
        "",
        "3) Diagrama retangular simplificado (NBR 6118:2023, 17.2.2 e)) x parábola-retângulo, seções líquida e bruta,",
        "   geometria de barras adotada, direção de b_w; referência: %s = %s." % (
            MODELO_REFERENCIA, DESCRICAO_MODELO[MODELO_REFERENCIA]),
    ]
    texto += ["   %-6s %s" % (modelo, DESCRICAO_MODELO[modelo]) for modelo in MODELOS_DIAGRAMA]
    texto += [
        "   Conferência do modelo de referência com a rotina: maior |Δ M_Rd| = %.2e kN·m (%d pilares); "
        "maior |Δ As,nec| = %.2e cm² (%d pilares)" % (diferenca_momento, len(linhas), diferenca_area, len(com_area)),
        "   a) Força normal resistente na compressão uniforme (reta b), %d pilares viáveis:" % len(linhas),
    ]
    texto += [estatisticas("      Δ%% %s" % modelo, [linha[coluna_variacao("N", modelo)] for linha in linhas])
              for modelo in MODELOS_DIAGRAMA[1:]]
    texto.append("   b) M_Rd com N = Nd, %d pilares viáveis:" % len(linhas))
    texto += [estatisticas("      Δ%% %s" % modelo, [linha[coluna_variacao("MRd", modelo)] for linha in linhas])
              for modelo in MODELOS_DIAGRAMA[1:]]
    texto.append("   c) As,nec para M_d,tot, lambda <= 90, governada pelo cálculo (As,nec > As,mín): %d pilares"
                 % len(pelo_calculo))
    texto += [estatisticas("      Δ%% %s" % modelo, [linha[coluna_variacao("As", modelo)] for linha in pelo_calculo])
              for modelo in MODELOS_DIAGRAMA[1:]]
    texto.append("   d) As,nec para M_d,tot, lambda <= 90, 0 < As,nec <= As,mín (As,calc = As,mín ou governado pela"
                 " direção de h): %d pilares" % len(pela_minima))
    texto += [estatisticas("      Δ%% %s" % modelo, [linha[coluna_variacao("As", modelo)] for linha in pela_minima])
              for modelo in MODELOS_DIAGRAMA[1:]]
    texto += [estatisticas("      Δ%% M_Rd com a armadura adotada, %s" % modelo,
                           [linha[coluna_variacao("MRd", modelo)] for linha in pela_minima])
              for modelo in MODELOS_DIAGRAMA[1:]]
    texto += ["      %s: maior redução absoluta de As,nec = %.2f cm²; As,nec (%s) <= As,mín em todos: %s" % (
        modelo, max(diferenca_absoluta(linha[coluna_modelo("As", MODELO_REFERENCIA)], linha[coluna_modelo("As", modelo)])
                    for linha in pela_minima), modelo,
        all(linha[coluna_modelo("As", modelo)] <= linha["As,mín (cm²)"] for linha in pela_minima))
        for modelo in MODELOS_DIAGRAMA[1:]]
    texto.append("      Maiores variações (RET-L):")
    texto += [texto_diagramas_pilar(linha, "RET-L")
              for linha in sorted(pela_minima, key=lambda linha: abs(linha[coluna_variacao("As", "RET-L")]),
                                  reverse=True)[:QUANTIDADE_MAIORES_VARIACOES]]
    return texto


def descricao_barras(resultado: nucleo.ResultadoPilar) -> str:
    candidato = resultado.candidato
    if candidato is None:
        return ""
    return "%d φ %.1f" % (candidato.numero_barras, processar_matriz.cm_para_mm(candidato.diametro))


def governa(resultado: nucleo.ResultadoPilar) -> str:
    return processar_matriz.parcela_governante(resultado.direcao_b.area_necessaria,
                                               resultado.direcao_h.area_necessaria, resultado.area_minima)


ROTULO_ADOTADA = NOME_HIPOTESE[HIPOTESE_ADOTADA]
ROTULO_ALTERNATIVA = NOME_HIPOTESE[HIPOTESE_ALTERNATIVA]
COLUNA_SITUACAO_ADOTADA = "Situação (%s, adotada)" % ROTULO_ADOTADA
COLUNA_SITUACAO_ALTERNATIVA = "Situação (%s)" % ROTULO_ALTERNATIVA
COLUNA_AREA_ADOTADA = "As,calc %s, adotada (cm²)" % ROTULO_ADOTADA
COLUNA_AREA_MESMA = "As,calc %s, mesma geometria (cm²)" % ROTULO_ALTERNATIVA
COLUNA_DELTA_MESMA = "Δ% mesma geometria"
COLUNA_AREA_COMPLETO = "As,calc %s, dimensionamento completo (cm²)" % ROTULO_ALTERNATIVA
COLUNA_DELTA_COMPLETO = "Δ% dimensionamento completo"


def linha_comparacao(caso, adotada: nucleo.ResultadoPilar, alternativa: nucleo.ResultadoPilar,
                     area_mesma_geometria) -> dict:
    viavel_adotada = adotada.status == SITUACAO_VIAVEL
    ambos_viaveis = viavel_adotada and alternativa.status == SITUACAO_VIAVEL
    return {
        "ID": caso.identificador, "b_w (cm)": caso.largura_bw, "h (cm)": caso.altura_h, "fck (MPa)": caso.fck_mpa,
        "λ": caso.esbeltez, "ν": caso.nu, "e1/b_w": caso.excentricidade_relativa,
        COLUNA_SITUACAO_ADOTADA: adotada.status, COLUNA_SITUACAO_ALTERNATIVA: alternativa.status,
        "As,mín (cm²)": adotada.area_minima,
        COLUNA_AREA_ADOTADA: adotada.area_calculada if viavel_adotada else "",
        "Barras (%s)" % ROTULO_ADOTADA: descricao_barras(adotada),
        COLUNA_AREA_MESMA: area_mesma_geometria if viavel_adotada else "",
        COLUNA_DELTA_MESMA: (variacao_percentual(adotada.area_calculada, area_mesma_geometria)
                             if viavel_adotada else ""),
        "Governa (%s)" % ROTULO_ADOTADA: governa(adotada) if ambos_viaveis else "",
        "Governa (%s)" % ROTULO_ALTERNATIVA: governa(alternativa) if ambos_viaveis else "",
        COLUNA_AREA_COMPLETO: alternativa.area_calculada if ambos_viaveis else "",
        COLUNA_DELTA_COMPLETO: (variacao_percentual(adotada.area_calculada, alternativa.area_calculada)
                                if ambos_viaveis else ""),
        "Barras (%s)" % ROTULO_ALTERNATIVA: descricao_barras(alternativa),
        "Mesmo detalhamento": (descricao_barras(adotada) == descricao_barras(alternativa)) if ambos_viaveis else "",
    }


def estatisticas(rotulo: str, valores: list[float]) -> str:
    if not valores:
        return "%s: sem casos" % rotulo
    return "%s: mínimo %+.2f %% | média %+.2f %% | máximo %+.2f %%" % (rotulo, min(valores),
                                                                       statistics.fmean(valores), max(valores))


def texto_pilar(linha: dict, chave_area: str, chave_variacao: str) -> str:
    return "  %s (λ %.0f, ν %.2f, e1/b_w %.2f, %.0f x %.0f cm, fck %.0f): %.3f -> %.3f cm² (%+.2f %%); %s -> %s" % (
        linha["ID"], linha["λ"], linha["ν"], linha["e1/b_w"], linha["b_w (cm)"], linha["h (cm)"],
        linha["fck (MPa)"], linha[COLUNA_AREA_ADOTADA], linha[chave_area], linha[chave_variacao],
        linha["Barras (%s)" % ROTULO_ADOTADA], linha["Barras (%s)" % ROTULO_ALTERNATIVA] or "inviável")


def resumo(linhas: list[dict]) -> list[str]:
    viaveis_adotada = [linha for linha in linhas if linha[COLUNA_SITUACAO_ADOTADA] == SITUACAO_VIAVEL]
    viaveis_alternativa = [linha for linha in linhas if linha[COLUNA_SITUACAO_ALTERNATIVA] == SITUACAO_VIAVEL]
    pelo_calculo = [linha for linha in viaveis_adotada
                    if governada_pelo_calculo(linha[COLUNA_AREA_ADOTADA], linha["As,mín (cm²)"])]
    mesma_geometria = [linha[COLUNA_DELTA_MESMA] for linha in pelo_calculo]
    maiores_mesma = sorted(pelo_calculo, key=lambda linha: abs(linha[COLUNA_DELTA_MESMA]),
                           reverse=True)[:QUANTIDADE_MAIORES_VARIACOES]
    mudancas_situacao = [linha for linha in linhas
                         if linha[COLUNA_SITUACAO_ADOTADA] != linha[COLUNA_SITUACAO_ALTERNATIVA]]
    comparaveis = [linha for linha in pelo_calculo if linha[COLUNA_DELTA_COMPLETO] != ""]
    completo = [linha[COLUNA_DELTA_COMPLETO] for linha in comparaveis]
    ambos = [linha for linha in linhas if linha["Governa (%s)" % ROTULO_ADOTADA] != ""]
    governa_mudou = [linha for linha in ambos
                     if linha["Governa (%s)" % ROTULO_ADOTADA] != linha["Governa (%s)" % ROTULO_ALTERNATIVA]]
    detalhamento_mudou = [linha for linha in ambos if not linha["Mesmo detalhamento"]]
    texto = [
        "Sensibilidade à hipótese [H05]: seção %s (adotada) x seção %s" % (ROTULO_ADOTADA, ROTULO_ALTERNATIVA),
        "Pilares avaliados: %d | viáveis com seção %s: %d | viáveis com seção %s: %d" % (
            len(linhas), ROTULO_ADOTADA, len(viaveis_adotada), ROTULO_ALTERNATIVA, len(viaveis_alternativa)),
        "",
        "1) Efeito isolado (mesma geometria de barras), %d pilares viáveis com As,calc > As,mín:" % len(pelo_calculo),
        estatisticas("   Δ%% As,calc (seção %s em relação à %s)" % (ROTULO_ALTERNATIVA, ROTULO_ADOTADA),
                     mesma_geometria),
        "   Maiores variações:",
    ]
    texto += [texto_pilar(linha, COLUNA_AREA_MESMA, COLUNA_DELTA_MESMA) for linha in maiores_mesma]
    texto += [
        "",
        "2) Dimensionamento completo com seção %s (barras podem mudar e, com elas, d'):" % ROTULO_ALTERNATIVA,
        "   Pilares que mudam de situação: %d" % len(mudancas_situacao),
    ]
    texto += ["     %s: %s -> %s" % (linha["ID"], linha[COLUNA_SITUACAO_ADOTADA], linha[COLUNA_SITUACAO_ALTERNATIVA])
              for linha in mudancas_situacao]
    texto += [
        estatisticas("   Δ%% As,calc (%d pilares viáveis nas duas e com As,calc > As,mín)" % len(comparaveis),
                     completo),
        "   Mudança da parcela que governa: %d | mudança das barras adotadas: %d (de %d viáveis nas duas)" % (
            len(governa_mudou), len(detalhamento_mudou), len(ambos)),
    ]
    return texto


def main() -> None:
    pasta_saida = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "saida"
    pasta_saida.mkdir(parents=True, exist_ok=True)
    casos = processar_matriz.montar_matriz()
    pilares = [caso.pilar for caso in casos]
    adotada = executar(nucleo.dimensionar_pilar, pilares, HIPOTESE_ADOTADA)
    alternativa = executar(nucleo.dimensionar_pilar, pilares, HIPOTESE_ALTERNATIVA)
    viaveis = [indice for indice, resultado in enumerate(adotada) if resultado.status == SITUACAO_VIAVEL]
    areas_viaveis = executar(area_calculada_na_geometria,
                             [(adotada[indice].pilar, adotada[indice].candidato) for indice in viaveis],
                             HIPOTESE_ALTERNATIVA)
    area_mesma_geometria = dict(zip(viaveis, areas_viaveis))
    linhas = [linha_comparacao(caso, adotada[indice], alternativa[indice], area_mesma_geometria.get(indice))
              for indice, caso in enumerate(casos)]
    with pasta_saida.joinpath(ARQUIVO_COMPARACAO).open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0].keys()), delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas)
    resultados_viaveis = [adotada[indice] for indice in viaveis]
    valores_diagramas = executar(comparacao_diagramas, resultados_viaveis, HIPOTESE_ADOTADA)
    linhas_diagramas = [linha_diagramas(casos[indice], adotada[indice], valores)
                        for indice, valores in zip(viaveis, valores_diagramas)]
    with pasta_saida.joinpath(ARQUIVO_DIAGRAMAS).open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas_diagramas[0].keys()), delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas_diagramas)
    texto = [*resumo(linhas), *resumo_diagramas(linhas_diagramas)]
    pasta_saida.joinpath(ARQUIVO_RESUMO).write_text("\n".join(texto) + "\n", encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
