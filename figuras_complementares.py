# -*- coding: utf-8 -*-
"""
figuras_complementares.py
=========================
Figuras do TCC que mostram como a rotina calcula e como ela é conferida (revisão de escopo de 23/10/2026),
em vetor (SVG) e em imagem de reserva (PNG, 300 dpi), no mesmo estilo de figuras_tcc.py:
  1) secao_v1         : estado final do equilíbrio da seção do pilar V1 do memorial (P156, direção de b_w):
                        seção com as barras e a linha neutra, deformações e tensões, com as forças resultantes
                        (valores de saida/validacao.json, memoriais; Seção 4.1.2 e Apêndice D); seção desenhada
                        em pé (b_w na horizontal, h na vertical), com a face comprimida à esquerda; deformações e
                        tensões com a profundidade na vertical;
  2) curvas_mn1r      : condição de equilíbrio do pilar-padrão acoplado a diagramas M, N, 1/r (NBR 6118:2023,
                        15.3.1 e 15.8.3.3.4), gama_f3 M_def(1/r) e M_ext(1/r), no V3 (P165, armadura definida pelo
                        momento resistente) e no P091 (armadura definida pela existência de equilíbrio estável);
  3) taxa_esbeltez    : taxa geométrica da armadura adotada dos pilares viáveis em função de lambda, por nu e e1/b_w
                        (saida/resultados_matriz.csv; Seção 4.2);
  4) auditor          : duas telas do auditor visual (saida/auditor_pilares.html, gerar_auditor.py), capturadas com o
                        navegador Chromium (Playwright): grandezas da direção de b_w do P165 com as etiquetas das
                        fórmulas e o item [F28] da zona de auditoria (Seção 4.5). Só PNG (captura de tela).
Grava também os dados das figuras 1 a 3 em CSV.

Uso:  python3 figuras_complementares.py [pasta_de_saida]     (padrão: saida/figuras)
      python3 figuras_complementares.py --fusco [pasta_de_saida]   (só a figura das curvas de Fusco, 1981)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import figuras_tcc as estilo  # noqa: E402
import pilares_nbr6118 as nucleo  # noqa: E402
import processar_matriz  # noqa: E402

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — INÍCIO
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [R01] Arquivos lidos: memoriais de validacao.json (pilares V1 a V4), resultados da matriz e auditor visual.
#   Variáveis: PASTA, ARQUIVO_VALIDACAO, ARQUIVO_RESULTADOS, ARQUIVO_AUDITOR.
#   Usada em: nível do módulo (l. 52, 53, 54, 102); figura_secao (l. 337); figura_taxa (l. 573);
#       capturar_telas (l. 651); main (l. 763)
PASTA = Path(__file__).resolve().parent
ARQUIVO_VALIDACAO = PASTA / "saida" / "validacao.json"
ARQUIVO_RESULTADOS = PASTA / "saida" / "resultados_matriz.csv"
ARQUIVO_AUDITOR = PASTA / "saida" / "auditor_pilares.html"

# [R02] Tamanho das figuras (largura da mancha, como em figuras_tcc.py [R01]) e cores: azul e laranja da rampa
#   categorial de figuras_tcc.py [R09] (validada para daltonismo), cinza para linhas de referência e preenchimento
#   claro da zona comprimida. O texto do SVG é gravado em curvas, como nos diagramas (figuras_tcc.py [R09]).
#   Variáveis: ALTURA_SECAO_CM, ALTURA_MN1R_CM, ALTURA_TAXA_CM, COR_PRINCIPAL, COR_SECUNDARIA, COR_REFERENCIA,
#   COR_ZONA_COMPRIMIDA, COR_BARRA.
#   Usada em: figura_secao (l. 365, 373, 379, 380, 388, 397, 407, 408, 409, 417, 419, 425, 426, 427, 428);
#       painel_mn1r (l. 493, 502, 505, 509); figura_mn1r (l. 542, 549); figura_taxa (l. 579, 585, 634);
#       figura_fusco (l. 725, 732, 736)
ALTURA_SECAO_CM = 12.8
ALTURA_MN1R_CM = 7.6
ALTURA_TAXA_CM = 20.0
COR_PRINCIPAL = "#2a78d6"
COR_SECUNDARIA = "#eb6834"
COR_REFERENCIA = "#6b6b6b"
COR_ZONA_COMPRIMIDA = "#dce9fa"
COR_BARRA = "#0b0b0b"

# [R03] Pilar da figura da seção: V1 do memorial (índice 0 de "memoriais" em validacao.json; P156 da matriz).
#   Pontos do perfil de tensões do concreto ao longo da zona comprimida; proporção das duas linhas da figura (seção
#   em escala, em pé, com b_w na horizontal e h na vertical, em cima; deformações e tensões embaixo, com a
#   profundidade a partir da face comprimida no eixo vertical, crescendo para baixo) e limites dos eixos (cm na
#   seção, ‰ e MPa nos diagramas); comprimento das setas das forças nas barras (na escala de tensões); posição do
#   texto à direita da seção (cm) e espaçamento entre as suas linhas (cm).
#   Variáveis: INDICE_MEMORIAL_SECAO, PILAR_MATRIZ_SECAO, PONTOS_PERFIL, PROPORCAO_LINHAS_SECAO, LIMITES_X_SECAO,
#   LIMITES_DEFORMACAO, LIMITES_TENSAO, COMPRIMENTO_SETA_MPA, ABSCISSA_TEXTO_SECAO, PASSO_TEXTO_SECAO.
#   Usada em: ordenada_texto (l. 185);
#       figura_secao (l. 337, 338, 345, 349, 351, 366, 393, 395, 398, 420, 434, 437, 442)
INDICE_MEMORIAL_SECAO = 0
PILAR_MATRIZ_SECAO = "P156"
PONTOS_PERFIL = 201
PROPORCAO_LINHAS_SECAO = (1.2, 1.0)
LIMITES_X_SECAO = (-12.0, 108.0)
LIMITES_DEFORMACAO = (-1.8, 6.2)
LIMITES_TENSAO = (-1.0, 40.0)
COMPRIMENTO_SETA_MPA = 9.0
ABSCISSA_TEXTO_SECAO = 37.0
PASSO_TEXTO_SECAO = 7.0


# [R04] Figura das curvas M, N, 1/r de Fusco (1981, Fig. 8.2.5-1) calculadas pela rotina, com os pontos
#   digitalizados da figura impressa (validacao_literatura.py, [N02]) e a reta do pilar-padrão com a inclinação
#   desenhada: arquivo lido, painéis (d′/h), taxas ω das curvas, limites dos eixos (10³ d/r e μ), altura da figura e
#   pontos de cada curva.
#   Variáveis: ARQUIVO_LITERATURA, PAINEIS_FUSCO, OMEGAS_FUSCO, LIMITES_X_FUSCO, LIMITES_MU_FUSCO, ALTURA_FUSCO_CM,
#   PONTOS_FUSCO.
#   Usada em: figura_fusco (l. 712, 717, 719, 721, 722, 723, 724, 726, 735, 740, 741)
ARQUIVO_LITERATURA = PASTA / "saida" / "validacao_literatura.json"
PAINEIS_FUSCO = (0.05, 0.15)
OMEGAS_FUSCO = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
LIMITES_X_FUSCO = (0.0, 4.0)
LIMITES_MU_FUSCO = (0.0, 0.5)
ALTURA_FUSCO_CM = 8.6
PONTOS_FUSCO = 400


# [F01] Curvatura da configuração de deformações com deformação nula na linha neutra: 1/r = eps_topo / x.
#   Variáveis: deformacao_topo (adimensional), linha_neutra (x, cm).  Fonte: NBR 6118:2023, 17.2.2 a) (seções planas).
#   Usada em: figura_secao (l. 348)
def curvatura_da_linha_neutra(deformacao_topo: float, linha_neutra: float) -> float:
    return deformacao_topo / linha_neutra


# [F02] Profundidade em que a deformação vale eps_c2 (fim do trecho parabólico do diagrama do concreto):
#   y = x (1 - eps_c2/eps_topo).  Variáveis: linha_neutra (cm), eps_c2, deformacao_topo.
#   Fonte: NBR 6118:2023, 8.2.10.1 (diagrama parábola-retângulo).
#   Usada em: figura_secao (l. 370)
def profundidade_eps_c2(linha_neutra: float, eps_c2: float, deformacao_topo: float) -> float:
    return linha_neutra * (1.0 - eps_c2 / deformacao_topo)


# [F03] Conversões de unidade das figuras: deformação em ‰ (x 1000); tensão de kN/cm² para MPa (x 10);
#   momento de kN·cm para kN·m (/100, [F02] do núcleo); curvatura de 1/cm para 10⁻³/m (x 10⁵) e para 1/m (x 100).
#   Variáveis: valor.
#   Usada em: figura_secao (l. 361, 362, 406, 410, 411, 413, 414, 425, 426, 429, 430, 439);
#       painel_mn1r (l. 492, 498, 502, 514, 517, 518); figura_mn1r (l. 548)
def por_mil(valor):
    return np.asarray(valor) * 1000.0


def kn_cm2_para_mpa(valor):
    return np.asarray(valor) * 10.0


def por_cm_para_mili_por_m(valor):
    return np.asarray(valor) * 1.0e5


def por_cm_para_por_m(valor):
    return np.asarray(valor) * 100.0


# [F04] Força numa camada de barras: F = (soma das frações de área das barras da camada) x As x sigma_s.
#   Variáveis: fracao_camada, area_aco (cm²), tensao (kN/cm²).
#   Usada em: figura_secao (l. 363)
def forca_camada(fracao_camada: float, area_aco: float, tensao: float) -> float:
    return fracao_camada * area_aco * tensao


# [F05] Parcela do concreto na seção líquida: resultante da seção bruta menos o concreto deslocado pelas barras
#   comprimidas: R_c = R_c,bruta - R_c,deslocado.  Variáveis: resultante_bruta, deslocado (kN).
#   Usada em: figura_secao (l. 364)
def resultante_liquida(resultante_bruta: float, deslocado: float) -> float:
    return resultante_bruta - deslocado


# [F06] Raio do círculo que representa a barra no desenho da seção: ϕ/2.  Variáveis: diametro (cm).
#   Limites do eixo vertical do desenho da seção (de -folga a h + folga); limites do eixo das profundidades dos
#   diagramas de deformações e tensões (invertido, face comprimida em cima: de altura + folga a -folga); ordenada
#   da linha n do texto ao lado da seção (topo - n x passo); centro de um lado da seção (comprimento / 2), onde
#   ficam as cotas b_w e h.
#   Variáveis: altura, comprimento (cm); topo (cm), linha; FOLGA_PROFUNDIDADE_CM, PASSO_TEXTO_SECAO.
#   Usada em: limites_profundidade_horizontal (l. 177); limites_profundidade (l. 181);
#       figura_secao (l. 378, 382, 384, 393, 395, 399, 446)
FOLGA_PROFUNDIDADE_CM = 1.5


def raio_barra(diametro: float) -> float:
    return diametro / 2.0


def limites_profundidade_horizontal(altura: float) -> tuple[float, float]:
    return -FOLGA_PROFUNDIDADE_CM, altura + FOLGA_PROFUNDIDADE_CM


def limites_profundidade(altura: float) -> tuple[float, float]:
    return altura + FOLGA_PROFUNDIDADE_CM, -FOLGA_PROFUNDIDADE_CM


def ordenada_texto(topo: float, linha: int) -> float:
    return topo - linha * PASSO_TEXTO_SECAO


def centro(comprimento: float) -> float:
    return comprimento / 2.0


# [R04] Pilares da figura do pilar-padrão acoplado: V3 do memorial (P165: 30 x 50 cm, C30, lambda = 110, nu = 0,40,
#   e1/b_w = 0,15; armadura definida por M_Rd = M_d,tot) e P091 (20 x 30 cm, C35, lambda = 110, nu = 0,20,
#   e1/b_w = 0,15; armadura definida pela existência de equilíbrio estável, Seção 4.2 do TCC); fração da armadura
#   necessária com que se mostra, no P091, que não há equilíbrio (a mesma de capitulo_4.py [R10]); pontos das curvas.
#   Variáveis: PILAR_MN1R_RESISTENCIA, PILAR_MN1R_ESTABILIDADE, FRACAO_REDUZIDA, PONTOS_CURVA_FIGURA.
#   Usada em: curva_do_pilar (l. 468); painel_mn1r (l. 484, 490); figura_mn1r (l. 536, 537, 538, 540, 543, 545)
PILAR_MN1R_RESISTENCIA = "P165"
PILAR_MN1R_ESTABILIDADE = "P091"
FRACAO_REDUZIDA = 0.99
PONTOS_CURVA_FIGURA = 1201

# [R07] Detalhe do resíduo da condição de equilíbrio no P091 (painel b): R(1/r) = gama_f3 M_def - M_ext ([F30] do
#   núcleo), ampliado em torno do ponto de tangência, em que R toca zero com As,nec e fica abaixo de zero com 99 %;
#   posição do detalhe e da anotação do ponto de equilíbrio em cada painel (fração dos eixos; a do P165 começa em
#   0,42 para que o texto não cruze a borda direita do painel).
#   Variáveis: POSICAO_DETALHE, LIMITES_DETALHE_CURVATURA, LIMITES_DETALHE_RESIDUO, POSICOES_ANOTACAO.
#   Usada em: painel_mn1r (l. 519); figura_mn1r (l. 546, 552, 553)
POSICAO_DETALHE = (0.21, 0.60, 0.30, 0.27)
LIMITES_DETALHE_CURVATURA = (4.0, 10.0)
LIMITES_DETALHE_RESIDUO = (-0.06, 0.03)
POSICOES_ANOTACAO = {"P165": (0.42, 0.20), "P091": (0.42, 0.10)}


# [F07] Momento externo do pilar-padrão acoplado em função da curvatura (a mesma expressão de [F28] e [F30] do
#   núcleo): M_ext(1/r) = alfa_b M1d,A + Nd (l_e²/10)(1/r); e momento da curva de deformação majorado:
#   gama_f3 M_def(1/r). Equilíbrio: gama_f3 M_def = M_ext (NBR 6118:2023, 15.3.1 e 15.8.3.3.4).
#   Variáveis: alfa_b, momento_primeira_ordem (kN·cm), forca_normal (kN), comprimento (cm), curvaturas (1/cm),
#   momentos (kN·cm).
#   Usada em: curva_do_pilar (l. 470); painel_mn1r (l. 485)
def momento_externo(alfa_b: float, momento_primeira_ordem: float, forca_normal: float, comprimento: float,
                    curvaturas) -> np.ndarray:
    return (alfa_b * momento_primeira_ordem
            + forca_normal * comprimento ** 2 / nucleo.DIVISOR_PILAR_PADRAO * np.asarray(curvaturas))


def momento_majorado(momentos) -> np.ndarray:
    return nucleo.GAMA_F3 * np.asarray(momentos)


# [F08] Área de aço reduzida para mostrar a falta de equilíbrio: f x As,nec.  Variáveis: fracao, area (cm²).
#   Usada em: curva_do_pilar (l. 466)
def area_reduzida(fracao: float, area: float) -> float:
    return fracao * area


# [R05] Taxa em função da esbeltez (Figura 7 do TCC): 3 linhas (ν) x 2 colunas (e1/b_w), eixo de λ proporcional
#   (LIMITES_LAMBDA_TAXA, rótulos em TAMANHO_ROTULO_LAMBDA_PT); traço por seção (b_w x h), cor e marcador por classe
#   do concreto (os de figuras_tcc.py [R09]); marca vazada quando a armadura mínima governa (coluna "Governa" igual
#   a GOVERNA_MINIMA); deslocamento horizontal por classe (DESLOCAMENTO_CLASSE_TAXA, em unidades de λ) só para
#   separar as marcas; ν no canto superior esquerdo de cada painel (POSICAO_ROTULO_NU, fração dos eixos); linha da
#   armadura máxima de 4 % Ac (fora da emenda); limite superior do eixo de ρ (LIMITE_TAXA_EIXO, em %).
#   Variáveis: ESTILOS_SECAO, TAMANHO_MARCADOR_TAXA, LIMITE_TAXA, LIMITES_LAMBDA_TAXA, TAMANHO_ROTULO_LAMBDA_PT,
#   DESLOCAMENTO_CLASSE_TAXA, GOVERNA_MINIMA, POSICAO_ROTULO_NU, LIMITE_TAXA_EIXO.
#   Usada em: figura_taxa (l. 581, 586, 595, 600, 601, 605, 606, 614, 620, 626, 628, 632)
ESTILOS_SECAO = {(20.0, 30.0): "-", (30.0, 50.0): (0, (4, 2)), (40.0, 80.0): (0, (1, 1.4))}
TAMANHO_MARCADOR_TAXA = 4.2
LIMITE_TAXA = nucleo.TAXA_MAXIMA_FORA_EMENDA
LIMITES_LAMBDA_TAXA = (25.0, 145.0)
TAMANHO_ROTULO_LAMBDA_PT = 8
DESLOCAMENTO_CLASSE_TAXA = {25.0: -1.6, 30.0: 0.0, 35.0: 1.6}
GOVERNA_MINIMA = "As,mín"
POSICAO_ROTULO_NU = (0.025, 0.975)
LIMITE_TAXA_EIXO = 4.5


# [F09] Taxa em porcentagem (100 rho) e abscissa deslocada de uma marca (λ + deslocamento da classe).
#   Variáveis: taxa, esbeltez, deslocamento.
#   Usada em: figura_taxa (l. 581, 595, 596)
def em_porcentagem(taxa):
    return np.asarray(taxa) * 100.0


def abscissa_deslocada(esbeltez: float, deslocamento: float) -> float:
    return esbeltez + deslocamento


# [R06] Telas do auditor: janela do navegador (largura em px e fator de escala da captura), pilar e linhas da tabela
#   "Esforços e 2ª ordem por direção" mostradas em (a), item da zona de auditoria mostrado em (b), margens e fonte
#   dos rótulos (a) e (b) da imagem composta.
#   Variáveis: LARGURA_JANELA_PX, ALTURA_JANELA_PX, ESCALA_CAPTURA, PILAR_TELA, LINHAS_TELA, FORMULA_TELA,
#   MARGEM_COMPOSICAO_PX, ARQUIVO_FONTE_ROTULO, CORPO_ROTULO_PX, ESPERA_MS; ESTILO_CAPTURA (quebra das linhas do
#   código na largura da janela, que é estreita para que o texto fique legível na largura da figura).
#   Usada em: tamanho_composicao (l. 295); ordenada_segunda_tela (l. 299); ordenada_rotulo_segunda_tela (l. 303);
#       capturar_telas (l. 649, 650, 652, 653, 654, 655, 657, 675, 676, 677, 680, 681, 682, 683);
#       figura_auditor (l. 693, 696)
LARGURA_JANELA_PX = 860
ALTURA_JANELA_PX = 900
ESCALA_CAPTURA = 2
PILAR_TELA = "P165"
LINHAS_TELA = ("ea (fluência)", "ecc", "M1d,A (com ecc, se houver)", "1/r", "Md,tot")
FORMULA_TELA = "F28"
MARGEM_COMPOSICAO_PX = 36
ARQUIVO_FONTE_ROTULO = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
CORPO_ROTULO_PX = 44
ESPERA_MS = 900
ESTILO_CAPTURA = "pre { white-space: pre-wrap !important; overflow-wrap: anywhere; }"


# [F10] Composição das duas telas: largura = maior largura; altura = soma das alturas + 3 margens (acima de (a), entre
#   (a) e (b) e acima de (b), onde ficam os rótulos); ordenada de (b) = margem + altura de (a) + 2 margens.
#   Variáveis: larguras, alturas (px), altura_a (px).
#   Usada em: figura_auditor (l. 691, 694, 698)
def tamanho_composicao(larguras: tuple[int, int], alturas: tuple[int, int]) -> tuple[int, int]:
    return max(larguras), sum(alturas) + 3 * MARGEM_COMPOSICAO_PX


def ordenada_segunda_tela(altura_a: int) -> int:
    return altura_a + 3 * MARGEM_COMPOSICAO_PX


def ordenada_rotulo_segunda_tela(altura_a: int) -> int:
    return altura_a + 2 * MARGEM_COMPOSICAO_PX


# [R08] Números escritos nas figuras (rótulos, títulos e legendas): casas fixas, vírgula decimal, sinal de menos
#   U+2212 e separador de milhar do SI (até quatro algarismos na parte inteira sem separador; com cinco ou mais,
#   espaço não separável), pela mesma regra dos eixos (figuras_tcc.py [R05]): 1421,2 kN; 26 071,6. Nunca ponto.
#   Unidade em ‰ separada do número por espaço não separável (ESPACO_UNIDADE), como em "3,5 ‰".
#   Variáveis: valor, casas; ESPACO_UNIDADE.
#   Usada em: figura_secao (l. 382, 384, 390, 396, 403, 404, 411, 414, 418, 430, 437, 440);
#       painel_mn1r (l. 489, 490, 491, 508, 517); figura_taxa (l. 626, 629); figura_fusco (l. 727, 739)
ESPACO_UNIDADE = "\u00a0"


def numero(valor: float, casas: int) -> str:
    return estilo.agrupar_milhares(("%.*f" % (casas, valor)).replace(".", ","))

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


def gravar_csv(caminho: Path, linhas: list[dict]) -> None:
    with caminho.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]), delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas)


def caso_da_matriz(identificador: str):
    return next(caso for caso in processar_matriz.montar_matriz() if caso.identificador == identificador)


# ----------------------------------------------------------------------------- 1) seção do V1
def figura_secao(pasta: Path) -> list[dict]:
    memorial = json.loads(ARQUIVO_VALIDACAO.read_text(encoding="utf-8"))["memoriais"][INDICE_MEMORIAL_SECAO]
    caso = caso_da_matriz(PILAR_MATRIZ_SECAO)
    resultado = nucleo.dimensionar_pilar(caso.pilar)
    pilar = resultado.pilar
    material = nucleo.Material(pilar.fck_mpa, pilar.fyk_mpa)
    secao = nucleo.secao_na_direcao(pilar.largura_bw, pilar.altura_h, resultado.candidato.x, resultado.candidato.y,
                                    nucleo.DIRECAO_B)
    if not np.allclose(secao.profundidades, memorial["camadas_y_cm"]):
        raise ValueError("as barras do %s não são as do memorial V1" % PILAR_MATRIZ_SECAO)
    linha_neutra = memorial["x_cm"]
    deformacao_topo = memorial["eps_topo"]
    curvatura = curvatura_da_linha_neutra(deformacao_topo, linha_neutra)
    profundidades = np.linspace(0.0, secao.altura, PONTOS_PERFIL)
    deformacoes = nucleo.deformacao_na_profundidade(deformacao_topo, curvatura, profundidades)
    zona = np.linspace(0.0, linha_neutra, PONTOS_PERFIL)
    tensoes = nucleo.tensao_concreto(nucleo.deformacao_na_profundidade(deformacao_topo, curvatura, zona),
                                     material.tensao_pico_resistencia, material.eps_c2, material.expoente)
    camadas = sorted(set(memorial["camadas_y_cm"]))
    forcas = []
    for profundidade in camadas:
        indices = [indice for indice, valor in enumerate(memorial["camadas_y_cm"]) if valor == profundidade]
        fracao = float(np.sum(secao.fracoes_area[indices]))
        tensao = memorial["sigma_camadas_kN_cm2"][indices[0]]
        forcas.append({"y (cm)": profundidade, "barras": len(indices),
                       "ε (‰)": float(por_mil(memorial["eps_camadas"][indices[0]])),
                       "σs (MPa)": float(kn_cm2_para_mpa(tensao)),
                       "F (kN)": forca_camada(fracao, memorial["As_nec_cm2"], tensao)})
    concreto = resultante_liquida(memorial["Rc_kN"], memorial["Rc_deslocado_kN"])
    figura = plt.figure(figsize=(estilo.cm_para_polegada(estilo.LARGURA_CM), estilo.cm_para_polegada(ALTURA_SECAO_CM)))
    grade = figura.add_gridspec(2, 2, height_ratios=PROPORCAO_LINHAS_SECAO, hspace=0.30, wspace=0.08)
    eixo_secao = figura.add_subplot(grade[0, :])
    eixo_deformacao = figura.add_subplot(grade[1, 0])
    eixo_tensao = figura.add_subplot(grade[1, 1], sharey=eixo_deformacao)
    profundidade_c2 = profundidade_eps_c2(linha_neutra, material.eps_c2, deformacao_topo)
    # (a) seção em escala, em pé: b_w (direção da flexão) na horizontal, face comprimida à esquerda; h na vertical.
    #     A linha neutra fica vertical, a x da face comprimida.
    eixo_secao.add_patch(plt.Rectangle((0.0, 0.0), linha_neutra, secao.largura, facecolor=COR_ZONA_COMPRIMIDA,
                                       edgecolor="none"))
    eixo_secao.add_patch(plt.Rectangle((0.0, 0.0), secao.altura, secao.largura, facecolor="none", edgecolor="black",
                                       linewidth=estilo.ESPESSURA_LINHA))
    for profundidade, posicao in zip(secao.profundidades, resultado.candidato.y):
        eixo_secao.add_patch(plt.Circle((profundidade, posicao), raio_barra(resultado.candidato.diametro),
                                        facecolor=COR_BARRA, edgecolor="none"))
    eixo_secao.plot([linha_neutra, linha_neutra], [0.0, secao.largura], color=COR_SECUNDARIA, linewidth=1.0,
                    linestyle=(0, (4, 2)))
    eixo_secao.annotate("$b_w$ = %s cm" % numero(secao.altura, 0), xy=(centro(secao.altura), 0.0), xytext=(0, -3),
                        textcoords="offset points", ha="center", va="top", fontsize=estilo.CORPO_SECUNDARIO_PT)
    eixo_secao.annotate("$h$ = %s cm" % numero(secao.largura, 0), xy=(0.0, centro(secao.largura)), xytext=(-4, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=estilo.CORPO_SECUNDARIO_PT,
                        rotation=90)
    eixo_secao.annotate("$x$", xy=(linha_neutra, secao.largura), xytext=(2, -2), textcoords="offset points",
                        ha="left", va="top", fontsize=estilo.CORPO_SECUNDARIO_PT, color=COR_SECUNDARIA)
    textos = ["flexão na direção de $b_w$ (a menor dimensão)"]
    textos += ["%d barras a %s cm da face comprimida" % (camada["barras"], numero(camada["y (cm)"], 2))
               for camada in forcas]
    for indice, texto in enumerate(textos):
        eixo_secao.text(ABSCISSA_TEXTO_SECAO, ordenada_texto(secao.largura, indice), texto, ha="left", va="top",
                        fontsize=estilo.CORPO_SECUNDARIO_PT)
    eixo_secao.text(ABSCISSA_TEXTO_SECAO, ordenada_texto(secao.largura, len(textos)),
                    "linha neutra (vertical), $x$ = %s cm" % numero(linha_neutra, 2), ha="left", va="top",
                    fontsize=estilo.CORPO_SECUNDARIO_PT, color=COR_SECUNDARIA)
    eixo_secao.set_xlim(*LIMITES_X_SECAO)
    eixo_secao.set_ylim(*limites_profundidade_horizontal(secao.largura))
    eixo_secao.set_aspect("equal", adjustable="box")
    eixo_secao.set_axis_off()
    eixo_secao.set_title("(a) Seção, com a face comprimida à esquerda: %d ϕ %s mm, $A_{s,calc}$ = %s cm²" % (
        resultado.candidato.numero_barras, numero(processar_matriz.cm_para_mm(resultado.candidato.diametro), 0),
        numero(memorial["As_nec_cm2"], 2)), fontsize=estilo.CORPO_SECUNDARIO_PT)
    # (b) deformações
    eixo_deformacao.plot(por_mil(deformacoes), profundidades, color="black", linewidth=estilo.ESPESSURA_LINHA)
    eixo_deformacao.axvline(0.0, color=COR_REFERENCIA, linewidth=0.5)
    eixo_deformacao.axhline(linha_neutra, color=COR_SECUNDARIA, linewidth=1.0, linestyle=(0, (4, 2)))
    eixo_deformacao.axhline(profundidade_c2, color=COR_REFERENCIA, linewidth=0.6, linestyle=(0, (1, 1.4)))
    eixo_deformacao.text(float(por_mil(material.eps_c2)), profundidade_c2, "  $\\varepsilon_{c2}$ = %s%s‰" % (
        numero(float(por_mil(material.eps_c2)), 1), ESPACO_UNIDADE), ha="left", va="bottom",
        fontsize=estilo.CORPO_SECUNDARIO_PT)
    eixo_deformacao.text(float(por_mil(deformacao_topo)), 0.0, "  $\\varepsilon_{cu}$ = %s%s‰" % (
        numero(float(por_mil(deformacao_topo)), 1), ESPACO_UNIDADE), ha="left", va="top",
        fontsize=estilo.CORPO_SECUNDARIO_PT)
    for camada in forcas:
        eixo_deformacao.plot(camada["ε (‰)"], camada["y (cm)"], marker="o", markersize=4, color=COR_PRINCIPAL)
        eixo_deformacao.text(camada["ε (‰)"], camada["y (cm)"], "  %s%s‰" % (numero(camada["ε (‰)"], 2), ESPACO_UNIDADE),
                             ha="left", va="center", fontsize=estilo.CORPO_SECUNDARIO_PT, color=COR_PRINCIPAL)
    eixo_deformacao.set_xlim(*LIMITES_DEFORMACAO)
    eixo_deformacao.set_xlabel("$\\varepsilon$ (‰), encurtamento positivo")
    eixo_deformacao.set_ylabel("Profundidade (cm)")
    eixo_deformacao.set_title("(b) Deformações", fontsize=estilo.CORPO_SECUNDARIO_PT)
    # (c) tensões do concreto e forças nas camadas de barras
    eixo_tensao.fill_betweenx(zona, 0.0, kn_cm2_para_mpa(tensoes), color=COR_ZONA_COMPRIMIDA)
    eixo_tensao.plot(kn_cm2_para_mpa(tensoes), zona, color=COR_PRINCIPAL, linewidth=estilo.ESPESSURA_LINHA)
    eixo_tensao.axvline(0.0, color=COR_REFERENCIA, linewidth=0.5)
    eixo_tensao.axhline(linha_neutra, color=COR_SECUNDARIA, linewidth=1.0, linestyle=(0, (4, 2)))
    eixo_tensao.text(float(kn_cm2_para_mpa(material.tensao_pico_resistencia)), profundidade_c2,
                     "  $0{,}85\\,\\eta_c f_{cd}$\n  = %s MPa" % numero(float(kn_cm2_para_mpa(
                         material.tensao_pico_resistencia)), 2), ha="left", va="center",
                     fontsize=estilo.CORPO_SECUNDARIO_PT)
    for camada in forcas:
        eixo_tensao.annotate("", xy=(0.0, camada["y (cm)"]), xytext=(COMPRIMENTO_SETA_MPA, camada["y (cm)"]),
                             arrowprops={"arrowstyle": "->" if camada["F (kN)"] > 0.0 else "<-", "color": "black",
                                         "linewidth": 0.9})
        eixo_tensao.text(COMPRIMENTO_SETA_MPA, camada["y (cm)"], "  barras: %s kN" % numero(camada["F (kN)"], 1),
                         ha="left", va="center", fontsize=estilo.CORPO_SECUNDARIO_PT)
    eixo_tensao.text(float(kn_cm2_para_mpa(material.tensao_pico_resistencia)), linha_neutra,
                     "  concreto (seção\n  líquida): %s kN" % numero(concreto, 1), ha="left", va="bottom",
                     fontsize=estilo.CORPO_SECUNDARIO_PT)
    eixo_tensao.set_xlim(*LIMITES_TENSAO)
    eixo_tensao.set_xlabel("$\\sigma_c$ (MPa)")
    eixo_tensao.set_title("(c) Tensões no concreto e forças nas barras", fontsize=estilo.CORPO_SECUNDARIO_PT)
    eixo_tensao.tick_params(labelleft=False)
    eixo_deformacao.set_ylim(*limites_profundidade(secao.altura))
    for eixo in (eixo_deformacao, eixo_tensao):
        eixo.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(estilo.texto_eixo))
        eixo.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(estilo.texto_eixo))
        for borda in eixo.spines.values():
            borda.set_linewidth(0.6)
    estilo.salvar(figura, pasta, "secao_v1")
    return [*forcas, {"y (cm)": "", "barras": "", "ε (‰)": "", "σs (MPa)": "concreto (seção líquida)",
                      "F (kN)": concreto}]


# ----------------------------------------------------------------------------- 2) pilar-padrão acoplado
def curva_do_pilar(identificador: str, fracao: float = 1.0) -> dict:
    caso = caso_da_matriz(identificador)
    resultado = nucleo.dimensionar_pilar(caso.pilar)
    pilar = resultado.pilar
    material = nucleo.Material(pilar.fck_mpa, pilar.fyk_mpa)
    secao = nucleo.secao_na_direcao(pilar.largura_bw, pilar.altura_h, resultado.candidato.x, resultado.candidato.y,
                                    nucleo.DIRECAO_B)
    direcao = resultado.direcao_b
    area = area_reduzida(fracao, direcao.area_necessaria)
    curvaturas, momentos = nucleo.curva_momento_curvatura(
        secao, material, area, nucleo.forca_normal_curva_deformacao(resultado.forca_normal), True, PONTOS_CURVA_FIGURA)
    return {"pilar": identificador, "resultado": resultado, "direcao": direcao, "area": area,
            "curvaturas": curvaturas, "momentos_majorados": momento_majorado(momentos),
            "residuo": nucleo.residuo_equilibrio_pilar_padrao(momentos, direcao.alfa_b, direcao.momento_primeira_ordem,
                                                              resultado.forca_normal,
                                                              resultado.comprimento_equivalente, curvaturas),
            "momento_resistente": nucleo.momento_resistente(secao, material, area, resultado.forca_normal),
            "equilibrio": nucleo.equilibrio_pilar_padrao_acoplado(
                secao, material, area, resultado.forca_normal, direcao.momento_primeira_ordem, direcao.alfa_b,
                resultado.comprimento_equivalente)}


def painel_mn1r(eixo, curvas: list[dict], titulo: str) -> list[dict]:
    principal = curvas[0]
    direcao, resultado = principal["direcao"], principal["resultado"]
    maior_curvatura = max(float(curva["curvaturas"][-1]) for curva in curvas)
    curvaturas_reta = np.linspace(0.0, maior_curvatura, PONTOS_CURVA_FIGURA)
    externo = momento_externo(direcao.alfa_b, direcao.momento_primeira_ordem, resultado.forca_normal,
                              resultado.comprimento_equivalente, curvaturas_reta)
    dados = []
    for indice, curva in enumerate(curvas):
        rotulo = ("$\\gamma_{f3} M_{def}$, $A_s$ = %s cm²" % numero(curva["area"], 2) if indice == 0 else
                  "$\\gamma_{f3} M_{def}$, %s %% de $A_s$ (%s cm²)" % (numero(estilo.porcentagem(FRACAO_REDUZIDA), 0),
                                                                     numero(curva["area"], 2)))
        eixo.plot(por_cm_para_mili_por_m(curva["curvaturas"]), nucleo.kn_cm_para_kn_m(curva["momentos_majorados"]),
                  color=COR_PRINCIPAL, linewidth=estilo.ESPESSURA_LINHA if indice == 0 else 0.8,
                  linestyle="-" if indice == 0 else (0, (2, 1.5)), label=rotulo)
        dados.append({"pilar": curva["pilar"], "As (cm²)": curva["area"],
                      "equilíbrio estável": curva["equilibrio"] is not None,
                      "1/r de equilíbrio (1/m)": "" if curva["equilibrio"] is None else float(
                          por_cm_para_por_m(curva["equilibrio"][0])),
                      "Md,tot (kN·m)": "" if curva["equilibrio"] is None else nucleo.kn_cm_para_kn_m(
                          curva["equilibrio"][1]),
                      "MRd (kN·m)": nucleo.kn_cm_para_kn_m(curva["momento_resistente"])})
    eixo.plot(por_cm_para_mili_por_m(curvaturas_reta), nucleo.kn_cm_para_kn_m(externo), color=COR_SECUNDARIA,
              linewidth=estilo.ESPESSURA_LINHA, linestyle=(0, (4, 2)), label="$M_{ext}$ = $\\alpha_b M_{1d,A}$ + "
              "$N_d (\\ell_e^2/10)(1/r)$")
    eixo.axhline(nucleo.kn_cm_para_kn_m(principal["momento_resistente"]), color=COR_REFERENCIA, linewidth=0.6,
                 linestyle=(0, (1, 1.4)))
    eixo.text(1.0, nucleo.kn_cm_para_kn_m(principal["momento_resistente"]),
              "$M_{Rd}$ = %s kN·m " % numero(nucleo.kn_cm_para_kn_m(principal["momento_resistente"]), 1),
              ha="right", va="top", fontsize=estilo.CORPO_SECUNDARIO_PT, color=COR_REFERENCIA,
              transform=eixo.get_yaxis_transform())
    if principal["equilibrio"] is None:
        raise ValueError("o %s não tem equilíbrio com a armadura necessária" % principal["pilar"])
    curvatura_eq, momento_eq = principal["equilibrio"]
    eixo.plot(por_cm_para_mili_por_m(curvatura_eq), nucleo.kn_cm_para_kn_m(momento_eq), marker="o", markersize=4.5,
              color="black", linestyle="none", label="Equilíbrio: $M_{d,tot}$")
    eixo.annotate("$1/r$ = %s · $10^{-3}$ m$^{-1}$\n$M_{d,tot}$ = %s kN·m" % (
        numero(float(por_cm_para_mili_por_m(curvatura_eq)), 1), numero(nucleo.kn_cm_para_kn_m(momento_eq), 1)),
        xy=(float(por_cm_para_mili_por_m(curvatura_eq)), nucleo.kn_cm_para_kn_m(momento_eq)),
        xytext=POSICOES_ANOTACAO[principal["pilar"]], textcoords="axes fraction", fontsize=estilo.CORPO_SECUNDARIO_PT,
        arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 0.5})
    eixo.set_title(titulo, fontsize=estilo.CORPO_SECUNDARIO_PT)
    eixo.set_xlabel("$1/r$ ($10^{-3}$ m$^{-1}$)")
    eixo.set_ylim(bottom=0.0)
    eixo.set_xlim(left=0.0)
    eixo.grid(True, color=estilo.CINZA_GRADE, linewidth=0.4)
    eixo.set_axisbelow(True)
    formatador = matplotlib.ticker.FuncFormatter(estilo.texto_eixo)
    eixo.xaxis.set_major_formatter(formatador)
    eixo.yaxis.set_major_formatter(formatador)
    eixo.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=estilo.CORPO_SECUNDARIO_PT,
                ncol=1)
    return dados


def figura_mn1r(pasta: Path) -> list[dict]:
    resistencia = curva_do_pilar(PILAR_MN1R_RESISTENCIA)
    estabilidade = curva_do_pilar(PILAR_MN1R_ESTABILIDADE)
    reduzida = curva_do_pilar(PILAR_MN1R_ESTABILIDADE, FRACAO_REDUZIDA)
    if reduzida["equilibrio"] is not None:
        raise ValueError("o %s tem equilíbrio com a armadura reduzida" % PILAR_MN1R_ESTABILIDADE)
    figura, (eixo_a, eixo_b) = plt.subplots(1, 2, figsize=(estilo.cm_para_polegada(estilo.LARGURA_CM),
                                                            estilo.cm_para_polegada(ALTURA_MN1R_CM)))
    dados = painel_mn1r(eixo_a, [resistencia], "(a) %s (V3): armadura definida por $M_{Rd}$" % PILAR_MN1R_RESISTENCIA)
    dados += painel_mn1r(eixo_b, [estabilidade, reduzida],
                         "(b) %s: armadura definida pelo equilíbrio estável" % PILAR_MN1R_ESTABILIDADE)
    detalhe = eixo_b.inset_axes(POSICAO_DETALHE)
    for indice, curva in enumerate((estabilidade, reduzida)):
        detalhe.plot(por_cm_para_mili_por_m(curva["curvaturas"]), nucleo.kn_cm_para_kn_m(curva["residuo"]),
                     color=COR_PRINCIPAL, linewidth=estilo.ESPESSURA_LINHA if indice == 0 else 0.8,
                     linestyle="-" if indice == 0 else (0, (2, 1.5)))
    detalhe.axhline(0.0, color="black", linewidth=0.6)
    detalhe.set_xlim(*LIMITES_DETALHE_CURVATURA)
    detalhe.set_ylim(*LIMITES_DETALHE_RESIDUO)
    detalhe.set_title("$R$ = $\\gamma_{f3} M_{def}$ − $M_{ext}$ (kN·m)", fontsize=estilo.CORPO_CELULA_PT)
    detalhe.tick_params(labelsize=estilo.CORPO_CELULA_PT, length=2)
    detalhe.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(estilo.texto_eixo))
    detalhe.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(estilo.texto_eixo))
    detalhe.grid(True, color=estilo.CINZA_GRADE, linewidth=0.3)
    for borda in detalhe.spines.values():
        borda.set_linewidth(0.5)
    dados += [{"pilar": curva["pilar"], "As (cm²)": curva["area"], "equilíbrio estável": "",
               "1/r de equilíbrio (1/m)": "maior R (kN·m)",
               "Md,tot (kN·m)": float(nucleo.kn_cm_para_kn_m(np.max(curva["residuo"]))), "MRd (kN·m)": ""}
              for curva in (estabilidade, reduzida)]
    eixo_a.set_ylabel("Momento (kN·m)")
    figura.subplots_adjust(wspace=0.22)
    estilo.salvar(figura, pasta, "curvas_mn1r")
    return dados


# ----------------------------------------------------------------------------- 3) taxa em função de lambda
def figura_taxa(pasta: Path) -> list[dict]:
    with ARQUIVO_RESULTADOS.open(encoding="utf-8") as arquivo:
        viaveis = [linha for linha in csv.DictReader(arquivo, delimiter=";") if linha["Situação"] == nucleo.STATUS_OK]
    linhas_painel = processar_matriz.FORCAS_ADIMENSIONAIS
    colunas_painel = processar_matriz.EXCENTRICIDADES_RELATIVAS
    figura, eixos = plt.subplots(len(linhas_painel), len(colunas_painel), sharex=True, sharey=True,
                                 figsize=(estilo.cm_para_polegada(estilo.LARGURA_CM),
                                          estilo.cm_para_polegada(ALTURA_TAXA_CM)))
    dados = []
    taxa_maxima = float(em_porcentagem(LIMITE_TAXA))
    for linha_eixo, nivel_nu in enumerate(linhas_painel):
        for coluna_eixo, relativa in enumerate(colunas_painel):
            eixo = eixos[linha_eixo][coluna_eixo]
            eixo.axhline(taxa_maxima, color=COR_REFERENCIA, linewidth=0.8, zorder=1)
            for (largura, altura), traco in ESTILOS_SECAO.items():
                for fck in processar_matriz.RESISTENCIAS_CONCRETO:
                    serie = sorted((float(linha["λ (dir. b_w)"]), float(linha["ρ adot"]), linha["ID"], linha["Governa"])
                                   for linha in viaveis if float(linha["b_w (cm)"]) == largura
                                   and float(linha["h (cm)"]) == altura and float(linha["fck (MPa)"]) == fck
                                   and float(linha["ν"]) == nivel_nu and float(linha["e1/b_w"]) == relativa)
                    if not serie:
                        continue
                    cor = estilo.CORES_CLASSE[fck]
                    abscissas = [abscissa_deslocada(item[0], DESLOCAMENTO_CLASSE_TAXA[fck]) for item in serie]
                    taxas = em_porcentagem([item[1] for item in serie])
                    eixo.plot(abscissas, taxas, color=cor, linestyle=traco, linewidth=0.9, zorder=2)
                    for abscissa, taxa_porcentagem, (esbeltez, _, identificador, governa) in zip(abscissas, taxas, serie):
                        eixo.plot(abscissa, taxa_porcentagem, linestyle="none", marker=estilo.MARCADORES_CLASSE[fck],
                                  markersize=TAMANHO_MARCADOR_TAXA, markeredgecolor=cor, markeredgewidth=0.9,
                                  markerfacecolor="white" if governa == GOVERNA_MINIMA else cor, zorder=3)
                        dados.append({"ID": identificador, "b_w (cm)": largura, "h (cm)": altura, "fck (MPa)": fck,
                                      "ν": nivel_nu, "e1/b_w": relativa, "λ": esbeltez,
                                      "ρ adot (%)": float(taxa_porcentagem), "Governa": governa})
            eixo.set_xlim(*LIMITES_LAMBDA_TAXA)
            eixo.set_ylim(0.0, LIMITE_TAXA_EIXO)
            eixo.set_xticks(processar_matriz.ESBELTEZES)
            formatador = matplotlib.ticker.FuncFormatter(estilo.texto_eixo)
            eixo.xaxis.set_major_formatter(formatador)
            eixo.yaxis.set_major_formatter(formatador)
            eixo.grid(True, color=estilo.CINZA_GRADE, linewidth=0.4)
            eixo.set_axisbelow(True)
            eixo.tick_params(labelsize=estilo.CORPO_SECUNDARIO_PT, length=2.5)
            eixo.tick_params(axis="x", labelbottom=True, labelsize=TAMANHO_ROTULO_LAMBDA_PT)
            for borda in eixo.spines.values():
                borda.set_linewidth(0.6)
            if linha_eixo == 0:
                eixo.set_title("$e_1/b_w$ = %s" % estilo.texto_fixo(relativa, estilo.CASAS_NIVEL_EXCENTRICIDADE),
                               fontsize=estilo.CORPO_SECUNDARIO_PT)
            eixo.text(POSICAO_ROTULO_NU[0], POSICAO_ROTULO_NU[1], "$\\nu$ = %s" % estilo.texto_fixo(nivel_nu, 1),
                      transform=eixo.transAxes, ha="left", va="top", fontsize=estilo.CORPO_SECUNDARIO_PT,
                      bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=4)
    figura.supylabel("$\\rho = A_{\\mathrm{s,adot}}/A_{\\mathrm{c}}$ (%)", fontsize=estilo.CORPO_SECUNDARIO_PT, x=0.02)
    figura.supxlabel("$\\lambda$ na direção de $b_w$", fontsize=estilo.CORPO_SECUNDARIO_PT, y=0.085)
    marcas = [matplotlib.lines.Line2D([], [], color=estilo.CORES_CLASSE[fck], marker=estilo.MARCADORES_CLASSE[fck],
                                      linestyle="none", markersize=TAMANHO_MARCADOR_TAXA, label="C%s" % numero(fck, 0))
              for fck in processar_matriz.RESISTENCIAS_CONCRETO]
    marcas += [matplotlib.lines.Line2D([], [], color="black", linestyle=ESTILOS_SECAO[secao], linewidth=0.9,
                                       label="%s × %s cm" % (numero(secao[0], 0), numero(secao[1], 0)))
               for secao in processar_matriz.SECOES]
    marcas += [matplotlib.lines.Line2D([], [], color="black", marker="o", markerfacecolor="white", linestyle="none",
                                       markersize=TAMANHO_MARCADOR_TAXA,
                                       label="marca vazada: $A_{\\mathrm{s,mín}}$ governa"),
               matplotlib.lines.Line2D([], [], color=COR_REFERENCIA, linewidth=0.8,
                                       label="$A_{\\mathrm{s,máx}}$ = 4 % $A_{\\mathrm{c}}$")]
    figura.legend(handles=marcas, loc="lower center", ncol=3, frameon=False, fontsize=estilo.CORPO_SECUNDARIO_PT,
                  bbox_to_anchor=(0.5, -0.01), columnspacing=1.4, handlelength=2.2)
    figura.subplots_adjust(hspace=0.16, wspace=0.05, bottom=0.135, left=0.075, right=0.985, top=0.965)
    estilo.salvar(figura, pasta, "taxa_esbeltez")
    return dados


# ----------------------------------------------------------------------------- 4) telas do auditor
def capturar_telas(pasta: Path) -> tuple[Path, Path]:
    from playwright.sync_api import sync_playwright
    tela_a, tela_b = pasta.joinpath("auditor_tela_a.png"), pasta.joinpath("auditor_tela_b.png")
    with sync_playwright() as playwright:
        navegador = playwright.chromium.launch()
        pagina = navegador.new_page(viewport={"width": LARGURA_JANELA_PX, "height": ALTURA_JANELA_PX},
                                    device_scale_factor=ESCALA_CAPTURA)
        pagina.goto(ARQUIVO_AUDITOR.as_uri())
        pagina.add_style_tag(content=ESTILO_CAPTURA)
        pagina.wait_for_timeout(ESPERA_MS)
        pagina.fill("#f-busca", PILAR_TELA)
        pagina.wait_for_timeout(ESPERA_MS)
        pagina.click("#lista > *")
        pagina.wait_for_timeout(ESPERA_MS)
        pagina.evaluate("""() => document.querySelectorAll('*').forEach(elemento => {
                               const posicao = getComputedStyle(elemento).position;
                               if (posicao === 'sticky' || posicao === 'fixed') { elemento.style.position = 'static'; }
                           })""")
        tabela = pagina.locator("#detalhe h3", has_text="Esforços e 2ª ordem por direção").locator(
            "xpath=following-sibling::*[1]")
        visiveis = tabela.evaluate(
            """(elemento, rotulos) => {
                   let contagem = 0;
                   elemento.querySelectorAll('tr').forEach(linha => {
                       const primeira = linha.querySelector('td');
                       if (!primeira) { return; }
                       const mostrar = rotulos.includes(primeira.textContent.trim());
                       linha.style.display = mostrar ? '' : 'none';
                       if (mostrar) { contagem = contagem + 1; }
                   });
                   return contagem;
               }""", list(LINHAS_TELA))
        if visiveis != len(LINHAS_TELA):
            raise ValueError("auditor: %d de %d linhas encontradas na tabela" % (visiveis, len(LINHAS_TELA)))
        tabela.screenshot(path=str(tela_a))
        pagina.click("button[data-aba='formulas']")
        pagina.wait_for_timeout(ESPERA_MS)
        pagina.fill("#busca-formula", FORMULA_TELA)
        pagina.wait_for_timeout(ESPERA_MS)
        pagina.locator("#formula-%s" % FORMULA_TELA).screenshot(path=str(tela_b))
        navegador.close()
    return tela_a, tela_b


def figura_auditor(pasta: Path) -> None:
    tela_a, tela_b = capturar_telas(pasta)
    with Image.open(tela_a) as imagem_a, Image.open(tela_b) as imagem_b:
        largura, altura = tamanho_composicao((imagem_a.width, imagem_b.width), (imagem_a.height, imagem_b.height))
        composta = Image.new("RGB", (largura, altura), "white")
        composta.paste(imagem_a, (0, MARGEM_COMPOSICAO_PX))
        composta.paste(imagem_b, (0, ordenada_segunda_tela(imagem_a.height)))
        desenho = ImageDraw.Draw(composta)
        fonte = ImageFont.truetype(ARQUIVO_FONTE_ROTULO, CORPO_ROTULO_PX)
        desenho.text((0, 0), "(a)", fill="black", font=fonte)
        desenho.text((0, ordenada_rotulo_segunda_tela(imagem_a.height)), "(b)", fill="black", font=fonte)
        composta.save(pasta.joinpath("auditor.png"), dpi=(estilo.RESOLUCAO_DPI, estilo.RESOLUCAO_DPI))
    tela_a.unlink()
    tela_b.unlink()


def reta_pilar_padrao(mu1: float, inclinacao: float, abscissas) -> np.ndarray:
    """Reta dos esforços externos do pilar-padrão no plano μ x 10³ d/r ([F06] de validacao_literatura.py)."""
    import validacao_literatura as literatura
    return np.asarray([literatura.soma(mu1, literatura.produto(inclinacao, abscissa)) for abscissa in abscissas])


def figura_fusco(pasta: Path) -> list[dict]:
    import validacao_literatura as literatura
    digitalizacao = json.loads(ARQUIVO_LITERATURA.read_text(encoding="utf-8"))["digitalizacao_fusco"]
    if digitalizacao.get("situacao") != "feita":
        raise ValueError("digitalização da Fig. 8.2.5-1 não disponível: rode validacao_literatura.py com o PDF")
    material = nucleo.Material(literatura.FUSCO["fck"])
    mu1 = literatura.valor_impresso(literatura.FUSCO["impresso"]["mu1"])
    figura, eixos = plt.subplots(1, len(PAINEIS_FUSCO), sharey=True,
                                 figsize=(estilo.cm_para_polegada(estilo.LARGURA_CM),
                                          estilo.cm_para_polegada(ALTURA_FUSCO_CM)))
    dados = []
    for letra, eixo, distancia_relativa in zip("ab", eixos, PAINEIS_FUSCO):
        for omega in OMEGAS_FUSCO:
            abscissas, mus = literatura.curva_fusco(material, distancia_relativa, omega, PONTOS_FUSCO)
            dentro = abscissas <= LIMITES_X_FUSCO[1]
            eixo.plot(abscissas[dentro], mus[dentro], color=COR_PRINCIPAL, linewidth=0.8,
                      label="Rotina deste trabalho: curvas de $\\omega$ = 0,1 a 1,0 (valor de $\\omega$ à direita de cada curva)" if omega == OMEGAS_FUSCO[0] else None)
            eixo.annotate(numero(omega, 1), xy=(float(abscissas[dentro][-1]), float(mus[dentro][-1])), xytext=(2, 0),
                          textcoords="offset points", ha="left", va="center", fontsize=estilo.CORPO_CELULA_PT,
                          color=estilo.TINTA_TEXTO, annotation_clip=False)
        pontos = digitalizacao["paineis"][str(distancia_relativa)]["pontos"]
        eixo.plot([ponto["abscissa"] for ponto in pontos], [ponto["mu_figura"] for ponto in pontos], linestyle="none",
                  marker="o", markersize=3.6, markerfacecolor="white", markeredgecolor=COR_SECUNDARIA,
                  markeredgewidth=0.9, label="Fusco (1981), Fig. 8.2.5-1: pontos lidos na figura impressa")
        inclinacao = literatura.inclinacao_desenhada(literatura.FUSCO["cota_reta_x4"][distancia_relativa], 4.0)
        abscissas_reta = np.linspace(LIMITES_X_FUSCO[0], LIMITES_X_FUSCO[1], 2)
        eixo.plot(abscissas_reta, reta_pilar_padrao(mu1, inclinacao, abscissas_reta), color=COR_REFERENCIA,
                  linewidth=0.8, linestyle=(0, (4, 2)),
                  label="Fusco (1981): reta do pilar-padrão, $\\mu_{1d}$ + tg $\\theta$ (10$^{3}$ $d$/$r$), com a inclinação desenhada no livro")
        eixo.set_title("(%s) $d′/h$ = %s" % (letra, numero(distancia_relativa, 2)), fontsize=estilo.CORPO_SECUNDARIO_PT)
        eixo.set_xlim(*LIMITES_X_FUSCO)
        eixo.set_ylim(*LIMITES_MU_FUSCO)
        eixo.set_xlabel("10$^{3}$ $d$/$r$")
        eixo.grid(True, color=estilo.CINZA_GRADE, linewidth=0.4)
        eixo.set_axisbelow(True)
        formatador = matplotlib.ticker.FuncFormatter(estilo.texto_eixo)
        eixo.xaxis.set_major_formatter(formatador)
        eixo.yaxis.set_major_formatter(formatador)
        for ponto in pontos:
            dados.append({"d'/h": distancia_relativa, "ω": ponto["omega"], "10³ d/r": ponto["abscissa"],
                          "μ figura": ponto["mu_figura"], "μ rotina": ponto["mu_rotina"],
                          "diferença": ponto["diferenca"]})
    eixos[0].set_ylabel("$\\mu$")
    alcas, rotulos = eixos[0].get_legend_handles_labels()
    figura.legend(alcas, rotulos, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False,
                  fontsize=estilo.CORPO_SECUNDARIO_PT)
    figura.subplots_adjust(wspace=0.16, bottom=0.2)
    estilo.salvar(figura, pasta, "fusco_rotina")
    return dados


def main() -> None:
    argumentos = [argumento for argumento in sys.argv[1:] if argumento != "--fusco"]
    pasta = Path(argumentos[0]) if argumentos else PASTA / "saida" / "figuras"
    pasta.mkdir(parents=True, exist_ok=True)
    estilo.configurar_estilo()
    plt.rcParams["svg.fonttype"] = estilo.TEXTO_SVG_DIAGRAMAS
    gravar_csv(pasta.joinpath("fusco_rotina_dados.csv"), figura_fusco(pasta))
    if "--fusco" in sys.argv:
        print("figura das curvas de Fusco (1981) gravada em", pasta)
        return
    gravar_csv(pasta.joinpath("secao_v1_dados.csv"), figura_secao(pasta))
    gravar_csv(pasta.joinpath("curvas_mn1r_dados.csv"), figura_mn1r(pasta))
    gravar_csv(pasta.joinpath("taxa_esbeltez_dados.csv"), figura_taxa(pasta))
    figura_auditor(pasta)
    print("figuras complementares gravadas em", pasta)


if __name__ == "__main__":
    main()
