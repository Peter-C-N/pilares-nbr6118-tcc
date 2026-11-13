# -*- coding: utf-8 -*-
"""
validacao_literatura.py
=======================
Nível 2 da validação: engenharia reversa de exemplos resolvidos da literatura (além dos quatro exemplos de
Bastos, 2017, que estão em validacao.py).

Exemplos refeitos:
    L1  Kimura ([2005?]), artigo técnico da TQS: pilar de 210 x 30 cm, C35, ℓe = 600 cm, Nd = 840 tf; curvatura
        aproximada, rigidez κ aproximada, pilar-padrão acoplado a diagramas N, M, 1/r (rigidez secante do diagrama
        de programa reproduzido na figura do item C.3) e método geral.
    L2  Scadelai e Pinheiro (2007), item 16.10: pilar P5 de 15 x 45 cm, C25, ℓe = 265 cm; curvatura aproximada,
        rigidez κ aproximada (iterações do texto), ábaco A-5 de Venturini (1987) e detalhamento (barras e estribos).
    L3  Fusco (1981), itens 8.2.5 e 8.2.6: pilar P5 de 20 x 150 cm, ℓe = 560 cm, λ = 97; diagramas M, N, 1/r da
        Fig. 8.2.5-1 (curvas digitalizadas da página do livro e comparadas com as da rotina), solução do
        pilar-padrão e excentricidade suplementar de fluência.
    L4  Franco (2010), item 1.4.4: pilar de 15 x 25 cm, C25, ℓe = 255 cm; excentricidade de fluência da
        NBR 6118 (15.8.4) e rigidez secante (com o MRd e a curvatura do programa MK-UFRJ como dados).
    L5  Carvalho e Pinheiro (2009), exemplos 5.1 a 5.5 e 5.10: pilares centrais curto (5.1), medianamente
        esbelto (5.2), de 12 cm com γn (5.3 e 5.4, curvatura aproximada e rigidez κ), esbelto com fluência e
        ábaco do pilar-padrão acoplado (5.5) e pilar lateral (5.10); da esbeltez à armadura adotada.

Para cada valor impresso na fonte são registrados: o valor impresso; o valor da cadeia da fonte (as contas do
exemplo refeitas com as regras e os arredondamentos da própria fonte, com os valores impressos das etapas
anteriores); o valor da rotina (núcleo pilares_nbr6118.py, com os dados do exemplo); e o código da origem da
diferença ([R07]). Desde 03/11/2026, também o valor da cadeia exata do autor (o caminho do autor, com as mesmas
fórmulas, dados, hipóteses e edição da norma, sem arredondar as etapas intermediárias e com as constantes exatas), a
comparação da rotina com ele e a nota curta das tabelas do Apêndice F do TCC ([N01], [R09], [F17]). Critérios de
comparação ([N01]):
    "impressão": conta feita na fonte e impressa com poucas casas; reproduzido quando a cadeia da fonte,
                 arredondada às casas impressas com a meia unidade para cima (em módulo), dá o número impresso;
    "programa":  valor calculado por um programa e publicado pela fonte (figuras de Kimura, [2005?], que não
                 nomeiam o programa); sem critério de reprodução, só o desvio da rotina e a origem identificada
                 da diferença. Em Franco (2010), o MRd e a curvatura do programa MK-UFRJ entram só como dados
                 da etapa seguinte;
    "leitura":   valor lido na fonte num ábaco ou diagrama, ou valor da rotina comparado com um valor impresso
                 de outra grandeza (MRd da rotina comparado com o Md,tot impresso); sem critério de reprodução,
                 só a diferença.
Nenhum limite de aceitação é fixado (item 1 do arquivo "Pessoal 2" do autor): cada diferença é apresentada com
a sua origem.

Saídas: saida/validacao_literatura.txt (relatório passo a passo), saida/validacao_literatura.json e, quando o
PDF de Fusco (1981) está disponível, saida/fusco_curvas_sobrepostas_05.png e _15.png (curvas da rotina sobre
a Fig. 8.2.5-1 do livro).

Uso:  python3 validacao_literatura.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import pilares_nbr6118 as nucleo
import validacao

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — INÍCIO
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [R01] Arquivos: PDF de Fusco (1981) com a Fig. 8.2.5-1 (procurado na ordem da lista; a variável de ambiente
#   PDF_FUSCO tem prioridade; senão, a pasta bibliografia ao lado dos programas, em "2 Livros", com o nome do autor e
#   o ano; o livro não faz parte do material e nenhum caminho absoluto fica gravado no código, alteração de
#   13/11/2026), página do PDF com a figura e saídas. O relatório registra só o nome do arquivo do PDF.
#   Variáveis: PASTA_SAIDA, CAMINHOS_PDF_FUSCO, PAGINA_PDF_FIGURA_FUSCO, ARQUIVO_TEXTO, ARQUIVO_JSON,
#   PREFIXO_SOBREPOSICAO.
#   Fonte: Fusco (1981), p. 288 (página 298 do arquivo PDF).
#   Usada em: nível do módulo (l. 83, 84); caminho_sobreposicao (l. 766); digitalizar_fusco (l. 1606, 1611, 1618);
#       escrever_relatorio (l. 2678, 2679); main (l. 2685, 2690, 2691)
PASTA_SAIDA = Path(__file__).resolve().parent / "saida"
CAMINHOS_PDF_FUSCO = (os.environ.get("PDF_FUSCO", ""),
                      str(Path(__file__).resolve().parent / "bibliografia" / "2 Livros" /
                          "FUSCO (1981) - Estruturas de concreto - solicitações normais.pdf"))
PAGINA_PDF_FIGURA_FUSCO = 298
ARQUIVO_TEXTO = PASTA_SAIDA / "validacao_literatura.txt"
ARQUIVO_JSON = PASTA_SAIDA / "validacao_literatura.json"
PREFIXO_SOBREPOSICAO = "fusco_curvas_sobrepostas_"

# [N01] Critério de reprodução dos valores impressos (critério "impressão"): a cadeia da fonte reproduz o valor
#   impresso quando, arredondada às casas impressas com a meia unidade para cima, em módulo (como fazem os
#   autores: 38,635 -> 38,64; 0,775 -> 0,78; 0,1615 -> 0,162), dá o número impresso, isto é, quando
#   |impresso| - m (1 + FOLGA_PONTO_FLUTUANTE) <= |cadeia| < |impresso| + m (1 - FOLGA_PONTO_FLUTUANTE), com o
#   mesmo sinal, em que m = FRACAO_RESOLUCAO x resolução (meia unidade). A resolução é a unidade do último
#   algarismo impresso, salvo quando informada no próprio dado (zeros finais de números inteiros, como 1890 ou
#   9800, que o autor arredonda a três algarismos significativos). A meia unidade exata acima do impresso não
#   reproduz (0,1615 não dá 0,161); a meia unidade exata abaixo reproduz (38,635 dá 38,64).
#   FOLGA_PONTO_FLUTUANTE (relativa) absorve o erro de representação binária desses valores (38,635 é guardado
#   como 38,63499...; 0,1615 pode ser guardado acima ou abaixo do valor decimal).
#   Critérios das linhas: "impressão" (valor impresso contado e conferido pela cadeia do autor); "programa" (valor
#   calculado por programa e publicado pelo autor) e "leitura" (valor lido em ábaco ou figura, ou valor da rotina
#   comparado com um valor impresso de outra grandeza, como o MRd da rotina comparado com o Md,tot impresso, o que
#   a nota da linha informa), comparados só pela diferença; "conferência" (valor impresso já contado em outra linha,
#   repetido para conferir uma fórmula da rotina com as entradas impressas do autor; não entra na contagem).
#   Cadeia exata do autor (segunda definição, gravada ao lado da primeira para comparação, nos campos "exata",
#   "reproduz_exata", "origem_exata" e "nota_exata", só nas linhas de critério "impressão"): as contas do autor
#   (mesmas fórmulas, dados, hipóteses e edição da norma que ele usou) refeitas sem as aproximações dele, isto é,
#   sem arredondar nenhuma etapa intermediária (cada etapa usa o valor completo calculado na etapa anterior da
#   própria cadeia, e não o impresso) e com constantes exatas (raiz(12) no lugar de 3,46; frações exatas, como
#   fcd = fck/1,4, fyd = fyk/1,15 e θ1 = 1/200; área de barra π φ²/4, a área nominal da ABNT NBR 7480, no lugar de
#   0,8, 1,25, 2,0 e 3,15 cm²). Leituras em ábaco ou figura e valores de programa publicados entram como dados.
#   Reproduz pelo mesmo arredondamento com a meia unidade para cima, em módulo ([F01]); a origem da não reprodução
#   vem de [R08] e a nota, da etapa.
#   Variáveis: FRACAO_RESOLUCAO, FOLGA_PONTO_FLUTUANTE, CRITERIO_IMPRESSAO, CRITERIO_PROGRAMA, CRITERIO_LEITURA,
#   CRITERIO_CONFERENCIA.
#   Usada em: limite_reproducao (l. 358); limite_inferior_reproducao (l. 362); limite_superior_reproducao (l. 366);
#       codigo_origem (l. 935, 937); Registro.adicionar (l. 1097, 1105);
#       executar_kimura (l. 1218, 1220, 1224, 1230, 1234, 1235, 1239, 1245, 1253, 1257, 1273, 1284, 1289, 1305, 1328, 1332, 1335, 1338, 1340, 1343, 1345, 1349, 1353, 1366);
#       executar_scadelai (l. 1395, 1396, 1397, 1399, 1401, 1403, 1406, 1409, 1414, 1423, 1430, 1434, 1438, 1452, 1460, 1463, 1472, 1504, 1508, 1512, 1519, 1531, 1537, 1547, 1550, 1553, 1557);
#       digitalizar_fusco (l. 1682);
#       executar_fusco (l. 1701, 1703, 1710, 1719, 1725, 1740, 1746, 1760, 1763, 1767, 1773, 1781, 1794, 1805, 1810, 1820, 1823);
#       executar_franco (l. 1849, 1853, 1857, 1864, 1875, 1890, 1897, 1905, 1916, 1921, 1937);
#       executar_carvalho (l. 2033, 2035, 2039, 2052, 2056, 2060, 2066, 2069, 2073, 2076, 2077, 2084, 2088, 2093, 2096, 2100, 2127, 2133, 2136, 2140, 2144, 2147, 2151, 2156, 2166, 2171, 2186, 2204, 2215, 2217, 2219, 2221, 2225, 2230, 2237, 2241, 2245, 2250, 2252, 2254, 2257, 2275, 2282, 2297, 2300, 2307, 2313, 2316, 2321, 2339, 2351, 2358, 2370, 2372, 2379, 2383, 2388, 2391, 2406, 2418, 2426, 2430, 2431, 2440, 2449, 2455, 2462, 2465, 2468, 2473, 2500, 2504, 2510, 2512, 2520, 2525, 2530, 2537, 2540, 2550, 2559, 2561, 2571, 2577)
FRACAO_RESOLUCAO = 0.5
FOLGA_PONTO_FLUTUANTE = 1e-9
CRITERIO_IMPRESSAO = "impressão"
CRITERIO_PROGRAMA = "programa"
CRITERIO_LEITURA = "leitura"
CRITERIO_CONFERENCIA = "conferência"

# [C01] Conversões de unidade. O documento da TQS usa tf = 10 kN (conferido no próprio texto: fck = 35 MPa é
#   escrito como 3500 tf/m², isto é, 35.000 kN/m² = 3.500 tf/m² com 1 tf = 10 kN).
#   Milésimos: Franco (2010, p. 34) escreve a curvatura do programa MK-UFRJ como 1/r = 27,4979/1000 (m⁻¹).
#   Variáveis: KN_POR_TF, CM_POR_M, MILESIMOS_POR_UNIDADE.
#   Fonte: Kimura ([2005?]), item A (dados do exemplo); Franco (2010, p. 34).
#   Usada em: tf_para_kn (l. 398); tfm_para_kncm (l. 402); kncm_para_tfm (l. 406); knm_para_kncm (l. 410);
#       kncm_para_knm (l. 414); por_metro_para_por_cm (l. 418); por_cm_para_por_metro (l. 422); cm_para_m (l. 426);
#       m_para_cm (l. 430); kncm2_para_tfm2 (l. 434); kncm2_para_knm2 (l. 442); milesimos_para_unidade (l. 446);
#       cm4_para_m4 (l. 450)
KN_POR_TF = 10.0
CM_POR_M = 100.0
MILESIMOS_POR_UNIDADE = 1000.0

# [R02] L1 — Kimura ([2005?]): pilar do mezanino, seção 210 x 30 cm, pé-direito duplo de 6,0 m na direção menos
#   rígida (ℓe = 600 cm), C35, CA-50, γc = 1,4, γs = 1,15, Es = 210.000 MPa, γf = 1,4, γf3 = 1,1, c = 2,5 cm,
#   Nd = 840 tf, M1d,A = 16,8 tf·m no topo e -2,8 tf·m na base (ambos menores que M1d,mín, logo αb = 1,0).
#   Armadura do item C.3: 16 φ16 (N2), 8 em cada face de 210 cm, estribo φ6,3 (N3), conforme o desenho da seção;
#   d' = c + φt + φl/2 = 3,93 cm ([F46] do núcleo). Valores impressos (texto) do passo a passo da NBR 6118:2003:
#   e1,mín, M1d,mín, λ (o item C repete o valor do item B, escrito como 3,46 x 6/0,3 = 69,3), λ1 (com ei = 1680/840 =
#   2,00 cm), ν, As,mín; C.1 (curvatura aproximada): 1/r, e2, etot, Md,tot; C.2 (rigidez κ aproximada): κ, e2,
#   Md,tot; C.3 (diagrama N, M, 1/r de programa, na figura do item, com as curvas de 1,1 fcd e 0,85 fcd): MRd,
#   ponto B da curva com 1,10 fcd (M = MRd/γf3 = 51,6 tf·m em 1/r = 5,85E-3 m⁻¹, guardado como 0.00585),
#   EIsec = 8817,8 tf·m², κ = 62,2 (lidos na figura), e2, Md,tot; C.4 (método geral): e2, Md,tot.
#   O artigo não nomeia o programa das figuras (o único citado é o TQS-Pilar, no item E). As seções 15.8.3.3.2 e 15.8.3.3.3 e a Figura 15.1 da NBR 6118:2003
#   têm o mesmo conteúdo das da edição de 2023 nos itens usados.
#   Variáveis: KIMURA (dicionário com dados e impressos; forças em tf, momentos em tf·m e comprimentos em cm).
#   Fonte: Kimura ([2005?]), "NBR6118:2003 - Mais sobre o dimensionamento de pilares", TQS Docs, Artigos
#   Técnicos (https://docs.tqs.com.br/Docs/Details?id=3335), itens A, C.1 a C.4 e D. A página não traz data; o
#   texto diz que a NBR 6118:2003 vai completar um ano de vigência exclusiva em 30 de março e cita o ENECE 2004.
#   Usada em: executar_kimura (l. 1196)
KIMURA = {
    "fonte": "Kimura ([2005?]), TQS Docs, itens A a D", "largura": 210.0, "altura": 30.0, "comprimento": 600.0,
    "fck": 35.0, "Nd_tf": 840.0, "e_inicial": 2.00, "cobrimento": 2.5, "estribo": 0.63, "barra": 1.6,
    "barras_por_face": 8,
    "impresso": {"e1_min": "2.40", "M1d_min": "20.2", "lambda": "69.3", "lambda1_calculado": "25.8",
                 "nu": "0.53", "As_min": "29", "curvatura": "0.016", "e2_curvatura": "5.81",
                 "etot_curvatura": "8.21", "Md_curvatura": "68.9", "kappa": "34.8", "e2_kappa": "3.82",
                 "Md_kappa": "52.3", "As_adotada": "32.2", "MRd": "56.8", "M_ponto_B": "51.6",
                 "curvatura_ponto_B": "0.00585", "EI_sec": "8817.8", "kappa_sec": "62.2", "e2_mn1r": "1.26",
                 "Md_mn1r": "30.7", "e2_geral": "1.52", "Md_geral": "33.0"},
}

# [R03] L2 — Scadelai e Pinheiro (2007), item 16.10: pilar P5 intermediário-extremidade, 15 x 45 cm, C25, CA-50,
#   cnom = 2,5 cm, d' = 4,0 cm, Nk = 650 kN, γn = 1,20 (b = 15 cm), Nd = γf γn Nk = 1092 kN, ℓe = ℓ0 + h =
#   250 + 15 = 265 cm (<= ℓ = 290 cm), momentos de topo e base Md = ±18,85 kN·m (menores que M1d,mín, logo
#   αb = 1,0 e λ1 com e1 = 0), flexão na direção de 15 cm. Ábaco A-5 de Venturini (1987): duas camadas,
#   d'/h = 0,25 (o texto adota 0,25 para d'/h = 0,27). Armadura adotada: 12 φ16 (6 por face de 45 cm), estribos
#   φ5 c/15 e estribos suplementares nas 4 barras centrais (20 φt = 10 cm). Rigidez κ: iterações impressas
#   (κ/ν e Md,tot de cada iteração, com a média dos dois últimos valores como nova estimativa).
#   Variáveis: SCADELAI (dados e impressos; forças em kN, momentos em kN·m, comprimentos em cm).
#   Fonte: Scadelai e Pinheiro (2007), p. 16.23-16.31 (Estruturas de concreto, cap. 16).
#   Usada em: executar_scadelai (l. 1376)
SCADELAI = {
    "fonte": "Scadelai e Pinheiro (2007), p. 16.23-16.31", "largura": 45.0, "altura": 15.0, "comprimento": 265.0,
    "fck": 25.0, "Nk": 650.0, "gama_n": 1.20, "M_topo_k": 11.22, "distancia": 4.0, "barra": 1.6,
    "barras_por_face": 6, "barras_adotadas": 12, "barra_alternativa": 2.0, "estribo": 0.5,
    "distancia_abaco": 0.25,
    "impresso": {"Nd": "1092", "nu": "0.91", "raio_giracao": "4.33", "lambda": "61.2", "Md_topo": "18.85",
                 "e_inicial": "1.73", "M1d_min": "21.29", "lambda1": "35", "curvatura": "0.0236",
                 "Md_curvatura": "39.39", "etot_curvatura": "3.61", "mu_curvatura": "0.22", "fator_As": "27.72",
                 "omega_curvatura": "0.90", "As_curvatura": "24.95", "rho_curvatura": "3.70",
                 "kappa_nu_1": "70.48", "Md_kappa_11": "38.21", "Md_kappa_20": "38.80", "kappa_nu_2": "69.90",
                 "Md_kappa_21": "38.47", "Md_kappa_30": "38.64", "etot_kappa": "3.54", "mu_kappa": "0.21",
                 "omega_kappa": "0.88", "As_kappa": "24.39", "rho_kappa": "3.61", "As_adotada": "24.0",
                 "estribo": "5", "espacamento_estribo": "15", "trecho_protegido": "10",
                 "barras_sem_protecao": "4"},
}

# [R04] L3 — Fusco (1981), itens 8.2.5 e 8.2.6: pilar P5 interno, seção 20 x 150 cm (flexão na direção de 20 cm),
#   ℓe = 560 cm, fck = 15 MPa, γc = 1,4, aço CA-50A (diagrama elastoplástico, como o da rotina), fyd = 435 MPa,
#   Nd = 2534 kN, e1 = ea = 2 cm (excentricidade acidental da NB-1/1978). Item 8.2.5: diagramas M, N, 1/r da
#   Fig. 8.2.5-1 para ν = 0,8, armadura em duas camadas com d'/h = 0,05 e 0,15, ω de 0 a 1,0 (μ = M/(Ac h fcd) em
#   ordenada; 10³ d/r em abscissa; a figura imprime o aço CA 50-A, d'/h e ν), refeitas com seção bruta, concreto com
#   0,85 fcd e força normal Nd (sem γf3), hipóteses não impressas, confirmadas pela digitalização; reta do
#   pilar-padrão μ = μ1d + tg θ (10³ d/r), tg θ = ν/10000 (ℓe/h)² (h/d); leituras ω = 0,65 (d/h = 0,95) e 0,90
#   (d/h = 0,85), interpolação ω = 0,78 para d/h = 0,90 e As = 57,6 cm². As cotas impressas na figura (0,264 e
#   0,280 em 10³ d/r = 4) dão as inclinações efetivamente desenhadas. Item 8.2.6: Nk = 1810 kN, carga de longa
#   duração 0,95 Nk, γn = 1,10, Fg = 1890 kN, φ = 2, e1g = ea = 2 cm, Ec = 0,9 x 6600 raiz(fck + 3,5) MPa,
#   As = 2 x 8 φ20 = 50,4 cm² (3,15 cm² por barra; na cadeia exata, 16 π 2,0²/4 = 50,27 cm²), d' = 3 cm,
#   Es = 210.000 MPa, Ii = Ic + α As (h/2 - d')², FE = 10 Ec Ii/ℓe²,
#   ec = e1g [exp(φ Fg/(FE - Fg)) - 1]; e1x = 3,19 cm, μ1d = 0,13; ω = 0,92 lido nos diagramas de interação de
#   barras esbeltas da Fig. 5.3.5-5 (outro método) e As = 67,9 cm².
#   Variáveis: FUSCO (dados e impressos; kN, cm, MPa onde indicado).
#   Fonte: Fusco (1981), p. 283-292 (páginas 293-302 do PDF), Fig. 8.2.5-1 (p. 288).
#   Usada em: curva_fusco (l. 1571); digitalizar_fusco (l. 1636, 1638); executar_fusco (l. 1689)
#   Usada também em: figuras_complementares.py: figura_fusco (l. 715, 716, 734)
FUSCO = {
    "fonte": "Fusco (1981), p. 283-292", "largura": 150.0, "altura": 20.0, "comprimento": 560.0, "fck": 15.0,
    "Nd": 2534.0, "e1": 2.0, "nu_diagrama": 0.80, "distancias_relativas": (0.05, 0.15),
    "fcd_impresso": 1.07, "fyd_impresso": 43.5, "cota_reta_x4": {0.05: 0.264, 0.15: 0.280},
    "omega_lido": {0.05: 0.65, 0.15: 0.90}, "relacao_d_interpolada": 0.90, "Nk": 1810.0,
    "fracao_longa_duracao": 0.95, "gama_n": 1.10, "phi": 2.0, "As_fluencia": 50.4, "distancia_fluencia": 3.0,
    "modulo_aco_mpa": 210000.0, "barras_fluencia": 16, "barra_fluencia": 2.0,
    "impresso": {"nu": "0.79", "mu1": "0.08", "tg_095": "0.066", "tg_085": "0.074", "omega_095": "0.65",
                 "omega_085": "0.90", "omega_090": "0.78", "As_090": "57.6", "Fg": "1890", "Ec": "25550",
                 "alfa": "8.22", "Ii": "120300", "FE": "9800", "ec": "1.19", "e1x": "3.19", "mu1_fluencia": "0.13",
                 "omega_fluencia": "0.92", "As_fluencia": "67.9"},
    "resolucao": {"Fg": 10.0, "Ec": 10.0, "Ii": 100.0, "FE": 100.0},
}

# [R05] L4 — Franco (2010), item 1.4.4 (2ª forma de consideração da fluência): seção 15 x 25 cm (flexão na
#   direção de 15 cm), C25, ℓe = 255 cm, Nd = 473,8 kN, e1,mín = 0,0195 m, φ = 2. O autor toma Msg = M1d,mín/1,4 e
#   Nsg = Nd/1,4 (toda a carga como quase permanente), Eci = 5600 raiz(25) = 28.000 MPa, Ic = 7,03E-5 m⁴
#   (0,25 x 0,15³/12 = 7,031E-5), θ1 = θ1,máx = 1/200 (escrito como o ângulo 0,2866°; theta1_autor, em radianos,
#   na cadeia exata) e ea = tg θ1 x ℓe/2.
#   Valores impressos: M1d,mín (p. 21; 9,23 = 9,2391 truncado), Msg, Nsg, Ne, ea, ecc; rigidez secante do item
#   1.4.4 (p. 34), com o MRd = 26,7 kN·m e a curvatura 1/r = 27,4979/1000 m⁻¹ do diagrama momento-curvatura do
#   anexo 1 (saída do programa MK-UFRJ, p. 23), que entram como dados: MRd/1,1 = 24,3 kN·m, (EI)sec = 882,7 kN·m² e
#   κsec = 58,59; M1d = Nd (e1,mín + ecc) e, com ν = 0,707 e λ = 58,89, Md,tot = 19,56 kN·m pela expressão da
#   rigidez κ.
#   Variáveis: FRANCO (dados e impressos; kN, m, kN·m).
#   Fonte: Franco (2010), p. 21 e 33-34.
#   Usada em: executar_franco (l. 1838)
FRANCO = {
    "fonte": "Franco (2010), p. 21 e 33-34", "largura": 25.0, "altura": 15.0, "comprimento": 255.0, "fck": 25.0,
    "Nd": 473.8, "e1_min": 0.0195, "phi": 2.0, "gama_f": 1.4, "angulo_graus": 0.2866, "Ic_impresso": 7.03e-5,
    "nu_impresso": 0.707, "lambda_impresso": 58.89,
    "MRd_programa": 26.7, "curvatura_programa_milesimos": 27.4979, "theta1_autor": 1.0 / 200.0,
    "impresso": {"M1d_min": "9.23", "Msg": "6.60", "Nsg": "338.4", "Ne": "3027", "ea": "0.006378", "ecc": "0.00741",
                 "M_ponto_B": "24.3", "EI_sec": "882.7", "kappa_sec": "58.59",
                 "M1d": "12.74", "Md_kappa_sec": "19.56"},
}

# [R06] L5 — Carvalho e Pinheiro (2009), Capítulo 5 (NBR 6118:2003). Pilares centrais com P = Nk = 200 kN, C20,
#   CA-50, cobrimento de 2 cm, estribo de 5 mm e barra de 10 mm (d' = 3 cm), pé-direito ℓp = 2,70 m mais 0,08 m de
#   revestimentos (ℓ0 = 2,78 m) e flexão só pela excentricidade mínima (e1 = 0, αb = 1):
#     5.1  seção 30 x 30 cm (pilar curto do texto), ℓe = 2,78 + 0,30 = 3,08 m, ábaco 2 (d'/h = 0,10);
#     5.2  seção 20 x 20 cm (medianamente esbelto), ℓe = 2,78 + 0,20 = 2,98 m, ábaco 3 (d'/h = 0,15);
#     5.3  seção 12 x 30 cm com γn = 1,35 (Tabela 13.1 da edição de 2003, que admitia 12 cm), ℓe,x = 2,90 m e
#          ℓe,y = 3,08 m, ábaco 5 (d'/h = 0,25) na direção de 12 cm e ábaco 6 na de 30 cm;
#     5.4  o pilar do 5.3 pela rigidez κ aproximada, com a expressão fechada 5.29 do livro;
#     5.5  seção 20 x 20 cm, ℓe = 6,0 m, C30, cobrimento de 1 cm (d' = 2 cm), Ngk = Nsg = 120 kN, φ = 2, θ1 = 1/200,
#          Ec = 0,85 x 5600 raiz(fck) e ábaco do pilar-padrão da Fig. 5.20 (de Fusco, 1995; impressos: aço
#          CA-50A, d'/h = 0,10 e As/2 em cada face; diagrama do concreto não impresso, refeito com 0,85 fcd);
#     5.10 pilar lateral P1 de 20 x 40 cm, C20, N = 816 kN, momento de 1ª ordem de 12,11 kN·m na direção de 20 cm
#          (ligação viga-pilar), αb = 0,4 adotado pelo texto, ℓe,x = 3,40 m, ℓe,y = 3,50 m, ℓ = 3,50 m e ábaco 5
#          (d'/h = 5/20 = 0,25), situação 1 (seção de extremidade, e = e1 + ea).
#   Os ábacos 1 a 5 do livro são de Pinheiro (1993), com armadura simétrica em duas faces (conferido no ábaco 2,
#   reproduzido no livro); as páginas dos ábacos 4 a 10 não constam do arquivo. Valores impressos em m, kN e kN·m.
#   Variáveis: CARVALHO (dados em cm, kN e MPa; impressos como no livro).
#   Fonte: Carvalho e Pinheiro (2009), p. 350-372 e 393-398.
#   Usada em: executar_carvalho (l. 2016)
CARVALHO = {
    "fonte": "Carvalho e Pinheiro (2009), p. 355-372 e 393-398",
    "fck": 20.0, "fyk": 500.0, "Nk": 200.0, "cobrimento": 2.0, "estribo": 0.5, "barra": 1.0, "pe_direito": 270.0,
    "revestimentos": 8.0, "lado_51": 30.0, "lado_52": 20.0, "esbeltez_curto": 35.0, "esbeltez_media": 90.0,
    "barras_adotadas_51": 4, "area_barra_autor_10": 0.8,
    "largura_53": 30.0, "altura_53": 12.0, "barra_53": 1.25, "barras_53": 12, "barras_por_face_53": 6,
    "area_barra_autor_125": 1.25,
    "fck_55": 30.0, "lado_55": 20.0, "comprimento_55": 600.0, "cobrimento_55": 1.0, "Nsg_55": 120.0, "phi_55": 2.0,
    "theta1_autor": 1.0 / 200.0, "inercia_autor_55_m4": 1.33e-4, "barras_55": 4, "barra_55": 1.25,
    "N_510": 816.0, "M1_510_knm": 12.11, "b_510": 20.0, "h_510": 40.0, "comprimento_x_510": 340.0,
    "comprimento_y_510": 350.0, "comprimento_510": 350.0, "alfa_b_510": 0.4, "distancia_abaco_510": 5.0,
    "distancia_abaco_53": 3.0, "distancia_abaco_51": 3.0, "distancia_abaco_52": 3.0, "distancia_abaco_55": 2.0,
    "impresso": {
        "le_51": "3.08", "fator_le_51": "10.66", "lado_min_51": "0.304", "lambda1_51": "25", "ea_51": "0.0077",
        "ea_ext_51": "0.0154", "e1min_51": "0.024", "Mk_51": "4.8", "dlinha_51": "0.1", "nu_51": "0.22",
        "mu_51": "0.018", "omega_51": "0", "rho_min_51": "0.0011", "As_min_51": "3.6", "As_4phi10": "3.2",
        "reducao_51": "11",
        "lado_min_52": "0.118", "le_52": "2.98", "lambda_52": "52", "e1min_52": "0.021", "nu_52": "0.49",
        "e2_52": "0.022", "dlinha_52": "0.15", "mu_52": "0.105", "omega_52": "0", "rho_min_52": "0.0011",
        "As_min_52": "1.6",
        "le_x_53": "2.90", "lambda_x_53": "84", "le_y_53": "3.08", "lambda_y_53": "35.5", "eax_53": "0.00725",
        "eax_ext_53": "0.0145", "e1x_min_53": "0.018", "eay_53": "0.0077", "e1y_min_53": "0.024", "gama_n_53": "1.35",
        "nu_53": "0.735", "e2x_53": "0.028", "e2y_53": "0.013", "dlinha_x_53": "0.25", "mu_x_53": "0.28",
        "omega_x_53": "1.17", "As_x_53": "13.84", "mu_y_53": "0.091", "omega_y_53": "0.25", "As_adot_53": "15",
        "rho_53": "0.0036", "As_min_53": "1.44",
        "k1_54": "-0.8375", "e2_54": "0.0292",
        "lambda_55": "104", "dlinha_55": "0.1", "ea_55": "0.015", "ea_ext_55": "0.03", "e1min_55": "0.021",
        "Ec_55": "26071", "Ne_55": "963", "ecc_55": "0.005", "nu_55": "0.33", "mu1_55": "0.043", "le_h_55": "30",
        "omega_55": "0.2", "As_55": "3.94", "rho_55": "0.01", "rho_min_55": "0.0024", "As_adot_55": "5",
        "lambda_y_510": "30", "lambda_x_510": "59", "e1x_510": "0.0148", "lambda1_x_510": "65", "theta_510": "187",
        "ea_ext_510": "0.0175", "ea_int_510": "0.00875", "emin_x_510": "0.021", "emin_y_510": "0.027",
        "ex_510": "3.23", "nu_510": "1.0", "mu_510": "0.161", "omega_510": "0.70"},
}

# [N02] Digitalização da Fig. 8.2.5-1 de Fusco (1981): a página é convertida em imagem a DPI_DIGITALIZACAO; cada
#   painel é recortado pelas frações RECORTES_PAINEIS (topo, base, esquerda, direita da página); a grade principal
#   é detectada pelas linhas de pixels escuros (abaixo de LIMIAR_ESCURO) que ocupam mais de FRACAO_LINHA_GRADE da
#   faixa central do painel; os eixos são calibrados por regressão linear nas linhas da grade (x = 1 a 4 nas
#   quatro últimas colunas; μ de 0,1 em 0,1 nas linhas, a última sendo μ = 0). Em cada abscissa de
#   ABSCISSAS_AMOSTRA (fora das linhas da grade), procura-se a curva impressa de cada ω na janela de
#   ±JANELA_MU em torno do μ da rotina, depois de apagar as faixas de ±MARGEM_GRADE pixels das linhas da grade
#   (de 0,02 em 0,02); o ponto só é registrado se houver um único traço na janela e se nenhuma das duas retas do
#   pilar-padrão passar dentro dela. ESPACAMENTO_GRADE_MENOR é a menor divisão da grade em μ (0,02).
#   Variáveis: DPI_DIGITALIZACAO, LIMIAR_ESCURO, FRACAO_LINHA_GRADE, RECORTES_PAINEIS, ABSCISSAS_AMOSTRA,
#   JANELA_MU, MARGEM_GRADE, ESPACAMENTO_GRADE_MENOR, OMEGAS_FIGURA, PONTOS_CURVA_FIGURA.
#   Usada em: valores_grade_menor (l. 724); intervalo_margem (l. 740); minimo_pixels_linha (l. 744);
#       dentro_janela (l. 748); omega_pilar_padrao_fusco.folga (l. 1588); rasterizar_pagina (l. 1597);
#       digitalizar_fusco (l. 1619, 1622, 1642, 1643, 1647); omega_tangente_abaco.folga (l. 1977)
DPI_DIGITALIZACAO = 300
LIMIAR_ESCURO = 120
FRACAO_LINHA_GRADE = 0.6
RECORTES_PAINEIS = {0.05: (0.33, 0.80, 0.04, 0.50), 0.15: (0.33, 0.80, 0.50, 1.00)}
ABSCISSAS_AMOSTRA = (2.1, 2.6, 3.1, 3.9)
JANELA_MU = 0.012
MARGEM_GRADE = 4
ESPACAMENTO_GRADE_MENOR = 0.02
OMEGAS_FIGURA = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
PONTOS_CURVA_FIGURA = 400


# [F01] Valor impresso, resolução e critério de reprodução ([N01]): valor = número do texto; resolução = unidade
#   do último algarismo (ou a resolução informada no dado); meia unidade m = FRACAO_RESOLUCAO x resolução (gravada
#   em "limite" no JSON); faixa de reprodução, em módulo: limite inferior |impresso| - m (1 + FOLGA_PONTO_FLUTUANTE)
#   (incluído) e limite superior |impresso| + m (1 - FOLGA_PONTO_FLUTUANTE) (excluído); sinais compatíveis: mesmo
#   sinal, ou impresso nulo; reproduz quando os sinais são compatíveis e limite inferior <= |autor| < limite
#   superior (arredondamento com a meia unidade para cima, em módulo). Desvio relativo da rotina em relação ao
#   impresso: (rotina - impresso)/impresso x 100 (%).
#   Variáveis: texto, resolucao_informada, resolucao, autor, impresso, meia_unidade, rotina.
#   Usada em: reproduz (l. 374, 375, 376); desvio_rotina_exata (l. 1025, 1026);
#       Registro.adicionar (l. 1092, 1098, 1100, 1102, 1103); Registro.adicionar_cadeia_exata (l. 1129);
#       executar_kimura (l. 1216, 1217, 1243, 1254, 1256, 1268, 1271, 1361, 1362, 1364);
#       executar_scadelai (l. 1385, 1387, 1418, 1421, 1429, 1431, 1449, 1458, 1459, 1464, 1474, 1475, 1477, 1478, 1479, 1481, 1482, 1509, 1513);
#       digitalizar_fusco (l. 1638, 1674);
#       executar_fusco (l. 1702, 1712, 1728, 1732, 1747, 1752, 1765, 1769, 1778, 1783, 1784, 1806, 1811, 1824, 1826);
#       executar_franco (l. 1854, 1885, 1886, 1902, 1911, 1918, 1927, 1929);
#       executar_carvalho (l. 2040, 2070, 2078, 2086, 2089, 2101, 2102, 2128, 2145, 2157, 2158, 2168, 2170, 2175, 2183, 2258, 2259, 2265, 2266, 2267, 2268, 2270, 2283, 2285, 2301, 2302, 2317, 2337, 2346, 2400, 2413, 2414, 2423, 2424, 2432, 2433, 2443, 2446, 2447, 2460, 2463, 2466, 2469, 2514, 2551, 2552, 2562, 2574)
#   Usada também em: figuras_complementares.py: figura_fusco (l. 716)
def valor_impresso(texto: str) -> float:
    return float(Decimal(texto))


def resolucao_impressa(texto: str, resolucao_informada: float | None) -> float:
    if resolucao_informada is not None:
        return resolucao_informada
    return float(Decimal(1).scaleb(Decimal(texto).as_tuple().exponent))


def limite_reproducao(resolucao: float) -> float:
    return FRACAO_RESOLUCAO * resolucao


def limite_inferior_reproducao(impresso: float, meia_unidade: float) -> float:
    return abs(impresso) - meia_unidade * (1.0 + FOLGA_PONTO_FLUTUANTE)


def limite_superior_reproducao(impresso: float, meia_unidade: float) -> float:
    return abs(impresso) + meia_unidade * (1.0 - FOLGA_PONTO_FLUTUANTE)


def sinais_compativeis(autor: float, impresso: float) -> bool:
    return impresso == 0.0 or (autor > 0.0) == (impresso > 0.0)


def reproduz(autor: float, impresso: float, meia_unidade: float) -> bool:
    return (sinais_compativeis(autor, impresso)
            and limite_inferior_reproducao(impresso, meia_unidade) <= abs(autor)
            < limite_superior_reproducao(impresso, meia_unidade))


def desvio_relativo(valor: float, referencia: float) -> float:
    return (valor - referencia) / referencia * 100.0


def diferenca(valor: float, referencia: float) -> float:
    return valor - referencia


# [F02] Conversões: tf -> kN e tf·m -> kN·cm (KN_POR_TF); m -> cm e kN·m -> kN·cm (CM_POR_M); 1/m -> 1/cm;
#   kN·cm² -> tf·m²; cm -> mm; kN/cm² -> kN/m²; milésimos -> unidades (MILESIMOS_POR_UNIDADE); cm⁴ -> m⁴ (momento
#   de inércia na cadeia exata de Franco, 2010, e do exemplo 5.5 de Carvalho e Pinheiro, 2009); e as inversas
#   para comparar com os valores impressos.
#   Variáveis: valor.
#   Usada em: executar_kimura (l. 1200, 1221, 1222, 1240, 1241, 1243, 1249, 1250, 1258, 1259, 1263, 1264, 1265, 1268, 1282, 1290, 1294, 1297, 1300, 1329, 1333, 1335, 1339, 1343, 1349, 1353, 1361, 1366, 1369);
#       executar_scadelai (l. 1389, 1402, 1407, 1408, 1415, 1416, 1418, 1421, 1424, 1425, 1426, 1428, 1431, 1474, 1479, 1492, 1495, 1498, 1499, 1501, 1502, 1503, 1509, 1532, 1537, 1548);
#       executar_franco (l. 1859, 1860, 1863, 1870, 1872, 1873, 1887, 1888, 1896, 1912, 1913, 1914, 1915, 1919, 1920);
#       texto_barras (l. 2011);
#       executar_carvalho (l. 2034, 2036, 2037, 2041, 2046, 2048, 2053, 2054, 2067, 2068, 2072, 2079, 2113, 2129, 2131, 2134, 2141, 2142, 2145, 2148, 2149, 2150, 2159, 2165, 2192, 2193, 2216, 2220, 2226, 2227, 2229, 2232, 2233, 2238, 2240, 2242, 2243, 2251, 2253, 2260, 2264, 2303, 2306, 2345, 2352, 2354, 2374, 2380, 2389, 2390, 2400, 2401, 2404, 2405, 2419, 2423, 2427, 2434, 2436, 2437, 2459, 2484, 2508, 2511, 2514, 2521, 2522, 2526, 2528, 2531, 2532, 2534, 2538, 2539, 2541, 2542, 2545, 2551, 2552)
def tf_para_kn(valor: float) -> float:
    return valor * KN_POR_TF


def tfm_para_kncm(valor: float) -> float:
    return valor * KN_POR_TF * CM_POR_M


def kncm_para_tfm(valor: float) -> float:
    return valor / (KN_POR_TF * CM_POR_M)


def knm_para_kncm(valor: float) -> float:
    return valor * CM_POR_M


def kncm_para_knm(valor: float) -> float:
    return valor / CM_POR_M


def por_metro_para_por_cm(valor: float) -> float:
    return valor / CM_POR_M


def por_cm_para_por_metro(valor: float) -> float:
    return valor * CM_POR_M


def cm_para_m(valor: float) -> float:
    return valor / CM_POR_M


def m_para_cm(valor: float) -> float:
    return valor * CM_POR_M


def kncm2_para_tfm2(valor: float) -> float:
    return valor / (KN_POR_TF * CM_POR_M ** 2)


def cm_para_mm(valor: float) -> float:
    return valor * 10.0


def kncm2_para_knm2(valor: float) -> float:
    return valor * CM_POR_M ** 2


def milesimos_para_unidade(valor: float) -> float:
    return valor / MILESIMOS_POR_UNIDADE


def cm4_para_m4(valor: float) -> float:
    return valor / CM_POR_M ** 4


# [F03] Grandezas adimensionais e armadura: μ = M/(Ac h fcd); ω = As fyd/(Ac fcd); As = ω Ac fcd/fyd; fator
#   Ac fcd/fyd; ρ = As/Ac (%); excentricidade e = M/N; μ1d = ν e1/h; raio de giração i = h/raiz(12); produto
#   de coeficientes (Nd = γf γn Nk); profundidade da camada oposta h - d' e profundidades de duas camadas iguais
#   (d' e h - d'); comprimento de face entre eixos das
#   barras de canto b - 2 d'; total de barras de n faces; λ1 sem os limites 35 e 90, (25 + 12,5 e1/h)/αb;
#   área exata de n barras, n π φ²/4 (área nominal da ABNT NBR 7480, Tabela B.1, sem arredondar; cadeia exata).
#   Variáveis: momento, area_concreto, altura, fcd, area_aco, fyd, omega, nu, forca, excentricidade, fatores,
#   distancia, dimensao, quantidade, numero_faces, excentricidade_relativa_1, alfa_b, numero_barras, diametro.
#   Fonte: Venturini e Rodrigues (1987); Fusco (1981, p. 287 e 291); Scadelai e Pinheiro (2007, p. 16.28);
#   ABNT NBR 7480:2024, Tabela B.1.
#   Usada em: area_de_omega (l. 487); secao_camadas (l. 1162); executar_kimura (l. 1206, 1212, 1252, 1303);
#       executar_scadelai (l. 1384, 1389, 1398, 1404, 1405, 1431, 1432, 1433, 1435, 1436, 1437, 1449, 1453, 1464, 1465, 1509, 1510, 1513, 1514, 1515, 1555, 1556);
#       curva_fusco (l. 1576, 1581);
#       executar_fusco (l. 1702, 1704, 1744, 1747, 1748, 1752, 1759, 1771, 1811, 1812, 1813, 1815, 1824, 1826);
#       executar_franco (l. 1931); omega_no_ponto (l. 1958, 1961); omega_tangente_abaco.folga (l. 1974, 1978);
#       omega_acoplado_norma (l. 1993, 1997);
#       executar_carvalho (l. 2022, 2050, 2065, 2078, 2080, 2082, 2139, 2157, 2160, 2162, 2210, 2235, 2236, 2258, 2261, 2262, 2283, 2285, 2301, 2304, 2305, 2387, 2432, 2435, 2436, 2460, 2463, 2496, 2509, 2514, 2519, 2535, 2536, 2560, 2562, 2566, 2569)
def momento_adimensional(momento: float, area_concreto: float, altura: float, fcd: float) -> float:
    return momento / (area_concreto * altura * fcd)


def momento_de_mu(mu: float, area_concreto: float, altura: float, fcd: float) -> float:
    return mu * area_concreto * altura * fcd


def taxa_mecanica(area_aco: float, fyd: float, area_concreto: float, fcd: float) -> float:
    return area_aco * fyd / (area_concreto * fcd)


def fator_area_aco(area_concreto: float, fcd: float, fyd: float) -> float:
    return area_concreto * fcd / fyd


def area_de_omega(omega: float, area_concreto: float, fcd: float, fyd: float) -> float:
    return omega * fator_area_aco(area_concreto, fcd, fyd)


def taxa_percentual(area_aco: float, area_concreto: float) -> float:
    return area_aco / area_concreto * 100.0


def excentricidade(momento: float, forca: float) -> float:
    return momento / forca


def mu_primeira_ordem(nu: float, excentricidade_1: float, altura: float) -> float:
    return nu * excentricidade_1 / altura


def raio_giracao(altura: float) -> float:
    return altura / math.sqrt(12.0)


def forca_de_coeficientes(*fatores: float) -> float:
    return math.prod(fatores)


def profundidade_camada_oposta(altura: float, distancia: float) -> float:
    return altura - distancia


def profundidades_duas_camadas(altura: float, distancia: float, barras_por_face: int) -> np.ndarray:
    return np.array([distancia] * barras_por_face + [altura - distancia] * barras_por_face)


def comprimento_face(dimensao: float, distancia: float) -> float:
    return dimensao - 2.0 * distancia


def numero_total(quantidade: int, numero_faces: int) -> int:
    return quantidade * numero_faces


def esbeltez_limite_sem_limites(excentricidade_relativa_1: float, alfa_b: float) -> float:
    return (25.0 + 12.5 * excentricidade_relativa_1) / alfa_b


def area_barras_exata(numero_barras: int, diametro: float) -> float:
    return numero_barras * math.pi * diametro ** 2 / 4.0


# [F04] Rigidez κ aproximada (NBR 6118:2023 e 2003, 15.8.3.3.3): κ/ν = 32 (1 + 5 Md,tot/(h Nd)); Md,tot =
#   αb M1d,A / [1 - λ²/(120 κ/ν)] >= M1d,A; solução direta da equação do 2º grau da norma:
#   A Md² + B Md + C = 0, A = 5 h, B = h² Nd - Nd ℓe²/320 - 5 h αb M1d,A, C = -Nd h² αb M1d,A (raiz positiva).
#   Iteração do texto de Scadelai e Pinheiro: nova estimativa = média das duas últimas.
#   Variáveis: momento_total, altura, forca, esbeltez, kappa_nu, alfa_b, momento_primeira_ordem,
#   comprimento_equivalente, valor_a, valor_b.
#   Fonte: NBR 6118:2023, 15.8.3.3.3; Scadelai e Pinheiro (2007, p. 16.30).
#   Usada em: executar_kimura (l. 1266, 1267, 1269, 1270, 1272);
#       executar_scadelai (l. 1474, 1476, 1478, 1479, 1480, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489);
#       executar_carvalho (l. 2347)
def kappa_sobre_nu(momento_total: float, altura: float, forca: float) -> float:
    return 32.0 * (1.0 + 5.0 * momento_total / (altura * forca))


def momento_total_kappa(alfa_b: float, momento_primeira_ordem: float, esbeltez: float, kappa_nu: float) -> float:
    return max(alfa_b * momento_primeira_ordem / (1.0 - esbeltez ** 2 / (120.0 * kappa_nu)), momento_primeira_ordem)


def momento_total_kappa_direto(altura: float, forca: float, comprimento_equivalente: float, alfa_b: float,
                               momento_primeira_ordem: float) -> float:
    coeficiente_a = 5.0 * altura
    coeficiente_b = (altura ** 2 * forca - forca * comprimento_equivalente ** 2 / 320.0
                     - 5.0 * altura * alfa_b * momento_primeira_ordem)
    coeficiente_c = -forca * altura ** 2 * alfa_b * momento_primeira_ordem
    return ((-coeficiente_b + math.sqrt(coeficiente_b ** 2 - 4.0 * coeficiente_a * coeficiente_c))
            / (2.0 * coeficiente_a))


def media(valor_a: float, valor_b: float) -> float:
    return (valor_a + valor_b) / 2.0


def kappa_de_kappa_nu(kappa_nu: float, nu: float) -> float:
    return kappa_nu * nu


# [F05] Excentricidade de 2ª ordem: e2 = Md,tot/Nd - e1 (αb = 1); soma de duas parcelas. As fórmulas da rigidez
#   secante do diagrama M, N, 1/r (reta AB da Figura 15.1, forma do item C.3 de Kimura) são as de [F55] de
#   validacao.py.
#   Variáveis: momento_total, forca, excentricidade_1, valor_a, valor_b.
#   Fonte: NBR 6118:2023, 15.8.3.3.2; Kimura ([2005?]), item C.2.
#   Usada em: executar_kimura (l. 1244, 1254, 1271, 1285, 1286, 1288, 1347); digitalizar_fusco (l. 1651);
#       executar_fusco (l. 1806, 1807, 1812, 1813, 1816);
#       executar_carvalho (l. 2024, 2030, 2125, 2158, 2162, 2211, 2212, 2259, 2261, 2262, 2287, 2302, 2304, 2305, 2349, 2433, 2435, 2436, 2544, 2551)
#   Usada também em: figuras_complementares.py: reta_pilar_padrao (l. 707)
def excentricidade_segunda_ordem(momento_total: float, forca: float, excentricidade_1: float) -> float:
    return momento_total / forca - excentricidade_1


def soma(valor_a: float, valor_b: float) -> float:
    return valor_a + valor_b


# [F06] Pilar-padrão no diagrama M, N, 1/r de Fusco (1981, p. 291): abscissa 10³ d/r = 1000 d (1/r); inclinação
#   da reta tg θ = ν/10000 (ℓe/h)² (h/d); μ externo = μ1d + tg θ (10³ d/r); inclinação desenhada a partir da
#   cota impressa em 10³ d/r = 4: tg θ = cota/4; folga da curva em relação à reta: max(μ_curva - μ_externo) (> 0:
#   a curva alcança a reta, há equilíbrio). Interpolação linear do texto entre d/h = 0,95 e 0,85.
#   Variáveis: curvatura, altura_util, nu, comprimento_equivalente, altura, mu1, inclinacao, abscissa, cota,
#   mu_curva, omega_a, omega_b, fracao.
#   Fonte: Fusco (1981), p. 288-291 e Fig. 8.2.5-1.
#   Usada em: abscissa_fusco (l. 603); inclinacao_reta_fusco (l. 607); executar_kimura (l. 1221, 1256, 1271);
#       executar_scadelai (l. 1441, 1458, 1459, 1520, 1526); curva_fusco (l. 1574, 1580);
#       omega_pilar_padrao_fusco.folga (l. 1589); digitalizar_fusco (l. 1637, 1651);
#       executar_fusco (l. 1707, 1708, 1711, 1727, 1728, 1729, 1730, 1731);
#       executar_franco (l. 1846, 1912, 1914, 1930, 1932); omega_tangente_abaco.folga (l. 1978);
#       executar_carvalho (l. 2036, 2037, 2041, 2046, 2048, 2070, 2072, 2097, 2129, 2131, 2287, 2314, 2474, 2526, 2528, 2545)
#   Usada também em: figuras_complementares.py: reta_pilar_padrao (l. 707);
#       figuras_complementares.py: figura_fusco (l. 734)
def abscissa_fusco(curvatura, altura_util: float):
    return 1000.0 * altura_util * np.asarray(curvatura, dtype=float)


def inclinacao_reta_fusco(nu: float, comprimento_equivalente: float, altura: float, altura_util: float) -> float:
    return nu / 10000.0 * (comprimento_equivalente / altura) ** 2 * altura / altura_util


def inclinacao_desenhada(cota: float, abscissa: float) -> float:
    return cota / abscissa


def folga_reta(mu_curva, mu1: float, inclinacao: float, abscissa) -> float:
    return float(np.max(np.asarray(mu_curva) - (mu1 + inclinacao * np.asarray(abscissa))))


def altura_util(altura: float, distancia_relativa: float) -> float:
    return altura * (1.0 - distancia_relativa)


def interpolar(valor_a: float, valor_b: float, fracao: float) -> float:
    return valor_a + fracao * (valor_b - valor_a)


def fracao_interpolacao(alvo: float, ponto_a: float, ponto_b: float) -> float:
    return (alvo - ponto_a) / (ponto_b - ponto_a)


def produto(valor_a: float, valor_b: float) -> float:
    return valor_a * valor_b


# [F07] Fluência na formulação de Fusco (1981, p. 290-291, NB-1/1978): Fg = γn fração Nk; Ec = 0,9 x 6600
#   raiz(fck + 3,5) (MPa); α = Es/Ec; Ii = b h³/12 + α As (h/2 - d')²; FE = 10 Ec Ii/ℓe² (Ec em kN/cm²);
#   ec = e1g [exp(φ Fg/(FE - Fg)) - 1] (escrita independente da [F34] do núcleo, que tem a mesma forma).
#   Variáveis: gama_n, fracao, forca_caracteristica, fck_mpa, modulo_aco, modulo_concreto, inercia_concreto,
#   alfa, area_aco, altura, distancia, modulo, inercia, comprimento_equivalente, excentricidade_1, phi, forca_g,
#   forca_e.
#   Fonte: Fusco (1981, p. 290-291).
#   Usada em: executar_fusco (l. 1762, 1765, 1766, 1769, 1772, 1778, 1780, 1785, 1787);
#       executar_franco (l. 1860, 1863); executar_carvalho (l. 2400, 2404)
def modulo_fusco(fck_mpa: float) -> float:
    return 0.9 * 6600.0 * math.sqrt(fck_mpa + 3.5)


def razao_modular(modulo_aco: float, modulo_concreto: float) -> float:
    return modulo_aco / modulo_concreto


def inercia_ideal(inercia_concreto: float, alfa: float, area_aco: float, altura: float, distancia: float) -> float:
    return inercia_concreto + alfa * area_aco * (altura / 2.0 - distancia) ** 2


def carga_flambagem(modulo: float, inercia: float, comprimento_equivalente: float) -> float:
    return 10.0 * modulo * inercia / comprimento_equivalente ** 2


def excentricidade_suplementar(excentricidade_1: float, phi: float, forca_g: float, forca_e: float) -> float:
    return excentricidade_1 * (math.exp(phi * forca_g / (forca_e - forca_g)) - 1.0)


def mpa_para_kncm2(valor: float) -> float:
    return valor / 10.0


# [F08] Fluência na leitura de Franco (2010, p. 33): ângulo θ1 impresso (graus) -> tg θ1; na cadeia exata, θ1 em
#   radianos (1/200) -> tg θ1; ea = tg θ1 x ℓe/2;
#   Msg = M1d,mín/γf e Nsg = Nd/γf (hipóteses do autor); M1d,mín = Nd e1,mín; M1d = Nd (e1,mín + ecc);
#   ecc = (Msg/Nsg + ea) [exp(φ Nsg/(Ne - Nsg)) - 1], escrita de forma independente da [F34] do núcleo.
#   Variáveis: angulo_graus, angulo_radianos, comprimento, momento, gama_f, forca, excentricidade_1,
#   excentricidade_cc, momento_sg, forca_sg, excentricidade_a, phi, forca_e.
#   Fonte: Franco (2010, p. 33-34).
#   Usada em: executar_franco (l. 1847, 1848, 1854, 1870, 1871, 1880, 1881, 1882, 1884, 1918, 1924, 1926, 1931, 1932, 1933);
#       executar_carvalho (l. 2040, 2041, 2046, 2048, 2053, 2074, 2128, 2129, 2131, 2151, 2152, 2226, 2229, 2232, 2233, 2255, 2256, 2272, 2274, 2281, 2373, 2374, 2413, 2417, 2440, 2441, 2466, 2467, 2514, 2519, 2531, 2534)
def tangente_graus(angulo_graus: float) -> float:
    return math.tan(math.radians(angulo_graus))


def tangente_radianos(angulo_radianos: float) -> float:
    return math.tan(angulo_radianos)


def excentricidade_imperfeicao(tangente: float, comprimento: float) -> float:
    return tangente * comprimento / 2.0


def dividir(valor: float, divisor: float) -> float:
    return valor / divisor


def momento_com_fluencia_franco(forca: float, excentricidade_1: float, excentricidade_cc: float) -> float:
    return forca * (excentricidade_1 + excentricidade_cc)


def excentricidade_cc_independente(momento_sg: float, forca_sg: float, excentricidade_a: float, phi: float,
                                   forca_e: float) -> float:
    return (momento_sg / forca_sg + excentricidade_a) * (math.exp(phi * forca_sg / (forca_e - forca_sg)) - 1.0)


# [F09] Digitalização ([N02]): regressão linear pixel = a valor + b nas linhas da grade; conversão pixel -> μ e
#   x -> pixel; centro de um traço vertical de pixels escuros; posições das linhas menores da grade.
#   Também: caminho da imagem de sobreposição de cada painel e linhas de caracteres do relatório.
#   Variáveis: valores, pixels, coeficientes, pixel, valor, inicio, fim, distancia_relativa, caractere, comprimento.
#   Usada em: digitalizar_fusco (l. 1620, 1621, 1625, 1627, 1630, 1632, 1633, 1634, 1635, 1644, 1652, 1654, 1655, 1658, 1667, 1671, 1675, 1678, 1679, 1683);
#       escrever_relatorio (l. 2599, 2618, 2633)
def ajuste_linear(valores, pixels) -> np.ndarray:
    return np.polyfit(np.asarray(valores, dtype=float), np.asarray(pixels, dtype=float), 1)


def valor_do_pixel(coeficientes: np.ndarray, pixel: float) -> float:
    return (pixel - coeficientes[1]) / coeficientes[0]


def pixel_do_valor(coeficientes: np.ndarray, valor: float) -> float:
    return float(np.polyval(coeficientes, valor))


def centro_traco(inicio: int, fim: int) -> float:
    return (inicio + fim - 1) / 2.0


def valores_grade_menor(limite_superior: float) -> np.ndarray:
    return np.arange(0.0, limite_superior + ESPACAMENTO_GRADE_MENOR / 2.0, ESPACAMENTO_GRADE_MENOR)


def valores_grade_principal(quantidade: int) -> list[float]:
    return [round(0.1 * (quantidade - 1 - indice), 1) for indice in range(quantidade)]


def pixel_inteiro(valor: float) -> int:
    return int(round(valor))


def faixa_em_pixels(fracao_inicio: float, fracao_fim: float, tamanho: int) -> tuple[int, int]:
    return int(fracao_inicio * tamanho), int(fracao_fim * tamanho)


def intervalo_margem(pixel: float) -> tuple[int, int]:
    return max(0, int(pixel) - MARGEM_GRADE), int(pixel) + MARGEM_GRADE + 1


def minimo_pixels_linha(tamanho: int) -> float:
    return FRACAO_LINHA_GRADE * tamanho


def dentro_janela(valor: float, centro: float) -> bool:
    return abs(valor - centro) <= JANELA_MU


def separar_grupos_consecutivos(indices) -> list[float]:
    grupos: list[list[int]] = []
    for indice in indices:
        if grupos and indice - grupos[-1][-1] <= 2:
            grupos[-1].append(indice)
        else:
            grupos.append([indice])
    return [sum(grupo) / len(grupo) for grupo in grupos]


def abscissas_principais() -> list[float]:
    return [1.0, 2.0, 3.0, 4.0]


def caminho_sobreposicao(distancia_relativa: float) -> Path:
    return PASTA_SAIDA / ("%s%02d.png" % (PREFIXO_SOBREPOSICAO, int(round(distancia_relativa * 100.0))))


def linha_de(caractere: str, comprimento: int) -> str:
    return caractere * comprimento


def colunas_vizinhas(coluna: int) -> tuple[int, int]:
    return coluna - 1, coluna + 2


def maximo_absoluto(valores) -> float:
    return float(np.max(np.abs(np.asarray(valores, dtype=float))))


def media_valores(valores) -> float:
    return float(np.mean(np.asarray(valores, dtype=float)))


# [C02] Constantes das contas de Carvalho e Pinheiro (2009): fator 3,46 no lugar de raiz(12) (p. 356; o mesmo fator
#   aparece em Kimura, [2005?], item B: 3,46 x 6/0,3); coeficiente
#   adicional γn = 1,95 - 0,05 b (b em cm; Tabela 13.1 da NBR 6118, igual nas edições de 2003 e 2023); divisor 3840
#   = 12 x 320 do coeficiente k1 da expressão 5.29 (p. 353); módulo usado no Ne do exemplo 5.5: 0,85 x 5600
#   raiz(fck) (Ecs da NBR 6118:2003, p. 371), com o fator 5600 do módulo inicial.
#   Menor dimensão admitida em casos especiais pela Seção 13.2.3 da NBR 6118:2023 (com γn): 14 cm.
#   Variáveis: FATOR_ESBELTEZ_AUTOR, GAMA_N_BASE, GAMA_N_INCLINACAO, DIVISOR_K1, FATOR_ECS_2003, FATOR_MODULO_INICIAL,
#   MENOR_DIMENSAO_ESPECIAL_2023.
#   Fonte: Carvalho e Pinheiro (2009), p. 353, 356, 366 e 371; Kimura ([2005?]), item B; NBR 6118:2023, 13.2.3,
#   Tabela 13.1 e 15.8.4.
#   Usada em: esbeltez_fator_autor (l. 819); fator_comprimento_autor (l. 823); coeficiente_gama_n (l. 827);
#       coeficiente_k1 (l. 865); modulo_autor_55 (l. 881); executar_carvalho (l. 2333)
FATOR_ESBELTEZ_AUTOR = 3.46
GAMA_N_BASE = 1.95
GAMA_N_INCLINACAO = 0.05
DIVISOR_K1 = 3840.0
FATOR_ECS_2003 = 0.85
FATOR_MODULO_INICIAL = 5600.0
MENOR_DIMENSAO_ESPECIAL_2023 = 14.0


# [F10] Cadeia de Carvalho e Pinheiro (2009): λ = 3,46 ℓe/h; produto 3,46 ℓe; γn = 1,95 - 0,05 b; taxa mínima
#   relativa ρ = 0,15 (fcd/fyd) ν; redução da área adotada em relação à mínima, (1 - As,adot/As,mín) x 100;
#   excentricidade de 2ª ordem e2 = (ℓe²/10)(1/r); excentricidade acidental da seção de extremidade, o dobro da
#   intermediária; inverso; maior de dois valores; denominador do desaprumo, 100 raiz(ℓ) (ℓ em m;
#   θ1 = 1/(100 raiz(ℓ)), 11.3.3.4.1 e 11.3.3.4.2). λ = 3,46 ℓe/h e e2 = (ℓe²/10)(1/r) também na cadeia de Kimura
#   ([2005?], itens B e C.1).
#   Variáveis: comprimento, altura, menor_dimensao, fcd, fyd, nu, area_adotada, area_minima,
#   curvatura, valor, comprimento_m.
#   Fonte: Carvalho e Pinheiro (2009), p. 356-367; Kimura ([2005?]), itens B e C.1; NBR 6118:2023, 11.3.3.4.1,
#   11.3.3.4.2, 15.8.3.3.2, 17.3.5.3.1 e Tabela 13.1.
#   Usada em: executar_kimura (l. 1223, 1242);
#       executar_carvalho (l. 2036, 2045, 2061, 2063, 2089, 2090, 2091, 2101, 2103, 2104, 2137, 2148, 2149, 2150, 2163, 2165, 2170, 2172, 2175, 2177, 2182, 2184, 2202, 2218, 2222, 2232, 2233, 2248, 2249, 2317, 2318, 2319, 2341, 2344, 2371, 2384, 2469, 2470, 2471, 2501, 2505, 2521, 2522, 2543)
def esbeltez_fator_autor(comprimento: float, altura: float) -> float:
    return FATOR_ESBELTEZ_AUTOR * comprimento / altura


def fator_comprimento_autor(comprimento: float) -> float:
    return FATOR_ESBELTEZ_AUTOR * comprimento


def coeficiente_gama_n(menor_dimensao: float) -> float:
    return GAMA_N_BASE - GAMA_N_INCLINACAO * menor_dimensao


def taxa_minima_relativa(fcd: float, fyd: float, nu: float) -> float:
    return 0.15 * fcd / fyd * nu


def reducao_percentual_area(area_adotada: float, area_minima: float) -> float:
    return (1.0 - area_adotada / area_minima) * 100.0


def excentricidade_curvatura(comprimento: float, curvatura: float) -> float:
    return comprimento ** 2 / 10.0 * curvatura


def dobro(valor: float) -> float:
    return 2.0 * valor


def inverso(valor: float) -> float:
    return 1.0 / valor


def maior(valor_a: float, valor_b: float) -> float:
    return max(valor_a, valor_b)


def fator_desaprumo(comprimento_m: float) -> float:
    return 100.0 * math.sqrt(comprimento_m)


# [F11] Rigidez κ aproximada na forma fechada de Carvalho e Pinheiro (2009, expressão 5.29, de Banki, 2003):
#   k1 = 1 - λ²/3840; e2 = [raiz(k1² h² + 10 e1 h (2 - k1) + 25 e1²) - 5 e1 - k1 h]/10. É a raiz positiva da
#   equação do 2º grau da Seção 15.8.3.3.3 com αb = 1, escrita em excentricidades (ℓe²/320 = λ² h²/3840).
#   Variáveis: esbeltez, k1, altura, excentricidade_1.
#   Fonte: Carvalho e Pinheiro (2009), p. 353 e 368; NBR 6118:2023, 15.8.3.3.3.
#   Usada em: executar_carvalho (l. 2337, 2338, 2345, 2350)
def coeficiente_k1(esbeltez: float) -> float:
    return 1.0 - esbeltez ** 2 / DIVISOR_K1


def excentricidade_kappa_fechada(k1: float, altura: float, excentricidade_1: float) -> float:
    return (math.sqrt(k1 ** 2 * altura ** 2 + 10.0 * excentricidade_1 * altura * (2.0 - k1)
                      + 25.0 * excentricidade_1 ** 2) - 5.0 * excentricidade_1 - k1 * altura) / 10.0


# [F12] Fluência e pilar-padrão acoplado do exemplo 5.5 de Carvalho e Pinheiro (2009): módulo 0,85 x 5600
#   raiz(fck) (MPa); reta dos esforços externos no plano μ x 1/r do pilar-padrão (Fig. 5.19 do livro):
#   μ = μ1 + inclinação (1/r), com inclinação = ν ℓe²/(10 h) (1/r em 1/cm, ℓe e h em cm); conversão de módulo
#   kN/cm² -> MPa (x 10).
#   Variáveis: fck_mpa, nu, comprimento, altura, valor.
#   Fonte: Carvalho e Pinheiro (2009), p. 349-351 e 371.
#   Usada em: omega_tangente_abaco (l. 1971); executar_carvalho (l. 2392, 2395, 2396, 2398, 2404)
def modulo_autor_55(fck_mpa: float) -> float:
    return FATOR_ECS_2003 * FATOR_MODULO_INICIAL * math.sqrt(fck_mpa)


def inclinacao_reta_curvatura(nu: float, comprimento: float, altura: float) -> float:
    return nu * comprimento ** 2 / (10.0 * altura)


def kncm2_para_mpa(valor: float) -> float:
    return valor * 10.0

# [R07] Origem da diferença entre a rotina e o valor impresso, por código (campo "origem" do TXT e do JSON; no TCC,
#   até 03/11/2026, coluna "Origem" do Apêndice F, retirada quando as tabelas passaram à cadeia exata do autor):
#   T arredondamento ou truncamento da fonte; H hipótese ou valor adotado pela fonte; N critério ou edição da
#   norma; E erro de conta ou de cópia da fonte; L leitura em ábaco ou figura; M método diferente; P valor de
#   programa; "–" sem diferença além do arredondamento da impressão. ORIGENS dá o código de cada etapa com
#   diferença (fonte, início do rótulo da etapa); as demais seguem [F13].
#   Variáveis: ORIGEM_LEGENDA, ORIGENS, SEM_ORIGEM.
#   Usada em: codigo_origem (l. 932, 936, 938, 939); codigo_origem_exata (l. 982); escrever_relatorio (l. 2605)
ORIGEM_LEGENDA = ("T, arredondamento ou truncamento da fonte; H, hipótese ou valor adotado pela fonte; N, critério "
                  "ou edição da norma; E, erro de conta ou de cópia da fonte; L, leitura em ábaco ou figura; M, método "
                  "diferente; P, valor de programa; –, sem diferença além do arredondamento da impressão")
SEM_ORIGEM = "–"
ORIGENS = (
    ("L1", "λ = 3,46 ℓe/h", "T"), ("L1", "C.1: e2", "T"), ("L1", "C.1: Md,tot", "T"),
    ("L1", "C.2: κ = 32", "T"), ("L1", "C.2: Md,tot = Nd etot", "E"),
    ("L1", "C.3: Md,tot pela curva de deformação", "M"), ("L1", "C.4: Md,tot pelo método geral", "M"),
    ("L2", "1/r = 0,005", "T"), ("L2", "Md,tot = αb M1d,A", "T"), ("L2", "As da rotina com d′ = 4,0 cm", "H"),
    ("L2", "MRd com 24,0 cm²", "H"), ("L2", "MRd com 12 ϕ 16", "H"), ("L2", "Rigidez κ, Md,tot adotado", "M"),
    ("L2", "Rigidez κ: etot", "M"), ("L2", "As adotada = 12 ϕ 16", "H"),
    ("L3", "ω interpolado", "M"), ("L3", "As = ω Ac fcd/fyd (fonte: ω = 0,78", "M"),
    ("L3", "ec = e1g", "E"), ("L3", "e1x = ea + ec", "E"), ("L3", "ω com fluência", "M"),
    ("L4", "M1d,mín = Nd e1,mín", "T"), ("L4", "Msg = M1d,mín/1,4", "T"), ("L4", "Ne = 10 Eci", "H"),
    ("L4", "ea = θ1", "H"), ("L4", "(EI)sec = (MRd/1,1)/(1/r)", "T"), ("L4", "M1d = Nd (e1,mín + ecc)", "T"),
    ("L5", "5.1: 3,46 ℓe", "H"), ("L5", "5.1: b >= 10,66/35", "T"), ("L5", "5.1: μ = (e/h) ν", "T"),
    ("L5", "5.1: As de 4 ϕ 10", "H"), ("L5", "5.1: redução de 4 ϕ 10", "H"), ("L5", "5.2: b >= 10,66/90", "H"),
    ("L5", "5.2: μ = [(e1,mín + e2)/h] ν", "T"), ("L5", "5.2: ρmín", "E"), ("L5", "5.3: λy = 3,46", "H"),
    ("L5", "5.3: e1x,mín", "T"), ("L5", "5.3: μx", "T"), ("L5", "5.3: μy", "T"), ("L5", "5.4: k1", "T; H"),
    ("L5", "5.5: ea = θ1", "H"), ("L5", "5.5: Ec = 0,85", "T; H"), ("L5", "5.5: Ne = 10 Ec", "H"),
    ("L5", "5.5: ecc = (Msg", "H"), ("L5", "5.5: μ1", "H"), ("L5", "5.5: ω pelo pilar-padrão acoplado", "M"),
    ("L5", "5.10: e na seção de extremidade", "N"), ("L5", "5.10: μx = ν e/h", "T; N"),
    ("L5", "5.10: ω da rotina com e = máx", "N"),
)
CODIGO_PROGRAMA = "P"
CODIGO_LEITURA = "L"


# [F13] Código da origem da diferença ([R07]): o de ORIGENS, se a etapa estiver lá; senão, P nos valores de
#   programa, L nas leituras, e "–" nas demais (conferência e etapas sem diferença além do arredondamento).
#   Variáveis: exemplo, etapa, criterio.
#   Usada em: Registro.adicionar (l. 1096)
def codigo_origem(exemplo: str, etapa: str, criterio: str) -> str:
    for fonte, inicio, codigo in ORIGENS:
        if exemplo.startswith(fonte + " ") and etapa.startswith(inicio):
            return codigo
    if criterio == CRITERIO_PROGRAMA:
        return CODIGO_PROGRAMA
    if criterio == CRITERIO_LEITURA:
        return CODIGO_LEITURA
    return SEM_ORIGEM


# [R08] Origem da não reprodução pela cadeia exata do autor ([N01]), com os códigos de [R07]: T etapa intermediária
#   arredondada ou truncada, ou constante aproximada pelo autor (3,46; área de barra; Ic; ângulo de θ1); E erro de
#   conta ou de cópia da fonte; H, N e M como em [R07]. ORIGENS_EXATA dá o código de cada etapa de critério
#   "impressão" cuja cadeia exata não dá o valor impresso (fonte, início do rótulo da etapa); a nota da etapa diz
#   qual aproximação do autor explica a diferença. ORIGEM_EXATA_NAO_TABELADA marca uma etapa que não reproduz e não
#   está na tabela (aviso no relatório).
#   Variáveis: ORIGENS_EXATA, ORIGEM_EXATA_NAO_TABELADA.
#   Usada em: origem_exata_tabelada (l. 974); codigo_origem_exata (l. 984)
ORIGENS_EXATA = (
    ("L1", "C.2: κ = 32", "T"), ("L1", "C.2: Md,tot = Nd etot", "E"),
    ("L2", "1/r = 0,005", "T"), ("L2", "Md,tot = αb M1d,A", "T"), ("L2", "As = (Ac fcd/fyd) ω, rigidez κ", "T"),
    ("L2", "Rigidez κ, 1ª iteração", "T"), ("L2", "Rigidez κ, estimativa", "T"), ("L2", "Rigidez κ, 2ª iteração", "T"),
    ("L2", "As adotada = 12 ϕ 16", "T"),
    ("L3", "As = ω Ac fcd/fyd (fonte: ω = 0,78", "T"), ("L3", "Ii = Ic", "T"), ("L3", "ec = e1g", "E"),
    ("L3", "e1x = ea + ec", "E"), ("L3", "As = ω Ac fcd/fyd (ω = 0,92)", "T"),
    ("L4", "M1d,mín = Nd e1,mín", "T"), ("L4", "Ne = 10 Eci", "T"), ("L4", "ea = θ1", "T"),
    ("L4", "M1d = Nd (e1,mín + ecc)", "T"), ("L4", "Md,tot = M1d/", "T"),
    ("L5", "5.1: 3,46 ℓe", "T"), ("L5", "5.1: b >= 10,66/35", "T"), ("L5", "5.1: μ = (e/h) ν", "T"),
    ("L5", "5.1: As de 4 ϕ 10", "T"), ("L5", "5.1: redução de 4 ϕ 10", "T"), ("L5", "5.2: b >= 10,66/90", "T"),
    ("L5", "5.2: μ = [(e1,mín + e2)/h] ν", "T"), ("L5", "5.2: ρmín", "E"), ("L5", "5.3: λy", "T"),
    ("L5", "5.3: e1x,mín", "T"), ("L5", "5.3: μx", "T"), ("L5", "5.3: μy", "T"), ("L5", "5.4: k1", "T"),
    ("L5", "5.5: Ec = 0,85", "T"), ("L5", "5.5: Ne = 10 Ec", "T"), ("L5", "5.5: μ1", "T"), ("L5", "5.10: μx", "T"),
)
ORIGEM_EXATA_NAO_TABELADA = "?"


# [F16] Código da origem pela cadeia exata ([R08]): o de ORIGENS_EXATA quando a etapa não reproduz;
#   ORIGEM_EXATA_NAO_TABELADA quando não reproduz e não está na tabela; SEM_ORIGEM ("–") quando reproduz.
#   origem_exata_tabelada dá o código da tabela (ou None) para os avisos de coerência de Registro.adicionar.
#   Variáveis: exemplo, etapa, reproduz_exata.
#   Usada em: codigo_origem_exata (l. 983); Registro.adicionar_cadeia_exata (l. 1131, 1133)
def origem_exata_tabelada(exemplo: str, etapa: str) -> str | None:
    for fonte, inicio, codigo in ORIGENS_EXATA:
        if exemplo.startswith(fonte + " ") and etapa.startswith(inicio):
            return codigo
    return None


def codigo_origem_exata(exemplo: str, etapa: str, reproduz_exata: bool) -> str:
    if reproduz_exata:
        return SEM_ORIGEM
    codigo = origem_exata_tabelada(exemplo, etapa)
    return ORIGEM_EXATA_NAO_TABELADA if codigo is None else codigo


# [R09] Comparação da rotina com a cadeia exata do autor (Apêndice F e Seção 4.1.2 do TCC, revisão de 03/11/2026):
#   nas linhas de critério "impressão" com valor da cadeia exata e da rotina, o desvio relativo
#   (rotina - exata)/exata x 100 (%) classifica a etapa em "sem diferença" (até LIMITE_PONTO_FLUTUANTE_PERCENTUAL,
#   arredondamento da aritmética de ponto flutuante), "abaixo do limite" (até LIMITE_COINCIDENCIA_PERCENTUAL) ou
#   "difere". Toda etapa que difere e toda etapa com erro da fonte (origem E da cadeia exata) devem ter a nota curta
#   das tabelas do TCC (campo "nota_tabela"); a falta dela gera aviso no relatório. As notas curtas dizem só o
#   essencial; a observação completa de cada linha continua no TXT e no JSON.
#   Variáveis: LIMITE_PONTO_FLUTUANTE_PERCENTUAL, LIMITE_COINCIDENCIA_PERCENTUAL, CLASSE_SEM_DIFERENCA,
#   CLASSE_ABAIXO_LIMITE, CLASSE_DIFERE, ORIGEM_ERRO_FONTE.
#   Usada em: classe_rotina_exata (l. 1032, 1033, 1034, 1035, 1036);
#       Registro.adicionar_comparacao_rotina_exata (l. 1118);
#       escrever_relatorio (l. 2612, 2613, 2659, 2661, 2662, 2664)
LIMITE_PONTO_FLUTUANTE_PERCENTUAL = 1e-9
LIMITE_COINCIDENCIA_PERCENTUAL = 1e-3
CLASSE_SEM_DIFERENCA = "sem diferença"
CLASSE_ABAIXO_LIMITE = "abaixo do limite"
CLASSE_DIFERE = "difere"
ORIGEM_ERRO_FONTE = "E"

# [R10] Nota curta comum às linhas de ea na seção de extremidade de Carvalho e Pinheiro (2009, exemplos 5.1, 5.3, 5.5
#   e 5.10), em que a rotina não tem valor: a ABNT NBR 6118:2023, Seção 11.3.3.4.3, substitui as imperfeições locais
#   pelo momento mínimo de 1ª ordem.
#   Variáveis: NOTA_TABELA_EA_EXTREMIDADE.
#   Usada em: executar_carvalho (l. 2063, 2234, 2385, 2529)
NOTA_TABELA_EA_EXTREMIDADE = ("A rotina não soma ea na seção de extremidade: o momento mínimo substitui as "
                              "imperfeições locais (Seção 11.3.3.4.3).")


# [F17] Classe da comparação rotina x cadeia exata ([R09]): desvio = (rotina - exata)/exata x 100 (desvio_relativo,
#   [F01]); |desvio| <= LIMITE_PONTO_FLUTUANTE_PERCENTUAL: sem diferença; <= LIMITE_COINCIDENCIA_PERCENTUAL: abaixo do
#   limite; acima: difere. Exata nula: compara a diferença absoluta com os mesmos limites. Sem um dos dois valores:
#   None.
#   Variáveis: exata, rotina, desvio.
#   Usada em: Registro.adicionar_comparacao_rotina_exata (l. 1113, 1114)
def desvio_rotina_exata(exata, rotina) -> float | None:
    if exata is None or rotina is None or isinstance(exata, bool) or isinstance(rotina, bool):
        return None
    if exata == 0.0:
        return diferenca(rotina, exata)
    return desvio_relativo(rotina, exata)


def classe_rotina_exata(desvio: float | None) -> str | None:
    if desvio is None:
        return None
    if abs(desvio) <= LIMITE_PONTO_FLUTUANTE_PERCENTUAL:
        return CLASSE_SEM_DIFERENCA
    if abs(desvio) <= LIMITE_COINCIDENCIA_PERCENTUAL:
        return CLASSE_ABAIXO_LIMITE
    return CLASSE_DIFERE


# [F14] Rigidez secante na cadeia de Franco (2010, p. 34), com os valores impressos das etapas anteriores:
#   (EI)sec = M_B/(1/r)_B, em que M_B = MRd/1,1 é o momento do ponto B impresso (24,3 kN·m) e (1/r)_B, a curvatura
#   do anexo 1 (programa MK-UFRJ), impressa como 27,4979/1000 m⁻¹. M_B = MRd/γf3 e κsec = (EI)sec/(Ac h² fcd)
#   são os de [F55] de validacao.py (validacao.momento_ponto_b e validacao.kappa_secante).
#   Variáveis: momento_ponto_b, curvatura_ponto_b.
#   Fonte: Franco (2010, p. 34); NBR 6118:2023, 15.3.1.
#   Usada em: executar_franco (l. 1902, 1903)
def rigidez_do_ponto_b(momento_ponto_b: float, curvatura_ponto_b: float) -> float:
    return momento_ponto_b / curvatura_ponto_b


# [F15] Separador de milhar dos números escritos nas notas e nos rótulos (regra do SI): parte inteira com até
#   quatro algarismos, sem separador (1285,7); com MINIMO_ALGARISMOS_AGRUPAMENTO (cinco) ou mais, grupos de
#   ALGARISMOS_POR_GRUPO (três) algarismos, a partir da vírgula, separados por SEPARADOR_MILHAR (espaço não
#   separável, U+00A0): 26 071,6; 120 300. Nunca ponto. O sinal e a parte decimal ficam como estão.
#   Variáveis: SEPARADOR_MILHAR, MINIMO_ALGARISMOS_AGRUPAMENTO, ALGARISMOS_POR_GRUPO, texto.
#   Fonte: regra do Sistema Internacional de Unidades adotada na redação do TCC.
#   Usada em: agrupar_milhares (l. 1065, 1068, 1069, 1070, 1072); num (l. 2002); texto_impresso (l. 2007)
SEPARADOR_MILHAR = "\u00a0"
MINIMO_ALGARISMOS_AGRUPAMENTO = 5
ALGARISMOS_POR_GRUPO = 3


def agrupar_milhares(texto: str) -> str:
    sinal = texto[:1] if texto[:1] in ("-", "−") else ""
    inteira, virgula, decimal = texto[len(sinal):].partition(",")
    if len(inteira) < MINIMO_ALGARISMOS_AGRUPAMENTO or not inteira.isdigit():
        return texto
    grupos = []
    while len(inteira) > ALGARISMOS_POR_GRUPO:
        grupos.insert(0, inteira[-ALGARISMOS_POR_GRUPO:])
        inteira = inteira[:-ALGARISMOS_POR_GRUPO]
    grupos.insert(0, inteira)
    return sinal + SEPARADOR_MILHAR.join(grupos) + virgula + decimal


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


class Registro:
    """Linhas de comparação (um valor impresso por linha) e verificações lógicas de todos os exemplos."""

    def __init__(self) -> None:
        self.linhas: list[dict] = []
        self.verificacoes: list[dict] = []
        self.digitalizacao: dict = {}
        self.avisos_cadeia_exata: list[str] = []

    def adicionar(self, exemplo: str, etapa: str, unidade: str, criterio: str, texto: str | None, autor,
                  rotina, observacao: str = "", resolucao_informada: float | None = None, exata=None,
                  nota_exata: str = "", nota_tabela: str = "") -> None:
        impresso = None if texto is None else valor_impresso(texto)
        linha = {"exemplo": exemplo, "etapa": etapa, "unidade": unidade, "criterio": criterio, "impresso": texto,
                 "autor": autor, "rotina": rotina, "reproduz": None, "limite": None,
                 "desvio_rotina_percentual": None, "diferenca_rotina": None, "observacao": observacao,
                 "origem": codigo_origem(exemplo, etapa, criterio)}
        if impresso is not None and criterio == CRITERIO_IMPRESSAO and autor is not None:
            limite = limite_reproducao(resolucao_impressa(texto, resolucao_informada))
            linha["limite"] = limite
            linha["reproduz"] = reproduz(autor, impresso, limite)
        if impresso is not None and rotina is not None and impresso != 0.0:
            linha["desvio_rotina_percentual"] = desvio_relativo(rotina, impresso)
            linha["diferenca_rotina"] = diferenca(rotina, impresso)
        linha.update({"exata": None, "reproduz_exata": None, "origem_exata": None, "nota_exata": ""})
        if impresso is not None and criterio == CRITERIO_IMPRESSAO:
            self.adicionar_cadeia_exata(linha, impresso, exata, nota_exata, resolucao_informada)
        self.adicionar_comparacao_rotina_exata(linha, nota_tabela)
        self.linhas.append(linha)

    def adicionar_comparacao_rotina_exata(self, linha: dict, nota_tabela: str) -> None:
        """Desvio e classe da rotina em relação à cadeia exata ([R09], [F17]) e nota curta das tabelas do TCC, com
        aviso quando falta a nota numa etapa que difere ou com erro da fonte."""
        desvio = desvio_rotina_exata(linha["exata"], linha["rotina"])
        classe = classe_rotina_exata(desvio)
        linha.update({"desvio_rotina_exata_percentual": desvio, "comparacao_rotina_exata": classe,
                      "nota_tabela": nota_tabela})
        identificacao = "%s | %s" % (linha["exemplo"], linha["etapa"])
        if (classe == CLASSE_DIFERE or linha["origem_exata"] == ORIGEM_ERRO_FONTE) and not nota_tabela:
            self.avisos_cadeia_exata.append("rotina difere da cadeia exata ou erro da fonte sem nota da tabela: "
                                            + identificacao)

    def adicionar_cadeia_exata(self, linha: dict, impresso: float, exata, nota_exata: str,
                               resolucao_informada: float | None) -> None:
        """Campos da cadeia exata do autor ([N01], [R08], [F16]) e avisos de coerência entre nota e tabela."""
        identificacao = "%s | %s" % (linha["exemplo"], linha["etapa"])
        if exata is None:
            self.avisos_cadeia_exata.append("sem valor da cadeia exata: " + identificacao)
            return
        reproduz_exata = reproduz(exata, impresso, limite_reproducao(resolucao_impressa(linha["impresso"],
                                                                                       resolucao_informada)))
        tabelada = origem_exata_tabelada(linha["exemplo"], linha["etapa"])
        linha.update({"exata": exata, "reproduz_exata": reproduz_exata,
                      "origem_exata": codigo_origem_exata(linha["exemplo"], linha["etapa"], reproduz_exata),
                      "nota_exata": "" if reproduz_exata else nota_exata})
        if reproduz_exata and tabelada is not None:
            self.avisos_cadeia_exata.append("reproduz pela cadeia exata, mas está em ORIGENS_EXATA: " + identificacao)
        if not reproduz_exata and (tabelada is None or not nota_exata):
            self.avisos_cadeia_exata.append("não reproduz pela cadeia exata e falta origem ou nota: " + identificacao)

    def verificar(self, exemplo: str, verificacao: str, atende: bool, texto_fonte: str, observacao: str = "") -> None:
        self.verificacoes.append({"exemplo": exemplo, "verificacao": verificacao, "rotina_atende": atende,
                                  "fonte": texto_fonte, "observacao": observacao})


class SecaoHipotese:
    """Fixa temporariamente a hipótese de seção do núcleo ([H05]: líquida ou bruta) durante um cálculo."""

    def __init__(self, liquida: bool) -> None:
        self.liquida = liquida
        self.anterior = nucleo.DESCONTAR_AREA_BARRAS

    def __enter__(self) -> None:
        self.anterior = nucleo.DESCONTAR_AREA_BARRAS
        nucleo.DESCONTAR_AREA_BARRAS = self.liquida

    def __exit__(self, *argumentos) -> None:
        nucleo.DESCONTAR_AREA_BARRAS = self.anterior


def secao_camadas(altura: float, largura: float, distancia: float, barras_por_face: int) -> nucleo.SecaoDirecao:
    """Duas camadas iguais de barras_por_face barras, nas profundidades d' e h - d'."""
    return nucleo.secao_por_camadas(altura, largura, profundidades_duas_camadas(altura, distancia, barras_por_face))


def curvatura_do_momento_deformacao(secao: nucleo.SecaoDirecao, material: nucleo.Material, area: float,
                                    forca_deformacao: float, alvo: float) -> float:
    """Curvatura (1/cm) em que o momento da curva de deformação (1,10 fcd e N dado) vale o alvo."""
    limite = nucleo.curvatura_ultima(secao, material, area, forca_deformacao, True)
    return brentq(lambda curvatura: validacao.diferenca_momento_curva(
        nucleo.momento_curva_deformacao(secao, material, area, forca_deformacao, curvatura), alvo),
        nucleo.CURVATURA_NULA, limite, xtol=1e-14)


def metodo_geral_constante(secao: nucleo.SecaoDirecao, material: nucleo.Material, area: float, forca: float,
                           momento_primeira_ordem: float, comprimento: float) -> tuple[float, str]:
    """Método geral de validacao.py ([F10] de validacao.py), com momento de 1ª ordem constante ao longo do pilar."""
    forca_deformacao = nucleo.forca_normal_curva_deformacao(forca)
    curvaturas, momentos = nucleo.curva_momento_curvatura(secao, material, area, forca_deformacao, True,
                                                          validacao.PONTOS_CURVA_GERAL)
    pico = int(np.argmax(momentos))
    curvaturas_subida = curvaturas[:pico + 1]
    momentos_subida = momentos[:pico + 1]

    def curvatura_inelastica(momentos_nos: np.ndarray):
        reduzidos = validacao.momentos_reduzidos(momentos_nos)
        if reduzidos.max() > momentos[pico]:
            return None
        return np.interp(reduzidos, momentos_subida, curvaturas_subida)

    momento, situacao, _ = validacao.linha_elastica(momento_primeira_ordem, forca, comprimento, curvatura_inelastica)
    return momento, situacao


# ----------------------------------------------------------------------------- L1 — Kimura ([2005?]), TQS
def executar_kimura(registro: Registro) -> dict:
    dados = KIMURA
    impresso = dados["impresso"]
    nome = "L1 Kimura ([2005?])"
    material = nucleo.Material(dados["fck"])
    forca = tf_para_kn(dados["Nd_tf"])
    altura = dados["altura"]
    largura = dados["largura"]
    comprimento = dados["comprimento"]
    area_concreto = nucleo.area_secao_retangular(largura, altura)
    momento_minimo = nucleo.momento_minimo_primeira_ordem(forca, altura)
    excentricidade_minima = excentricidade(momento_minimo, forca)
    esbeltez = nucleo.indice_esbeltez(comprimento, altura)
    alfa_b = nucleo.coeficiente_alfa_b(nucleo.momento_excentricidade(forca, dados["e_inicial"]), 0.0,
                                       nucleo.momentos_menores_que_minimo(
                                           nucleo.momento_excentricidade(forca, dados["e_inicial"]), momento_minimo))
    momento_inicial = nucleo.momento_excentricidade(forca, dados["e_inicial"])
    limite_calculado = esbeltez_limite_sem_limites(nucleo.excentricidade_relativa(momento_inicial, forca, altura),
                                                   alfa_b)
    nu = nucleo.forca_normal_adimensional(forca, area_concreto, material.fcd)
    area_minima = nucleo.armadura_minima(forca, material.fyd, area_concreto)
    excentricidade_minima_impressa = valor_impresso(impresso["e1_min"])
    nu_impresso = valor_impresso(impresso["nu"])
    registro.adicionar(nome, "e1,mín = 1,5 + 0,03 h", "cm", CRITERIO_IMPRESSAO, impresso["e1_min"],
                       excentricidade_minima, excentricidade_minima, exata=excentricidade_minima)
    registro.adicionar(nome, "M1d,mín = Nd e1,mín", "tf·m", CRITERIO_IMPRESSAO, impresso["M1d_min"],
                       kncm_para_tfm(produto(forca, excentricidade_minima_impressa)), kncm_para_tfm(momento_minimo),
                       exata=kncm_para_tfm(momento_minimo))
    esbeltez_autor = esbeltez_fator_autor(comprimento, altura)
    registro.adicionar(nome, "λ = 3,46 ℓe/h (item B)", "", CRITERIO_IMPRESSAO, impresso["lambda"], esbeltez_autor,
                       esbeltez,
                       "O item B escreve 3,46 × 6/0,3 = 69,3 (o item C repete λ = 69,3); 3,46 × 6/0,3 = %s, e o 69,3 "
                       "impresso é o valor com raiz(12), %s (coluna “rotina”)." % (num(esbeltez_autor, 1),
                                                                                  num(esbeltez, 2)),
                       exata=esbeltez)
    registro.adicionar(nome, "λ1 = (25 + 12,5 e1/h)/αb, antes do limite de 35", "", CRITERIO_IMPRESSAO,
                       impresso["lambda1_calculado"], limite_calculado, limite_calculado,
                       "Com ei = 1680/840 = 2,00 cm no meio do lance; o texto e a rotina adotam λ1 = 35.",
                       exata=limite_calculado)
    registro.adicionar(nome, "ν = Nd/(Ac fcd)", "", CRITERIO_IMPRESSAO, impresso["nu"], nu, nu, exata=nu)
    registro.adicionar(nome, "As,mín = 0,15 Nd/fyd >= 0,004 Ac", "cm²", CRITERIO_IMPRESSAO, impresso["As_min"],
                       area_minima, area_minima, exata=area_minima)
    curvatura = nucleo.curvatura_aproximada(nu, altura)
    momento_curvatura = nucleo.momento_total_pilar_padrao(alfa_b, momento_minimo, forca, comprimento, curvatura)
    registro.adicionar(nome, "C.1: 1/r = 0,005/[h (ν + 0,5)]", "m⁻¹", CRITERIO_IMPRESSAO, impresso["curvatura"],
                       por_cm_para_por_metro(nucleo.curvatura_aproximada(nu_impresso, altura)),
                       por_cm_para_por_metro(curvatura), exata=por_cm_para_por_metro(curvatura))
    excentricidade_2_autor = excentricidade_curvatura(comprimento,
                                                      por_metro_para_por_cm(valor_impresso(impresso["curvatura"])))
    excentricidade_2_rotina = excentricidade_segunda_ordem(momento_curvatura, forca, excentricidade_minima)
    registro.adicionar(nome, "C.1: e2 = (ℓe²/10)(1/r)", "cm", CRITERIO_IMPRESSAO, impresso["e2_curvatura"],
                       excentricidade_2_autor, excentricidade_2_rotina,
                       "O texto escreve (36/10) × 0,016 = 0,0581 m; com o 0,016 impresso, %s m. O 0,0581 vem da "
                       "curvatura sem arredondar, 1/r = %s m⁻¹, como na coluna “rotina”: a fonte leva à etapa "
                       "seguinte o valor sem arredondar." % (num(cm_para_m(excentricidade_2_autor), 4),
                                                             num(por_cm_para_por_metro(curvatura), 6)),
                       exata=excentricidade_2_rotina)
    excentricidade_total_rotina = excentricidade(momento_curvatura, forca)
    registro.adicionar(nome, "C.1: etot = e1,mín + e2", "cm", CRITERIO_IMPRESSAO, impresso["etot_curvatura"],
                       soma(excentricidade_minima_impressa, valor_impresso(impresso["e2_curvatura"])),
                       excentricidade_total_rotina, exata=excentricidade_total_rotina)
    momento_curvatura_autor = produto(forca, valor_impresso(impresso["etot_curvatura"]))
    registro.adicionar(nome, "C.1: Md,tot (curvatura aproximada)", "tf·m", CRITERIO_IMPRESSAO,
                       impresso["Md_curvatura"], kncm_para_tfm(momento_curvatura_autor),
                       kncm_para_tfm(momento_curvatura),
                       "O texto escreve 840 × 0,0821 = 68,9; com o etot impresso (2,40 + 5,81 = 8,21 cm), %s (%s "
                       "com uma casa). O 68,9 vem do etot sem arredondar: 840 × %s = %s (coluna “rotina”): a fonte "
                       "leva à etapa seguinte o valor sem arredondar."
                       % (num(kncm_para_tfm(momento_curvatura_autor), 3), num(kncm_para_tfm(momento_curvatura_autor), 1),
                          num(cm_para_m(excentricidade_total_rotina), 5), num(kncm_para_tfm(momento_curvatura), 2)),
                       exata=kncm_para_tfm(momento_curvatura))
    momento_kappa = momento_total_kappa_direto(altura, forca, comprimento, alfa_b, momento_minimo)
    kappa = kappa_de_kappa_nu(kappa_sobre_nu(momento_kappa, altura, forca), nu)
    momento_kappa_impresso = tfm_para_kncm(valor_impresso(impresso["Md_kappa"]))
    kappa_autor = kappa_de_kappa_nu(kappa_sobre_nu(momento_kappa_impresso, altura, forca), nu_impresso)
    kappa_nu_sem_arredondar = kappa_de_kappa_nu(kappa_sobre_nu(momento_kappa_impresso, altura, forca), nu)
    momento_kappa_autor = produto(forca, soma(excentricidade_minima_impressa, valor_impresso(impresso["e2_kappa"])))
    kappa_com_e2_arredondado = kappa_de_kappa_nu(kappa_sobre_nu(momento_kappa_autor, altura, forca), nu)
    registro.adicionar(nome, "C.2: κ = 32 (1 + 5 Md,tot/(h Nd)) ν", "", CRITERIO_IMPRESSAO, impresso["kappa"],
                       kappa_autor, kappa,
                       "Cadeia da fonte com o ν (0,53) e o Md,tot (52,3 tf·m) impressos: %s (%s com uma casa). O "
                       "34,8 impresso vem do ν sem arredondar (%s): %s; a fonte leva à etapa seguinte o valor sem "
                       "arredondar. Coluna “rotina”: ν e Md,tot da rotina, sem arredondamentos."
                       % (num(kappa_autor, 2), num(kappa_autor, 1), num(nu, 4), num(kappa_nu_sem_arredondar, 2)),
                       exata=kappa,
                       nota_exata="Com o Md,tot da solução exata (%s tf·m), κ = %s, que dá %s. O autor leva ao κ o "
                                  "Md,tot do e2 arredondado (840 × 0,0622 = 52,25 tf·m), com o qual κ = %s."
                                  % (num(kncm_para_tfm(momento_kappa), 2), num(kappa, 3), num(kappa, 1),
                                     num(kappa_com_e2_arredondado, 3)))
    registro.adicionar(nome, "C.2: e2 (rigidez κ aproximada)", "cm", CRITERIO_IMPRESSAO, impresso["e2_kappa"],
                       excentricidade_segunda_ordem(momento_kappa, forca, excentricidade_minima),
                       excentricidade_segunda_ordem(momento_kappa, forca, excentricidade_minima),
                       "Solução direta da equação do 2º grau da Seção 15.8.3.3.3.",
                       exata=excentricidade_segunda_ordem(momento_kappa, forca, excentricidade_minima))
    registro.adicionar(nome, "C.2: Md,tot = Nd etot (etot = 2,40 + 3,82)", "tf·m", CRITERIO_IMPRESSAO,
                       impresso["Md_kappa"], kncm_para_tfm(momento_kappa_autor), kncm_para_tfm(momento_kappa),
                       "O texto escreve 840 × 0,0622 = 52,3; o produto é 52,248, que dá 52,2 com uma casa "
                       "(arredondamento errado ou deslize de conta da fonte). A solução exata da equação do 2º grau "
                       "da norma é a da coluna “rotina”.",
                       exata=kncm_para_tfm(momento_kappa),
                       nota_exata="A solução exata dá %s tf·m (52,2), e o produto impresso, 52,248 (52,2): o 52,3 não "
                                  "decorre de nenhuma das contas (arredondamento errado ou deslize de conta)."
                                  % num(kncm_para_tfm(momento_kappa), 3),
                       nota_tabela="Erro da fonte: o texto escreve 840 × 0,0622 = 52,3, mas o produto é 52,248 tf·m, "
                                   "que dá 52,2 com uma casa, como a cadeia exata (%s tf·m)."
                                   % num(kncm_para_tfm(momento_kappa), 3))
    estribo = dados["estribo"]
    distancia = nucleo.distancia_eixo_barra(dados["cobrimento"], estribo, dados["barra"])
    numero_barras = numero_total(dados["barras_por_face"], 2)
    area = nucleo.area_barras(dados["barra"], numero_barras)
    registro.adicionar(nome, "C.3: As adotada = 16 ϕ 16", "cm²", CRITERIO_IMPRESSAO, impresso["As_adotada"], area,
                       area, exata=area)
    secao = secao_camadas(altura, largura, distancia, dados["barras_por_face"])
    resultados = {}
    for liquida, rotulo_secao in ((False, "seção bruta"), (True, "seção líquida")):
        with SecaoHipotese(liquida):
            momento_resistente = nucleo.momento_resistente(secao, material, area, forca)
            forca_deformacao = nucleo.forca_normal_curva_deformacao(forca)
            curvatura_b = curvatura_do_momento_deformacao(secao, material, area, forca_deformacao,
                                                          validacao.momento_ponto_b(momento_resistente))
            rigidez = validacao.rigidez_secante(momento_resistente, curvatura_b)
            kappa_sec = validacao.kappa_secante(rigidez, area_concreto, altura, material.fcd)
            momento_secante = validacao.momento_total_kappa_secante(alfa_b, momento_minimo, esbeltez, nu, kappa_sec)
            equilibrio = nucleo.equilibrio_pilar_padrao_acoplado(secao, material, area, forca, momento_minimo, alfa_b,
                                                                 comprimento)
            momento_geral, situacao_geral = metodo_geral_constante(secao, material, area, forca, momento_minimo,
                                                                    comprimento)
        resultados[rotulo_secao] = {"MRd_kNcm": momento_resistente, "curvatura_B_1_cm": curvatura_b,
                                    "EI_sec_kNcm2": rigidez, "kappa_sec": kappa_sec,
                                    "Md_tot_secante_kNcm": momento_secante,
                                    "curvatura_equilibrio_1_cm": equilibrio[0], "Md_tot_equilibrio_kNcm": equilibrio[1],
                                    "Md_tot_metodo_geral_kNcm": momento_geral, "situacao_metodo_geral": situacao_geral}
        sufixo = " (%s)" % rotulo_secao
        registro.adicionar(nome, "C.3: MRd com 16 ϕ 16" + sufixo, "tf·m", CRITERIO_PROGRAMA, impresso["MRd"], None,
                           kncm_para_tfm(momento_resistente),
                           "MRd do diagrama da figura do item C.3. A seção bruta é a hipótese provável do programa; a "
                           "rotina usa a seção líquida (Seção 3.5).")
        registro.adicionar(nome, "C.3: M do ponto B = MRd/γf3" + sufixo, "tf·m", CRITERIO_PROGRAMA,
                           impresso["M_ponto_B"], None, kncm_para_tfm(validacao.momento_ponto_b(momento_resistente)))
        registro.adicionar(nome, "C.3: 1/r do ponto B (curva com 1,10 fcd e Nd/γf3)" + sufixo, "m⁻¹",
                           CRITERIO_PROGRAMA, impresso["curvatura_ponto_B"], None, por_cm_para_por_metro(curvatura_b),
                           "A figura escreve 5,85E-3 (m⁻¹). Os parâmetros internos do programa que gerou a figura não "
                           "são informados no artigo.")
        registro.adicionar(nome, "C.3: (EI)sec = (MRd/γf3)/(1/r)B" + sufixo, "tf·m²", CRITERIO_PROGRAMA,
                           impresso["EI_sec"], None, kncm2_para_tfm2(rigidez))
        registro.adicionar(nome, "C.3: κsec = (EI)sec/(Ac h² fcd)" + sufixo, "", CRITERIO_PROGRAMA,
                           impresso["kappa_sec"], None, kappa_sec)
        registro.adicionar(nome, "C.3: Md,tot pela rigidez secante (forma do item C.3)" + sufixo, "tf·m",
                           CRITERIO_PROGRAMA, impresso["Md_mn1r"], None, kncm_para_tfm(momento_secante),
                           "κsec da curva de deformação calculada pela rotina.")
        registro.adicionar(nome, "C.3: e2 pela rigidez secante" + sufixo, "cm", CRITERIO_PROGRAMA,
                           impresso["e2_mn1r"], None,
                           excentricidade_segunda_ordem(momento_secante, forca, excentricidade_minima))
        registro.adicionar(nome, "C.3: Md,tot pela curva de deformação, sem linearização (método da rotina)" + sufixo,
                           "tf·m", CRITERIO_PROGRAMA, impresso["Md_mn1r"], None, kncm_para_tfm(equilibrio[1]),
                           "A rotina usa a própria curva de deformação (menor curvatura de equilíbrio); o item C.3 a "
                           "lineariza pela secante AB (EIsec), que a Seção 15.3.1 admite a favor da segurança.")
        registro.adicionar(nome, "C.4: Md,tot pelo método geral (M1d,mín constante)" + sufixo, "tf·m",
                           CRITERIO_PROGRAMA, impresso["Md_geral"], None, kncm_para_tfm(momento_geral),
                           "Método geral de validacao.py com o momento mínimo constante ao longo do pilar. Pela figura "
                           "do item C.4, o programa usou M1d,mín = 20,2 tf·m com o sinal do diagrama de 1ª ordem "
                           "(negativo nas duas estações da base); o texto não informa as imperfeições nem o modelo do "
                           "método geral.",
                           nota_tabela="Método geral da rotina com o momento mínimo constante ao longo do pilar; o "
                                       "artigo não informa as imperfeições nem o modelo do método geral do programa.")
    momento_kappa_impresso_sec = validacao.momento_total_kappa_secante(
        alfa_b, tfm_para_kncm(valor_impresso(impresso["M1d_min"])), valor_impresso(impresso["lambda"]), nu_impresso,
        valor_impresso(impresso["kappa_sec"]))
    momento_kappa_secante_exato = validacao.momento_total_kappa_secante(alfa_b, momento_minimo, esbeltez, nu,
                                                                        valor_impresso(impresso["kappa_sec"]))
    registro.adicionar(nome, "C.3: Md,tot = αb M1d,A/[1 - λ² ν/(120 κsec)] com o κsec impresso (62,2)", "tf·m",
                       CRITERIO_IMPRESSAO, impresso["Md_mn1r"], kncm_para_tfm(momento_kappa_impresso_sec), None,
                       "Confirma a forma do item C.3: a rigidez secante entra na expressão da rigidez κ. Cadeia da "
                       "fonte com M1d,mín, λ e ν impressos (20,2 tf·m, 69,3 e 0,53).",
                       exata=kncm_para_tfm(momento_kappa_secante_exato))
    return {"dados": {chave: valor for chave, valor in dados.items() if chave != "impresso"},
            "d_linha_cm": distancia, "As_cm2": area, "por_hipotese_de_secao": resultados}


# ----------------------------------------------------------------------------- L2 — Scadelai e Pinheiro (2007)
def executar_scadelai(registro: Registro) -> dict:
    dados = SCADELAI
    impresso = dados["impresso"]
    nome = "L2 Scadelai e Pinheiro (2007)"
    material = nucleo.Material(dados["fck"])
    altura = dados["altura"]
    largura = dados["largura"]
    comprimento = dados["comprimento"]
    area_concreto = nucleo.area_secao_retangular(largura, altura)
    forca = forca_de_coeficientes(nucleo.GAMA_F, dados["gama_n"], dados["Nk"])
    forca_impressa = valor_impresso(impresso["Nd"])
    nu = nucleo.forca_normal_adimensional(forca, area_concreto, material.fcd)
    nu_impresso = valor_impresso(impresso["nu"])
    esbeltez = nucleo.indice_esbeltez(comprimento, altura)
    momento_topo = knm_para_kncm(forca_de_coeficientes(nucleo.GAMA_F, dados["gama_n"], dados["M_topo_k"]))
    momento_minimo = nucleo.momento_minimo_primeira_ordem(forca, altura)
    menores = nucleo.momentos_menores_que_minimo(momento_topo, momento_minimo)
    alfa_b = nucleo.coeficiente_alfa_b(momento_topo, momento_topo, menores)
    limite_rotina = nucleo.esbeltez_limite(nucleo.excentricidade_relativa(momento_topo, forca, altura), alfa_b)
    limite_autor = nucleo.esbeltez_limite(0.0, alfa_b)
    registro.adicionar(nome, "Nd = γf γn Nk", "kN", CRITERIO_IMPRESSAO, impresso["Nd"], forca, forca, exata=forca)
    registro.adicionar(nome, "ν = Nd/(Ac fcd)", "", CRITERIO_IMPRESSAO, impresso["nu"], nu, nu, exata=nu)
    registro.adicionar(nome, "i = h/raiz(12)", "cm", CRITERIO_IMPRESSAO, impresso["raio_giracao"],
                       raio_giracao(altura), raio_giracao(altura), exata=raio_giracao(altura))
    registro.adicionar(nome, "λ = ℓe/i", "", CRITERIO_IMPRESSAO, impresso["lambda"], esbeltez, esbeltez,
                       exata=esbeltez)
    registro.adicionar(nome, "Md,topo = γf γn Mk", "kN·m", CRITERIO_IMPRESSAO, impresso["Md_topo"],
                       kncm_para_knm(momento_topo), kncm_para_knm(momento_topo), exata=kncm_para_knm(momento_topo))
    registro.adicionar(nome, "ei = Md/Nd", "cm", CRITERIO_IMPRESSAO, impresso["e_inicial"],
                       excentricidade(momento_topo, forca), excentricidade(momento_topo, forca),
                       exata=excentricidade(momento_topo, forca))
    registro.adicionar(nome, "M1d,mín = Nd (0,015 + 0,03 h)", "kN·m", CRITERIO_IMPRESSAO, impresso["M1d_min"],
                       kncm_para_knm(momento_minimo), kncm_para_knm(momento_minimo),
                       exata=kncm_para_knm(momento_minimo))
    registro.adicionar(nome, "λ1 (35 <= λ1 <= 90)", "", CRITERIO_IMPRESSAO, impresso["lambda1"], limite_autor,
                       limite_rotina, "O texto usa e1 = 0 (λ1 = 25 -> 35); a rotina usa e1/h do momento aplicado "
                                      "(λ1 = 26,4 -> 35): o mesmo resultado.", exata=limite_autor)
    curvatura_autor = nucleo.curvatura_aproximada(nu_impresso, altura)
    curvatura = nucleo.curvatura_aproximada(nu, altura)
    registro.adicionar(nome, "1/r = 0,005/[h (ν + 0,5)]", "m⁻¹", CRITERIO_IMPRESSAO, impresso["curvatura"],
                       por_cm_para_por_metro(curvatura_autor), por_cm_para_por_metro(curvatura),
                       "Cadeia da fonte com ν impresso (0,91).", exata=por_cm_para_por_metro(curvatura),
                       nota_exata="O autor leva adiante o ν impresso (0,91) no lugar de %s." % num(nu, 4))
    momento_impresso_minimo = knm_para_kncm(valor_impresso(impresso["M1d_min"]))
    momento_curvatura_autor = nucleo.momento_total_pilar_padrao(
        1.0, momento_impresso_minimo, forca_impressa, comprimento,
        por_metro_para_por_cm(valor_impresso(impresso["curvatura"])))
    momento_curvatura = nucleo.momento_total_pilar_padrao(alfa_b, momento_minimo, forca, comprimento, curvatura)
    registro.adicionar(nome, "Md,tot = αb M1d,A + Nd (ℓe²/10)(1/r)", "kN·m", CRITERIO_IMPRESSAO,
                       impresso["Md_curvatura"], kncm_para_knm(momento_curvatura_autor),
                       kncm_para_knm(momento_curvatura), "Cadeia da fonte com 1/r impresso (0,0236).",
                       exata=kncm_para_knm(momento_curvatura),
                       nota_exata="O autor leva adiante o 1/r impresso (0,0236 m⁻¹) no lugar de %s m⁻¹."
                                  % num(por_cm_para_por_metro(curvatura), 6))
    etot_impresso = valor_impresso(impresso["etot_curvatura"])
    registro.adicionar(nome, "etot = Md,tot/Nd", "cm", CRITERIO_IMPRESSAO, impresso["etot_curvatura"],
                       excentricidade(knm_para_kncm(valor_impresso(impresso["Md_curvatura"])), forca_impressa),
                       excentricidade(momento_curvatura, forca), exata=excentricidade(momento_curvatura, forca))
    mu_rotina = momento_adimensional(momento_curvatura, area_concreto, altura, material.fcd)
    registro.adicionar(nome, "μ = ν etot/h", "", CRITERIO_IMPRESSAO, impresso["mu_curvatura"],
                       mu_primeira_ordem(nu_impresso, etot_impresso, altura), mu_rotina,
                       exata=mu_primeira_ordem(nu, excentricidade(momento_curvatura, forca), altura))
    fator = fator_area_aco(area_concreto, material.fcd, material.fyd)
    registro.adicionar(nome, "Ac fcd/fyd", "cm²", CRITERIO_IMPRESSAO, impresso["fator_As"], fator, fator,
                       exata=fator)
    resultados = {}
    distancia_abaco = produto(dados["distancia_abaco"], altura)
    secao_abaco = secao_camadas(altura, largura, distancia_abaco, dados["barras_por_face"])
    secao_real = secao_camadas(altura, largura, dados["distancia"], dados["barras_por_face"])
    for chave_omega, chave_mu, chave_area, chave_rho, momento_rotina, rotulo in (
            ("omega_curvatura", "mu_curvatura", "As_curvatura", "rho_curvatura", momento_curvatura,
             "curvatura aproximada"),
            ("omega_kappa", "mu_kappa", "As_kappa", "rho_kappa", None, "rigidez κ")):
        forca_ponto = nucleo.forca_normal_de_nu(nu_impresso, area_concreto, material.fcd)
        momento_ponto = momento_de_mu(valor_impresso(impresso[chave_mu]), area_concreto, altura, material.fcd)
        with SecaoHipotese(False):
            area_ponto = nucleo.armadura_flexo_compressao(secao_abaco, material, forca_ponto, momento_ponto)
        registro.adicionar(nome, "ω do ábaco A-5 no ponto (ν; μ) impresso, %s" % rotulo, "", CRITERIO_LEITURA,
                           impresso[chave_omega], None, taxa_mecanica(area_ponto, material.fyd, area_concreto,
                                                                      material.fcd),
                           "Rotina na hipótese do ábaco: seção bruta, duas camadas, d′/h = 0,25."
                           + (" A linha do texto traz também 0,86 numa etapa da conta (As = ... · 0,86 = 27,72 · "
                              "0,88)." if chave_omega == "omega_kappa" else ""))
        area_autor = produto(valor_impresso(impresso["fator_As"]), valor_impresso(impresso[chave_omega]))
        area_exata = produto(fator, valor_impresso(impresso[chave_omega]))
        registro.adicionar(nome, "As = (Ac fcd/fyd) ω, %s" % rotulo, "cm²", CRITERIO_IMPRESSAO, impresso[chave_area],
                           area_autor, None, exata=area_exata,
                           nota_exata="O autor usa o fator impresso, 27,72 cm², no lugar de %s cm²." % num(fator, 4))
        registro.adicionar(nome, "ρ = As/Ac, %s" % rotulo, "%", CRITERIO_IMPRESSAO, impresso[chave_rho],
                           taxa_percentual(valor_impresso(impresso[chave_area]), area_concreto), None,
                           exata=taxa_percentual(area_exata, area_concreto))
        if momento_rotina is not None:
            for liquida, rotulo_secao in ((False, "seção bruta"), (True, "seção líquida")):
                with SecaoHipotese(liquida):
                    area_real = nucleo.armadura_flexo_compressao(secao_real, material, forca, momento_rotina)
                resultados["As_d4_" + rotulo_secao] = area_real
                registro.adicionar(nome, "As da rotina com d′ = 4,0 cm (d′/h = 0,27), %s" % rotulo_secao, "cm²",
                                   CRITERIO_LEITURA, impresso[chave_area], None, area_real,
                                   "O texto entra no ábaco de d′/h = 0,25 com d′/h = 0,27.")
    kappa_nu_1 = kappa_sobre_nu(knm_para_kncm(valor_impresso(impresso["Md_curvatura"])), altura, forca_impressa)
    esbeltez_impressa = valor_impresso(impresso["lambda"])
    momento_11 = momento_total_kappa(1.0, momento_impresso_minimo, esbeltez_impressa,
                                     valor_impresso(impresso["kappa_nu_1"]))
    momento_20 = media(valor_impresso(impresso["Md_curvatura"]), valor_impresso(impresso["Md_kappa_11"]))
    kappa_nu_2 = kappa_sobre_nu(knm_para_kncm(valor_impresso(impresso["Md_kappa_20"])), altura, forca_impressa)
    momento_21 = momento_total_kappa(1.0, momento_impresso_minimo, esbeltez_impressa,
                                     valor_impresso(impresso["kappa_nu_2"]))
    momento_30 = media(valor_impresso(impresso["Md_kappa_20"]), valor_impresso(impresso["Md_kappa_21"]))
    momento_kappa_rotina = momento_total_kappa_direto(altura, forca, comprimento, alfa_b, momento_minimo)
    kappa_nu_1_exato = kappa_sobre_nu(momento_curvatura, altura, forca)
    momento_11_exato = momento_total_kappa(1.0, momento_minimo, esbeltez, kappa_nu_1_exato)
    momento_20_exato = media(momento_curvatura, momento_11_exato)
    kappa_nu_2_exato = kappa_sobre_nu(momento_20_exato, altura, forca)
    momento_21_exato = momento_total_kappa(1.0, momento_minimo, esbeltez, kappa_nu_2_exato)
    momento_30_exato = media(momento_20_exato, momento_21_exato)
    nota_iteracao_exata = ("O autor parte do Md,tot impresso (39,39 kN·m, com o 1/r arredondado) e leva adiante os "
                           "valores impressos de cada iteração; a cadeia exata parte de %s kN·m."
                           % num(kncm_para_knm(momento_curvatura), 3))
    nota_tabela_kappa = ("A fonte itera a rigidez κ e adota a média das duas últimas iterações (%s kN·m na cadeia "
                         "exata); a rotina resolve diretamente a equação do 2º grau da Seção 15.8.3.3.3 (%s kN·m)."
                         % (num(kncm_para_knm(momento_30_exato), 2), num(kncm_para_knm(momento_kappa_rotina), 2)))
    for chave, rotulo, autor, unidade, exata in (
            ("kappa_nu_1", "1ª iteração: κ/ν com Md,tot = 39,39", kappa_nu_1, "", kappa_nu_1_exato),
            ("Md_kappa_11", "1ª iteração: Md,tot", kncm_para_knm(momento_11), "kN·m", kncm_para_knm(momento_11_exato)),
            ("Md_kappa_20", "estimativa: média", momento_20, "kN·m", kncm_para_knm(momento_20_exato)),
            ("kappa_nu_2", "2ª iteração: κ/ν", kappa_nu_2, "", kappa_nu_2_exato),
            ("Md_kappa_21", "2ª iteração: Md,tot", kncm_para_knm(momento_21), "kN·m", kncm_para_knm(momento_21_exato)),
            ("Md_kappa_30", "Md,tot adotado (média)", momento_30, "kN·m", kncm_para_knm(momento_30_exato))):
        rotina = kncm_para_knm(momento_kappa_rotina) if chave == "Md_kappa_30" else None
        registro.adicionar(nome, "Rigidez κ, " + rotulo, unidade, CRITERIO_IMPRESSAO, impresso[chave], autor, rotina,
                           "Coluna “rotina”: solução direta da equação do 2º grau da Seção 15.8.3.3.3."
                           if rotina is not None else "", exata=exata, nota_exata=nota_iteracao_exata,
                           nota_tabela=nota_tabela_kappa if rotina is not None else "")
    registro.adicionar(nome, "Rigidez κ: etot = Md,tot/Nd", "cm", CRITERIO_IMPRESSAO, impresso["etot_kappa"],
                       excentricidade(knm_para_kncm(valor_impresso(impresso["Md_kappa_30"])), forca_impressa),
                       excentricidade(momento_kappa_rotina, forca), exata=excentricidade(momento_30_exato, forca),
                       nota_tabela=nota_tabela_kappa)
    registro.adicionar(nome, "Rigidez κ: μ = ν etot/h", "", CRITERIO_IMPRESSAO, impresso["mu_kappa"],
                       mu_primeira_ordem(nu_impresso, valor_impresso(impresso["etot_kappa"]), altura),
                       momento_adimensional(momento_kappa_rotina, area_concreto, altura, material.fcd),
                       exata=mu_primeira_ordem(nu, excentricidade(momento_30_exato, forca), altura),
                       nota_tabela=nota_tabela_kappa)
    numero_adotado = dados["barras_adotadas"]
    area_adotada = nucleo.area_barras(dados["barra"], numero_adotado)
    registro.adicionar(nome, "As adotada = 12 ϕ 16", "cm²", CRITERIO_IMPRESSAO, impresso["As_adotada"],
                       produto(numero_adotado, 2.0), area_adotada, "O texto usa 2,0 cm² por barra de 16 mm.",
                       exata=area_adotada,
                       nota_exata="O autor usa 2,0 cm² por barra de 16 mm (área nominal π 1,6²/4 = 2,011 cm²).")
    with SecaoHipotese(True):
        momento_resistente_adotado = nucleo.momento_resistente(secao_real, material, area_adotada, forca)
    resultados["MRd_12phi16_liquida_kNcm"] = momento_resistente_adotado
    area_adotada_texto = produto(numero_adotado, 2.0)
    with SecaoHipotese(False):
        momento_resistente_abaco = nucleo.momento_resistente(secao_abaco, material, area_adotada_texto, forca)
    resultados["MRd_24cm2_abaco_kNcm"] = momento_resistente_abaco
    registro.adicionar(nome, "MRd com 24,0 cm² na hipótese do ábaco (bruta, d′/h = 0,25), comparado com Md,tot",
                       "kN·m", CRITERIO_LEITURA, impresso["Md_curvatura"], None,
                       kncm_para_knm(momento_resistente_abaco),
                       "MRd da área adotada no texto (24,0 cm², menor que os 24,95 cm² calculados), calculado pela "
                       "rotina na hipótese do ábaco e comparado com o Md,tot impresso, de outra grandeza: comparação só "
                       "pela diferença, como as leituras.")
    registro.adicionar(nome, "MRd com 12 ϕ 16, d′ = 4,0 cm (seção líquida), comparado com Md,tot", "kN·m",
                       CRITERIO_LEITURA, impresso["Md_curvatura"], None, kncm_para_knm(momento_resistente_adotado),
                       "MRd da armadura adotada no texto, calculado pela rotina e comparado com o Md,tot impresso, de "
                       "outra grandeza: comparação só pela diferença, como as leituras.")
    menor_dimensao = min(largura, altura)
    registro.verificar(nome, "ϕ 16 atende 10 mm <= ϕl <= b/8 (18.4.2.1)",
                       nucleo.diametro_longitudinal_permitido(dados["barra"], menor_dimensao), dados["fonte"])
    registro.verificar(nome, "ϕ 20 (alternativa 8 ϕ 20 do texto) atende 10 mm <= ϕl <= b/8 = 18,75 mm (18.4.2.1)",
                       nucleo.diametro_longitudinal_permitido(dados["barra_alternativa"], menor_dimensao),
                       dados["fonte"], "A rotina não admite ϕ 20 em seção de 15 cm.")
    estribo = nucleo.diametro_estribo(dados["barra"])
    registro.adicionar(nome, "ϕt = máx(5 mm; ϕl/4)", "mm", CRITERIO_IMPRESSAO, impresso["estribo"],
                       cm_para_mm(estribo), cm_para_mm(estribo), exata=cm_para_mm(estribo))
    espacamento = nucleo.espacamento_maximo_estribos(menor_dimensao, dados["barra"])
    registro.adicionar(nome, "st,máx = mín(20 cm; b; 12 ϕl)", "cm", CRITERIO_IMPRESSAO,
                       impresso["espacamento_estribo"], espacamento, espacamento, exata=espacamento)
    trecho = nucleo.trecho_protegido_estribo(dados["estribo"])
    registro.adicionar(nome, "Trecho protegido 20 ϕt", "cm", CRITERIO_IMPRESSAO, impresso["trecho_protegido"],
                       trecho, trecho, exata=trecho)
    face = comprimento_face(largura, dados["distancia"])
    sem_protecao = numero_total(nucleo.barras_sem_protecao_face(face, dados["barras_por_face"], dados["estribo"]), 2)
    registro.adicionar(nome, "Barras que exigem estribo suplementar (2 faces)", "", CRITERIO_IMPRESSAO,
                       impresso["barras_sem_protecao"], sem_protecao, sem_protecao, exata=sem_protecao)
    face_espacamento = nucleo.espacamento_eixos_face(face, dados["barras_por_face"])
    registro.verificar(nome, "Espaçamento entre eixos <= mín(2b; 40 cm) (18.4.2.2)",
                       face_espacamento <= nucleo.espacamento_maximo_eixos(menor_dimensao), dados["fonte"],
                       "s = %s cm" % num(face_espacamento, 2))
    resultados.update({"Nd_kN": forca, "Md_tot_curvatura_kNcm": momento_curvatura,
                       "Md_tot_kappa_kNcm": momento_kappa_rotina, "As_adotada_cm2": area_adotada})
    return resultados


# ----------------------------------------------------------------------------- L3 — Fusco (1981)
def curva_fusco(material: nucleo.Material, distancia_relativa: float, omega: float, pontos: int) -> tuple:
    """Curva μ x 10³ d/r da Fig. 8.2.5-1 calculada pela rotina: seção bruta, duas camadas, 0,85 fcd e N = ν Ac fcd."""
    dados = FUSCO
    altura = dados["altura"]
    area_concreto = nucleo.area_secao_retangular(dados["largura"], altura)
    distancia = produto(distancia_relativa, altura)
    secao = secao_camadas(altura, dados["largura"], distancia, 1)
    area = area_de_omega(omega, area_concreto, material.fcd, material.fyd)
    forca = nucleo.forca_normal_de_nu(dados["nu_diagrama"], area_concreto, material.fcd)
    with SecaoHipotese(False):
        curvaturas, momentos = nucleo.curva_momento_curvatura(secao, material, area, forca, False, pontos)
    abscissas = abscissa_fusco(curvaturas, altura_util(altura, distancia_relativa))
    return abscissas, momento_adimensional(momentos, area_concreto, altura, material.fcd)


def omega_pilar_padrao_fusco(material: nucleo.Material, distancia_relativa: float, mu1: float,
                             inclinacao: float) -> float:
    """Menor ω cuja curva da rotina alcança a reta μ = μ1d + tg θ (10³ d/r) (construção de Fusco, 1981, p. 291)."""
    def folga(omega: float) -> float:
        abscissas, mus = curva_fusco(material, distancia_relativa, omega, PONTOS_CURVA_FIGURA)
        return folga_reta(mus, mu1, inclinacao, abscissas)
    return brentq(folga, 0.05, 2.0, xtol=1e-6)


def rasterizar_pagina(caminho_pdf: str, pagina: int, pasta: str) -> Path | None:
    if shutil.which("pdftoppm") is None:
        return None
    prefixo = Path(pasta) / "pagina"
    subprocess.run(["pdftoppm", "-f", str(pagina), "-l", str(pagina), "-r", str(DPI_DIGITALIZACAO), "-png",
                    caminho_pdf, str(prefixo)], check=True)
    encontrados = sorted(Path(pasta).glob("pagina*.png"))
    return encontrados[0] if encontrados else None


def digitalizar_fusco(material: nucleo.Material, registro: Registro) -> None:
    """Compara as curvas impressas da Fig. 8.2.5-1 (página do PDF) com as calculadas pela rotina ([N02])."""
    from PIL import Image, ImageDraw
    caminho = next((caminho for caminho in CAMINHOS_PDF_FUSCO if caminho and Path(caminho).exists()), None)
    if caminho is None:
        registro.digitalizacao = {"situacao": "PDF de Fusco (1981) não encontrado; digitalização não feita"}
        return
    with tempfile.TemporaryDirectory() as pasta:
        imagem_pagina = rasterizar_pagina(caminho, PAGINA_PDF_FIGURA_FUSCO, pasta)
        if imagem_pagina is None:
            registro.digitalizacao = {"situacao": "pdftoppm indisponível; digitalização não feita"}
            return
        pagina = Image.open(imagem_pagina).convert("RGB")
    largura_pagina, altura_pagina = pagina.size
    cinza = np.asarray(pagina.convert("L"), dtype=int)
    resumo = {"situacao": "feita", "pdf": Path(caminho).name, "pagina_pdf": PAGINA_PDF_FIGURA_FUSCO, "paineis": {}}
    for distancia_relativa, recorte in RECORTES_PAINEIS.items():
        topo, base = faixa_em_pixels(recorte[0], recorte[1], altura_pagina)
        esquerda, direita = faixa_em_pixels(recorte[2], recorte[3], largura_pagina)
        escuro = cinza[topo:base, esquerda:direita] < LIMIAR_ESCURO
        faixa_colunas = escuro[300:1300, :]
        contagem_colunas = faixa_colunas.sum(axis=0)
        colunas = separar_grupos_consecutivos(
            [indice for indice in range(contagem_colunas.size)
             if contagem_colunas[indice] > minimo_pixels_linha(faixa_colunas.shape[0])])
        faixa_linhas = escuro[:, 200:850]
        contagem_linhas = faixa_linhas.sum(axis=1)
        linhas = separar_grupos_consecutivos(
            [indice for indice in range(contagem_linhas.size)
             if contagem_linhas[indice] > minimo_pixels_linha(faixa_linhas.shape[1])])
        calibracao_x = ajuste_linear(abscissas_principais(), colunas[-4:])
        calibracao_mu = ajuste_linear(valores_grade_principal(len(linhas)), linhas)
        grade_menor = [pixel_do_valor(calibracao_mu, valor) for valor in valores_grade_menor(0.48)]
        cota = FUSCO["cota_reta_x4"][distancia_relativa]
        inclinacao = inclinacao_desenhada(cota, 4.0)
        mu1 = valor_impresso(FUSCO["impresso"]["mu1"])
        pontos = []
        recorte_imagem = pagina.crop((esquerda, topo, direita, base))
        desenho = ImageDraw.Draw(recorte_imagem)
        for omega in OMEGAS_FIGURA:
            abscissas, mus = curva_fusco(material, distancia_relativa, omega, PONTOS_CURVA_FIGURA)
            tracado = [(pixel_do_valor(calibracao_x, abscissa), pixel_do_valor(calibracao_mu, mu))
                       for abscissa, mu in zip(abscissas, mus) if abscissa <= 4.0]
            desenho.line(tracado, fill=(220, 0, 0), width=3)
            for abscissa in ABSCISSAS_AMOSTRA:
                if abscissa > abscissas[-1]:
                    continue
                mu_rotina = float(np.interp(abscissa, abscissas, mus))
                retas = (produto(inclinacao, abscissa), soma(mu1, produto(inclinacao, abscissa)))
                if any(dentro_janela(reta, mu_rotina) for reta in retas):
                    continue
                coluna = pixel_inteiro(pixel_do_valor(calibracao_x, abscissa))
                inicio_coluna, fim_coluna = colunas_vizinhas(coluna)
                perfil = escuro[:, inicio_coluna:fim_coluna].any(axis=1).copy()
                for pixel in grade_menor:
                    inicio, fim = intervalo_margem(pixel)
                    perfil[inicio:fim] = False
                tracos = []
                indice = 0
                while indice < perfil.size:
                    if perfil[indice]:
                        fim = indice
                        while fim < perfil.size and perfil[fim]:
                            fim += 1
                        tracos.append(valor_do_pixel(calibracao_mu, centro_traco(indice, fim)))
                        indice = fim
                    else:
                        indice += 1
                proximos = [mu for mu in tracos if dentro_janela(mu, mu_rotina)]
                if len(proximos) == 1:
                    pontos.append({"omega": omega, "abscissa": abscissa, "mu_rotina": mu_rotina,
                                   "mu_figura": proximos[0], "diferenca": diferenca(proximos[0], mu_rotina)})
        recorte_imagem.save(caminho_sobreposicao(distancia_relativa))
        diferencas = [ponto["diferenca"] for ponto in pontos]
        resumo["paineis"][str(distancia_relativa)] = {
            "pontos": pontos, "quantidade": len(pontos), "maior_diferenca_absoluta": maximo_absoluto(diferencas),
            "diferenca_media": media_valores(diferencas), "calibracao_x": list(calibracao_x),
            "calibracao_mu": list(calibracao_mu)}
        registro.adicionar("L3 Fusco (1981)", "Fig. 8.2.5-1, d′/h = %s: maior |μ figura - μ rotina| em %d pontos"
                           % (num(distancia_relativa, 2), len(pontos)), "", CRITERIO_LEITURA, None, None,
                           maximo_absoluto(diferencas),
                           "Curvas impressas digitalizadas da página do livro; menor divisão da grade: 0,02.")
    registro.digitalizacao = resumo


def executar_fusco(registro: Registro) -> dict:
    dados = FUSCO
    impresso = dados["impresso"]
    nome = "L3 Fusco (1981)"
    material = nucleo.Material(dados["fck"])
    altura = dados["altura"]
    largura = dados["largura"]
    comprimento = dados["comprimento"]
    area_concreto = nucleo.area_secao_retangular(largura, altura)
    fcd_impresso = dados["fcd_impresso"]
    fyd_impresso = dados["fyd_impresso"]
    nu_autor = nucleo.forca_normal_adimensional(dados["Nd"], area_concreto, fcd_impresso)
    nu = nucleo.forca_normal_adimensional(dados["Nd"], area_concreto, material.fcd)
    registro.adicionar(nome, "νd = Nd/(Ac fcd)", "", CRITERIO_IMPRESSAO, impresso["nu"], nu_autor, nu, exata=nu)
    mu1_autor = mu_primeira_ordem(valor_impresso(impresso["nu"]), dados["e1"], altura)
    registro.adicionar(nome, "μ1d = νd e1/h", "", CRITERIO_IMPRESSAO, impresso["mu1"], mu1_autor,
                       mu_primeira_ordem(nu, dados["e1"], altura), exata=mu_primeira_ordem(nu, dados["e1"], altura))
    resultados = {"omega_rotina": {}}
    for distancia_relativa, chave_tg, chave_omega in ((0.05, "tg_095", "omega_095"), (0.15, "tg_085", "omega_085")):
        util = altura_util(altura, distancia_relativa)
        inclinacao = inclinacao_reta_fusco(dados["nu_diagrama"], comprimento, altura, util)
        registro.adicionar(nome, "tg θ = ν/10000 (ℓe/h)² (h/d), d′/h = %s" % num(distancia_relativa, 2), "",
                           CRITERIO_IMPRESSAO, impresso[chave_tg], inclinacao, inclinacao, exata=inclinacao)
        inclinacao_figura = inclinacao_desenhada(dados["cota_reta_x4"][distancia_relativa], 4.0)
        mu1 = valor_impresso(impresso["mu1"])
        omega_texto = omega_pilar_padrao_fusco(material, distancia_relativa, mu1, inclinacao)
        omega_figura = omega_pilar_padrao_fusco(material, distancia_relativa, mu1, inclinacao_figura)
        resultados["omega_rotina"][str(distancia_relativa)] = {"tg_texto": inclinacao, "omega_tg_texto": omega_texto,
                                                              "tg_desenhada": inclinacao_figura,
                                                              "omega_tg_desenhada": omega_figura}
        registro.adicionar(nome, "ω da reta tangente, d′/h = %s, tg θ do texto (%s)"
                           % (num(distancia_relativa, 2), num(inclinacao, 4)), "", CRITERIO_LEITURA, impresso[chave_omega], None,
                           omega_texto, "Rotina com os dados impressos na figura (CA 50-A, d′/h, ν = 0,8) e com seção bruta e "
                                        "0,85 fcd, hipóteses não impressas, confirmadas pela coincidência das curvas.")
        registro.adicionar(nome, "ω da reta tangente, d′/h = %s, tg θ desenhada (cota %s/4 = %s)"
                           % (num(distancia_relativa, 2), num(dados["cota_reta_x4"][distancia_relativa], 3),
                              num(inclinacao_figura, 4)), "",
                           CRITERIO_LEITURA, impresso[chave_omega], None, omega_figura,
                           "A cota impressa na figura em 10³ d/r = 4 dá a inclinação efetivamente desenhada.")
    fracao = fracao_interpolacao(dados["relacao_d_interpolada"], 0.95, 0.85)
    omega_interpolado = interpolar(valor_impresso(impresso["omega_095"]), valor_impresso(impresso["omega_085"]), fracao)
    distancia_interpolada = altura_util(1.0, dados["relacao_d_interpolada"])
    inclinacao_interpolada = inclinacao_reta_fusco(dados["nu_diagrama"], comprimento, altura,
                                                   altura_util(altura, distancia_interpolada))
    omega_direto = omega_pilar_padrao_fusco(material, distancia_interpolada, valor_impresso(impresso["mu1"]),
                                            inclinacao_interpolada)
    resultados["omega_rotina"]["0.1"] = {"tg": inclinacao_interpolada, "omega": omega_direto}
    nota_tabela_interpolacao = ("A fonte interpola entre as leituras de d′/h = 0,05 e 0,15 (%s e %s) a taxa de "
                                "d′/h = 0,10 (%s, impressa como %s); a rotina a calcula diretamente com d′/h = 0,10 "
                                "(%s)." % (texto_impresso(impresso["omega_095"]), texto_impresso(impresso["omega_085"]),
                                           num(omega_interpolado, 3), texto_impresso(impresso["omega_090"]),
                                           num(omega_direto, 4)))
    registro.adicionar(nome, "ω interpolado para d/h = 0,90", "", CRITERIO_IMPRESSAO, impresso["omega_090"],
                       omega_interpolado, omega_direto,
                       "Coluna “rotina”: curvas da rotina calculadas diretamente com d′/h = 0,10.",
                       exata=omega_interpolado, nota_tabela=nota_tabela_interpolacao)
    area_090_exata = area_de_omega(omega_interpolado, area_concreto, material.fcd, material.fyd)
    registro.adicionar(nome, "As = ω Ac fcd/fyd (fonte: ω = 0,78; rotina: ω = %s)" % num(omega_direto, 4), "cm²",
                       CRITERIO_IMPRESSAO, impresso["As_090"],
                       area_de_omega(valor_impresso(impresso["omega_090"]), area_concreto, fcd_impresso,
                                     fyd_impresso), area_de_omega(omega_direto, area_concreto, material.fcd,
                                                                  material.fyd),
                       "Coluna “rotina”: ω da linha anterior (%s), e não 0,78; com ω = 0,78 e o fcd e o fyd sem "
                       "arredondar, a rotina daria %s cm²."
                       % (num(omega_direto, 4), num(area_de_omega(valor_impresso(impresso["omega_090"]), area_concreto,
                                                                  material.fcd, material.fyd), 2)),
                       nota_tabela=nota_tabela_interpolacao,
                       exata=area_090_exata,
                       nota_exata="O autor leva adiante o ω arredondado (0,78, no lugar de 0,775) e usa fcd = 1,07 e "
                                  "fyd = 43,5 kN/cm² (15/1,4 = %s e 50/1,15 = %s)."
                                  % (num(material.fcd, 4), num(material.fyd, 3)))
    forca_g = forca_de_coeficientes(dados["gama_n"], dados["fracao_longa_duracao"], dados["Nk"])
    registro.adicionar(nome, "Fg = γn 0,95 Nk", "kN", CRITERIO_IMPRESSAO, impresso["Fg"], forca_g, forca_g,
                       resolucao_informada=dados["resolucao"]["Fg"], exata=forca_g)
    modulo = modulo_fusco(dados["fck"])
    registro.adicionar(nome, "Ec = 0,9 × 6600 raiz(fck + 3,5)", "MPa", CRITERIO_IMPRESSAO, impresso["Ec"], modulo,
                       None, resolucao_informada=dados["resolucao"]["Ec"], exata=modulo)
    alfa = razao_modular(dados["modulo_aco_mpa"], valor_impresso(impresso["Ec"]))
    alfa_exato = razao_modular(dados["modulo_aco_mpa"], modulo)
    registro.adicionar(nome, "α = Es/Ec", "", CRITERIO_IMPRESSAO, impresso["alfa"], alfa, None, exata=alfa_exato)
    inercia_concreto = nucleo.momento_inercia_retangulo(largura, altura)
    inercia = inercia_ideal(inercia_concreto, valor_impresso(impresso["alfa"]), dados["As_fluencia"], altura,
                            dados["distancia_fluencia"])
    area_fluencia_exata = area_barras_exata(dados["barras_fluencia"], dados["barra_fluencia"])
    inercia_exata = inercia_ideal(inercia_concreto, alfa_exato, area_fluencia_exata, altura, dados["distancia_fluencia"])
    registro.adicionar(nome, "Ii = Ic + α As (h/2 - d′)²", "cm⁴", CRITERIO_IMPRESSAO, impresso["Ii"], inercia, None,
                       resolucao_informada=dados["resolucao"]["Ii"], exata=inercia_exata,
                       nota_exata="O autor usa As = 50,4 cm² (3,15 cm² por barra de 20 mm); com a área nominal, "
                                  "16 π 2,0²/4 = %s cm², Ii = %s cm⁴." % (num(area_fluencia_exata, 2),
                                                                          num(inercia_exata, 0)))
    forca_e = carga_flambagem(mpa_para_kncm2(valor_impresso(impresso["Ec"])), valor_impresso(impresso["Ii"]),
                              comprimento)
    forca_e_exata = carga_flambagem(mpa_para_kncm2(modulo), inercia_exata, comprimento)
    registro.adicionar(nome, "FE = 10 Ec Ii/ℓe²", "kN", CRITERIO_IMPRESSAO, impresso["FE"], forca_e, None,
                       resolucao_informada=dados["resolucao"]["FE"], exata=forca_e_exata)
    forca_g_impressa = valor_impresso(impresso["Fg"])
    forca_e_impressa = valor_impresso(impresso["FE"])
    excentricidade_autor = excentricidade_suplementar(dados["e1"], dados["phi"], forca_g_impressa, forca_e_impressa)
    excentricidade_rotina = nucleo.excentricidade_fluencia(0.0, forca_g, dados["e1"], dados["phi"], forca_e_exata)
    excentricidade_exata = excentricidade_suplementar(dados["e1"], dados["phi"], forca_g, forca_e_exata)
    nota_tabela_fluencia = ("Deslize de conta da fonte: a expressão impressa dá %s cm (%s cm na cadeia exata), e o "
                            "texto escreve 2 × exp(−0,52) = 1,19 cm e leva esse valor a e1x. Na rotina, ec vem da "
                            "expressão da Seção 15.8.4, com Msg = 0, ea = e1g e o Fg e o FE da formulação de Fusco "
                            "(1981) sem arredondar (%s kN e %s kN)."
                            % (num(excentricidade_autor, 2), num(excentricidade_exata, 3), num(forca_g, 2),
                               num(forca_e_exata, 1)))
    registro.adicionar(nome, "ec = e1g [exp(φ Fg/(FE - Fg)) - 1]", "cm", CRITERIO_IMPRESSAO, impresso["ec"],
                       excentricidade_autor, excentricidade_rotina,
                       "A expressão impressa, com os valores impressos, dá 1,23 cm; o texto escreve o resultado "
                       "intermediário como 2 · exp(−0,52) = 1,19 cm, que não decorre da expressão (exp(0,478) - 1 = "
                       "0,613). Coluna “rotina”: expressão de ecc da Seção 15.8.4, implementada no núcleo, com "
                       "Msg = 0, ea = e1g = 2 cm e o Fg e o FE da formulação de Fusco (1981) sem arredondar "
                       "(%s kN e %s kN; a rotina não calcula o FE dessa formulação)."
                       % (num(forca_g, 2), num(forca_e_exata, 1)), exata=excentricidade_exata,
                       nota_exata="A expressão impressa dá %s cm; o texto escreve 2 · exp(−0,52) = 1,19 cm "
                                  "(deslize de conta)." % num(excentricidade_exata, 3),
                       nota_tabela=nota_tabela_fluencia)
    registro.adicionar(nome, "e1x = ea + ec", "cm", CRITERIO_IMPRESSAO, impresso["e1x"],
                       soma(dados["e1"], valor_impresso(impresso["ec"])), soma(dados["e1"], excentricidade_rotina),
                       exata=soma(dados["e1"], excentricidade_exata),
                       nota_exata="O autor soma o ec errado (1,19 cm, deslize de conta da etapa anterior).",
                       nota_tabela=nota_tabela_fluencia)
    registro.adicionar(nome, "μ1d = νd e1x/h", "", CRITERIO_IMPRESSAO, impresso["mu1_fluencia"],
                       mu_primeira_ordem(valor_impresso(impresso["nu"]), valor_impresso(impresso["e1x"]), altura),
                       mu_primeira_ordem(nu, soma(dados["e1"], excentricidade_rotina), altura),
                       exata=mu_primeira_ordem(nu, soma(dados["e1"], excentricidade_exata), altura))
    omega_fluencia = omega_pilar_padrao_fusco(material, distancia_interpolada,
                                              mu_primeira_ordem(dados["nu_diagrama"],
                                                                soma(dados["e1"], excentricidade_rotina), altura),
                                              inclinacao_interpolada)
    resultados["omega_rotina_fluencia"] = omega_fluencia
    registro.adicionar(nome, "ω com fluência (Fig. 5.3.5-5 no texto; pilar-padrão no diagrama M, N, 1/r na rotina)", "",
                       CRITERIO_LEITURA, impresso["omega_fluencia"], None, omega_fluencia,
                       "Métodos diferentes: o texto lê diagramas de interação de barras esbeltas; a rotina aplica "
                       "a construção do item 8.2.5 com d′/h = 0,10 e e1x calculado.")
    registro.adicionar(nome, "As = ω Ac fcd/fyd (ω = 0,92)", "cm²", CRITERIO_IMPRESSAO, impresso["As_fluencia"],
                       area_de_omega(valor_impresso(impresso["omega_fluencia"]), area_concreto, fcd_impresso,
                                     fyd_impresso), None,
                       exata=area_de_omega(valor_impresso(impresso["omega_fluencia"]), area_concreto, material.fcd,
                                           material.fyd),
                       nota_exata="O autor usa fcd = 1,07 e fyd = 43,5 kN/cm² (15/1,4 = %s e 50/1,15 = %s)."
                                  % (num(material.fcd, 4), num(material.fyd, 3)))
    digitalizar_fusco(material, registro)
    resultados.update({"excentricidade_fluencia_rotina_cm": excentricidade_rotina, "fcd": material.fcd,
                       "fyd": material.fyd})
    return resultados


# ----------------------------------------------------------------------------- L4 — Franco (2010)
def executar_franco(registro: Registro) -> dict:
    dados = FRANCO
    impresso = dados["impresso"]
    nome = "L4 Franco (2010)"
    material = nucleo.Material(dados["fck"])
    altura = dados["altura"]
    largura = dados["largura"]
    comprimento = dados["comprimento"]
    forca = dados["Nd"]
    momento_minimo_autor = produto(forca, dados["e1_min"])
    momento_sg_autor = dividir(momento_minimo_autor, dados["gama_f"])
    forca_sg_autor = dividir(forca, dados["gama_f"])
    registro.adicionar(nome, "M1d,mín = Nd e1,mín = 473,8 × 0,0195", "kN·m", CRITERIO_IMPRESSAO,
                       impresso["M1d_min"], momento_minimo_autor, momento_minimo_autor,
                       "9,2391 é impresso como 9,23 (truncamento).", exata=momento_minimo_autor,
                       nota_exata="O autor trunca 9,2391 para 9,23.")
    registro.adicionar(nome, "Msg = M1d,mín/1,4 (hipótese da fonte)", "kN·m", CRITERIO_IMPRESSAO, impresso["Msg"],
                       dividir(valor_impresso(impresso["M1d_min"]), dados["gama_f"]), momento_sg_autor,
                       "O texto escreve 9,23/1,4 = 6,60; com o 9,23 impresso, 6,593. O 6,60 vem do valor sem "
                       "arredondar: 9,2391/1,4 = 6,599 (coluna “rotina”).", exata=momento_sg_autor)
    registro.adicionar(nome, "Nsg = Nd/1,4 (hipótese da fonte)", "kN", CRITERIO_IMPRESSAO, impresso["Nsg"],
                       forca_sg_autor, None, exata=forca_sg_autor)
    modulo_autor = kncm2_para_knm2(nucleo.modulo_elasticidade_inicial(dados["fck"], 1.0))
    forca_e_autor = carga_flambagem(modulo_autor, dados["Ic_impresso"], cm_para_m(comprimento))
    inercia = nucleo.momento_inercia_retangulo(largura, altura)
    forca_e_rotina = nucleo.forca_euler_fluencia(material.modulo_inicial, inercia, comprimento)
    forca_e_exata = carga_flambagem(modulo_autor, cm4_para_m4(inercia), cm_para_m(comprimento))
    registro.adicionar(nome, "Ne = 10 Eci Ic/ℓe²", "kN", CRITERIO_IMPRESSAO, impresso["Ne"], forca_e_autor,
                       forca_e_rotina, "A fonte usa Ic = 7,03 × 10⁻⁵ m⁴ (0,25 × 0,15³/12 = 7,031 × 10⁻⁵ m⁴).",
                       exata=forca_e_exata,
                       nota_exata="O autor usa Ic = 7,03 × 10⁻⁵ m⁴, arredondado de 7,03125 × 10⁻⁵ m⁴, com o qual "
                                  "Ne = %s kN; com o Ic exato, %s kN." % (num(forca_e_autor, 1),
                                                                         num(forca_e_exata, 1)))
    excentricidade_a_autor = excentricidade_imperfeicao(tangente_graus(dados["angulo_graus"]), cm_para_m(comprimento))
    excentricidade_a_exata = excentricidade_imperfeicao(tangente_radianos(dados["theta1_autor"]),
                                                        cm_para_m(comprimento))
    excentricidade_a_rotina = cm_para_m(nucleo.excentricidade_acidental(nucleo.desaprumo_local(comprimento),
                                                                        comprimento))
    registro.adicionar(nome, "ea = θ1 ℓe/2, θ1 = θ1,máx = 1/200", "m", CRITERIO_IMPRESSAO, impresso["ea"],
                       excentricidade_a_autor, excentricidade_a_rotina,
                       "A fonte escreve θ1 como o ângulo 0,2866° (arctg 1/200 = 0,28648°).",
                       exata=excentricidade_a_exata,
                       nota_exata="O autor usa o ângulo arredondado 0,2866° (tg = %s) no lugar de θ1 = 1/200 "
                                  "(tg = %s)." % (num(tangente_graus(dados["angulo_graus"]), 7),
                                                  num(tangente_radianos(dados["theta1_autor"]), 7)))
    excentricidade_cc_exata = excentricidade_cc_independente(momento_sg_autor, forca_sg_autor, excentricidade_a_exata,
                                                             dados["phi"], forca_e_exata)
    excentricidade_cc_autor = excentricidade_cc_independente(
        valor_impresso(impresso["Msg"]), valor_impresso(impresso["Nsg"]), valor_impresso(impresso["ea"]),
        dados["phi"], valor_impresso(impresso["Ne"]))
    excentricidade_cc_rotina = cm_para_m(nucleo.excentricidade_fluencia(
        knm_para_kncm(momento_sg_autor), forca_sg_autor, m_para_cm(excentricidade_a_rotina), dados["phi"],
        forca_e_rotina))
    registro.adicionar(nome, "ecc = (Msg/Nsg + ea)[exp(φ Nsg/(Ne - Nsg)) - 1]", "m", CRITERIO_IMPRESSAO,
                       impresso["ecc"], excentricidade_cc_autor, excentricidade_cc_rotina,
                       "Coluna “rotina”: θ1, ea, Ne e ecc da rotina (Seções 11.3.3.4.1, 11.3.3.4.2 e 15.8.4, "
                       "implementadas no núcleo), com o Msg e o Nsg da fonte.", exata=excentricidade_cc_exata)
    momento_resistente_programa = dados["MRd_programa"]
    momento_ponto_b_sem_arredondar = validacao.momento_ponto_b(momento_resistente_programa)
    curvatura_programa = milesimos_para_unidade(dados["curvatura_programa_milesimos"])
    registro.adicionar(nome, "MRd/1,1 = 26,7/1,1 (MRd do MK-UFRJ, anexo 1)", "kN·m", CRITERIO_IMPRESSAO,
                       impresso["M_ponto_B"], momento_ponto_b_sem_arredondar, None,
                       "MRd = 26,7 kN·m e 1/r = 27,4979/1000 m⁻¹ vêm do diagrama momento-curvatura do anexo 1 "
                       "(programa MK-UFRJ, p. 23 e 34) e entram como dados da etapa; 1,1 = γf3.",
                       exata=momento_ponto_b_sem_arredondar)
    rigidez_autor = rigidez_do_ponto_b(valor_impresso(impresso["M_ponto_B"]), curvatura_programa)
    rigidez_sem_arredondar = rigidez_do_ponto_b(momento_ponto_b_sem_arredondar, curvatura_programa)
    registro.adicionar(nome, "(EI)sec = (MRd/1,1)/(1/r), com 1/r = 27,4979/1000 m⁻¹ (anexo 1)", "kN·m²",
                       CRITERIO_IMPRESSAO, impresso["EI_sec"], rigidez_autor, None,
                       "O texto escreve 24,3 × 1000/27,4979 = 882,7; com o 24,3 impresso, %s. O 882,7 vem do valor "
                       "sem arredondar: 26,7/1,1 = %s e %s × 1000/27,4979 = %s; a fonte leva à etapa seguinte o valor "
                       "sem arredondar." % (num(rigidez_autor, 1), num(momento_ponto_b_sem_arredondar, 4),
                                            num(momento_ponto_b_sem_arredondar, 4), num(rigidez_sem_arredondar, 1)),
                       exata=rigidez_sem_arredondar)
    kappa_autor = validacao.kappa_secante(valor_impresso(impresso["EI_sec"]),
                                          produto(cm_para_m(altura), cm_para_m(largura)), cm_para_m(altura),
                                          kncm2_para_knm2(material.fcd))
    kappa_exato = validacao.kappa_secante(rigidez_sem_arredondar, produto(cm_para_m(altura), cm_para_m(largura)),
                                          cm_para_m(altura), kncm2_para_knm2(material.fcd))
    registro.adicionar(nome, "κsec = (EI)sec/(Ac h² fcd)", "", CRITERIO_IMPRESSAO, impresso["kappa_sec"], kappa_autor,
                       None, exata=kappa_exato)
    momento_autor = momento_com_fluencia_franco(forca, dados["e1_min"], valor_impresso(impresso["ecc"]))
    momento_rotina = kncm_para_knm(nucleo.momento_primeira_ordem_com_fluencia(
        knm_para_kncm(momento_minimo_autor), forca, m_para_cm(excentricidade_cc_rotina)))
    registro.adicionar(nome, "M1d = Nd (e1,mín + ecc)", "kN·m", CRITERIO_IMPRESSAO, impresso["M1d"], momento_autor,
                       momento_rotina, "473,8 × (0,0195 + 0,00741) = 12,74996 é impresso como 12,74 (truncamento); "
                                       "12,74 é também a soma do M1d,mín impresso, 9,23, com 473,8 × 0,00741 = 3,511.",
                       exata=momento_com_fluencia_franco(forca, dados["e1_min"], excentricidade_cc_exata),
                       nota_exata="O autor trunca %s para 12,74."
                                  % num(momento_com_fluencia_franco(forca, dados["e1_min"], excentricidade_cc_exata), 4))
    momento_total_autor = validacao.momento_total_kappa_secante(1.0, valor_impresso(impresso["M1d"]),
                                                                dados["lambda_impresso"], dados["nu_impresso"],
                                                                valor_impresso(impresso["kappa_sec"]))
    area_concreto = produto(largura, altura)
    esbeltez_exata = dividir(comprimento, raio_giracao(altura))
    nu_exato = dividir(forca, produto(area_concreto, material.fcd))
    momento_primeira_ordem_exato = momento_com_fluencia_franco(forca, dados["e1_min"], excentricidade_cc_exata)
    momento_total_exato = validacao.momento_total_kappa_secante(1.0, momento_primeira_ordem_exato, esbeltez_exata,
                                                                nu_exato, kappa_exato)
    registro.adicionar(nome, "Md,tot = M1d/[1 - λ² ν/(120 κsec)] (κsec com a 1/r do MK-UFRJ, anexo 1)", "kN·m",
                       CRITERIO_IMPRESSAO, impresso["Md_kappa_sec"], momento_total_autor, None,
                       "Mesma forma do item C.3 de Kimura ([2005?]). O 1/r vem do diagrama momento-curvatura do "
                       "anexo 1, saída do programa MK-UFRJ (p. 23 e 34).", exata=momento_total_exato,
                       nota_exata="O autor leva adiante M1d = 12,74 kN·m (truncado), λ = 58,89, ν = 0,707 e "
                                  "κsec = 58,59 impressos; com os valores exatos (%s kN·m, %s, %s e %s), %s kN·m."
                                  % (num(momento_primeira_ordem_exato, 4), num(esbeltez_exata, 3),
                                     num(nu_exato, 4), num(kappa_exato, 3), num(momento_total_exato, 3)))
    registro.verificar(nome, "θ1 = 1/(100 raiz(ℓ)) limitado a 1/200 (Seções 11.3.3.4.1 e 11.3.3.4.2)",
                       nucleo.desaprumo_local(comprimento) == nucleo.THETA1_MAX, dados["fonte"],
                       "1/(100 raiz(2,55)) = 0,00626 > 1/200.")
    return {"Ne_rotina_kN": forca_e_rotina, "ea_rotina_m": excentricidade_a_rotina,
            "ecc_rotina_m": excentricidade_cc_rotina}


# ----------------------------------------------------------------------------- L5 — Carvalho e Pinheiro (2009)
def omega_no_ponto(material: nucleo.Material, altura: float, largura: float, distancia: float, nu: float,
                   mu: float) -> float:
    """ω da rotina no ponto (ν; μ) de um ábaco: seção bruta, duas camadas simétricas a d' das faces."""
    area_concreto = nucleo.area_secao_retangular(largura, altura)
    secao = secao_camadas(altura, largura, distancia, 1)
    forca = nucleo.forca_normal_de_nu(nu, area_concreto, material.fcd)
    momento = momento_de_mu(mu, area_concreto, altura, material.fcd)
    with SecaoHipotese(False):
        area = nucleo.armadura_flexo_compressao(secao, material, forca, momento)
    return taxa_mecanica(area, material.fyd, area_concreto, material.fcd)


def omega_tangente_abaco(material: nucleo.Material, altura: float, largura: float, distancia: float, nu: float,
                         mu1: float, comprimento: float) -> float:
    """Menor ω cuja curva μ x 1/r (0,85 fcd, N = ν Ac fcd, seção bruta) alcança a reta μ = μ1 + ν ℓe²/(10 h) (1/r)
    (construção do ábaco do pilar-padrão acoplado, Carvalho e Pinheiro, 2009, Fig. 5.19 e 5.20)."""
    area_concreto = nucleo.area_secao_retangular(largura, altura)
    secao = secao_camadas(altura, largura, distancia, 1)
    forca = nucleo.forca_normal_de_nu(nu, area_concreto, material.fcd)
    inclinacao = inclinacao_reta_curvatura(nu, comprimento, altura)

    def folga(omega: float) -> float:
        area = area_de_omega(omega, area_concreto, material.fcd, material.fyd)
        with SecaoHipotese(False):
            curvaturas, momentos = nucleo.curva_momento_curvatura(secao, material, area, forca, False,
                                                                  PONTOS_CURVA_FIGURA)
        return folga_reta(momento_adimensional(momentos, area_concreto, altura, material.fcd), mu1, inclinacao,
                          curvaturas)

    if folga(0.0) >= 0.0:
        return 0.0
    return brentq(folga, 0.0, 2.0, xtol=1e-6)


def omega_acoplado_norma(material: nucleo.Material, altura: float, largura: float, distancia: float, nu: float,
                         mu1: float, comprimento: float) -> tuple[float, float]:
    """ω e Md,tot do pilar-padrão acoplado da rotina (curva com 1,10 fcd e Nd/γf3; 15.3.1 e 15.8.3.3.4) no ponto
    (ν; μ1) do ábaco, com a seção bruta e duas camadas."""
    area_concreto = nucleo.area_secao_retangular(largura, altura)
    secao = secao_camadas(altura, largura, distancia, 1)
    forca = nucleo.forca_normal_de_nu(nu, area_concreto, material.fcd)
    momento = momento_de_mu(mu1, area_concreto, altura, material.fcd)
    with SecaoHipotese(False):
        area, momento_total, _ = nucleo.armadura_pilar_padrao_acoplado(secao, material, forca, momento, 1.0,
                                                                       comprimento)
    return taxa_mecanica(area, material.fyd, area_concreto, material.fcd), momento_total


def num(valor: float, casas: int) -> str:
    """Número com vírgula decimal e separador de milhar do SI ([F15]), para as observações que vão para o TCC."""
    return agrupar_milhares(("%.*f" % (casas, valor)).replace(".", ","))


def texto_impresso(texto: str) -> str:
    """Valor impresso (texto do dicionário, com ponto decimal) escrito como no TCC: vírgula e separador do SI."""
    return agrupar_milhares(texto.replace(".", ","))


def texto_barras(candidato) -> str:
    return "%d ϕ %s (%s cm²)" % (candidato.numero_barras, ("%.1f" % cm_para_mm(candidato.diametro)).replace(".0", "")
                                 .replace(".", ","), ("%.2f" % candidato.area).replace(".", ","))


def executar_carvalho(registro: Registro) -> dict:
    dados = CARVALHO
    impresso = dados["impresso"]
    nome = "L5 Carvalho e Pinheiro (2009)"
    fonte = dados["fonte"]
    material = nucleo.Material(dados["fck"])
    forca_k = dados["Nk"]
    forca = forca_de_coeficientes(nucleo.GAMA_F, forca_k)
    distancia = nucleo.distancia_eixo_barra(dados["cobrimento"], dados["estribo"], dados["barra"])
    vao_livre = soma(dados["pe_direito"], dados["revestimentos"])
    theta_autor = dados["theta1_autor"]
    resultados = {}

    # 5.1 — pilar central "curto", 30 x 30 cm
    lado = dados["lado_51"]
    comprimento = soma(vao_livre, lado)
    area_concreto = nucleo.area_secao_retangular(lado, lado)
    esbeltez = nucleo.indice_esbeltez(comprimento, lado)
    registro.adicionar(nome, "5.1: ℓe = ℓ0 + h = 2,78 + 0,30", "m", CRITERIO_IMPRESSAO, impresso["le_51"],
                       cm_para_m(comprimento), cm_para_m(comprimento), exata=cm_para_m(comprimento))
    registro.adicionar(nome, "5.1: 3,46 ℓe (λ = 3,46 ℓe/b)", "m", CRITERIO_IMPRESSAO, impresso["fator_le_51"],
                       cm_para_m(fator_comprimento_autor(comprimento)), cm_para_m(produto(esbeltez, lado)),
                       "Coluna “rotina”: raiz(12) ℓe.", exata=cm_para_m(produto(esbeltez, lado)),
                       nota_exata="O autor usa 3,46 no lugar de raiz(12) (3,46 × 3,08 = 10,6568).")
    registro.adicionar(nome, "5.1: b >= 10,66/35 (pilar curto)", "m", CRITERIO_IMPRESSAO, impresso["lado_min_51"],
                       dividir(valor_impresso(impresso["fator_le_51"]), dados["esbeltez_curto"]),
                       cm_para_m(dividir(produto(esbeltez, lado), dados["esbeltez_curto"])),
                       "Com o 10,66 impresso, 0,3046; o 0,304 vem do produto sem arredondar, 3,08 × 3,46 = 10,6568, e "
                       "10,6568/35 = 0,3045 (a fonte leva à etapa seguinte o valor sem arredondar). O texto adota "
                       "b = 0,30 m, com λ = %s pelo fator 3,46 e %s com raiz(12), acima de λ1 = 35."
                       % (num(esbeltez_fator_autor(comprimento, lado), 2), num(esbeltez, 2)),
                       exata=cm_para_m(dividir(produto(esbeltez, lado), dados["esbeltez_curto"])),
                       nota_exata="O autor usa 3,46 no lugar de raiz(12): 3,46 × 3,08/35 = 0,3045, que dá 0,304; "
                                  "com raiz(12), %s." % num(cm_para_m(dividir(produto(esbeltez, lado),
                                                                              dados["esbeltez_curto"])), 5))
    limite = esbeltez_limite_sem_limites(0.0, 1.0)
    registro.adicionar(nome, "5.1: λ1 = (25 + 12,5 e1/h)/αb com e1 = 0, antes do limite de 35", "",
                       CRITERIO_IMPRESSAO, impresso["lambda1_51"], limite, limite, exata=limite)
    excentricidade_a_autor = excentricidade_imperfeicao(theta_autor, cm_para_m(comprimento))
    excentricidade_a_rotina = cm_para_m(nucleo.excentricidade_acidental(nucleo.desaprumo_local(comprimento),
                                                                        comprimento))
    registro.adicionar(nome, "5.1: ea = θ1 ℓ/2 (seção intermediária)", "m", CRITERIO_IMPRESSAO, impresso["ea_51"],
                       excentricidade_a_autor, excentricidade_a_rotina,
                       "θ1 = 1/(100 raiz(3,08)) = 1/175 > 1/200: θ1 = 1/200 no texto e na rotina.",
                       exata=excentricidade_a_autor)
    registro.adicionar(nome, "5.1: ea na seção de extremidade (dobro)", "m", CRITERIO_IMPRESSAO, impresso["ea_ext_51"],
                       dobro(excentricidade_a_autor), None,
                       "A rotina não soma ea: o momento mínimo substitui as imperfeições locais (11.3.3.4.3).",
                       exata=dobro(excentricidade_a_autor), nota_tabela=NOTA_TABELA_EA_EXTREMIDADE)
    momento_minimo = nucleo.momento_minimo_primeira_ordem(forca, lado)
    excentricidade_minima = excentricidade(momento_minimo, forca)
    registro.adicionar(nome, "5.1: e1,mín = 0,015 + 0,03 h", "m", CRITERIO_IMPRESSAO, impresso["e1min_51"],
                       cm_para_m(excentricidade_minima), cm_para_m(excentricidade_minima),
                       exata=cm_para_m(excentricidade_minima))
    registro.adicionar(nome, "5.1: M = N e1,mín (valor característico)", "kN·m", CRITERIO_IMPRESSAO, impresso["Mk_51"],
                       produto(forca_k, valor_impresso(impresso["e1min_51"])), None,
                       "O momento de cálculo, usado pela rotina, é M1d,mín = 1,4 × 4,8 = 6,72 kN·m.",
                       exata=produto(forca_k, cm_para_m(excentricidade_minima)))
    registro.adicionar(nome, "5.1: d′/h = (c + ϕt + ϕl/2)/h", "", CRITERIO_IMPRESSAO, impresso["dlinha_51"],
                       dividir(distancia, lado), dividir(distancia, lado), exata=dividir(distancia, lado))
    nu = nucleo.forca_normal_adimensional(forca, area_concreto, material.fcd)
    registro.adicionar(nome, "5.1: ν = Nd/(b h fcd)", "", CRITERIO_IMPRESSAO, impresso["nu_51"], nu, nu, exata=nu)
    registro.adicionar(nome, "5.1: μ = (e/h) ν", "", CRITERIO_IMPRESSAO, impresso["mu_51"],
                       mu_primeira_ordem(valor_impresso(impresso["nu_51"]), valor_impresso(impresso["e1min_51"]),
                                         cm_para_m(lado)),
                       mu_primeira_ordem(nu, excentricidade_minima, lado),
                       "Coluna “rotina”: μ do momento mínimo de 1ª ordem.",
                       exata=mu_primeira_ordem(nu, excentricidade_minima, lado),
                       nota_exata="O autor leva adiante o ν impresso (0,22) no lugar de %s." % num(nu, 4))
    registro.adicionar(nome, "5.1: ω do ábaco 2 no ponto (ν; μ) impresso", "", CRITERIO_LEITURA, impresso["omega_51"],
                       None, omega_no_ponto(material, lado, lado, dados["distancia_abaco_51"],
                                            valor_impresso(impresso["nu_51"]), valor_impresso(impresso["mu_51"])),
                       "Rotina na hipótese do ábaco: seção bruta, duas camadas, d′/h = 0,10.")
    registro.adicionar(nome, "5.1: ρmín = 0,15 (fcd/fyd) ν", "", CRITERIO_IMPRESSAO, impresso["rho_min_51"],
                       taxa_minima_relativa(material.fcd, material.fyd, valor_impresso(impresso["nu_51"])),
                       taxa_minima_relativa(material.fcd, material.fyd, nu),
                       exata=taxa_minima_relativa(material.fcd, material.fyd, nu))
    area_minima = nucleo.armadura_minima(forca, material.fyd, area_concreto)
    registro.adicionar(nome, "5.1: As,mín = 0,004 Ac (governa)", "cm²", CRITERIO_IMPRESSAO, impresso["As_min_51"],
                       area_minima, area_minima, exata=area_minima)
    area_texto = nucleo.area_barras(dados["barra"], dados["barras_adotadas_51"])
    registro.adicionar(nome, "5.1: As de 4 ϕ 10", "cm²", CRITERIO_IMPRESSAO, impresso["As_4phi10"],
                       produto(dados["barras_adotadas_51"], dados["area_barra_autor_10"]), area_texto,
                       "O texto usa 0,8 cm² por barra de 10 mm.", exata=area_texto,
                       nota_exata="O autor usa 0,8 cm² por barra de 10 mm (área nominal π 1,0²/4 = 0,785 cm²).")
    registro.adicionar(nome, "5.1: redução de 4 ϕ 10 em relação a As,mín (%)", "%", CRITERIO_IMPRESSAO,
                       impresso["reducao_51"], reducao_percentual_area(valor_impresso(impresso["As_4phi10"]),
                                                                       valor_impresso(impresso["As_min_51"])),
                       reducao_percentual_area(area_texto, area_minima),
                       exata=reducao_percentual_area(area_texto, area_minima),
                       nota_exata="Decorre da área de 0,8 cm² por barra de 10 mm (com a área nominal, 3,14 cm²).")
    pilar = nucleo.PilarTeste("C&P 5.1", lado, lado, dados["fck"], dados["fyk"], forca, 0.0, comprimento,
                              cobrimento=dados["cobrimento"])
    resultado = nucleo.dimensionar_pilar(pilar)
    registro.verificar(nome, "5.1: λ = raiz(12) ℓe/b <= λ1 = 35 com b = 30 cm (pilar curto do texto)",
                       resultado.direcao_b.metodo == nucleo.METODO_DISPENSADO, fonte,
                       "λ = %s: a rotina aplica a curvatura aproximada (Md,tot = %s kN·m) e chega à mesma "
                       "As,calc = As,mín = %s cm²." % (num(resultado.direcao_b.esbeltez, 2),
                                                      num(kncm_para_knm(resultado.direcao_b.momento_total), 2),
                                                      num(resultado.area_calculada, 2)))
    registro.verificar(nome, "5.1: 4 ϕ 10 (%s cm²) >= As,mín = %s cm² (17.3.5.3.1)" % (num(area_texto, 2), num(area_minima, 2)),
                       area_texto >= area_minima, fonte,
                       "O texto aceita 11 %% a menos que As,mín; a rotina adota %s, a menor área que atende."
                       % texto_barras(resultado.candidato))
    resultados["5.1"] = {"lambda": resultado.direcao_b.esbeltez, "metodo": resultado.direcao_b.metodo,
                         "Md_tot_kNcm": resultado.direcao_b.momento_total, "As_calc_cm2": resultado.area_calculada,
                         "barras": texto_barras(resultado.candidato), "As_adot_cm2": resultado.candidato.area}

    # 5.2 — pilar central medianamente esbelto, 20 x 20 cm
    lado = dados["lado_52"]
    comprimento = soma(vao_livre, lado)
    area_concreto = nucleo.area_secao_retangular(lado, lado)
    registro.adicionar(nome, "5.2: b >= 10,66/90, com o ℓe do 5.1", "m", CRITERIO_IMPRESSAO,
                       impresso["lado_min_52"], dividir(valor_impresso(impresso["fator_le_51"]), dados["esbeltez_media"]),
                       cm_para_m(dividir(produto(esbeltez, dados["lado_51"]), dados["esbeltez_media"])),
                       "O texto adota b = 0,20 m (menor dimensão de 19 cm sem γn).",
                       exata=cm_para_m(dividir(produto(esbeltez, dados["lado_51"]), dados["esbeltez_media"])),
                       nota_exata="O autor usa 3,46 no lugar de raiz(12) (3,46 × 3,08/90 = 0,1184).")
    registro.adicionar(nome, "5.2: ℓe = ℓ0 + h = 2,78 + 0,20", "m", CRITERIO_IMPRESSAO, impresso["le_52"],
                       cm_para_m(comprimento), cm_para_m(comprimento), exata=cm_para_m(comprimento))
    esbeltez = nucleo.indice_esbeltez(comprimento, lado)
    registro.adicionar(nome, "5.2: λ = 3,46 ℓe/b", "", CRITERIO_IMPRESSAO, impresso["lambda_52"],
                       esbeltez_fator_autor(comprimento, lado), esbeltez, exata=esbeltez)
    momento_minimo = nucleo.momento_minimo_primeira_ordem(forca, lado)
    excentricidade_minima = excentricidade(momento_minimo, forca)
    registro.adicionar(nome, "5.2: e1,mín = 0,015 + 0,03 h", "m", CRITERIO_IMPRESSAO, impresso["e1min_52"],
                       cm_para_m(excentricidade_minima), cm_para_m(excentricidade_minima),
                       exata=cm_para_m(excentricidade_minima))
    nu = nucleo.forca_normal_adimensional(forca, area_concreto, material.fcd)
    registro.adicionar(nome, "5.2: ν = Nd/(b h fcd)", "", CRITERIO_IMPRESSAO, impresso["nu_52"], nu, nu, exata=nu)
    curvatura_autor = por_cm_para_por_metro(nucleo.curvatura_aproximada(valor_impresso(impresso["nu_52"]), lado))
    curvatura = nucleo.curvatura_aproximada(nu, lado)
    registro.adicionar(nome, "5.2: e2 = (ℓe²/10) 0,005/[(ν + 0,5) h], com (ν + 0,5) >= 1", "m", CRITERIO_IMPRESSAO,
                       impresso["e2_52"], excentricidade_curvatura(cm_para_m(comprimento), curvatura_autor),
                       cm_para_m(excentricidade_curvatura(comprimento, curvatura)),
                       exata=cm_para_m(excentricidade_curvatura(comprimento, curvatura)))
    registro.adicionar(nome, "5.2: d′/h", "", CRITERIO_IMPRESSAO, impresso["dlinha_52"], dividir(distancia, lado),
                       dividir(distancia, lado), exata=dividir(distancia, lado))
    pilar = nucleo.PilarTeste("C&P 5.2", lado, lado, dados["fck"], dados["fyk"], forca, 0.0, comprimento,
                              cobrimento=dados["cobrimento"])
    resultado = nucleo.dimensionar_pilar(pilar)
    registro.adicionar(nome, "5.2: μ = [(e1,mín + e2)/h] ν", "", CRITERIO_IMPRESSAO, impresso["mu_52"],
                       mu_primeira_ordem(valor_impresso(impresso["nu_52"]),
                                         soma(valor_impresso(impresso["e1min_52"]), valor_impresso(impresso["e2_52"])),
                                         cm_para_m(lado)),
                       momento_adimensional(resultado.direcao_b.momento_total, area_concreto, lado, material.fcd),
                       "Coluna “rotina”: Md,tot da curvatura aproximada da rotina.",
                       exata=mu_primeira_ordem(nu, soma(excentricidade_minima,
                                                        excentricidade_curvatura(comprimento, curvatura)), lado),
                       nota_exata="O autor leva adiante o e2 impresso (0,022 m) no lugar de %s m."
                                  % num(cm_para_m(excentricidade_curvatura(comprimento, curvatura)), 6))
    registro.adicionar(nome, "5.2: ω do ábaco 3 no ponto (ν; μ) impresso", "", CRITERIO_LEITURA, impresso["omega_52"],
                       None, omega_no_ponto(material, lado, lado, dados["distancia_abaco_52"],
                                            valor_impresso(impresso["nu_52"]), valor_impresso(impresso["mu_52"])),
                       "Rotina na hipótese do ábaco: seção bruta, duas camadas, d′/h = 0,15.")
    taxa_minima_autor_52 = taxa_minima_relativa(material.fcd, material.fyd, valor_impresso(impresso["nu_52"]))
    registro.adicionar(nome, "5.2: ρmín = 0,15 (fcd/fyd) ν", "", CRITERIO_IMPRESSAO, impresso["rho_min_52"],
                       taxa_minima_autor_52, taxa_minima_relativa(material.fcd, material.fyd, nu),
                       "Erro de cópia da fonte: o texto usa o ν = 0,22 do exemplo 5.1, com o qual ρmín = %s; com o "
                       "ν = 0,49 impresso no exemplo 5.2, ρmín = %s, também menor que 0,4 %%."
                       % (num(taxa_minima_relativa(material.fcd, material.fyd, valor_impresso(impresso["nu_51"])), 4),
                          num(taxa_minima_autor_52, 4)),
                       exata=taxa_minima_relativa(material.fcd, material.fyd, nu),
                       nota_exata="Erro de cópia da fonte: o texto usa o ν = 0,22 do exemplo 5.1 no lugar do ν do "
                                  "exemplo 5.2 (%s)." % num(nu, 4),
                       nota_tabela="Erro de cópia da fonte: o texto usa o ν = 0,22 do exemplo 5.1, com o qual "
                                   "ρmín = %s; com o ν do exemplo 5.2 (%s), ρmín = %s."
                                   % (num(taxa_minima_relativa(material.fcd, material.fyd,
                                                               valor_impresso(impresso["nu_51"])), 4), num(nu, 2),
                                      num(taxa_minima_relativa(material.fcd, material.fyd, nu), 4)))
    area_minima = nucleo.armadura_minima(forca, material.fyd, area_concreto)
    registro.adicionar(nome, "5.2: As,mín = 0,004 Ac (governa)", "cm²", CRITERIO_IMPRESSAO, impresso["As_min_52"],
                       area_minima, area_minima, exata=area_minima)
    registro.verificar(nome, "5.2: armadura adotada pela rotina igual à do texto (4 ϕ 10)",
                       resultado.candidato.numero_barras == dados["barras_adotadas_51"]
                       and resultado.candidato.diametro == dados["barra"], fonte,
                       "Rotina: %s; MRd = %s kN·m >= Md,tot = %s kN·m." % (
                           texto_barras(resultado.candidato), num(kncm_para_knm(resultado.momentos_resistentes["b"]), 2),
                           num(kncm_para_knm(resultado.direcao_b.momento_total), 2)))
    resultados["5.2"] = {"lambda": esbeltez, "Md_tot_kNcm": resultado.direcao_b.momento_total,
                         "As_calc_cm2": resultado.area_calculada, "barras": texto_barras(resultado.candidato),
                         "MRd_kNcm": resultado.momentos_resistentes["b"]}

    # 5.3 e 5.4 — seção de 12 x 30 cm com γn
    altura = dados["altura_53"]
    largura = dados["largura_53"]
    area_concreto = nucleo.area_secao_retangular(largura, altura)
    gama_n = coeficiente_gama_n(altura)
    registro.adicionar(nome, "5.3: γn = 1,95 - 0,05 b (Tabela 5.1 do livro; Tabela 13.1 da NBR 6118:2003)", "",
                       CRITERIO_IMPRESSAO, impresso["gama_n_53"], gama_n, None,
                       "A rotina não aplica γn: o escopo do trabalho exige menor dimensão de pelo menos 19 cm. Nas "
                       "etapas dos exemplos 5.3 e 5.4, as funções do núcleo recebem Nd = γn γf Nk.", exata=gama_n,
                       nota_tabela="A rotina não aplica γn, porque o escopo deste trabalho exige menor dimensão de "
                                   "pelo menos 19 cm; nos exemplos 5.3 e 5.4, as funções do núcleo recebem "
                                   "Nd = γn γf Nk.")
    forca_53 = forca_de_coeficientes(gama_n, nucleo.GAMA_F, forca_k)
    comprimento_x = soma(vao_livre, altura)
    comprimento_y = soma(vao_livre, largura)
    esbeltez_x = nucleo.indice_esbeltez(comprimento_x, altura)
    esbeltez_y = nucleo.indice_esbeltez(comprimento_y, largura)
    registro.adicionar(nome, "5.3: ℓe,x = ℓ0 + h = 2,78 + 0,12", "m", CRITERIO_IMPRESSAO, impresso["le_x_53"],
                       cm_para_m(comprimento_x), cm_para_m(comprimento_x), exata=cm_para_m(comprimento_x))
    registro.adicionar(nome, "5.3: λx = 3,46 ℓe,x/0,12", "", CRITERIO_IMPRESSAO, impresso["lambda_x_53"],
                       esbeltez_fator_autor(comprimento_x, altura), esbeltez_x, exata=esbeltez_x)
    registro.adicionar(nome, "5.3: ℓe,y = ℓ0 + b = 2,78 + 0,30", "m", CRITERIO_IMPRESSAO, impresso["le_y_53"],
                       cm_para_m(comprimento_y), cm_para_m(comprimento_y), exata=cm_para_m(comprimento_y))
    registro.adicionar(nome, "5.3: λy = 3,46 ℓe,y/0,30", "", CRITERIO_IMPRESSAO, impresso["lambda_y_53"],
                       esbeltez_fator_autor(comprimento_y, largura), esbeltez_y, exata=esbeltez_y,
                       nota_exata="O autor usa 3,46 no lugar de raiz(12) (3,46 × 3,08/0,30 = 35,52).")
    for chave, rotulo, comprimento_direcao in (("eax_53", "x", comprimento_x), ("eay_53", "y", comprimento_y)):
        registro.adicionar(nome, "5.3: ea,%s = θ1 ℓ/2 (seção intermediária)" % rotulo, "m", CRITERIO_IMPRESSAO,
                           impresso[chave], excentricidade_imperfeicao(theta_autor, cm_para_m(comprimento_direcao)),
                           cm_para_m(nucleo.excentricidade_acidental(nucleo.desaprumo_local(comprimento_direcao),
                                                                     comprimento_direcao)),
                           exata=excentricidade_imperfeicao(theta_autor, cm_para_m(comprimento_direcao)))
    registro.adicionar(nome, "5.3: ea,x na seção de extremidade (dobro)", "m", CRITERIO_IMPRESSAO,
                       impresso["eax_ext_53"],
                       dobro(excentricidade_imperfeicao(theta_autor, cm_para_m(comprimento_x))), None,
                       exata=dobro(excentricidade_imperfeicao(theta_autor, cm_para_m(comprimento_x))),
                       nota_tabela=NOTA_TABELA_EA_EXTREMIDADE)
    excentricidade_minima_x = excentricidade(nucleo.momento_minimo_primeira_ordem(forca_53, altura), forca_53)
    excentricidade_minima_y = excentricidade(nucleo.momento_minimo_primeira_ordem(forca_53, largura), forca_53)
    registro.adicionar(nome, "5.3: e1x,mín = 0,015 + 0,03 × 0,12", "m", CRITERIO_IMPRESSAO, impresso["e1x_min_53"],
                       cm_para_m(excentricidade_minima_x), cm_para_m(excentricidade_minima_x),
                       "0,0186 é impresso como 0,018 (truncamento), valor usado nas etapas seguintes do texto.",
                       exata=cm_para_m(excentricidade_minima_x), nota_exata="O autor trunca 0,0186 para 0,018.")
    registro.adicionar(nome, "5.3: e1y,mín = 0,015 + 0,03 × 0,30", "m", CRITERIO_IMPRESSAO, impresso["e1y_min_53"],
                       cm_para_m(excentricidade_minima_y), cm_para_m(excentricidade_minima_y),
                       exata=cm_para_m(excentricidade_minima_y))
    nu = nucleo.forca_normal_adimensional(forca_53, area_concreto, material.fcd)
    registro.adicionar(nome, "5.3: ν = γn Nd/(b h fcd)", "", CRITERIO_IMPRESSAO, impresso["nu_53"], nu, nu, exata=nu)
    curvatura_x = nucleo.curvatura_aproximada(nu, altura)
    curvatura_y = nucleo.curvatura_aproximada(nu, largura)
    excentricidade_2x = excentricidade_curvatura(comprimento_x, curvatura_x)
    excentricidade_2y = excentricidade_curvatura(comprimento_y, curvatura_y)
    registro.adicionar(nome, "5.3: e2x = (ℓe,x²/10) 0,005/[(ν + 0,5) h]", "m", CRITERIO_IMPRESSAO, impresso["e2x_53"],
                       cm_para_m(excentricidade_2x), cm_para_m(excentricidade_2x), exata=cm_para_m(excentricidade_2x))
    registro.adicionar(nome, "5.3: e2y = (ℓe,y²/10) 0,005/[(ν + 0,5) b]", "m", CRITERIO_IMPRESSAO, impresso["e2y_53"],
                       cm_para_m(excentricidade_2y), cm_para_m(excentricidade_2y), exata=cm_para_m(excentricidade_2y))
    registro.adicionar(nome, "5.3: d′/h na direção x", "", CRITERIO_IMPRESSAO, impresso["dlinha_x_53"],
                       dividir(dados["distancia_abaco_53"], altura), dividir(distancia, altura),
                       exata=dividir(dados["distancia_abaco_53"], altura))
    registro.adicionar(nome, "5.3: μx = [(e1x,mín + e2x)/h] ν", "", CRITERIO_IMPRESSAO, impresso["mu_x_53"],
                       mu_primeira_ordem(valor_impresso(impresso["nu_53"]),
                                         soma(valor_impresso(impresso["e1x_min_53"]), valor_impresso(impresso["e2x_53"])),
                                         cm_para_m(altura)),
                       mu_primeira_ordem(nu, soma(excentricidade_minima_x, excentricidade_2x), altura),
                       exata=mu_primeira_ordem(nu, soma(excentricidade_minima_x, excentricidade_2x), altura),
                       nota_exata="O autor leva adiante e1x,mín = 0,018 (truncado de 0,0186) e e2x = 0,028 "
                                  "(de %s)." % num(cm_para_m(excentricidade_2x), 5))
    omega_lido = valor_impresso(impresso["omega_x_53"])
    nu_lido = valor_impresso(impresso["nu_53"])
    mu_lido = valor_impresso(impresso["mu_x_53"])
    mu_equivalente = brentq(lambda mu: diferenca(omega_no_ponto(material, altura, largura, dados["distancia_abaco_53"],
                                                                nu_lido, mu), omega_lido), mu_lido, 0.5, xtol=1e-6)
    distancia_equivalente = brentq(lambda distancia_teste: diferenca(omega_no_ponto(
        material, altura, largura, distancia_teste, nu_lido, mu_lido), omega_lido), dados["distancia_abaco_53"],
        dividir(altura, 3.0), xtol=1e-6)
    resultados["omega_5.3"] = {"mu_equivalente": mu_equivalente,
                               "d_linha_h_equivalente": dividir(distancia_equivalente, altura)}
    registro.adicionar(nome, "5.3: ωx do ábaco 5 no ponto (ν; μ) impresso", "", CRITERIO_LEITURA,
                       impresso["omega_x_53"], None,
                       omega_no_ponto(material, altura, largura, dados["distancia_abaco_53"], nu_lido, mu_lido),
                       "Rotina na hipótese do ábaco: seção bruta, duas camadas, d′/h = 0,25. O ábaco 5 não consta do "
                       "arquivo do livro; na rotina, ω = 1,17 corresponde a μ = %s (com d′/h = 0,25) ou a d′/h = "
                       "%s (com μ = 0,28). Com o mesmo ábaco, a leitura do exemplo 5.10 fica próxima da rotina."
                       % (num(mu_equivalente, 3), num(dividir(distancia_equivalente, altura), 3)))
    registro.adicionar(nome, "5.3: As = b h (fcd/fyd) ωx", "cm²", CRITERIO_IMPRESSAO, impresso["As_x_53"],
                       area_de_omega(valor_impresso(impresso["omega_x_53"]), area_concreto, material.fcd, material.fyd),
                       None,
                       exata=area_de_omega(valor_impresso(impresso["omega_x_53"]), area_concreto, material.fcd,
                                           material.fyd))
    momento_total_x = produto(forca_53, soma(excentricidade_minima_x, excentricidade_2x))
    distancia_real = nucleo.distancia_eixo_barra(dados["cobrimento"], nucleo.diametro_estribo(dados["barra_53"]),
                                                 dados["barra_53"])
    secao_real = secao_camadas(altura, largura, distancia_real, dados["barras_por_face_53"])
    resultados["5.3"] = {"Nd_gama_n_kN": forca_53, "Md_tot_x_kNcm": momento_total_x, "d_linha_cm": distancia_real}
    for liquida, rotulo_secao in ((False, "seção bruta"), (True, "seção líquida")):
        with SecaoHipotese(liquida):
            area_rotina = nucleo.armadura_flexo_compressao(secao_real, material, forca_53, momento_total_x)
        resultados["5.3"]["As_x_" + rotulo_secao] = area_rotina
        registro.adicionar(nome, "5.3: As da rotina na direção x, 12 ϕ 12,5 em duas faces (d′ = 3,1 cm), %s"
                           % rotulo_secao, "cm²", CRITERIO_LEITURA, impresso["As_x_53"], None, area_rotina,
                           "Md,tot = γn Nd (e1x,mín + e2x) com e1x,mín = 1,86 cm; o texto usa 1,8 cm e o ábaco de "
                           "d′/h = 0,25.")
    registro.adicionar(nome, "5.3: μy = [(e1y,mín + e2y)/b] ν", "", CRITERIO_IMPRESSAO, impresso["mu_y_53"],
                       mu_primeira_ordem(valor_impresso(impresso["nu_53"]),
                                         soma(valor_impresso(impresso["e1y_min_53"]), valor_impresso(impresso["e2y_53"])),
                                         cm_para_m(largura)),
                       mu_primeira_ordem(nu, soma(excentricidade_minima_y, excentricidade_2y), largura),
                       exata=mu_primeira_ordem(nu, soma(excentricidade_minima_y, excentricidade_2y), largura),
                       nota_exata="O autor leva adiante e2y = 0,013 (de %s)." % num(cm_para_m(excentricidade_2y), 5))
    registro.adicionar(nome, "5.3: ωy do ábaco 6 (d′/h = 0,20)", "", CRITERIO_LEITURA, impresso["omega_y_53"], None,
                       None, "O ábaco 6 do capítulo 4 do livro não consta do arquivo (p. 300 a 307 ausentes): arranjo "
                             "das barras não conferido.",
                       nota_tabela="O ábaco 6 não consta do arquivo do livro (p. 300 a 307 ausentes), e o arranjo das "
                                   "barras não pôde ser conferido.")
    area_adotada = nucleo.area_barras(dados["barra_53"], dados["barras_53"])
    registro.adicionar(nome, "5.3: As adotada = 12 ϕ 12,5", "cm²", CRITERIO_IMPRESSAO, impresso["As_adot_53"],
                       produto(dados["barras_53"], dados["area_barra_autor_125"]), area_adotada,
                       "O texto usa 1,25 cm² por barra de 12,5 mm.", exata=area_adotada)
    registro.adicionar(nome, "5.3: ρ = 0,15 (fcd/fyd) ν", "", CRITERIO_IMPRESSAO, impresso["rho_53"],
                       taxa_minima_relativa(material.fcd, material.fyd, valor_impresso(impresso["nu_53"])),
                       taxa_minima_relativa(material.fcd, material.fyd, nu),
                       exata=taxa_minima_relativa(material.fcd, material.fyd, nu))
    area_minima = nucleo.armadura_minima(forca_53, material.fyd, area_concreto)
    registro.adicionar(nome, "5.3: As,mín = 0,004 Ac", "cm²", CRITERIO_IMPRESSAO, impresso["As_min_53"], area_minima,
                       area_minima, exata=area_minima)
    pilar = nucleo.PilarTeste("C&P 5.3", altura, largura, dados["fck"], dados["fyk"], forca_53, 0.0, comprimento_x,
                              cobrimento=dados["cobrimento"])
    try:
        nucleo.dimensionar_pilar(pilar)
        recusada = False
    except ValueError:
        recusada = True
    registro.verificar(nome, "5.3: a rotina recusa a seção de 12 cm (menor dimensão < 19 cm)", recusada, fonte,
                       "As etapas foram refeitas com as funções do núcleo e Nd = γn γf Nk.")
    registro.verificar(nome, "5.3: menor dimensão de 12 cm admitida pela Seção 13.2.3 da ABNT NBR 6118:2023 "
                             "(mínimo de 14 cm com γn)", altura >= MENOR_DIMENSAO_ESPECIAL_2023, fonte,
                       "A edição de 2003, usada no livro, admitia 12 cm (γn = 1,35).")
    registro.verificar(nome, "5.3: ϕ 12,5 atende 10 mm <= ϕl <= b/8 = 15 mm (18.4.2.1)",
                       nucleo.diametro_longitudinal_permitido(dados["barra_53"], altura), fonte)
    k1_autor = coeficiente_k1(valor_impresso(impresso["lambda_x_53"]))
    k1_rotina = coeficiente_k1(esbeltez_x)
    registro.adicionar(nome, "5.4: k1 = 1 - λ²/3840", "", CRITERIO_IMPRESSAO, impresso["k1_54"], k1_autor, k1_rotina,
                       "O texto usa λ = 84, arredondado de %s (fator 3,46); a coluna “rotina” usa λ = %s, com "
                       "raiz(12)." % (num(esbeltez_fator_autor(comprimento_x, altura), 2), num(esbeltez_x, 2)),
                       exata=k1_rotina,
                       nota_exata="O autor usa λ = 84, arredondado (com 3,46) de %s; com raiz(12), λ = %s."
                                  % (num(esbeltez_fator_autor(comprimento_x, altura), 2), num(esbeltez_x, 2)))
    excentricidade_kappa_autor = excentricidade_kappa_fechada(k1_autor, cm_para_m(altura),
                                                              valor_impresso(impresso["e1x_min_53"]))
    momento_kappa = momento_total_kappa_direto(altura, forca_53, comprimento_x, 1.0,
                                               nucleo.momento_minimo_primeira_ordem(forca_53, altura))
    excentricidade_kappa_rotina = excentricidade_segunda_ordem(momento_kappa, forca_53, excentricidade_minima_x)
    excentricidade_fechada_exata = excentricidade_kappa_fechada(k1_rotina, altura, excentricidade_minima_x)
    registro.adicionar(nome, "5.4: e2 pela rigidez κ (expressão 5.29)", "m", CRITERIO_IMPRESSAO, impresso["e2_54"],
                       excentricidade_kappa_autor, cm_para_m(excentricidade_kappa_rotina),
                       "Coluna “rotina”: solução direta da equação do 2º grau da Seção 15.8.3.3.3 com λ e e1,mín "
                       "exatos.", exata=cm_para_m(excentricidade_fechada_exata))
    registro.verificar(nome, "5.4: a expressão 5.29 do livro coincide com a solução direta da Seção 15.8.3.3.3 "
                             "(mesmos dados)",
                       math.isclose(excentricidade_fechada_exata, excentricidade_kappa_rotina,
                                    rel_tol=FOLGA_PONTO_FLUTUANTE), fonte,
                       "e2 = %s cm (5.29) e %s cm (rotina)." % (num(excentricidade_fechada_exata, 9),
                                                                num(excentricidade_kappa_rotina, 9)))
    resultados["5.4"] = {"k1": k1_rotina, "e2_kappa_cm": excentricidade_kappa_rotina}

    # 5.5 — pilar esbelto com fluência, 20 x 20 cm, C30
    material_55 = nucleo.Material(dados["fck_55"])
    lado = dados["lado_55"]
    comprimento = dados["comprimento_55"]
    area_concreto = nucleo.area_secao_retangular(lado, lado)
    distancia_55 = nucleo.distancia_eixo_barra(dados["cobrimento_55"], dados["estribo"], dados["barra"])
    esbeltez = nucleo.indice_esbeltez(comprimento, lado)
    registro.adicionar(nome, "5.5: λ = 3,46 ℓe/b", "", CRITERIO_IMPRESSAO, impresso["lambda_55"],
                       esbeltez_fator_autor(comprimento, lado), esbeltez, exata=esbeltez)
    registro.adicionar(nome, "5.5: d′/h = (c + ϕt + ϕl/2)/h", "", CRITERIO_IMPRESSAO, impresso["dlinha_55"],
                       dividir(distancia_55, lado), dividir(distancia_55, lado), exata=dividir(distancia_55, lado))
    excentricidade_a_autor = excentricidade_imperfeicao(theta_autor, cm_para_m(comprimento))
    excentricidade_a_rotina = nucleo.excentricidade_acidental(nucleo.desaprumo_local(comprimento), comprimento)
    nota_tabela_55 = ("Os autores adotam θ1 = 1/200 e, em Ne, o módulo secante da edição de 2003 (0,85 Eci); a "
                      "rotina usa θ1 = 1/(100 √6) = 1/245, pelas Seções 11.3.3.4.1 e 11.3.3.4.2, e Eci, pela "
                      "Seção 15.8.4.")
    registro.adicionar(nome, "5.5: ea = θ1 ℓ/2 (seção intermediária)", "m", CRITERIO_IMPRESSAO, impresso["ea_55"],
                       excentricidade_a_autor, cm_para_m(excentricidade_a_rotina),
                       "O texto adota θ1 = θ1,máx = 1/200; pelas Seções 11.3.3.4.1 e 11.3.3.4.2, θ1 = 1/(100 raiz(6)) = 1/245, "
                       "entre 1/300 e 1/200.", exata=excentricidade_a_autor, nota_tabela=nota_tabela_55)
    registro.adicionar(nome, "5.5: ea na seção de extremidade (dobro)", "m", CRITERIO_IMPRESSAO, impresso["ea_ext_55"],
                       dobro(excentricidade_a_autor), None, exata=dobro(excentricidade_a_autor),
                       nota_tabela=NOTA_TABELA_EA_EXTREMIDADE)
    momento_minimo = nucleo.momento_minimo_primeira_ordem(forca, lado)
    excentricidade_minima = excentricidade(momento_minimo, forca)
    registro.adicionar(nome, "5.5: e1,mín = 0,015 + 0,03 h", "m", CRITERIO_IMPRESSAO, impresso["e1min_55"],
                       cm_para_m(excentricidade_minima), cm_para_m(excentricidade_minima),
                       exata=cm_para_m(excentricidade_minima))
    registro.adicionar(nome, "5.5: Ec = 0,85 × 5600 raiz(fck)", "MPa", CRITERIO_IMPRESSAO, impresso["Ec_55"],
                       modulo_autor_55(dados["fck_55"]), kncm2_para_mpa(material_55.modulo_inicial),
                       "%s é impresso como %s (truncamento). A Seção 15.8.4 usa Eci = 5600 raiz(fck) (coluna "
                       "“rotina”); o texto usa 0,85 Eci (Ecs da edição de 2003)."
                       % (num(modulo_autor_55(dados["fck_55"]), 1), texto_impresso(impresso["Ec_55"])),
                       exata=modulo_autor_55(dados["fck_55"]),
                       nota_exata="O autor trunca %s para %s."
                                  % (num(modulo_autor_55(dados["fck_55"]), 1), texto_impresso(impresso["Ec_55"])),
                       nota_tabela=nota_tabela_55)
    forca_e_autor = carga_flambagem(kncm2_para_knm2(mpa_para_kncm2(valor_impresso(impresso["Ec_55"]))),
                                    dados["inercia_autor_55_m4"], cm_para_m(comprimento))
    inercia_55 = nucleo.momento_inercia_retangulo(lado, lado)
    forca_e_rotina = nucleo.forca_euler_fluencia(material_55.modulo_inicial, inercia_55, comprimento)
    forca_e_exata = carga_flambagem(kncm2_para_knm2(mpa_para_kncm2(modulo_autor_55(dados["fck_55"]))),
                                    cm4_para_m4(inercia_55), cm_para_m(comprimento))
    registro.adicionar(nome, "5.5: Ne = 10 Ec Ic/ℓe²", "kN", CRITERIO_IMPRESSAO, impresso["Ne_55"], forca_e_autor,
                       forca_e_rotina, "Cadeia da fonte com Ec impresso e Ic = 1,33 × 10⁻⁴ m⁴ (0,20⁴/12 = 1,333 × "
                                       "10⁻⁴; o texto escreve só 10⁻⁴).", exata=forca_e_exata,
                       nota_exata="O autor usa Ec truncado (26 071 MPa) e Ic = 1,33 × 10⁻⁴ m⁴ (de 1,3333 × 10⁻⁴); "
                                  "com os valores exatos, %s kN." % num(forca_e_exata, 1), nota_tabela=nota_tabela_55)
    forca_sg = dados["Nsg_55"]
    phi = dados["phi_55"]
    excentricidade_cc_autor = excentricidade_cc_independente(0.0, forca_sg, valor_impresso(impresso["ea_55"]), phi,
                                                             valor_impresso(impresso["Ne_55"]))
    excentricidade_cc_rotina = nucleo.excentricidade_fluencia(0.0, forca_sg, excentricidade_a_rotina, phi,
                                                              forca_e_rotina)
    excentricidade_cc_exata = excentricidade_cc_independente(0.0, forca_sg, excentricidade_a_autor, phi, forca_e_exata)
    registro.adicionar(nome, "5.5: ecc = (Msg/Nsg + ea)[exp(φ Nsg/(Ne - Nsg)) - 1]", "m", CRITERIO_IMPRESSAO,
                       impresso["ecc_55"], excentricidade_cc_autor, cm_para_m(excentricidade_cc_rotina),
                       "Coluna “rotina”: expressão de ecc da Seção 15.8.4, implementada no núcleo, com Msg = 0, o Nsg "
                       "(120 kN) e o φ (2) do texto e o ea e o "
                       "Ne da rotina (θ1 = 1/245; Eci).", exata=excentricidade_cc_exata, nota_tabela=nota_tabela_55)
    excentricidade_cc_formula = nucleo.excentricidade_fluencia(0.0, forca_sg, m_para_cm(valor_impresso(
        impresso["ea_55"])), phi, valor_impresso(impresso["Ne_55"]))
    registro.adicionar(nome, "5.5: ecc pela expressão da Seção 15.8.4 implementada no núcleo, com ea, Nsg e Ne "
                             "impressos", "m", CRITERIO_CONFERENCIA,
                       impresso["ecc_55"], excentricidade_cc_autor, cm_para_m(excentricidade_cc_formula),
                       "Mesma expressão: a diferença para 0,005 é só o arredondamento da impressão.")
    nu = nucleo.forca_normal_adimensional(forca, area_concreto, material_55.fcd)
    registro.adicionar(nome, "5.5: ν = Nd/(b h fcd)", "", CRITERIO_IMPRESSAO, impresso["nu_55"], nu, nu, exata=nu)
    registro.adicionar(nome, "5.5: μ1 = [(e1,mín + ecc)/h] ν", "", CRITERIO_IMPRESSAO, impresso["mu1_55"],
                       mu_primeira_ordem(valor_impresso(impresso["nu_55"]),
                                         soma(valor_impresso(impresso["e1min_55"]), valor_impresso(impresso["ecc_55"])),
                                         cm_para_m(lado)),
                       mu_primeira_ordem(nu, soma(excentricidade_minima, excentricidade_cc_rotina), lado),
                       exata=mu_primeira_ordem(nu, soma(cm_para_m(excentricidade_minima), excentricidade_cc_exata),
                                               cm_para_m(lado)),
                       nota_exata="O autor leva adiante ν = 0,33 (de %s) e ecc = 0,005 m (de %s m)."
                                  % (num(nu, 4), num(excentricidade_cc_exata, 5)), nota_tabela=nota_tabela_55)
    registro.adicionar(nome, "5.5: ℓe/h", "", CRITERIO_IMPRESSAO, impresso["le_h_55"], dividir(comprimento, lado),
                       dividir(comprimento, lado), exata=dividir(comprimento, lado))
    omega_abaco = omega_tangente_abaco(material_55, lado, lado, dados["distancia_abaco_55"],
                                       valor_impresso(impresso["nu_55"]), valor_impresso(impresso["mu1_55"]),
                                       comprimento)
    omega_norma, momento_norma = omega_acoplado_norma(material_55, lado, lado, dados["distancia_abaco_55"],
                                                      valor_impresso(impresso["nu_55"]),
                                                      valor_impresso(impresso["mu1_55"]), comprimento)
    registro.adicionar(nome, "5.5: ω do ábaco da Fig. 5.20 na construção do ábaco (0,85 fcd, sem γf3)", "",
                       CRITERIO_LEITURA, impresso["omega_55"], None, omega_abaco,
                       "Rotina com os dados impressos no ábaco (CA-50A, d′/h = 0,10, As/2 em cada face) e com seção bruta e "
                       "0,85 fcd, hipóteses não impressas; reta tangente à curva de μ em função de 1/r (Fig. 5.19). "
                       "As curvas do ábaco são espaçadas de 0,2: o valor da rotina fica entre as curvas 0 e 0,2, e o "
                       "texto lê a curva de 0,2.")
    registro.adicionar(nome, "5.5: ω pelo pilar-padrão acoplado da rotina (1,10 fcd e Nd/γf3; 15.3.1)", "",
                       CRITERIO_LEITURA, impresso["omega_55"], None, omega_norma,
                       "No mesmo ponto (ν; μ1), com a curva de deformação da Seção 15.3.1, mais rígida que a de "
                       "0,85 fcd, o pilar fica em equilíbrio sem armadura de cálculo (Md,tot = %s kN·m), e governa "
                       "As,mín."
                       % num(kncm_para_knm(momento_norma), 2))
    area_55_exata = area_de_omega(valor_impresso(impresso["omega_55"]), area_concreto, material_55.fcd,
                                  material_55.fyd)
    registro.adicionar(nome, "5.5: As = b h (fcd/fyd) ω", "cm²", CRITERIO_IMPRESSAO, impresso["As_55"],
                       area_de_omega(valor_impresso(impresso["omega_55"]), area_concreto, material_55.fcd,
                                     material_55.fyd), None, exata=area_55_exata)
    registro.adicionar(nome, "5.5: ρ = As/(b h)", "", CRITERIO_IMPRESSAO, impresso["rho_55"],
                       dividir(valor_impresso(impresso["As_55"]), area_concreto), None,
                       exata=dividir(area_55_exata, area_concreto))
    registro.adicionar(nome, "5.5: ρmín = 0,15 (fcd/fyd) ν", "", CRITERIO_IMPRESSAO, impresso["rho_min_55"],
                       taxa_minima_relativa(material_55.fcd, material_55.fyd, valor_impresso(impresso["nu_55"])),
                       taxa_minima_relativa(material_55.fcd, material_55.fyd, nu),
                       exata=taxa_minima_relativa(material_55.fcd, material_55.fyd, nu))
    area_adotada_55 = nucleo.area_barras(dados["barra_55"], dados["barras_55"])
    registro.adicionar(nome, "5.5: As adotada = 4 ϕ 12,5", "cm²", CRITERIO_IMPRESSAO, impresso["As_adot_55"],
                       produto(dados["barras_55"], dados["area_barra_autor_125"]), area_adotada_55,
                       exata=area_adotada_55)
    pilar = nucleo.PilarTeste("C&P 5.5", lado, lado, dados["fck_55"], dados["fyk"], forca, 0.0, comprimento,
                              cobrimento=dados["cobrimento_55"])
    resultado = nucleo.dimensionar_pilar(pilar)
    registro.verificar(nome, "5.5: dimensionamento completo pela rotina, com as hipóteses do trabalho (φ = %s e "
                             "Nsg da Seção 3.6; θ1 e Eci da norma)" % num(nucleo.COEFICIENTE_FLUENCIA, 1),
                       resultado.status == nucleo.STATUS_OK, fonte,
                       "ecc = %s cm; Md,tot = %s kN·m; As,calc = As,mín = %s cm²; barras adotadas: %s." % (
                           num(resultado.direcao_b.excentricidade_fluencia, 3),
                           num(kncm_para_knm(resultado.direcao_b.momento_total), 2),
                           num(resultado.area_calculada, 2), texto_barras(resultado.candidato)))
    resultados["5.5"] = {"ea_rotina_cm": excentricidade_a_rotina, "Ne_rotina_kN": forca_e_rotina,
                         "ecc_rotina_cm": excentricidade_cc_rotina, "omega_construcao_abaco": omega_abaco,
                         "omega_pilar_padrao_acoplado": omega_norma, "Md_tot_acoplado_kNcm": momento_norma,
                         "dimensionamento": {"ecc_cm": resultado.direcao_b.excentricidade_fluencia,
                                             "Md_tot_kNcm": resultado.direcao_b.momento_total,
                                             "As_calc_cm2": resultado.area_calculada,
                                             "barras": texto_barras(resultado.candidato)}}

    # 5.10 — pilar lateral P1, 20 x 40 cm, situação 1 (seção de extremidade, flexão na direção de 20 cm)
    forca_k = dados["N_510"]
    forca = forca_de_coeficientes(nucleo.GAMA_F, forca_k)
    altura = dados["b_510"]
    largura = dados["h_510"]
    area_concreto = nucleo.area_secao_retangular(largura, altura)
    registro.adicionar(nome, "5.10: λy = 3,46 ℓe,y/0,40", "", CRITERIO_IMPRESSAO, impresso["lambda_y_510"],
                       esbeltez_fator_autor(dados["comprimento_y_510"], largura),
                       nucleo.indice_esbeltez(dados["comprimento_y_510"], largura),
                       exata=nucleo.indice_esbeltez(dados["comprimento_y_510"], largura))
    registro.adicionar(nome, "5.10: λx = 3,46 ℓe,x/0,20", "", CRITERIO_IMPRESSAO, impresso["lambda_x_510"],
                       esbeltez_fator_autor(dados["comprimento_x_510"], altura),
                       nucleo.indice_esbeltez(dados["comprimento_x_510"], altura),
                       exata=nucleo.indice_esbeltez(dados["comprimento_x_510"], altura))
    momento_1 = knm_para_kncm(dados["M1_510_knm"])
    excentricidade_1 = excentricidade(momento_1, forca_k)
    registro.adicionar(nome, "5.10: e1x = M1/N = 12,11/816", "m", CRITERIO_IMPRESSAO, impresso["e1x_510"],
                       cm_para_m(excentricidade_1), cm_para_m(excentricidade_1), exata=cm_para_m(excentricidade_1))
    registro.adicionar(nome, "5.10: λ1x = (25 + 12,5 e1x/h)/αb, αb = 0,4", "", CRITERIO_IMPRESSAO,
                       impresso["lambda1_x_510"],
                       esbeltez_limite_sem_limites(dividir(m_para_cm(valor_impresso(impresso["e1x_510"])), altura),
                                                   dados["alfa_b_510"]),
                       nucleo.esbeltez_limite(nucleo.excentricidade_relativa(momento_1, forca_k, altura),
                                              dados["alfa_b_510"]),
                       "αb = 0,4 adotado pelo texto para o pilar lateral (p. 387).",
                       exata=esbeltez_limite_sem_limites(dividir(excentricidade_1, altura), dados["alfa_b_510"]))
    registro.adicionar(nome, "5.10: 100 raiz(ℓ) (θ1 = 1/187)", "", CRITERIO_IMPRESSAO, impresso["theta_510"],
                       fator_desaprumo(cm_para_m(dados["comprimento_510"])), None,
                       exata=fator_desaprumo(cm_para_m(dados["comprimento_510"])))
    registro.verificar(nome, "5.10: θ1 = 1/187 limitado a θ1,máx = 1/200 (11.3.3.4.1)",
                       nucleo.desaprumo_local(dados["comprimento_510"]) == nucleo.THETA1_MAX, fonte)
    registro.adicionar(nome, "5.10: ea,extr = ℓ θ1", "m", CRITERIO_IMPRESSAO, impresso["ea_ext_510"],
                       produto(cm_para_m(dados["comprimento_510"]), theta_autor), None,
                       "A rotina não soma ea na extremidade (11.3.3.4.3).",
                       exata=produto(cm_para_m(dados["comprimento_510"]), theta_autor),
                       nota_tabela=NOTA_TABELA_EA_EXTREMIDADE)
    registro.adicionar(nome, "5.10: ea,inter = (ℓ/2) θ1", "m", CRITERIO_IMPRESSAO, impresso["ea_int_510"],
                       excentricidade_imperfeicao(theta_autor, cm_para_m(dados["comprimento_510"])),
                       cm_para_m(nucleo.excentricidade_acidental(nucleo.desaprumo_local(dados["comprimento_510"]),
                                                                 dados["comprimento_510"])),
                       exata=excentricidade_imperfeicao(theta_autor, cm_para_m(dados["comprimento_510"])))
    excentricidade_minima_x = excentricidade(nucleo.momento_minimo_primeira_ordem(forca, altura), forca)
    excentricidade_minima_y = excentricidade(nucleo.momento_minimo_primeira_ordem(forca, largura), forca)
    registro.adicionar(nome, "5.10: emín,x = 0,015 + 0,03 × 0,20", "m", CRITERIO_IMPRESSAO, impresso["emin_x_510"],
                       cm_para_m(excentricidade_minima_x), cm_para_m(excentricidade_minima_x),
                       exata=cm_para_m(excentricidade_minima_x))
    registro.adicionar(nome, "5.10: emín,y = 0,015 + 0,03 × 0,40", "m", CRITERIO_IMPRESSAO, impresso["emin_y_510"],
                       cm_para_m(excentricidade_minima_y), cm_para_m(excentricidade_minima_y),
                       exata=cm_para_m(excentricidade_minima_y))
    excentricidade_rotina = maior(excentricidade_1, excentricidade_minima_x)
    excentricidade_extremidade_exata = soma(excentricidade_1,
                                            m_para_cm(produto(cm_para_m(dados["comprimento_510"]), theta_autor)))
    nota_tabela_510 = ("Os autores somam ea,extr a e1 na seção de extremidade, na leitura que fazem da edição de 2003 "
                       "da norma (Carvalho; Pinheiro, 2009, p. 387); a rotina usa o momento mínimo, que substitui as "
                       "imperfeições locais (Seção 11.3.3.4.3): e = máx(e1; e1,mín) = %s cm."
                       % num(excentricidade_rotina, 2))
    registro.adicionar(nome, "5.10: e na seção de extremidade, direção x", "cm", CRITERIO_IMPRESSAO,
                       impresso["ex_510"], soma(m_para_cm(valor_impresso(impresso["e1x_510"])),
                                                m_para_cm(valor_impresso(impresso["ea_ext_510"]))),
                       excentricidade_rotina,
                       "O texto soma ea,extr a e1 (1,48 + 1,75), na leitura que os autores fazem da NBR 6118:2003 "
                       "(Carvalho; Pinheiro, 2009, p. 387); a rotina substitui as imperfeições locais pelo momento mínimo "
                       "(11.3.3.4.3): e = máx(e1; e1,mín) = 2,10 cm. Nessa direção, λ < λ1: sem 2ª ordem.",
                       exata=excentricidade_extremidade_exata, nota_tabela=nota_tabela_510)
    nu = nucleo.forca_normal_adimensional(forca, area_concreto, material.fcd)
    registro.adicionar(nome, "5.10: ν = Nd/(b h fcd)", "", CRITERIO_IMPRESSAO, impresso["nu_510"], nu, nu, exata=nu)
    mu_rotina = mu_primeira_ordem(nu, excentricidade_rotina, altura)
    registro.adicionar(nome, "5.10: μx = ν e/h", "", CRITERIO_IMPRESSAO, impresso["mu_510"],
                       mu_primeira_ordem(valor_impresso(impresso["nu_510"]), valor_impresso(impresso["ex_510"]),
                                         altura), mu_rotina,
                       "Com o ν = 1,0 impresso, 0,1615, que daria 0,162; o 0,161 vem do ν sem arredondar, 0,9996: "
                       "0,9996 × 3,23/20 = 0,1614.",
                       exata=mu_primeira_ordem(nu, excentricidade_extremidade_exata, altura),
                       nota_exata="O autor leva adiante o e impresso (3,23 cm, de %s cm): com o e exato, %s."
                                  % (num(excentricidade_extremidade_exata, 4),
                                     num(mu_primeira_ordem(nu, excentricidade_extremidade_exata, altura), 5)),
                       nota_tabela=nota_tabela_510)
    registro.adicionar(nome, "5.10: ω do ábaco 5 no ponto (ν; μ) impresso", "", CRITERIO_LEITURA,
                       impresso["omega_510"], None,
                       omega_no_ponto(material, altura, largura, dados["distancia_abaco_510"],
                                      valor_impresso(impresso["nu_510"]), valor_impresso(impresso["mu_510"])),
                       "Rotina na hipótese do ábaco: seção bruta, duas camadas, d′/h = 0,25.")
    omega_rotina = omega_no_ponto(material, altura, largura, dados["distancia_abaco_510"], nu, mu_rotina)
    registro.adicionar(nome, "5.10: ω da rotina com e = máx(e1; e1,mín) = 2,10 cm", "", CRITERIO_LEITURA,
                       impresso["omega_510"], None, omega_rotina,
                       "Diferença de critério (soma de ea ou momento mínimo), e não de cálculo.",
                       nota_tabela=nota_tabela_510)
    resultados["5.10"] = {"e_rotina_cm": excentricidade_rotina, "mu_rotina": mu_rotina, "omega_rotina": omega_rotina}
    return resultados


# ----------------------------------------------------------------------------- relatório
def texto_numero(valor) -> str:
    if valor is None:
        return "–"
    if isinstance(valor, bool):
        return "sim" if valor else "não"
    if isinstance(valor, str):
        return valor
    if valor == 0.0:
        return "0"
    return "%.6g" % valor


def escrever_relatorio(registro: Registro, resultados: dict) -> None:
    linhas_texto = ["VALIDAÇÃO DA ROTINA COM EXEMPLOS RESOLVIDOS DA LITERATURA (Nível 2)", linha_de("=", 72), "",
                    "Critérios: “impressão” = conta da fonte (reproduz: a cadeia da fonte, com os valores impressos "
                    "das etapas anteriores e arredondada às casas impressas com a meia unidade para cima, em módulo, "
                    "dá o valor impresso); “programa” = resultado de um programa; “leitura” = leitura gráfica na "
                    "fonte, ou valor da rotina comparado com um valor impresso de outra grandeza; “conferência” = "
                    "valor já contado, repetido para conferir uma fórmula. Desvio da rotina = (rotina - "
                    "impresso)/impresso × 100. Origem: %s." % ORIGEM_LEGENDA,
                    "Cadeia exata (colunas “Exata” e “Repr.ex.”, só no critério “impressão”): as contas do autor "
                    "sem arredondar etapas intermediárias e com constantes exatas (raiz(12), frações exatas, área "
                    "nominal das barras); reproduz pelo mesmo arredondamento; origem exata pelos mesmos códigos.",
                    "Rotina × cadeia exata (resumo no fim): desvio (rotina - exata)/exata × 100; “sem diferença” até "
                    "%s %%, “abaixo do limite” até %s %%, “difere” acima. “Nota da tabela do TCC”: nota curta das "
                    "tabelas do Apêndice F, só onde a rotina difere da cadeia exata, onde há erro da fonte ou onde a "
                    "linha precisa de explicação." % (texto_numero(LIMITE_PONTO_FLUTUANTE_PERCENTUAL),
                                                       texto_numero(LIMITE_COINCIDENCIA_PERCENTUAL)), ""]
    exemplo_atual = None
    for linha in registro.linhas:
        if linha["exemplo"] != exemplo_atual:
            exemplo_atual = linha["exemplo"]
            linhas_texto += ["", exemplo_atual, linha_de("-", len(exemplo_atual)),
                             "%-70s %-6s %-10s %-11s %-12s %-12s %-8s %-9s %-12s %-8s" % (
                                 "Etapa", "Unid.", "Critério", "Impresso", "Fonte", "Rotina", "Reproduz", "Desvio %",
                                 "Exata", "Repr.ex.")]
        linhas_texto.append("%-70s %-6s %-10s %-11s %-12s %-12s %-8s %-9s %-12s %-8s  origem %s; exata %s" % (
            linha["etapa"][:70], linha["unidade"], linha["criterio"], texto_numero(linha["impresso"]),
            texto_numero(linha["autor"]), texto_numero(linha["rotina"]), texto_numero(linha["reproduz"]),
            texto_numero(linha["desvio_rotina_percentual"]), texto_numero(linha["exata"]),
            texto_numero(linha["reproduz_exata"]), linha["origem"], texto_numero(linha["origem_exata"])))
        if linha["observacao"]:
            linhas_texto.append("      obs.: " + linha["observacao"])
        if linha["nota_exata"]:
            linhas_texto.append("      cadeia exata: " + linha["nota_exata"])
        if linha["nota_tabela"]:
            linhas_texto.append("      nota da tabela do TCC: " + linha["nota_tabela"])
    linhas_texto += ["", "Verificações lógicas", linha_de("-", 20)]
    for verificacao in registro.verificacoes:
        linhas_texto.append("%s | %s | rotina: %s %s" % (verificacao["exemplo"], verificacao["verificacao"],
                                                         texto_numero(verificacao["rotina_atende"]),
                                                         ("| " + verificacao["observacao"])
                                                         if verificacao["observacao"] else ""))
    impressos = [linha for linha in registro.linhas if linha["reproduz"] is not None]
    reproduzidos = [linha for linha in impressos if linha["reproduz"]]
    linhas_texto += ["", "Resumo: %d valores impressos com critério “impressão”; %d reproduzidos pela cadeia da fonte."
                     % (len(impressos), len(reproduzidos))]
    for linha in impressos:
        if not linha["reproduz"]:
            linhas_texto.append("  não reproduzido: %s | %s | impresso %s, cadeia da fonte %s | %s" % (
                linha["exemplo"], linha["etapa"], linha["impresso"], texto_numero(linha["autor"]),
                linha["observacao"]))
    exatas = [linha for linha in registro.linhas if linha["reproduz_exata"] is not None]
    linhas_texto.append("Cadeia exata: %d valores impressos com critério “impressão”; %d reproduzidos."
                        % (len(exatas), len([linha for linha in exatas if linha["reproduz_exata"]])))
    for linha in exatas:
        if not linha["reproduz_exata"]:
            linhas_texto.append("  não reproduzido pela cadeia exata: %s | %s | impresso %s, cadeia exata %s | "
                                "origem %s | %s" % (linha["exemplo"], linha["etapa"], linha["impresso"],
                                                    texto_numero(linha["exata"]), linha["origem_exata"],
                                                    linha["nota_exata"]))
    comparadas = [linha for linha in exatas if linha["comparacao_rotina_exata"] is not None]
    contagens = {classe: len([linha for linha in comparadas if linha["comparacao_rotina_exata"] == classe])
                 for classe in (CLASSE_SEM_DIFERENCA, CLASSE_ABAIXO_LIMITE, CLASSE_DIFERE)}
    linhas_texto.append("Rotina × cadeia exata: %d etapas com os dois valores; %d sem diferença; %d abaixo do limite; "
                        "%d diferem." % (len(comparadas), contagens[CLASSE_SEM_DIFERENCA],
                                         contagens[CLASSE_ABAIXO_LIMITE], contagens[CLASSE_DIFERE]))
    for linha in comparadas:
        if linha["comparacao_rotina_exata"] != CLASSE_SEM_DIFERENCA:
            linhas_texto.append("  %s: %s | %s | exata %s, rotina %s, desvio %s %% | %s" % (
                linha["comparacao_rotina_exata"], linha["exemplo"], linha["etapa"], texto_numero(linha["exata"]),
                texto_numero(linha["rotina"]), texto_numero(linha["desvio_rotina_exata_percentual"]),
                linha["nota_tabela"]))
    for aviso in registro.avisos_cadeia_exata:
        linhas_texto.append("  AVISO da cadeia exata: " + aviso)
    if registro.digitalizacao.get("situacao") == "feita":
        for painel, dados in registro.digitalizacao["paineis"].items():
            linhas_texto.append("Fig. 8.2.5-1, d′/h = %s: %d pontos digitalizados; maior |Δμ| = %.4f; Δμ médio = %+.4f"
                                % (painel, dados["quantidade"], dados["maior_diferenca_absoluta"],
                                   dados["diferenca_media"]))
    else:
        linhas_texto.append("Digitalização da Fig. 8.2.5-1: " + registro.digitalizacao.get("situacao", ""))
    ARQUIVO_TEXTO.write_text("\n".join(linhas_texto) + "\n", encoding="utf-8")
    ARQUIVO_JSON.write_text(json.dumps({"linhas": registro.linhas, "verificacoes": registro.verificacoes,
                                        "digitalizacao_fusco": registro.digitalizacao, "resultados": resultados},
                                       ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def main() -> None:
    PASTA_SAIDA.mkdir(exist_ok=True)
    registro = Registro()
    resultados = {"L1": executar_kimura(registro), "L2": executar_scadelai(registro), "L3": executar_fusco(registro),
                  "L4": executar_franco(registro), "L5": executar_carvalho(registro)}
    escrever_relatorio(registro, resultados)
    print(ARQUIVO_TEXTO)
    print(ARQUIVO_TEXTO.read_text(encoding="utf-8").split("Resumo:")[1])


if __name__ == "__main__":
    main()
