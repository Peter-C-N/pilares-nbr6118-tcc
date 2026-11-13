# -*- coding: utf-8 -*-
"""
pilares_nbr6118.py
==================
Dimensionamento de pilares retangulares de concreto armado à flexão normal composta reta, com
índice de esbeltez lambda <= 140, segundo a ABNT NBR 6118:2023.

Trabalho de Conclusão de Curso — Engenharia Civil / UFMT (FAET)
Autor: Pedro Costaneski Neto  |  Orientador: Prof. Dr. Milton Soares Filho
Coorientador: Prof. Dr. Alberto Rodrigues Dalmaso

Unidades internas (todas as fórmulas usam este sistema, salvo indicação no próprio comentário):
    forças em kN; comprimentos em cm; tensões em kN/cm² (1 kN/cm² = 10 MPa);
    momentos em kN·cm; deformações específicas adimensionais (0,0035 = 3,5 ‰); curvaturas em 1/cm.

Organização do arquivo:
    1) ZONA DE AUDITORIA — constantes normativas (C), hipóteses do trabalho (H), parâmetros
       numéricos (N), fórmulas (F), regras de negócio (R) e métodos numéricos (M), todos como
       constantes ou funções puras. Acima de cada item há um comentário de rastreabilidade com: o que
       calcula, variáveis, fonte (seção da NBR 6118:2023 e equação do TCC) e onde é usado (função e
       linha). As linhas "Usada em" são geradas e conferidas automaticamente por rastreabilidade.py.
    2) MOTOR DE CÁLCULO — apenas organiza os cálculos chamando os itens da zona de auditoria;
       não contém fórmulas escritas diretamente.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import brentq

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — INÍCIO
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────── A) CONSTANTES NORMATIVAS (C)
# [C01] Coeficientes de ponderação das resistências no ELU, combinações normais.
#   Variáveis: GAMA_C (concreto), GAMA_S (aço).  Fonte: NBR 6118:2023, Tabela 12.1.
#   Usada em: resistencia_calculo_concreto (l. 416); resistencia_calculo_aco (l. 426)
#   Usada também em: validacao.py: conferir_mn1r_independente (l. 2108);
#       processar_matriz.py: escrever_hipoteses (l. 788); verificar_matriz.py: forca_exata (l. 175)
GAMA_C = 1.4
GAMA_S = 1.15

# [C02] Coeficiente de ponderação das ações no ELU (ações permanentes e variáveis desfavoráveis).
#   Variáveis: GAMA_F.  Fonte: NBR 6118:2023, Tabela 11.1.
#   Usada em: parcelas_forca_caracteristica (l. 967)
#   Usada também em: validacao.py: cadeia_exata_bastos (l. 1133); validacao.py: conferir_mn1r_independente (l. 2088);
#       processar_matriz.py: escrever_hipoteses (l. 788); validacao_literatura.py: executar_scadelai (l. 1384, 1389);
#       validacao_literatura.py: executar_carvalho (l. 2022, 2210, 2496)
GAMA_F = 1.4

# [C03] Parcela gama_f3 da formulação de segurança da relação momento-curvatura.
#   Variáveis: GAMA_F3.  Fonte: NBR 6118:2023, Seção 15.3.1.
#   Usada em: forca_normal_curva_deformacao (l. 879); residuo_equilibrio_pilar_padrao (l. 898)
#   Usada também em: validacao.py: momentos_reduzidos (l. 307); validacao.py: momento_ponto_b (l. 1261);
#       processar_matriz.py: escrever_hipoteses (l. 796); gerar_auditor.py: reta_momento_externo (l. 149);
#       gerar_auditor.py: momento_reduzido (l. 168); figuras_complementares.py: momento_majorado (l. 228);
#       memorial_pilares.py: majorado_gama_f3 (l. 282)
GAMA_F3 = 1.1

# [C04] Resistência do concreto na curva momento-curvatura usada no cálculo das deformações
#   ("curva obtida com 1,10 fcd e NRd/gama_f3") e redução da resistência no diagrama de cálculo
#   (0,85 eta_c fcd, curva que define o esforço resistente).
#   Variáveis: FATOR_CURVA_DEFORMACAO, ALFA_C_DIAGRAMA.  Fonte: NBR 6118:2023, 8.2.10.1 e 15.3.1.
#   Fórmula: σ_{pico} = 1,10 · f_{cd} (curva de deformação);  σ_{pico} = 0,85 · η_{c} · f_{cd} (diagrama de cálculo)
#   Usada em: tensao_pico_concreto (l. 467, 468)
FATOR_CURVA_DEFORMACAO = 1.10
ALFA_C_DIAGRAMA = 0.85

# [C05] Módulo de elasticidade do aço de armadura passiva (kN/cm²; 210 GPa).
#   Variáveis: MODULO_ACO.  Fonte: NBR 6118:2023, 8.3.5.
#   Usada em: tensao_aco (l. 531)
#   Usada também em: validacao.py: forca_compressao_centrada (l. 122);
#       validacao.py: forca_compressao_centrada_secao_liquida (l. 131); validacao.py: equilibrio_manual (l. 179);
#       validacao.py: esforcos_por_fibras (l. 236); validacao.py: rigidez_tangente_inicial (l. 260);
#       validacao.py: esforcos_obliquos_por_fibras (l. 769); figuras_tcc.py: figura_dominios (l. 579)
MODULO_ACO = 21000.0

# [C06] Alongamento máximo da armadura tracionada no ELU (pivô A, 10 ‰).
#   Variáveis: DEFORMACAO_ULTIMA_ACO.  Fonte: NBR 6118:2023, 17.2.2, Figura 17.1.
#   Usada em: configuracao_estado_limite (l. 722, 723, 724); intervalo_deformacao_topo (l. 1775)
DEFORMACAO_ULTIMA_ACO = 0.010

# [C07] Divisor do pilar-padrão (deformada senoidal: l_e² / pi² ≈ l_e² / 10).
#   Variáveis: DIVISOR_PILAR_PADRAO.  Fonte: NBR 6118:2023, 15.8.3.3.2.
#   Fórmula: \frac{ℓ_{e}^{2}}{π^{2}} ≈ \frac{ℓ_{e}^{2}}{10}
#   Usada em: momento_total_pilar_padrao (l. 865); residuo_equilibrio_pilar_padrao (l. 896)
#   Usada também em: gerar_auditor.py: reta_momento_externo (l. 148);
#       figuras_complementares.py: momento_externo (l. 224); memorial_pilares.py: momento_externo (l. 278)
DIVISOR_PILAR_PADRAO = 10.0

# [C08] Limites da esbeltez-limite lambda_1 e das faixas de aplicação dos métodos de 2ª ordem.
#   Variáveis: LAMBDA1_MIN, LAMBDA1_MAX, LAMBDA_LIMITE_CURVATURA (curvatura aproximada; acima dele a
#   fluência é obrigatória), LAMBDA_LIMITE_PILAR_PADRAO (pilar-padrão acoplado a diagramas M, N, 1/r;
#   acima dele o método geral é obrigatório).
#   Fonte: NBR 6118:2023, 15.8.2, 15.8.3.3.2, 15.8.3.3.4 e 15.8.4.
#   Usada em: esbeltez_limite (l. 812); metodo_segunda_ordem (l. 1456, 1458); validar_entradas (l. 1645)
#   Usada também em: validacao.py: nivel_2 (l. 1889, 1890); verificar_matriz.py: conferir_entradas (l. 310, 311);
#       verificar_matriz.py: conferir_resultados (l. 381); figuras_tcc.py: nível do módulo (l. 249)
LAMBDA1_MIN = 35.0
LAMBDA1_MAX = 90.0
LAMBDA_LIMITE_CURVATURA = 90.0
LAMBDA_LIMITE_PILAR_PADRAO = 140.0

# [C09] Taxas-limite de armadura longitudinal de pilares: TAXA_MINIMA_GEOMETRICA (0,4 % de Ac),
#   FATOR_ARMADURA_MINIMA_FORCA (0,15 Nd/fyd), TAXA_MAXIMA_COM_EMENDA (8 % de Ac, inclusive na região
#   de emenda) e TAXA_MAXIMA_FORA_EMENDA (4 %: hipótese de traspasse de todas as barras na mesma
#   seção, que dobra a armadura na emenda).  Fonte: NBR 6118:2023, 17.3.5.3.1 e 17.3.5.3.2.
#   Fórmula: ρ_{mín} = 0,4 %;  A_{s,mín} ≥ \frac{0,15 · N_{d}}{f_{yd}}
#   Fórmula: ρ_{máx} = 8 % (com emenda);  ρ_{máx} = 4 % (fora da emenda)
#   Usada em: armadura_minima (l. 1054); armadura_limite_busca (l. 1078); PilarTeste (l. 2306)
#   Usada também em: processar_matriz.py: nível do módulo (l. 322); figuras_tcc.py: nível do módulo (l. 168);
#       figuras_complementares.py: nível do módulo (l. 248)
TAXA_MINIMA_GEOMETRICA = 0.004
FATOR_ARMADURA_MINIMA_FORCA = 0.15
TAXA_MAXIMA_COM_EMENDA = 0.08
TAXA_MAXIMA_FORA_EMENDA = 0.04

# [C10] Regras de detalhamento das armaduras longitudinais e transversais (cm):
#   DIAMETRO_MINIMO_LONGITUDINAL (10 mm), FRACAO_DIAMETRO_MAXIMO (1/8 da menor dimensão),
#   ESPACAMENTO_LIVRE_MINIMO_ABSOLUTO (20 mm), FATOR_AGREGADO_ESPACAMENTO (1,2 d_max),
#   ESPACAMENTO_MAXIMO_ABSOLUTO (400 mm), FATOR_ESPACAMENTO_MAXIMO (2 x menor dimensão),
#   DIAMETRO_MINIMO_ESTRIBO (5 mm), FRACAO_DIAMETRO_ESTRIBO (1/4 da barra longitudinal).
#   Fonte: NBR 6118:2023, 18.4.2.1, 18.4.2.2 e 18.4.3.
#   Usada em: diametro_longitudinal_permitido (l. 1111); diametro_estribo (l. 1122);
#       espacamento_livre_minimo (l. 1133, 1134); espacamento_maximo_eixos (l. 1145)
#   Usada também em: memorial_pilares.py: diametro_maximo_permitido (l. 306)
DIAMETRO_MINIMO_LONGITUDINAL = 1.0
FRACAO_DIAMETRO_MAXIMO = 1.0 / 8.0
ESPACAMENTO_LIVRE_MINIMO_ABSOLUTO = 2.0
FATOR_AGREGADO_ESPACAMENTO = 1.2
ESPACAMENTO_MAXIMO_ABSOLUTO = 40.0
FATOR_ESPACAMENTO_MAXIMO = 2.0
DIAMETRO_MINIMO_ESTRIBO = 0.5
FRACAO_DIAMETRO_ESTRIBO = 0.25

# [C11] Desaprumo para imperfeições locais: limites de theta_1 (1/300 e 1/200).
#   Variáveis: THETA1_MIN, THETA1_MAX.  Fonte: NBR 6118:2023, 11.3.3.4.1 e 11.3.3.4.2.
#   Fórmula: \frac{1}{300} ≤ θ_{1} ≤ \frac{1}{200}
#   Usada em: desaprumo_local (l. 911)
#   Usada também em: validacao_literatura.py: executar_franco (l. 1945);
#       validacao_literatura.py: executar_carvalho (l. 2524)
THETA1_MIN = 1.0 / 300.0
THETA1_MAX = 1.0 / 200.0

# [C13] Dimensões mínimas da seção, na validação das entradas (Módulo 1 da Figura 3 do TCC, Figura 3.1 do PP):
#   menor dimensão >= 19 cm (sem o coeficiente adicional gama_n) e área >= 360 cm².
#   Variáveis: DIMENSAO_MINIMA_PILAR (cm), AREA_MINIMA_PILAR (cm²).  Fonte: NBR 6118:2023, 13.2.3.
#   Fórmula: b_{mín} ≥ 19 cm;  A_{c} ≥ 360 cm²
#   Usada em: validar_entradas (l. 1634, 1635, 1636, 1637)
#   Usada também em: verificar_matriz.py: conferir_entradas (l. 300, 301)
DIMENSAO_MINIMA_PILAR = 19.0
AREA_MINIMA_PILAR = 360.0

# [C14] Estribos de pilares: limite absoluto do espaçamento longitudinal (200 mm), múltiplo do diâmetro da barra
#   longitudinal no limite do espaçamento para CA-50 (12 φl), extensão protegida contra a flambagem a partir de
#   cada canto do estribo poligonal (20 φt) e número máximo de barras nesse trecho, sem contar a de canto (2).
#   Variáveis: ESPACAMENTO_MAXIMO_ESTRIBO_ABSOLUTO (cm), FATOR_ESPACAMENTO_ESTRIBO_CA50,
#   FATOR_TRECHO_PROTEGIDO, BARRAS_MAXIMAS_TRECHO_PROTEGIDO.
#   Fonte: NBR 6118:2023, 18.4.3 (espaçamento) e 18.2.4 (proteção contra a flambagem das barras).
#   Fórmula: s_{t} ≤ mín(200 mm; b_{mín}; 12 · φ_{ℓ});  trecho protegido = 20 · φ_{t}
#   Usada em: espacamento_maximo_estribos (l. 1201, 1202); trecho_protegido_estribo (l. 1213);
#       barras_sem_protecao_face (l. 1707)
#   Usada também em: memorial_pilares.py: doze_diametros (l. 310)
ESPACAMENTO_MAXIMO_ESTRIBO_ABSOLUTO = 20.0
FATOR_ESPACAMENTO_ESTRIBO_CA50 = 12.0
FATOR_TRECHO_PROTEGIDO = 20.0
BARRAS_MAXIMAS_TRECHO_PROTEGIDO = 2

# [C12] Bitolas comerciais de aço (cm): barras longitudinais CA-50 e estribos (5 mm em CA-60; demais em CA-50).
#   Variáveis: BITOLAS_LONGITUDINAIS, BITOLAS_ESTRIBO.  Fonte: ABNT NBR 7480 (diâmetros nominais).
#   Usada em: diametro_estribo (l. 1123); gerar_candidatos (l. 2334); candidato_referencia (l. 2355)
#   Usada também em: processar_matriz.py: escrever_hipoteses (l. 828, 830);
#       verificar_matriz.py: conferir_resultados (l. 410, 411)
BITOLAS_LONGITUDINAIS = (1.0, 1.25, 1.6, 2.0, 2.5, 3.2)
BITOLAS_ESTRIBO = (0.5, 0.63, 0.8, 1.0)

# ───────────────────────────────────────────────────────────── B) HIPÓTESES DO TRABALHO (H)
# [H01] Aço CA-50 (fyk = 500 MPa) e agregado graúdo de granito ou gnaisse (alfa_E = 1,0).
#   Variáveis: RESISTENCIA_ACO_MPA, ALFA_E_AGREGADO.  Fonte: Seção 3.6 do TCC; NBR 6118:2023, 8.2.8.
#   Usada em: Material.__init__ (l. 1833)
#   Usada também em: validacao.py: conferir_mn1r_independente (l. 2090);
#       processar_matriz.py: criar_pilar (l. 378, 382); processar_matriz.py: escrever_hipoteses (l. 803, 805);
#       figuras_tcc.py: nível do módulo (l. 77)
RESISTENCIA_ACO_MPA = 500.0
ALFA_E_AGREGADO = 1.0

# [H02] Cobrimento nominal (cm) e dimensão máxima característica do agregado graúdo (cm; brita 1).
#   Variáveis: COBRIMENTO_NOMINAL, DIAMETRO_MAXIMO_AGREGADO.  Fonte: Seção 3.6 do TCC.
#   Usada em: PilarTeste (l. 2304, 2305)
#   Usada também em: processar_matriz.py: escrever_hipoteses (l. 806, 809);
#       verificar_matriz.py: conferir_resultados (l. 410)
COBRIMENTO_NOMINAL = 2.5
DIAMETRO_MAXIMO_AGREGADO = 1.9

# [H03] Fluência (lambda > 90) e combinação de ações: coeficiente de fluência phi, fração permanente
#   f_g = g/(g + q) da força normal característica e fator de redução psi_2 da combinação quase
#   permanente (edifício residencial).  phi = 2,5 é o valor da Tabela 8.1 (concretos C20 a C45, U = 55 %,
#   t0 = 60 dias, 2Ac/u = 20 cm); na seção 40 x 80 (2Ac/u = 26,7 cm), a interpolação até 60 cm (2,3) dá
#   2,47, que arredonda a 2,5; nas seções menores (12,0 e 18,8 cm) adota-se a coluna de 20 cm.
#   Variáveis: COEFICIENTE_FLUENCIA, FRACAO_PERMANENTE, PSI2.
#   Fonte: Seção 3.6 do TCC; NBR 6118:2023, 8.2.11 (Tabela 8.1) e 11.8.3.2 (Tabela 11.2).
#   Usada em: solicitacao_direcao (l. 2124, 2125, 2130)
#   Usada também em: validacao.py: conferir_mn1r_independente (l. 2089, 2091);
#       processar_matriz.py: linha_resultado (l. 428); processar_matriz.py: escrever_hipoteses (l. 792, 793, 794);
#       verificar_matriz.py: conferir_resultados (l. 443, 535); validacao_literatura.py: executar_carvalho (l. 2480);
#       memorial_pilares.py: dados_direcao (l. 431, 438); memorial_pilares.py: calcular_memorial (l. 525)
COEFICIENTE_FLUENCIA = 2.5
FRACAO_PERMANENTE = 0.70
PSI2 = 0.30

# [H04] Vinculação e momentos de 1ª ordem: pilar biapoiado sem cargas transversais (beta = 1, l_e = l),
#   com momentos de 1ª ordem iguais no topo e na base (M_B / M_A = +1, curvatura simples) na direção de b_w
#   ([H07]), gerados pela excentricidade e1 da força normal; na direção de h não há momento aplicado (somente o
#   momento mínimo, verificado isoladamente pela envoltória mínima).
#   Variáveis: VINCULACAO_BIAPOIADO, FATOR_COMPRIMENTO_BIAPOIADO (beta), RAZAO_MOMENTOS_EXTREMIDADE (M_B/M_A).
#   Fonte: Seção 3.6 do PP/TCC (vinculação, alfa_b em função dos momentos de extremidade); NBR 6118:2023,
#   15.6, 15.8.2 a) e 15.3.2.
#   Usada em: validar_entradas (l. 1642); solicitacao_direcao (l. 2108); PilarTeste (l. 2302)
#   Usada também em: validacao.py: nivel_2 (l. 1899); processar_matriz.py: criar_pilar (l. 386);
#       processar_matriz.py: entradas_lancaveis (l. 526); verificar_matriz.py: conferir_entradas (l. 294, 295, 322)
VINCULACAO_BIAPOIADO = "biapoiado"
FATOR_COMPRIMENTO_BIAPOIADO = 1.0
RAZAO_MOMENTOS_EXTREMIDADE = 1.0

# [H05] Área de concreto considerada no equilíbrio da seção: com DESCONTAR_AREA_BARRAS = True (adotado), a
#   resultante do concreto é a da seção b h menos a parcela de concreto deslocada pelas barras situadas na
#   zona comprimida ([F61]), isto é, a seção líquida; com False, usa-se a seção bruta b h. As hipóteses básicas
#   da NBR 6118:2023 (17.2.2, alíneas a a g) não tratam desse desconto, e as menções da norma à seção bruta
#   referem-se a propriedades geométricas (raio de giração, momento de inércia), não à resistência da seção.
#   Na omissão da norma, adota-se a seção líquida, que representa sem dupla contagem as alíneas b) e e)
#   (barras e concreto não ocupam o mesmo lugar) e é a forma de Fusco (1981, p. 226: A_cc = A_c - A_s);
#   o efeito da hipótese é medido por sensibilidade_secao_liquida.py.  Variáveis: DESCONTAR_AREA_BARRAS.
#   Fonte: NBR 6118:2023, 17.2.2 (omissa); Seção 3.5 do TCC.
#   Usada em: esforcos_resistentes (l. 1892); esforcos_resistentes_obliquos (l. 2196)
#   Usada também em: validacao.py: hipotese_secao (l. 1369, 1370, 1374); validacao.py: nivel_1 (l. 1442, 1460);
#       validacao.py: nivel_1_flexao_obliqua (l. 1528);
#       validacao.py: nivel_1_flexao_obliqua.fibras_no_parametro (l. 1655);
#       validacao.py: memorial_pilar (l. 1704, 1726, 1735);
#       validacao.py: momento_fibras_com_forca.residuo_forca (l. 2067);
#       validacao.py: momento_fibras_com_forca (l. 2074); sensibilidade_secao_liquida.py: nível do módulo (l. 70, 71);
#       sensibilidade_secao_liquida.py: definir_hipotese (l. 171);
#       sensibilidade_secao_liquida.py: esforcos_modelo (l. 201, 205);
#       validacao_literatura.py: SecaoHipotese.__init__ (l. 1150);
#       validacao_literatura.py: SecaoHipotese.__enter__ (l. 1153, 1154);
#       validacao_literatura.py: SecaoHipotese.__exit__ (l. 1157); memorial_pilares.py: estado_ultimo (l. 384);
#       memorial_pilares.py: calcular_memorial (l. 538)
DESCONTAR_AREA_BARRAS = True

# [H06] Leitura da verificação do momento mínimo com 2ª ordem (NBR 6118:2023, 15.3.2). O primeiro período de
#   15.3.2 considera a verificação atendida quando a envoltória resistente engloba a envoltória mínima com 2ª
#   ordem (elipse da Figura 15.2, semieixos M_d,tot,mín,xx e M_d,tot,mín,yy); o segundo período admite realizá-la
#   "através de duas análises à flexão composta normal, calculadas de forma isolada". O dimensionamento da rotina
#   segue o segundo período (duas análises isoladas, uma em cada direção principal, [R08] e [R09]), coerente com
#   o escopo de flexão normal composta reta; a envoltória elíptica completa é verificada à parte ([R12]), com a
#   armadura adotada, e o seu fator de atendimento é informado, sem governar a armadura. Para comparação com
#   softwares que verificam a elipse, [R12] informa também o primeiro arranjo que engloba a elipse.
#   Variáveis: LEITURA_ENVOLTORIA_MINIMA (descrição da leitura adotada, para relatórios).
#   Fonte: NBR 6118:2023, 11.3.3.4.3 (Figura 11.3) e 15.3.2 (Figura 15.2); Seção 3.5 do TCC.
#   Usada em: (sem uso neste script)
#   Usada também em: processar_matriz.py: escrever_hipoteses (l. 817); gerar_auditor.py: main (l. 453)
LEITURA_ENVOLTORIA_MINIMA = ("armadura dimensionada por duas análises isoladas de flexão composta normal "
                             "(NBR 6118:2023, 15.3.2, segundo período); envoltória elíptica (Figura 15.2) "
                             "verificada à parte, com a armadura adotada, sem governar a armadura")

# [H07] Notação das dimensões da seção. Neste trabalho, b_w é a menor dimensão da seção e h a maior; a
#   excentricidade e1 e a flexão de 1ª ordem atuam na direção de b_w (direção de menor inércia, a do lambda da
#   matriz), e na direção de h atua só o momento mínimo. Na NBR 6118:2023 (por exemplo, M1d,mín = Nd (0,015 +
#   0,03 h) em 11.3.3.4.3, lambda_1 com e1/h em 15.8.2 e 1/r = 0,005/[h (nu + 0,5)] em 15.8.3.3.2), nos ábacos
#   de flexão composta e nos exemplos de Bastos (2017, p. 83-84: pilar P5, 15 x 50 cm, momento na direção da
#   largura, hy = 15 cm), h é a altura da seção NA DIREÇÃO CONSIDERADA, isto é, a dimensão medida no plano do
#   momento, que não é necessariamente a maior. Por isso, nas fórmulas de uma direção (parâmetro "altura"), o h
#   da norma vale b_w na direção de b_w e h na direção de h. Coordenadas: x ao longo de b_w e y ao longo de h.
#   Variáveis: NOTACAO_DIMENSOES (descrição da notação, para relatórios).
#   Fonte: NBR 6118:2023, 11.3.3.4.3, 15.8.2 e 15.8.3.3.2; Quadro 1 do TCC (Tabela 3.1 do PP): seção b_w x h;
#   Bastos (2017, p. 83-84).
#   Usada em: (sem uso neste script)
#   Usada também em: processar_matriz.py: escrever_hipoteses (l. 811); gerar_auditor.py: main (l. 452)
NOTACAO_DIMENSOES = ("b_w = menor dimensão (direção de e1, da flexão de 1ª ordem e do λ da matriz); h = maior "
                     "dimensão (só momento mínimo). Nas fórmulas da NBR 6118:2023, nos ábacos e em Bastos (2017), h "
                     "é a altura da seção na direção considerada: vale b_w na direção de b_w e h na direção de h")

# ───────────────────────────────────────────────────────────── C) PARÂMETROS NUMÉRICOS (N)
# [N01] Precisão das buscas: área de aço (cm²) e parâmetro de domínio t.
#   Variáveis: TOLERANCIA_AREA, TOLERANCIA_PARAMETRO.
#   Usada em: tolerancia_area_reta_b (l. 1791); folga_tolerancia_area (l. 1805);
#       parametro_ultimo_para_forca (l. 1926); armadura_flexo_compressao (l. 1971);
#       armadura_pilar_padrao_acoplado (l. 2071)
#   Usada também em: verificar_matriz.py: nao_menor_que_area (l. 190);
#       sensibilidade_secao_liquida.py: momento_resistente_modelo (l. 232);
#       sensibilidade_secao_liquida.py: area_necessaria_modelo (l. 259);
#       conferir_busca_exaustiva.py: nível do módulo (l. 45)
TOLERANCIA_AREA = 1e-3
TOLERANCIA_PARAMETRO = 1e-12

# [N02] Discretização e iterações: pontos da curva momento-curvatura, iterações da bisseção
#   vetorizada, folga do intervalo de busca da deformação do topo e limite de kappa·h abaixo do qual a
#   curvatura é tratada como nula (deformação uniforme).
#   Variáveis: PONTOS_CURVA, ITERACOES_BISSECAO, FOLGA_INTERVALO_DEFORMACAO, CURVATURA_NULA.
#   Usada em: intervalo_deformacao_topo (l. 1775, 1776); deformacoes_topo_para_forca (l. 1985);
#       curva_momento_curvatura (l. 1998); envoltoria_resistente (l. 2223)
#   Usada também em: validacao.py: comparacao_secante (l. 1960);
#       validacao_literatura.py: curvatura_do_momento_deformacao (l. 1171)
PONTOS_CURVA = 81
ITERACOES_BISSECAO = 60
FOLGA_INTERVALO_DEFORMACAO = 0.01
CURVATURA_NULA = 1e-10

# [N03] Integração do concreto para curvaturas muito pequenas: quando |k| h < LIMITE_FORMA_FECHADA eps_c2,
#   a forma fechada de [F16] perde precisão por cancelamento numérico (diferenças de primitivas quase
#   iguais divididas por k e k²; erro relativo de 1,5e-4 em k h = 2e-7 e de 20 % em k h = 2e-8, conferido
#   contra o modelo de fibras em validacao.py) e usa-se a quadratura de Gauss-Legendre por trechos [F58],
#   com PONTOS_GAUSS pontos por trecho (exata para o diagrama com n = 2, polinômio de grau 2).
#   Variáveis: LIMITE_FORMA_FECHADA, PONTOS_GAUSS, NOS_GAUSS, PESOS_GAUSS (nós e pesos em [-1, 1]).
#   Usada em: nível do módulo (l. 320); esforcos_concreto_retangular (l. 582)
LIMITE_FORMA_FECHADA = 0.05
PONTOS_GAUSS = 8
NOS_GAUSS, PESOS_GAUSS = np.polynomial.legendre.leggauss(PONTOS_GAUSS)

# [N04] Discretização da envoltória resistente na flexão oblíqua e da verificação da envoltória mínima ([R12]):
#   ANGULOS_LINHA_NEUTRA: inclinações alfa da normal à linha neutra ([F63]) estritamente entre 0 e 90°, com passo
#   PASSO_ANGULO_LINHA_NEUTRA (os extremos 0 e 90° são os M_Rd das duas direções principais, calculados pela
#   flexão normal composta); DIRECOES_ENVOLTORIA: direções psi do vetor momento, de 0 a 90°, com passo
#   PASSO_DIRECAO_ENVOLTORIA, nas quais se comparam o raio resistente e o da elipse. Por simetria dupla da seção
#   e dos arranjos ([R02]), basta o primeiro quadrante. O refinamento (passos pela metade) altera o fator de
#   [F72] em menos de 1e-5 nos pilares testados em validacao.py (tolerância do teste: 1e-4).
#   Variáveis: PASSO_ANGULO_LINHA_NEUTRA, PASSO_DIRECAO_ENVOLTORIA (graus), ANGULOS_LINHA_NEUTRA,
#   DIRECOES_ENVOLTORIA (rad).
#   Usada em: nível do módulo (l. 338, 339); envoltoria_resistente (l. 2209);
#       verificar_envoltoria_minima (l. 2280, 2282, 2283, 2286)
#   Usada também em: validacao.py: discretizacao_angulos (l. 1485, 1486, 1490);
#       validacao.py: discretizacao_envoltoria (l. 1496, 1497, 1498, 1502);
#       validacao.py: nivel_1_flexao_obliqua (l. 1615); gerar_auditor.py: graficos_do_pilar (l. 394)
PASSO_ANGULO_LINHA_NEUTRA = 0.5
PASSO_DIRECAO_ENVOLTORIA = 0.125
ANGULOS_LINHA_NEUTRA = np.radians(np.arange(PASSO_ANGULO_LINHA_NEUTRA, 90.0, PASSO_ANGULO_LINHA_NEUTRA))
DIRECOES_ENVOLTORIA = np.radians(np.linspace(0.0, 90.0, round(90.0 / PASSO_DIRECAO_ENVOLTORIA) + 1))

# [N05] Tolerância nas fronteiras de esbeltez ([R01] e [R10]): o lambda é recalculado a partir de l_e ([F19]) em ponto
#   flutuante, e o valor exato do nível pode voltar com erro de 1 ulp (por exemplo, lambda = 130 em b_w = 30 cm volta
#   como 129,99999999999997). A tolerância absoluta de 1e-9 no lambda trata como igual ao limite um valor que difere
#   dele só por esse erro, sem efeito físico (1e-9 em lambda = 90 corresponde a 5e-9 cm em l_e = 520 cm).
#   Variáveis: TOLERANCIA_FRONTEIRA_ESBELTEZ.
#   Fórmula: |λ − λ_{limite}| ≤ 10^{−9} conta como λ = λ_{limite}
#   Usada em: metodo_segunda_ordem (l. 1454, 1456, 1458); validar_entradas (l. 1645)
TOLERANCIA_FRONTEIRA_ESBELTEZ = 1e-9


# ───────────────────────────────────────────────────────────── D) FÓRMULAS (F)
# [F01] Conversão de tensão: MPa -> kN/cm² (1 kN/cm² = 10 MPa).  Variáveis: valor_mpa.
#   Fórmula: f (kN/cm²) = \frac{f (MPa)}{10}
#   TCC: sem equação numerada no texto
#   Usada em: modulo_elasticidade_inicial (l. 548); Material.__init__ (l. 1836, 1837)
#   Usada também em: validacao.py: conferir_mn1r_independente (l. 2108); figuras_tcc.py: nível do módulo (l. 77)
def mpa_para_kn_cm2(valor_mpa: float) -> float:
    return valor_mpa / 10.0


# [F02] Conversão de momento: kN·cm -> kN·m.  Variáveis: valor_kn_cm.
#   Fórmula: M (kN·m) = \frac{M (kN·cm)}{100}
#   TCC: sem equação numerada no texto
#   Usada em: (sem uso neste script)
#   Usada também em: processar_matriz.py: linha_resultado (l. 438, 439, 442, 445, 446, 457, 471, 472, 473, 475);
#       processar_matriz.py: entradas_lancaveis (l. 530, 532, 534); processar_matriz.py: linha_inviavel (l. 742, 743);
#       verificar_matriz.py: conferir_resultados (l. 449, 451);
#       sensibilidade_secao_liquida.py: valor_em_unidade_de_saida (l. 295);
#       sensibilidade_secao_liquida.py: linha_diagramas (l. 305); figuras_tcc.py: figura_interacao (l. 666, 669);
#       figuras_tcc.py: diagrama_pilar (l. 767);
#       figuras_complementares.py: painel_mn1r (l. 492, 499, 501, 502, 505, 507, 508, 514, 517, 518);
#       figuras_complementares.py: figura_mn1r (l. 548, 563)
def kn_cm_para_kn_m(valor_kn_cm: float) -> float:
    return valor_kn_cm / 100.0


# [F03] Área da seção retangular: A_c = b h.  Variáveis: largura, altura (cm).
#   Fórmula: A_{c} = b · h
#   TCC: sem equação numerada no texto
#   Usada em: limites_busca_armadura (l. 1949); solicitacao_direcao (l. 2119); dimensionar_arranjo (l. 2401)
#   Usada também em: validacao.py: nivel_1 (l. 1382); validacao.py: memorial_pilar (l. 1693);
#       validacao.py: nivel_2 (l. 1892); validacao.py: nivel_2_bastos (l. 1992);
#       processar_matriz.py: criar_pilar (l. 379); verificar_matriz.py: conferir_entradas (l. 284, 301);
#       sensibilidade_secao_liquida.py: area_calculada_na_geometria (l. 184);
#       sensibilidade_secao_liquida.py: area_necessaria_modelo (l. 241);
#       conferir_busca_exaustiva.py: area_calculada_candidato (l. 76);
#       conferir_busca_exaustiva.py: escolha_exaustiva (l. 87); figuras_tcc.py: figura_interacao (l. 654);
#       validacao_literatura.py: executar_kimura (l. 1204); validacao_literatura.py: executar_scadelai (l. 1383);
#       validacao_literatura.py: curva_fusco (l. 1573); validacao_literatura.py: executar_fusco (l. 1696);
#       validacao_literatura.py: omega_no_ponto (l. 1955); validacao_literatura.py: omega_tangente_abaco (l. 1968);
#       validacao_literatura.py: omega_acoplado_norma (l. 1990);
#       validacao_literatura.py: executar_carvalho (l. 2031, 2126, 2201, 2367, 2499)
def area_secao_retangular(largura: float, altura: float) -> float:
    return largura * altura


# [F04] Momento de inércia da seção retangular em relação ao eixo perpendicular à direção da flexão:
#   I_c = b h³ / 12 (b e h da seção vista na direção considerada, [H07]: h na direção da flexão).
#   Variáveis: largura, altura (cm).
#   Fórmula: I_{c} = \frac{b · h^{3}}{12}
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2127)
#   Usada também em: validacao_literatura.py: executar_fusco (l. 1768);
#       validacao_literatura.py: executar_franco (l. 1861); validacao_literatura.py: executar_carvalho (l. 2402);
#       memorial_pilares.py: dados_direcao (l. 436)
def momento_inercia_retangulo(largura: float, altura: float) -> float:
    return largura * altura ** 3 / 12.0


# [F05] Resistência de cálculo do concreto à compressão: fcd = fck / gama_c.
#   Variáveis: fck (kN/cm²).  Fonte: NBR 6118:2023, 12.3.3.
#   Fórmula: f_{cd} = \frac{f_{ck}}{γ_{c}}
#   TCC: sem equação numerada no texto
#   Usada em: Material.__init__ (l. 1836)
def resistencia_calculo_concreto(fck: float) -> float:
    return fck / GAMA_C


# [F06] Resistência de cálculo do aço ao escoamento: fyd = fyk / gama_s.
#   Variáveis: fyk (kN/cm²).  Fonte: NBR 6118:2023, 12.3.1.
#   Fórmula: f_{yd} = \frac{f_{yk}}{γ_{s}}
#   TCC: sem equação numerada no texto
#   Usada em: Material.__init__ (l. 1837)
#   Usada também em: figuras_tcc.py: nível do módulo (l. 77)
def resistencia_calculo_aco(fyk: float) -> float:
    return fyk / GAMA_S


# [F07] Coeficiente eta_c: 1,0 para fck <= 40 MPa; (40 / fck)^(1/3) para fck > 40 MPa.
#   Variáveis: fck_mpa.  Fonte: NBR 6118:2023, 8.2.10.1.
#   Fórmula: η_{c} = 1,0 (f_{ck} ≤ 40 MPa);  η_{c} = (\frac{40}{f_{ck}})^{1/3} (f_{ck} > 40 MPa)
#   TCC: sem equação numerada no texto
#   Usada em: Material.__init__ (l. 1838)
def coeficiente_eta_c(fck_mpa: float) -> float:
    if fck_mpa <= 40.0:
        return 1.0
    return (40.0 / fck_mpa) ** (1.0 / 3.0)


# [F08] Parâmetros do diagrama parábola-retângulo (eps_c2, eps_cu, n).
#   fck <= 50 MPa: eps_c2 = 2,0 ‰; eps_cu = 3,5 ‰; n = 2.
#   50 < fck <= 90 MPa: eps_c2 = 2,0 ‰ + 0,085 ‰ (fck - 50)^0,53; eps_cu = 2,6 ‰ + 35 ‰ [(90 - fck)/100]^4;
#   n = 1,4 + 23,4 [(90 - fck)/100]^4.   Variáveis: fck_mpa.  Fonte: NBR 6118:2023, 8.2.10.1.
#   Fórmula: f_{ck} ≤ 50 MPa:  ε_{c2} = 2,0 ‰;  ε_{cu} = 3,5 ‰;  n = 2
#   Fórmula: 50 < f_{ck} ≤ 90 MPa:  ε_{c2} = 2,0 ‰ + 0,085 ‰ · (f_{ck} − 50)^{0,53}
#   Fórmula: ε_{cu} = 2,6 ‰ + 35 ‰ · (\frac{90 − f_{ck}}{100})^{4};  n = 1,4 + 23,4 · (\frac{90 − f_{ck}}{100})^{4}
#   TCC: sem equação numerada no texto
#   Usada em: Material.__init__ (l. 1839)
def parametros_parabola_retangulo(fck_mpa: float) -> tuple[float, float, float]:
    if fck_mpa <= 50.0:
        return 0.002, 0.0035, 2.0
    fator_alta_resistencia = ((90.0 - fck_mpa) / 100.0) ** 4
    deformacao_inicio_patamar = 0.002 + 0.000085 * (fck_mpa - 50.0) ** 0.53
    deformacao_ultima = 0.0026 + 0.035 * fator_alta_resistencia
    expoente = 1.4 + 23.4 * fator_alta_resistencia
    return deformacao_inicio_patamar, deformacao_ultima, expoente


# [F09] Tensão de pico do diagrama: sigma_pico = 0,85 eta_c fcd (curva de resistência, M_Rd) ou
#   1,10 fcd (curva de deformação do método M, N, 1/r).
#   Variáveis: fcd (kN/cm²), eta_c, curva_deformacao (bool).  Fonte: NBR 6118:2023, 8.2.10.1 e 15.3.1.
#   Fórmula: σ_{pico} = 0,85 · η_{c} · f_{cd} (resistência);  σ_{pico} = 1,10 · f_{cd} (curva de deformação)
#   TCC: sem equação numerada no texto
#   Usada em: Material.__init__ (l. 1841, 1842)
def tensao_pico_concreto(fcd: float, eta_c: float, curva_deformacao: bool) -> float:
    if curva_deformacao:
        return FATOR_CURVA_DEFORMACAO * fcd
    return ALFA_C_DIAGRAMA * eta_c * fcd


# [F10] Tensão no concreto (compressão positiva; tração desprezada), diagrama parábola-retângulo:
#   sigma_c = sigma_pico [1 - (1 - eps/eps_c2)^n] para 0 <= eps <= eps_c2; sigma_c = sigma_pico além.
#   Variáveis: deformacao, tensao_pico, eps_c2, expoente.  Fonte: NBR 6118:2023, 8.2.10.1 e 17.2.2.
#   Fórmula: σ_{c} = σ_{pico} · [1 − (1 − \frac{ε}{ε_{c2}})^{n}]  (0 ≤ ε ≤ ε_{c2})
#   Fórmula: σ_{c} = σ_{pico} (ε > ε_{c2});  σ_{c} = 0 (ε < 0, tração desprezada)
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_concreto_gauss (l. 624); esforcos_concreto_deslocado (l. 662);
#       esforcos_concreto_obliquo (l. 1326); forcas_barras_obliquo (l. 1348)
#   Usada também em: figuras_complementares.py: figura_secao (l. 352); memorial_pilares.py: estado_ultimo (l. 383)
def tensao_concreto(deformacao, tensao_pico: float, eps_c2: float, expoente: float) -> np.ndarray:
    trecho_parabola = np.clip(np.asarray(deformacao, dtype=float), 0.0, eps_c2)
    return tensao_pico * (1.0 - (1.0 - trecho_parabola / eps_c2) ** expoente)


# [F11] Primitiva P0(eps) = integral de 0 a eps de sigma_c(e) de, em forma fechada (integração exata
#   da resultante de compressão do concreto), com eps* = min(max(eps, 0), eps_c2):
#   P0 = sigma_pico {eps* + eps_c2/(n+1) [(1 - eps*/eps_c2)^(n+1) - 1] + max(eps - eps_c2, 0)}.
#   Variáveis: deformacao, tensao_pico, eps_c2, expoente.  Fonte: integração do diagrama de 8.2.10.1.
#   Fórmula: P_{0}(ε) = ∫_{0}^{ε} σ_{c}(e) de, com ε* = mín(máx(ε; 0); ε_{c2})
#   Fórmula: P_{0} = σ_{pico} · {ε* + \frac{ε_{c2}}{n + 1} · [(1 − \frac{ε*}{ε_{c2}})^{n+1} − 1] + máx(ε − ε_{c2}; 0)}
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_concreto_retangular (l. 585, 586)
def primitiva_tensao_concreto(deformacao, tensao_pico: float, eps_c2: float, expoente: float) -> np.ndarray:
    deformacao = np.asarray(deformacao, dtype=float)
    trecho_parabola = np.clip(deformacao, 0.0, eps_c2)
    complemento = 1.0 - trecho_parabola / eps_c2
    integral_parabola = trecho_parabola + eps_c2 / (expoente + 1.0) * (complemento ** (expoente + 1.0) - 1.0)
    integral_patamar = np.maximum(deformacao - eps_c2, 0.0)
    return tensao_pico * (integral_parabola + integral_patamar)


# [F12] Primitiva P1(eps) = integral de 0 a eps de sigma_c(e)·e de, em forma fechada (momento da
#   resultante do concreto), com eps* como em [F11]:
#   P1 = sigma_pico {eps*²/2 + eps_c2² [((1 - eps*/eps_c2)^(n+1) - 1)/(n+1) - ((1 - eps*/eps_c2)^(n+2) - 1)/(n+2)]
#        + (max(eps, eps_c2)² - eps_c2²)/2}.
#   Variáveis: deformacao, tensao_pico, eps_c2, expoente.  Fonte: integração do diagrama de 8.2.10.1.
#   Fórmula: P_{1}(ε) = ∫_{0}^{ε} σ_{c}(e) · e de = σ_{pico} · (A + B + C), com ε* de [F11]:
#   Fórmula: A = \frac{ε*^{2}}{2};  C = \frac{máx(ε; ε_{c2})^{2} − ε_{c2}^{2}}{2}
#   Fórmula: B = ε_{c2}^{2} · [\frac{(1 − ε*/ε_{c2})^{n+1} − 1}{n + 1} − \frac{(1 − ε*/ε_{c2})^{n+2} − 1}{n + 2}]
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_concreto_retangular (l. 587, 588)
def primitiva_momento_tensao_concreto(deformacao, tensao_pico: float, eps_c2: float,
                                      expoente: float) -> np.ndarray:
    deformacao = np.asarray(deformacao, dtype=float)
    trecho_parabola = np.clip(deformacao, 0.0, eps_c2)
    complemento = 1.0 - trecho_parabola / eps_c2
    integral_parabola = (trecho_parabola ** 2 / 2.0
                         + eps_c2 ** 2 * ((complemento ** (expoente + 1.0) - 1.0) / (expoente + 1.0)
                                          - (complemento ** (expoente + 2.0) - 1.0) / (expoente + 2.0)))
    integral_patamar = (np.maximum(deformacao, eps_c2) ** 2 - eps_c2 ** 2) / 2.0
    return tensao_pico * (integral_parabola + integral_patamar)


# [F13] Tensão na armadura passiva: diagrama bilinear, sigma_s = E_s eps_s com |sigma_s| <= fyd.
#   Variáveis: deformacao, fyd.  Fonte: NBR 6118:2023, 8.3.6.
#   Fórmula: σ_{s} = E_{s} · ε_{s}, com |σ_{s}| ≤ f_{yd}
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_armadura (l. 644); forcas_barras_obliquo (l. 1345)
#   Usada também em: memorial_pilares.py: estado_ultimo (l. 382)
def tensao_aco(deformacao, fyd: float) -> np.ndarray:
    return np.clip(MODULO_ACO * np.asarray(deformacao, dtype=float), -fyd, fyd)


# [F14] Módulo de elasticidade tangente inicial do concreto (resultado em kN/cm²):
#   E_ci = alfa_E 5600 raiz(fck) [MPa] para fck <= 50 MPa;
#   E_ci = 21,5·10³ alfa_E (fck/10 + 1,25)^(1/3) [MPa] para fck > 50 MPa.
#   Variáveis: fck_mpa, alfa_e.  Fonte: NBR 6118:2023, 8.2.8.
#   Fórmula: E_{ci} = α_{E} · 5600 · \sqrt{f_{ck}} (MPa; f_{ck} ≤ 50 MPa)
#   Fórmula: E_{ci} = 21,5 · 10^{3} · α_{E} · (\frac{f_{ck}}{10} + 1,25)^{1/3} (MPa; f_{ck} > 50 MPa)
#   TCC: sem equação numerada no texto
#   Usada em: Material.__init__ (l. 1840)
#   Usada também em: validacao_literatura.py: executar_franco (l. 1859)
def modulo_elasticidade_inicial(fck_mpa: float, alfa_e: float) -> float:
    if fck_mpa <= 50.0:
        modulo_mpa = alfa_e * 5600.0 * math.sqrt(fck_mpa)
    else:
        modulo_mpa = 21.5e3 * alfa_e * (fck_mpa / 10.0 + 1.25) ** (1.0 / 3.0)
    return mpa_para_kn_cm2(modulo_mpa)


# [F15] Hipótese das seções planas: deformação na profundidade y, medida a partir da face mais
#   comprimida: eps(y) = eps_topo - (1/r) y.  Variáveis: deformacao_topo, curvatura, profundidade.
#   Fonte: NBR 6118:2023, 17.2.2 a).
#   Fórmula: ε(y) = ε_{topo} − \frac{1}{r} · y
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_armadura (l. 642); esforcos_concreto_deslocado (l. 660); esforcos_concreto_obliquo (l. 1326);
#       forcas_barras_obliquo (l. 1344)
#   Usada também em: figuras_complementares.py: figura_secao (l. 350, 352);
#       memorial_pilares.py: estado_ultimo (l. 381)
def deformacao_na_profundidade(deformacao_topo, curvatura, profundidade) -> np.ndarray:
    return deformacao_topo - curvatura * profundidade


# [F16] Resultantes do concreto em seção retangular (força N_c e momento M_c em relação ao centro
#   da seção), por integração exata do diagrama parábola-retângulo (k = 1/r; b e h da seção vista na direção
#   considerada, [H07]; o mesmo vale para [F17] e [F18]):
#   N_c = (b/k) [P0(eps_topo) - P0(eps_base)];
#   M_c = (b/k) {(h/2 - eps_topo/k) [P0(eps_topo) - P0(eps_base)] + [P1(eps_topo) - P1(eps_base)]/k};
#   para |k| h < LIMITE_FORMA_FECHADA eps_c2 (curvatura muito pequena, inclusive k = 0), usa-se [F58] ([N03]).
#   Variáveis: deformacao_topo, curvatura (aceitam vetores), altura (na direção da flexão), largura,
#   tensao_pico, eps_c2, expoente.  Fonte: NBR 6118:2023, 17.2.2 e 8.2.10.1.
#   Fórmula: N_{c} = \frac{b}{k} · ΔP_{0},  k = \frac{1}{r},  ε_{base} = ε_{topo} − k · h
#   Fórmula: M_{c} = \frac{b}{k} · [(\frac{h}{2} − \frac{ε_{topo}}{k}) · ΔP_{0} + \frac{ΔP_{1}}{k}]
#   Fórmula: ΔP_{i} = P_{i}(ε_{topo}) − P_{i}(ε_{base})  (P_{0} de [F11], P_{1} de [F12])
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_resistentes (l. 1888)
#   Usada também em: validacao.py: nivel_1 (l. 1405); memorial_pilares.py: estado_ultimo (l. 377)
def esforcos_concreto_retangular(deformacao_topo, curvatura, altura: float, largura: float, tensao_pico: float,
                                 eps_c2: float, expoente: float) -> tuple[np.ndarray, np.ndarray]:
    deformacao_topo, curvatura = np.broadcast_arrays(np.asarray(deformacao_topo, dtype=float),
                                                     np.asarray(curvatura, dtype=float))
    pequena = np.abs(curvatura) * altura < LIMITE_FORMA_FECHADA * eps_c2
    curvatura_segura = np.where(pequena, 1.0, curvatura)
    deformacao_base = deformacao_topo - curvatura_segura * altura
    diferenca_0 = (primitiva_tensao_concreto(deformacao_topo, tensao_pico, eps_c2, expoente)
                   - primitiva_tensao_concreto(deformacao_base, tensao_pico, eps_c2, expoente))
    diferenca_1 = (primitiva_momento_tensao_concreto(deformacao_topo, tensao_pico, eps_c2, expoente)
                   - primitiva_momento_tensao_concreto(deformacao_base, tensao_pico, eps_c2, expoente))
    forca = largura / curvatura_segura * diferenca_0
    momento = largura / curvatura_segura * ((altura / 2.0 - deformacao_topo / curvatura_segura) * diferenca_0
                                            + diferenca_1 / curvatura_segura)
    if np.any(pequena):
        forca = np.array(forca, dtype=float)
        momento = np.array(momento, dtype=float)
        forca[pequena], momento[pequena] = esforcos_concreto_gauss(deformacao_topo[pequena], curvatura[pequena],
                                                                   altura, largura, tensao_pico, eps_c2, expoente)
    return forca, momento


# [F58] Resultantes do concreto por quadratura de Gauss-Legendre por trechos: a altura [0, h] é dividida
#   nas profundidades em que eps(y) = eps_topo - k y vale 0 e eps_c2 (mudanças de expressão do diagrama
#   de [F10]); em cada trecho [y_a, y_b]:  N = b soma_j w_j sigma_c(eps(y_j)) (y_b - y_a)/2  e
#   M = b soma_j w_j sigma_c(eps(y_j)) (h/2 - y_j) (y_b - y_a)/2, com y_j = (y_a + y_b)/2 + t_j (y_b - y_a)/2.
#   Variáveis: deformacao_topo, curvatura (vetores de mesmo tamanho), altura, largura, tensao_pico, eps_c2,
#   expoente; NOS_GAUSS (t_j) e PESOS_GAUSS (w_j) de [N03].  Fonte: integração numérica do diagrama de
#   8.2.10.1 (NBR 6118:2023, 17.2.2).
#   Fórmula: N = b · ∑_{j} w_{j} · σ_{c}(ε(y_{j})) · \frac{y_{b} − y_{a}}{2}
#   Fórmula: M = b · ∑_{j} w_{j} · σ_{c}(ε(y_{j})) · (\frac{h}{2} − y_{j}) · \frac{y_{b} − y_{a}}{2}
#   Fórmula: y_{j} = \frac{y_{a} + y_{b}}{2} + t_{j} · \frac{y_{b} − y_{a}}{2}
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_concreto_retangular (l. 595)
def esforcos_concreto_gauss(deformacao_topo: np.ndarray, curvatura: np.ndarray, altura: float, largura: float,
                            tensao_pico: float, eps_c2: float, expoente: float) -> tuple[np.ndarray, np.ndarray]:
    tem_curvatura = curvatura != 0.0
    curvatura_segura = np.where(tem_curvatura, curvatura, 1.0)
    corte_zero = np.where(tem_curvatura, deformacao_topo / curvatura_segura, altura)
    corte_pico = np.where(tem_curvatura, (deformacao_topo - eps_c2) / curvatura_segura, altura)
    cortes = np.sort(np.clip(np.stack([corte_zero, corte_pico], axis=-1), 0.0, altura), axis=-1)
    limites = np.concatenate([np.zeros(cortes.shape[:-1] + (1,)), cortes,
                              np.full(cortes.shape[:-1] + (1,), altura)], axis=-1)
    centro_trecho = (limites[..., 1:] + limites[..., :-1]) / 2.0
    meia_altura_trecho = (limites[..., 1:] - limites[..., :-1]) / 2.0
    profundidades = centro_trecho[..., None] + meia_altura_trecho[..., None] * NOS_GAUSS
    tensoes = tensao_concreto(deformacao_topo[..., None, None] - curvatura[..., None, None] * profundidades,
                              tensao_pico, eps_c2, expoente)
    pesos = meia_altura_trecho[..., None] * PESOS_GAUSS
    forca = largura * (tensoes * pesos).sum(axis=(-1, -2))
    momento = largura * (tensoes * pesos * (altura / 2.0 - profundidades)).sum(axis=(-1, -2))
    return forca, momento


# [F17] Resultantes das armaduras (força e momento em relação ao centro da seção):
#   N_s = soma(A_si sigma_si);  M_s = soma[A_si sigma_si (h/2 - y_i)], com eps_si dado por [F15].
#   Variáveis: deformacao_topo, curvatura (aceitam vetores), altura, profundidades (y_i), areas (A_si), fyd.
#   Fonte: NBR 6118:2023, 17.2.2 b).
#   Fórmula: N_{s} = ∑ A_{si} · σ_{si};  M_{s} = ∑ A_{si} · σ_{si} · (\frac{h}{2} − y_{i})
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_resistentes (l. 1899)
#   Usada também em: sensibilidade_secao_liquida.py: esforcos_modelo (l. 212)
def esforcos_armadura(deformacao_topo, curvatura, altura: float, profundidades: np.ndarray, areas: np.ndarray,
                      fyd: float) -> tuple[np.ndarray, np.ndarray]:
    deformacoes = deformacao_na_profundidade(np.asarray(deformacao_topo, dtype=float)[..., None],
                                             np.asarray(curvatura, dtype=float)[..., None], profundidades)
    forcas = areas * tensao_aco(deformacoes, fyd)
    return forcas.sum(axis=-1), (forcas * (altura / 2.0 - profundidades)).sum(axis=-1)


# [F61] Parcela de concreto deslocada pelas barras (correção para seção líquida, [H05]): em cada barra, a
#   tensão que o concreto teria na posição da barra multiplicada pela área da barra:
#   N_desl = soma[A_si sigma_c(eps_si)];  M_desl = soma[A_si sigma_c(eps_si) (h/2 - y_i)],
#   com eps_si de [F15] e sigma_c de [F10] (nula na tração, de modo que só as barras comprimidas contam).
#   Variáveis: deformacao_topo, curvatura (aceitam vetores), altura, profundidades (y_i), areas (A_si),
#   tensao_pico, eps_c2, expoente.  Fonte: NBR 6118:2023, 17.2.2 (equilíbrio da seção) e 8.2.10.1.
#   Fórmula: N_{desl} = ∑ A_{si} · σ_{c}(ε_{si});  M_{desl} = ∑ A_{si} · σ_{c}(ε_{si}) · (\frac{h}{2} − y_{i})
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_resistentes (l. 1893)
def esforcos_concreto_deslocado(deformacao_topo, curvatura, altura: float, profundidades: np.ndarray,
                                areas: np.ndarray, tensao_pico: float, eps_c2: float,
                                expoente: float) -> tuple[np.ndarray, np.ndarray]:
    deformacoes = deformacao_na_profundidade(np.asarray(deformacao_topo, dtype=float)[..., None],
                                             np.asarray(curvatura, dtype=float)[..., None], profundidades)
    forcas = areas * tensao_concreto(deformacoes, tensao_pico, eps_c2, expoente)
    return forcas.sum(axis=-1), (forcas * (altura / 2.0 - profundidades)).sum(axis=-1)


# [F62] Desconto de uma parcela: valor líquido = valor bruto - parcela descontada (seção líquida, [H05]).
#   Variáveis: parcela_bruta, parcela_descontada.
#   Fórmula: valor líquido = valor bruto − parcela descontada
#   TCC: sem equação numerada no texto
#   Usada em: forcas_barras_obliquo (l. 1348); esforcos_resistentes (l. 1897, 1898)
def descontar_parcela(parcela_bruta, parcela_descontada):
    return parcela_bruta - parcela_descontada


# [F59] Áreas das barras da seção a partir da área total e da fração de cada barra ([F52]): A_si = A_s f_i.
#   Variáveis: area_aco (A_s), fracoes_area (f_i).
#   Fórmula: A_{si} = A_{s} · f_{i}
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_resistentes (l. 1891); esforcos_resistentes_obliquos (l. 2192)
#   Usada também em: validacao.py: nivel_1 (l. 1430, 1432); validacao.py: nivel_1_flexao_obliqua (l. 1514, 1638);
#       validacao.py: memorial_pilar (l. 1700); validacao.py: conferir_mn1r_independente (l. 2109);
#       sensibilidade_secao_liquida.py: esforcos_modelo (l. 207)
def areas_barras_secao(area_aco: float, fracoes_area: np.ndarray) -> np.ndarray:
    return area_aco * fracoes_area


# [F60] Esforço resistente total da seção: soma das parcelas do concreto e da armadura (N = N_c + N_s;
#   M = M_c + M_s).  Variáveis: parcela_concreto, parcela_aco.  Fonte: NBR 6118:2023, 17.2.2.
#   Fórmula: N = N_{c} + N_{s};  M = M_{c} + M_{s}
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_resistentes (l. 1901); esforcos_resistentes_obliquos (l. 2198, 2199)
#   Usada também em: sensibilidade_secao_liquida.py: esforcos_modelo (l. 214)
def soma_concreto_aco(parcela_concreto, parcela_aco):
    return parcela_concreto + parcela_aco


# [F18] Configuração última de deformações (ELU) percorrendo os domínios por um parâmetro t:
#   t em [0, 1]: pivô A, eps = -10 ‰ na barra mais afastada (y = d): reta a e domínios 1 e 2;
#        eps_topo = -10 ‰ + t (10 ‰ + eps_cu); 1/r = (eps_topo + 10 ‰) / d.
#   t em [1, 2]: pivô B, eps_topo = eps_cu, x = x_23 + (t - 1)(h - x_23), x_23 = eps_cu d/(eps_cu + 10 ‰);
#        1/r = eps_cu / x: domínios 3, 4 e 4a.
#   t em [2, 3]: pivô C, eps = eps_c2 a y_C = (eps_cu - eps_c2) h / eps_cu do topo; eps_base = (t - 2) eps_c2;
#        1/r = (eps_c2 - eps_base)/(h - y_C); eps_topo = eps_c2 + (1/r) y_C: domínio 5 e reta b.
#   Retorna (eps_topo, 1/r).  Variáveis: parametro, altura, profundidade_util (d), eps_c2, eps_cu (parametro,
#   altura e d aceitam vetores; com escalares, retorna escalares). Na flexão oblíqua ([F64]), h e d são a altura
#   projetada H(alfa) e a profundidade da barra mais tracionada, medidas na direção perpendicular à linha neutra.
#   Fonte: NBR 6118:2023, 17.2.2, Figura 17.1.
#   Fórmula: t ∈ [0; 1] (pivô A):  ε_{topo} = −10 ‰ + t · (10 ‰ + ε_{cu});  \frac{1}{r} = \frac{ε_{topo} + 10 ‰}{d}
#   Fórmula: t ∈ [1; 2] (pivô B):  ε_{topo} = ε_{cu};  x = x_{23} + (t − 1) · (h − x_{23})
#   Fórmula: x_{23} = \frac{ε_{cu} · d}{ε_{cu} + 10 ‰};  \frac{1}{r} = \frac{ε_{cu}}{x}
#   Fórmula: t ∈ [2; 3] (pivô C):  y_{C} = \frac{(ε_{cu} − ε_{c2}) · h}{ε_{cu}};  ε_{base} = (t − 2) · ε_{c2}
#   Fórmula: \frac{1}{r} = \frac{ε_{c2} − ε_{base}}{h − y_{C}};  ε_{topo} = ε_{c2} + \frac{1}{r} · y_{C}
#   TCC: sem equação numerada no texto
#   Usada em: configuracao_no_parametro (l. 1905); envoltoria_resistente.configuracao (l. 2215)
#   Usada também em: validacao.py: nivel_1_flexao_obliqua (l. 1521, 1544, 1560, 1574, 1605, 1606);
#       validacao.py: nivel_1_flexao_obliqua.fibras_no_parametro (l. 1651)
def configuracao_estado_limite(parametro, altura, profundidade_util, eps_c2: float, eps_cu: float):
    escalar = np.ndim(parametro) == 0 and np.ndim(altura) == 0 and np.ndim(profundidade_util) == 0
    parametro = np.asarray(parametro, dtype=float)
    altura = np.asarray(altura, dtype=float)
    profundidade_util = np.asarray(profundidade_util, dtype=float)
    deformacao_topo_a = -DEFORMACAO_ULTIMA_ACO + np.clip(parametro, 0.0, 1.0) * (DEFORMACAO_ULTIMA_ACO + eps_cu)
    curvatura_a = (deformacao_topo_a + DEFORMACAO_ULTIMA_ACO) / profundidade_util
    linha_neutra_23 = eps_cu * profundidade_util / (eps_cu + DEFORMACAO_ULTIMA_ACO)
    linha_neutra = linha_neutra_23 + (np.clip(parametro, 1.0, 2.0) - 1.0) * (altura - linha_neutra_23)
    curvatura_b = eps_cu / linha_neutra
    profundidade_pivo_c = (eps_cu - eps_c2) * altura / eps_cu
    deformacao_base = (np.clip(parametro, 2.0, 3.0) - 2.0) * eps_c2
    curvatura_c = (eps_c2 - deformacao_base) / (altura - profundidade_pivo_c)
    deformacao_topo_c = eps_c2 + curvatura_c * profundidade_pivo_c
    deformacao_topo = np.where(parametro <= 1.0, deformacao_topo_a,
                               np.where(parametro <= 2.0, eps_cu, deformacao_topo_c))
    curvatura = np.where(parametro <= 1.0, curvatura_a, np.where(parametro <= 2.0, curvatura_b, curvatura_c))
    if escalar:
        return float(deformacao_topo), float(curvatura)
    return deformacao_topo, curvatura


# [F19] Índice de esbeltez de seção retangular: lambda = l_e / i, com i = h / raiz(12) (h da norma: altura na
#   direção considerada, [H07]; b_w na direção de b_w).
#   Variáveis: comprimento_equivalente (cm), altura (cm, na direção considerada).  Fonte: 15.8.2.
#   Fórmula: λ = \frac{ℓ_{e}}{i},  i = \frac{h}{\sqrt{12}}  ⇒  λ = \frac{\sqrt{12} · ℓ_{e}}{h}
#   TCC: sem equação numerada no texto
#   Usada em: validar_entradas (l. 1645); solicitacao_direcao (l. 2104); dimensionar_arranjo (l. 2407, 2408)
#   Usada também em: verificar_matriz.py: conferir_entradas (l. 276, 306);
#       validacao_literatura.py: executar_kimura (l. 1207); validacao_literatura.py: executar_scadelai (l. 1388);
#       validacao_literatura.py: executar_carvalho (l. 2032, 2135, 2213, 2214, 2369, 2502, 2503, 2506, 2507)
def indice_esbeltez(comprimento_equivalente: float, altura: float) -> float:
    return comprimento_equivalente * math.sqrt(12.0) / altura


# [F20] Comprimento equivalente correspondente a um índice de esbeltez (inversa de [F19]):
#   l_e = lambda h / raiz(12), h na direção do lambda ([H07]).  Variáveis: esbeltez, altura (cm).  Fonte: 15.8.2.
#   Fórmula: ℓ_{e} = \frac{λ · h}{\sqrt{12}}
#   TCC: sem equação numerada no texto
#   Usada em: (sem uso neste script)
#   Usada também em: processar_matriz.py: criar_pilar (l. 385)
def comprimento_para_esbeltez(esbeltez: float, altura: float) -> float:
    return esbeltez * altura / math.sqrt(12.0)


# [F21] Momento mínimo de primeira ordem: M1d,min = Nd (0,015 + 0,03 h), h em metros; com Nd em kN e
#   h em cm, M1d,min [kN·cm] = Nd (1,5 + 0,03 h); h da norma = altura na direção considerada ([H07]).
#   Variáveis: forca_normal_calculo, altura (cm, na direção considerada).  Fonte: 11.3.3.4.3.
#   Fórmula: M_{1d,mín} = N_{d} · (0,015 + 0,03 · h)  (h em m)
#   Fórmula: M_{1d,mín} (kN·cm) = N_{d} · (1,5 + 0,03 · h)  (N_{d} em kN, h em cm)
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2105)
#   Usada também em: validacao.py: nivel_2 (l. 1896); verificar_matriz.py: conferir_entradas (l. 324);
#       validacao_literatura.py: executar_kimura (l. 1205); validacao_literatura.py: executar_scadelai (l. 1390);
#       validacao_literatura.py: executar_carvalho (l. 2064, 2138, 2235, 2236, 2348, 2386, 2535, 2536)
def momento_minimo_primeira_ordem(forca_normal_calculo: float, altura: float) -> float:
    return forca_normal_calculo * (1.5 + 0.03 * altura)


# [F22] Excentricidade relativa de primeira ordem: e1/h = M1d,A / (Nd h), h na direção considerada ([H07]; na
#   direção de b_w, e1/b_w).  Variáveis: momento_primeira_ordem, forca_normal_calculo, altura.  Fonte: 15.8.2.
#   Fórmula: \frac{e_{1}}{h} = \frac{M_{1d,A}}{N_{d} · h}
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2110)
#   Usada também em: validacao_literatura.py: executar_kimura (l. 1212);
#       validacao_literatura.py: executar_scadelai (l. 1393); validacao_literatura.py: executar_carvalho (l. 2516);
#       memorial_pilares.py: dados_direcao (l. 419)
def excentricidade_relativa(momento_primeira_ordem: float, forca_normal_calculo: float, altura: float) -> float:
    return momento_primeira_ordem / (forca_normal_calculo * altura)


# [F23] Coeficiente alfa_b de pilar biapoiado sem cargas transversais:
#   alfa_b = 0,60 + 0,40 M_B/M_A, com 0,40 <= alfa_b <= 1,0; alfa_b = 1,0 quando os momentos de 1ª ordem
#   são menores que o momento mínimo.  Variáveis: momento_a, momento_b (com sinal), momentos_menores_que_minimo.
#   Fonte: NBR 6118:2023, 15.8.2 a) e d).
#   Fórmula: α_{b} = 0,60 + 0,40 · \frac{M_{B}}{M_{A}},  com 0,40 ≤ α_{b} ≤ 1,0;  α_{b} = 1,0 se M_{1d} < M_{1d,mín}
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2108)
#   Usada também em: validacao.py: nivel_2 (l. 1898); validacao_literatura.py: executar_kimura (l. 1208);
#       validacao_literatura.py: executar_scadelai (l. 1392)
def coeficiente_alfa_b(momento_a: float, momento_b: float, momentos_menores_que_minimo: bool) -> float:
    if momentos_menores_que_minimo or momento_a == 0.0:
        return 1.0
    return min(1.0, max(0.40, 0.60 + 0.40 * momento_b / momento_a))


# [F24] Esbeltez-limite: lambda_1 = (25 + 12,5 e1/h) / alfa_b, com 35 <= lambda_1 <= 90.
#   Variáveis: excentricidade_relativa_1 (e1/h, h na direção considerada, [H07]), alfa_b.
#   Fonte: NBR 6118:2023, 15.8.2.
#   Fórmula: λ_{1} = \frac{25 + 12,5 · e_{1}/h}{α_{b}},  com 35 ≤ λ_{1} ≤ 90
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2110)
#   Usada também em: validacao_literatura.py: executar_scadelai (l. 1393, 1394);
#       validacao_literatura.py: executar_carvalho (l. 2516)
def esbeltez_limite(excentricidade_relativa_1: float, alfa_b: float) -> float:
    return min(LAMBDA1_MAX, max(LAMBDA1_MIN, (25.0 + 12.5 * excentricidade_relativa_1) / alfa_b))


# [F25] Força normal adimensional: nu = Nd / (Ac fcd).
#   Variáveis: forca_normal_calculo, area_concreto, fcd.  Fonte: NBR 6118:2023, 15.8.3.3.2.
#   Fórmula: ν = \frac{N_{d}}{A_{c} · f_{cd}}
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2119); dimensionar_arranjo (l. 2409)
#   Usada também em: validacao.py: nivel_2 (l. 1901); validacao.py: nivel_2_bastos (l. 2003);
#       verificar_matriz.py: conferir_entradas (l. 283); validacao_literatura.py: executar_kimura (l. 1214);
#       validacao_literatura.py: executar_scadelai (l. 1386); validacao_literatura.py: executar_fusco (l. 1699, 1700);
#       validacao_literatura.py: executar_carvalho (l. 2075, 2143, 2244, 2429, 2558)
def forca_normal_adimensional(forca_normal_calculo: float, area_concreto: float, fcd: float) -> float:
    return forca_normal_calculo / (area_concreto * fcd)


# [F26] Força normal de cálculo a partir de nu (inversa de [F25]): Nd = nu Ac fcd.
#   Variáveis: nu, area_concreto, fcd.  Fonte: 15.8.3.3.2.
#   Fórmula: N_{d} = ν · A_{c} · f_{cd}
#   TCC: sem equação numerada no texto
#   Usada em: (sem uso neste script)
#   Usada também em: validacao.py: nivel_2_bastos (l. 2031); processar_matriz.py: criar_pilar (l. 383);
#       validacao_literatura.py: executar_scadelai (l. 1448); validacao_literatura.py: curva_fusco (l. 1577);
#       validacao_literatura.py: omega_no_ponto (l. 1957); validacao_literatura.py: omega_tangente_abaco (l. 1970);
#       validacao_literatura.py: omega_acoplado_norma (l. 1992)
def forca_normal_de_nu(nu: float, area_concreto: float, fcd: float) -> float:
    return nu * area_concreto * fcd


# [F27] Curvatura aproximada da seção crítica: 1/r = 0,005 / [h (nu + 0,5)] <= 0,005 / h.
#   Variáveis: nu, altura (cm, h da norma na direção considerada, [H07]; resultado em 1/cm).  Fonte: 15.8.3.3.2.
#   Fórmula: \frac{1}{r} = \frac{0,005}{h · (ν + 0,5)} ≤ \frac{0,005}{h}
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2120)
#   Usada também em: validacao.py: nivel_2 (l. 1901); validacao_literatura.py: executar_kimura (l. 1237, 1240);
#       validacao_literatura.py: executar_scadelai (l. 1412, 1413);
#       validacao_literatura.py: executar_carvalho (l. 2145, 2146, 2246, 2247);
#       memorial_pilares.py: dados_direcao (l. 427)
def curvatura_aproximada(nu: float, altura: float) -> float:
    return min(0.005 / (altura * (nu + 0.5)), 0.005 / altura)


# [F28] Momento total do pilar-padrão: M_d,tot = alfa_b M1d,A + Nd (l_e²/10) (1/r) >= M1d,A.
#   Variáveis: alfa_b, momento_primeira_ordem (M1d,A), forca_normal_calculo, comprimento_equivalente, curvatura.
#   Fonte: NBR 6118:2023, 15.8.3.3.2 (curvatura de [F27]) e 15.8.3.3.4 (curvatura do diagrama M, N, 1/r).
#   Fórmula: M_{d,tot} = α_{b} · M_{1d,A} + N_{d} · \frac{ℓ_{e}^{2}}{10} · \frac{1}{r} ≥ M_{1d,A}
#   TCC: Eq. (1)
#   Usada em: equilibrio_pilar_padrao_acoplado (l. 2029, 2039); solicitacao_direcao (l. 2121)
#   Usada também em: validacao.py: nivel_2 (l. 1906); validacao_literatura.py: executar_kimura (l. 1238);
#       validacao_literatura.py: executar_scadelai (l. 1419, 1422)
def momento_total_pilar_padrao(alfa_b: float, momento_primeira_ordem: float, forca_normal_calculo: float,
                               comprimento_equivalente: float, curvatura: float) -> float:
    momento = (alfa_b * momento_primeira_ordem
               + forca_normal_calculo * comprimento_equivalente ** 2 / DIVISOR_PILAR_PADRAO * curvatura)
    return max(momento, momento_primeira_ordem)


# [F29] Força normal da curva de deformação do método M, N, 1/r: N = Nd / gama_f3.
#   Variáveis: forca_normal_calculo.  Fonte: NBR 6118:2023, 15.3.1 (Figura 15.1).
#   Fórmula: N = \frac{N_{d}}{γ_{f3}}
#   TCC: sem equação numerada no texto
#   Usada em: equilibrio_pilar_padrao_acoplado (l. 2018)
#   Usada também em: validacao.py: metodo_geral (l. 1778); validacao.py: comparacao_secante (l. 1956);
#       gerar_auditor.py: curva_mn1r (l. 359); figuras_complementares.py: curva_do_pilar (l. 468);
#       validacao_literatura.py: metodo_geral_constante (l. 1177); validacao_literatura.py: executar_kimura (l. 1312);
#       memorial_pilares.py: dados_direcao (l. 439)
def forca_normal_curva_deformacao(forca_normal_calculo: float) -> float:
    return forca_normal_calculo / GAMA_F3


# [F30] Resíduo da condição de equilíbrio do pilar-padrão acoplado a diagramas M, N, 1/r, na
#   formulação de segurança com gama_f3:  R(1/r) = gama_f3 M_def(1/r) - [alfa_b M1d,A + Nd (l_e²/10)(1/r)],
#   sendo M_def o momento da curva de deformação ([F09] com 1,10 fcd e N de [F29]). Equilíbrio: R = 0.
#   Variáveis: momento_curva_deformacao, alfa_b, momento_primeira_ordem, forca_normal_calculo,
#   comprimento_equivalente, curvatura (aceitam vetores).  Fonte: NBR 6118:2023, 15.3.1 e 15.8.3.3.4.
#   Fórmula: R(1/r) = γ_{f3} · M_{def}(1/r) − [α_{b} · M_{1d,A} + N_{d} · \frac{ℓ_{e}^{2}}{10} · \frac{1}{r}]
#   Fórmula: equilíbrio: R(1/r) = 0
#   TCC: sem equação numerada no texto
#   Usada em: equilibrio_pilar_padrao_acoplado (l. 2022);
#       equilibrio_pilar_padrao_acoplado.residuo_equilibrio (l. 2034)
#   Usada também em: figuras_complementares.py: curva_do_pilar (l. 471); memorial_pilares.py: dados_direcao (l. 460)
def residuo_equilibrio_pilar_padrao(momento_curva_deformacao, alfa_b: float, momento_primeira_ordem: float,
                                    forca_normal_calculo: float, comprimento_equivalente: float, curvatura):
    momento_externo = (alfa_b * momento_primeira_ordem
                       + forca_normal_calculo * comprimento_equivalente ** 2 / DIVISOR_PILAR_PADRAO
                       * np.asarray(curvatura, dtype=float))
    return GAMA_F3 * np.asarray(momento_curva_deformacao, dtype=float) - momento_externo


# [F31] Desaprumo local: theta_1 = 1 / (100 raiz(l)), l em metros, com 1/300 <= theta_1 <= 1/200.
#   Variáveis: comprimento (cm).  Fonte: NBR 6118:2023, 11.3.3.4.1 e 11.3.3.4.2.
#   Fórmula: θ_{1} = \frac{1}{100 · \sqrt{ℓ}} (ℓ em m),  com \frac{1}{300} ≤ θ_{1} ≤ \frac{1}{200}
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2129)
#   Usada também em: verificar_matriz.py: conferir_entradas (l. 323);
#       validacao_literatura.py: executar_franco (l. 1873, 1945);
#       validacao_literatura.py: executar_carvalho (l. 2054, 2227, 2375, 2524, 2532);
#       memorial_pilares.py: dados_direcao (l. 432)
def desaprumo_local(comprimento: float) -> float:
    return min(THETA1_MAX, max(THETA1_MIN, 1.0 / (100.0 * math.sqrt(comprimento / 100.0))))


# [F32] Excentricidade por falta de retilineidade do eixo do pilar: e_a = theta_1 l / 2.
#   Variáveis: desaprumo (theta_1), comprimento (cm).  Fonte: NBR 6118:2023, 11.3.3.4.2, Figura 11.2.
#   Fórmula: e_{a} = \frac{θ_{1} · ℓ}{2}
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2129)
#   Usada também em: verificar_matriz.py: conferir_entradas (l. 323);
#       validacao_literatura.py: executar_franco (l. 1873);
#       validacao_literatura.py: executar_carvalho (l. 2054, 2227, 2375, 2532)
def excentricidade_acidental(desaprumo: float, comprimento: float) -> float:
    return desaprumo * comprimento / 2.0


# [F33] Força de flambagem para a fluência: N_e = 10 E_ci I_c / l_e².
#   Variáveis: modulo_inicial (kN/cm²), inercia (cm⁴), comprimento_equivalente (cm).  Fonte: 15.8.4.
#   Fórmula: N_{e} = \frac{10 · E_{ci} · I_{c}}{ℓ_{e}^{2}}
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2127)
#   Usada também em: validacao_literatura.py: executar_franco (l. 1862);
#       validacao_literatura.py: executar_carvalho (l. 2403)
def forca_euler_fluencia(modulo_inicial: float, inercia: float, comprimento_equivalente: float) -> float:
    return 10.0 * modulo_inicial * inercia / comprimento_equivalente ** 2


# [F34] Excentricidade adicional de fluência: e_cc = (M_sg/N_sg + e_a) [e^(phi N_sg / (N_e - N_sg)) - 1],
#   somada à excentricidade de 1ª ordem como se fosse imediata (infinita se N_sg >= N_e).
#   Variáveis: momento_quase_permanente (M_sg), forca_quase_permanente_kn (N_sg), excentricidade_acidental_cm
#   (e_a), coeficiente_fluencia (phi), forca_euler (N_e).  Fonte: NBR 6118:2023, 15.8.4.
#   Fórmula: e_{cc} = (\frac{M_{sg}}{N_{sg}} + e_{a}) · [e^{φ · N_{sg}/(N_{e} − N_{sg})} − 1]
#   Fórmula: e_{cc} infinita se N_{sg} ≥ N_{e}
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2130)
#   Usada também em: validacao_literatura.py: executar_fusco (l. 1786);
#       validacao_literatura.py: executar_franco (l. 1887); validacao_literatura.py: executar_carvalho (l. 2415, 2423)
def excentricidade_fluencia(momento_quase_permanente: float, forca_quase_permanente_kn: float,
                            excentricidade_acidental_cm: float, coeficiente_fluencia: float,
                            forca_euler: float) -> float:
    if forca_quase_permanente_kn >= forca_euler:
        return math.inf
    expoente = coeficiente_fluencia * forca_quase_permanente_kn / (forca_euler - forca_quase_permanente_kn)
    return ((momento_quase_permanente / forca_quase_permanente_kn + excentricidade_acidental_cm)
            * (math.exp(expoente) - 1.0))


# [F35] Força normal característica e parcelas permanente e variável (dados de entrada dos softwares):
#   N_k = Nd / gama_f;  N_gk = f_g N_k;  N_qk = (1 - f_g) N_k.
#   Variáveis: forca_normal_calculo, fracao_permanente.  Fonte: NBR 6118:2023, 11.7 e Tabela 11.1.
#   Fórmula: N_{k} = \frac{N_{d}}{γ_{f}};  N_{gk} = f_{g} · N_{k};  N_{qk} = (1 − f_{g}) · N_{k}
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2124)
#   Usada também em: processar_matriz.py: linha_resultado (l. 427);
#       verificar_matriz.py: conferir_resultados (l. 442, 534); memorial_pilares.py: dados_direcao (l. 430);
#       memorial_pilares.py: calcular_memorial (l. 524)
def parcelas_forca_caracteristica(forca_normal_calculo: float, fracao_permanente: float) -> tuple[float, float, float]:
    forca_caracteristica = forca_normal_calculo / GAMA_F
    return (forca_caracteristica, fracao_permanente * forca_caracteristica,
            (1.0 - fracao_permanente) * forca_caracteristica)


# [F36] Força normal da combinação quase permanente: N_sg = N_gk + psi_2 N_qk.
#   Variáveis: forca_permanente (N_gk), forca_variavel (N_qk), psi2.  Fonte: NBR 6118:2023, 11.8.3.2.
#   Fórmula: N_{sg} = N_{gk} + ψ_{2} · N_{qk}
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2125)
def forca_quase_permanente(forca_permanente: float, forca_variavel: float, psi2: float) -> float:
    return forca_permanente + psi2 * forca_variavel


# [F53] Momento de 1ª ordem aplicado pela excentricidade da força normal: M = N e1
#   (com Nd: M1d,A = M1d,B = Nd e1, momentos iguais no topo e na base — [H04]; com N_sg: M_sg = N_sg e1,
#   pois o momento de 1ª ordem decorre da própria força normal excêntrica em qualquer combinação).
#   Variáveis: forca_normal (kN), excentricidade_primeira_ordem (e1, cm).  Fonte: NBR 6118:2023, 15.8.2 e
#   15.8.4 (M_sg da combinação quase permanente); Seção 3.6 do TCC.
#   Fórmula: M = N · e_{1}  (M_{1d,A} = M_{1d,B} = N_{d} · e_{1};  M_{sg} = N_{sg} · e_{1})
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2106, 2126)
#   Usada também em: validacao.py: nivel_2 (l. 1895); processar_matriz.py: entradas_lancaveis (l. 530, 532, 534);
#       verificar_matriz.py: conferir_entradas (l. 325); verificar_matriz.py: conferir_resultados (l. 450, 452);
#       validacao_literatura.py: executar_kimura (l. 1208, 1210, 1211)
def momento_excentricidade(forca_normal: float, excentricidade_primeira_ordem: float) -> float:
    return forca_normal * excentricidade_primeira_ordem


# [F54] Momento de 1ª ordem na extremidade B a partir do da extremidade A e da razão M_B/M_A ([H04]):
#   M_B = (M_B/M_A) M_A (sinal positivo = curvatura simples).
#   Variáveis: momento_a, razao_momentos.  Fonte: NBR 6118:2023, 15.8.2 a).
#   Fórmula: M_{B} = \frac{M_{B}}{M_{A}} · M_{A}  (razão de [H04] = +1, curvatura simples)
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2108)
#   Usada também em: validacao.py: nivel_2 (l. 1899)
def momento_extremidade_b(momento_a: float, razao_momentos: float) -> float:
    return razao_momentos * momento_a


# [F55] Momento de 1ª ordem de cálculo adotado: M1d,A = max(Nd e1 ; M1d,min) — o momento mínimo
#   substitui o efeito das imperfeições locais e é o piso do momento de 1ª ordem.
#   Variáveis: momento_aplicado (Nd e1, [F53]), momento_minimo ([F21]).  Fonte: NBR 6118:2023, 11.3.3.4.3.
#   Fórmula: M_{1d,A} = máx(N_{d} · e_{1}; M_{1d,mín})
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2107)
#   Usada também em: validacao.py: nivel_2 (l. 1897); memorial_pilares.py: dados_direcao (l. 421)
def momento_primeira_ordem_adotado(momento_aplicado: float, momento_minimo: float) -> float:
    return max(momento_aplicado, momento_minimo)


# [F56] Momento de 1ª ordem acrescido do efeito da fluência (90 < lambda <= 140):
#   M1d,A,fluência = M1d,A + Nd e_cc  (e_cc de [F34] tratado como efeito imediato somado ao de 1ª ordem).
#   Variáveis: momento_primeira_ordem (M1d,A de [F55]), forca_normal_calculo, excentricidade_fluencia_cm (e_cc).
#   Fonte: NBR 6118:2023, 15.8.4 ("como se fosse um efeito imediato, que se soma à excentricidade e1");
#   Scadelai e Pinheiro (2007, item 16.6: "acrescentando-se ao momento de 1ª ordem M1d a parcela
#   relativa à excentricidade suplementar").
#   Fórmula: M_{1d,A,fluência} = M_{1d,A} + N_{d} · e_{cc}
#   TCC: sem equação numerada no texto
#   Usada em: solicitacao_direcao (l. 2132)
#   Usada também em: validacao_literatura.py: executar_franco (l. 1919)
def momento_primeira_ordem_com_fluencia(momento_primeira_ordem: float, forca_normal_calculo: float,
                                        excentricidade_fluencia_cm: float) -> float:
    return momento_primeira_ordem + forca_normal_calculo * excentricidade_fluencia_cm


# [F57] Comprimento real do pilar a partir do comprimento equivalente: l = l_e / beta (beta = 1 no pilar
#   biapoiado, [H04]); é o "pé-direito" a lançar nos softwares.
#   Variáveis: comprimento_equivalente (cm), fator_comprimento (beta).  Fonte: NBR 6118:2023, 15.6; Seção 3.6 do TCC.
#   Fórmula: ℓ = \frac{ℓ_{e}}{β}
#   TCC: sem equação numerada no texto
#   Usada em: (sem uso neste script)
#   Usada também em: processar_matriz.py: entradas_lancaveis (l. 526); verificar_matriz.py: conferir_entradas (l. 322)
def comprimento_real(comprimento_equivalente: float, fator_comprimento: float) -> float:
    return comprimento_equivalente / fator_comprimento


# [F37] Armadura longitudinal mínima de pilares: As,min = 0,15 Nd / fyd >= 0,004 Ac.
#   Variáveis: forca_normal_calculo, fyd, area_concreto.  Fonte: NBR 6118:2023, 17.3.5.3.1.
#   Fórmula: A_{s,mín} = máx(\frac{0,15 · N_{d}}{f_{yd}}; 0,004 · A_{c})
#   TCC: sem equação numerada no texto
#   Usada em: dimensionar_arranjo (l. 2405)
#   Usada também em: sensibilidade_secao_liquida.py: area_calculada_na_geometria (l. 193);
#       conferir_busca_exaustiva.py: area_calculada_candidato (l. 75);
#       validacao_literatura.py: executar_kimura (l. 1215);
#       validacao_literatura.py: executar_carvalho (l. 2092, 2185, 2320)
def armadura_minima(forca_normal_calculo: float, fyd: float, area_concreto: float) -> float:
    return max(FATOR_ARMADURA_MINIMA_FORCA * forca_normal_calculo / fyd, TAXA_MINIMA_GEOMETRICA * area_concreto)


# [F38] Armadura longitudinal máxima fora da região de emenda: As,max = taxa Ac, com taxa = TAXA_MAXIMA_FORA_EMENDA
#   (0,04, adotada: 8 % na região de emenda com traspasse de todas as barras na mesma seção) ou, só na análise de
#   sensibilidade dos pilares inviáveis, TAXA_MAXIMA_COM_EMENDA (0,08, limite sem sobreposição de barras, por exemplo
#   com emendas por luvas, mantidas as demais regras de detalhamento).
#   Variáveis: area_concreto, taxa_maxima.  Fonte: NBR 6118:2023, 17.3.5.3.2; Bastos (2017, p. 81).
#   Fórmula: A_{s,máx} = ρ_{máx} · A_{c}  (ρ_{máx} = 0,04 adotado; 0,08 só na sensibilidade)
#   TCC: sem equação numerada no texto
#   Usada em: dimensionar_arranjo (l. 2406)
#   Usada também em: conferir_busca_exaustiva.py: escolha_exaustiva (l. 87)
def armadura_maxima(area_concreto: float, taxa_maxima: float) -> float:
    return taxa_maxima * area_concreto


# [F39] Limite superior da busca de armadura: 8 % de Ac (acima disso a seção é insuficiente).
#   Variáveis: area_concreto.  Fonte: NBR 6118:2023, 17.3.5.3.2.
#   Fórmula: A_{s,busca} = 0,08 · A_{c}
#   TCC: sem equação numerada no texto
#   Usada em: limites_busca_armadura (l. 1949)
#   Usada também em: validacao.py: area_necessaria_metodo_geral (l. 1819);
#       sensibilidade_secao_liquida.py: area_necessaria_modelo (l. 241)
def armadura_limite_busca(area_concreto: float) -> float:
    return TAXA_MAXIMA_COM_EMENDA * area_concreto


# [F40] Área de um conjunto de n barras iguais: A_s = n pi phi² / 4.
#   Variáveis: diametro (cm), numero_barras.
#   Fórmula: A_{s} = n · \frac{π · φ^{2}}{4}
#   TCC: sem equação numerada no texto
#   Usada em: montar_candidato (l. 2326)
#   Usada também em: validacao_literatura.py: executar_kimura (l. 1304);
#       validacao_literatura.py: executar_scadelai (l. 1518);
#       validacao_literatura.py: executar_carvalho (l. 2095, 2312, 2472)
def area_barras(diametro: float, numero_barras: int) -> float:
    return numero_barras * math.pi * diametro ** 2 / 4.0


# [F41] Taxa geométrica de armadura: rho = As / Ac.  Variáveis: area_aco, area_concreto.
#   Fonte: NBR 6118:2023, 17.3.5.3.
#   Fórmula: ρ = \frac{A_{s}}{A_{c}}
#   TCC: sem equação numerada no texto
#   Usada em: (sem uso neste script)
#   Usada também em: processar_matriz.py: linha_resultado (l. 470)
def taxa_geometrica(area_aco: float, area_concreto: float) -> float:
    return area_aco / area_concreto


# [F42] Bitola longitudinal permitida: 10 mm <= phi_l <= (menor dimensão) / 8.
#   Variáveis: diametro, menor_dimensao (cm).  Fonte: NBR 6118:2023, 18.4.2.1.
#   Fórmula: 10 mm ≤ φ_{ℓ} ≤ \frac{b_{mín}}{8}  (b_{mín} = menor dimensão da seção)
#   TCC: sem equação numerada no texto
#   Usada em: gerar_candidatos (l. 2335); candidato_referencia (l. 2355)
#   Usada também em: validacao_literatura.py: executar_scadelai (l. 1542, 1544);
#       validacao_literatura.py: executar_carvalho (l. 2336); memorial_pilares.py: dados_detalhamento (l. 503)
def diametro_longitudinal_permitido(diametro: float, menor_dimensao: float) -> bool:
    return DIAMETRO_MINIMO_LONGITUDINAL <= diametro <= FRACAO_DIAMETRO_MAXIMO * menor_dimensao + 1e-9


# [F43] Diâmetro do estribo: menor bitola comercial >= max(5 mm, phi_l / 4).
#   Variáveis: diametro_longitudinal (cm).  Fonte: NBR 6118:2023, 18.4.3.
#   Fórmula: φ_{t} = menor bitola comercial ≥ máx(5 mm; \frac{φ_{ℓ}}{4})
#   TCC: sem equação numerada no texto
#   Usada em: montar_candidato (l. 2322)
#   Usada também em: validacao_literatura.py: executar_scadelai (l. 1546);
#       validacao_literatura.py: executar_carvalho (l. 2288)
def diametro_estribo(diametro_longitudinal: float) -> float:
    exigido = max(DIAMETRO_MINIMO_ESTRIBO, FRACAO_DIAMETRO_ESTRIBO * diametro_longitudinal)
    return next(bitola for bitola in BITOLAS_ESTRIBO if bitola >= exigido - 1e-9)


# [F44] Espaçamento livre mínimo entre faces de barras: a_min = max(20 mm, phi_l, 1,2 d_max).
#   Variáveis: diametro_longitudinal (cm), diametro_maximo_agregado (cm).  Fonte: 18.4.2.2.
#   Fórmula: a_{mín} = máx(20 mm; φ_{ℓ}; 1,2 · d_{máx})
#   TCC: sem equação numerada no texto
#   Usada em: gerar_candidatos (l. 2337)
#   Usada também em: memorial_pilares.py: dados_detalhamento (l. 485)
def espacamento_livre_minimo(diametro_longitudinal: float, diametro_maximo_agregado: float) -> float:
    return max(ESPACAMENTO_LIVRE_MINIMO_ABSOLUTO, diametro_longitudinal,
               FATOR_AGREGADO_ESPACAMENTO * diametro_maximo_agregado)


# [F45] Espaçamento máximo entre eixos de barras: s_max = min(2 x menor dimensão, 400 mm).
#   Variáveis: menor_dimensao (cm).  Fonte: NBR 6118:2023, 18.4.2.2.
#   Fórmula: s_{máx} = mín(2 · b_{mín}; 400 mm)
#   TCC: sem equação numerada no texto
#   Usada em: montar_candidato (l. 2325); gerar_candidatos (l. 2332)
#   Usada também em: validacao.py: nivel_1_flexao_obliqua (l. 1511);
#       validacao_literatura.py: executar_scadelai (l. 1561); memorial_pilares.py: dados_detalhamento (l. 470)
def espacamento_maximo_eixos(menor_dimensao: float) -> float:
    return min(FATOR_ESPACAMENTO_MAXIMO * menor_dimensao, ESPACAMENTO_MAXIMO_ABSOLUTO)


# [F46] Distância da face ao eixo das barras longitudinais de canto: d' = c + phi_t + phi_l / 2.
#   Variáveis: cobrimento, diametro_estribo_cm, diametro_longitudinal (cm).  Fonte: NBR 6118:2023, 7.4.7.
#   Fórmula: d' = c + φ_{t} + \frac{φ_{ℓ}}{2}
#   TCC: sem equação numerada no texto
#   Usada em: montar_candidato (l. 2323)
#   Usada também em: validacao_literatura.py: executar_kimura (l. 1302);
#       validacao_literatura.py: executar_carvalho (l. 2023, 2288, 2368)
def distancia_eixo_barra(cobrimento: float, diametro_estribo_cm: float, diametro_longitudinal: float) -> float:
    return cobrimento + diametro_estribo_cm + diametro_longitudinal / 2.0


# [F47] Espaçamento entre eixos de n barras igualmente distribuídas num trecho de face de
#   comprimento L (entre os eixos das barras extremas): s = L / (n - 1).
#   Variáveis: comprimento_face, numero_barras_face.
#   Fórmula: s = \frac{L}{n − 1}
#   TCC: sem equação numerada no texto
#   Usada em: posicoes_barras (l. 1507); espacamentos_atendem (l. 1525); barras_sem_protecao_face (l. 1700)
#   Usada também em: validacao_literatura.py: executar_scadelai (l. 1559);
#       memorial_pilares.py: dados_detalhamento (l. 488)
def espacamento_eixos_face(comprimento_face: float, numero_barras_face: int) -> float:
    return comprimento_face / (numero_barras_face - 1)


# [F48] Espaçamento livre na região de emenda por traspasse de todas as barras na mesma seção
#   (duas barras justapostas em cada posição): a = s - 2 phi_l.
#   Variáveis: espacamento_eixos, diametro_longitudinal.  Fonte: NBR 6118:2023, 18.4.2.2 (os valores
#   mínimos aplicam-se também às regiões de emenda) e hipótese do trabalho (Seção 3.6 do TCC).
#   Fórmula: a = s − 2 · φ_{ℓ}
#   TCC: sem equação numerada no texto
#   Usada em: espacamentos_atendem (l. 1528)
#   Usada também em: memorial_pilares.py: dados_detalhamento (l. 490)
def espacamento_livre_emenda(espacamento_eixos: float, diametro_longitudinal: float) -> float:
    return espacamento_eixos - 2.0 * diametro_longitudinal


# [F49] Número mínimo de barras numa face para respeitar s_max: n_min = max(2, ceil(L / s_max) + 1).
#   Variáveis: comprimento_face, espacamento_maximo.  Fonte: NBR 6118:2023, 18.4.2.2 (barra em cada vértice).
#   Fórmula: n_{mín} = máx(2; ⌈\frac{L}{s_{máx}}⌉ + 1)
#   TCC: sem equação numerada no texto
#   Usada em: posicoes_barras (l. 1499)
def barras_minimas_face(comprimento_face: float, espacamento_maximo: float) -> int:
    return max(2, math.ceil(comprimento_face / espacamento_maximo - 1e-9) + 1)


# [F74] Espaçamento longitudinal máximo dos estribos de pilares com barras de CA-50:
#   s_t,máx = mín(200 mm; menor dimensão da seção; 12 φl).
#   Variáveis: menor_dimensao (cm), diametro_longitudinal (cm).  Fonte: NBR 6118:2023, 18.4.3.
#   Fórmula: s_{t,máx} = mín(20 cm; b_{mín}; 12 · φ_{ℓ})
#   TCC: sem equação numerada no texto
#   Usada em: (sem uso neste script)
#   Usada também em: validacao_literatura.py: executar_scadelai (l. 1549);
#       memorial_pilares.py: dados_detalhamento (l. 499)
def espacamento_maximo_estribos(menor_dimensao: float, diametro_longitudinal: float) -> float:
    return min(ESPACAMENTO_MAXIMO_ESTRIBO_ABSOLUTO, menor_dimensao,
               FATOR_ESPACAMENTO_ESTRIBO_CA50 * diametro_longitudinal)


# [F75] Extensão protegida contra a flambagem a partir de cada canto do estribo poligonal: 20 φt.
#   Variáveis: diametro_estribo_cm (φt, cm).  Fonte: NBR 6118:2023, 18.2.4.
#   Fórmula: ℓ_{prot} = 20 · φ_{t}
#   TCC: sem equação numerada no texto
#   Usada em: barras_sem_protecao_face (l. 1701)
#   Usada também em: validacao_literatura.py: executar_scadelai (l. 1552);
#       memorial_pilares.py: dados_detalhamento (l. 500)
def trecho_protegido_estribo(diametro_estribo_cm: float) -> float:
    return FATOR_TRECHO_PROTEGIDO * diametro_estribo_cm


# [F52] Fração da área total de aço em cada barra (barras de mesmo diâmetro): f_i = 1 / n.
#   Variáveis: numero_barras.  Fonte: hipótese de barras iguais (Seção 3.6 do TCC).
#   Fórmula: f_{i} = \frac{1}{n}
#   TCC: sem equação numerada no texto
#   Usada em: secao_na_direcao (l. 1872); secao_por_camadas (l. 1881); esforcos_resistentes_obliquos (l. 2192)
#   Usada também em: validacao.py: nivel_1_flexao_obliqua (l. 1514, 1638)
def fracao_area_barras_iguais(numero_barras: int) -> np.ndarray:
    return np.full(numero_barras, 1.0 / numero_barras)


# [F50] Desvio percentual: Delta% = |X_Python - X_software| / X_software x 100.
#   Variáveis: valor_python, valor_software.  Fonte: Seção 3.9 do TCC.
#   Fórmula: Δ% = \frac{|X_{Python} − X_{software}|}{X_{software}} · 100
#   TCC: Eq. (2)
#   Usada em: (sem uso neste script)
#   Usada também em: validacao.py: registrar (l. 1346, 1351)
def desvio_percentual(valor_python: float, valor_software: float) -> float:
    return abs(valor_python - valor_software) / valor_software * 100.0


# ── Flexão composta oblíqua: envoltória resistente para a verificação da envoltória mínima (15.3.2, [R12])
# Coordenadas da seção: x ao longo de b_w e y ao longo de h, a partir do vértice (0, 0), o mais comprimido
# (as mesmas de posicoes_barras, [R02]); momentos em relação ao centro da seção (b_w/2, h/2), positivos quando
# comprimem o lado do vértice (0, 0): M_b (flexão na direção de b_w, como na direção "b", a de e1) e M_h (na de h).

# [F63] Vetor unitário normal à linha neutra, do vértice mais comprimido (0, 0) para o interior da seção, para a
#   inclinação alfa entre essa normal e o eixo de b_w: n_b = cos(alfa) (componente ao longo de b_w) e
#   n_h = sen(alfa) (ao longo de h). alfa = 0: flexão na direção de b_w (a de e1); alfa = 90°: na direção de h.
#   Variáveis: angulo (alfa, rad; aceita vetores).  Fonte: NBR 6118:2023, 17.2.2 (flexão composta oblíqua:
#   linha neutra inclinada em relação aos eixos principais).
#   Fórmula: n_{b} = cos α;  n_{h} = sen α
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_resistentes_obliquos (l. 2187); envoltoria_resistente (l. 2210)
#   Usada também em: validacao.py: nivel_1_flexao_obliqua (l. 1517, 1538, 1556, 1571)
def direcao_normal_linha_neutra(angulo) -> tuple[np.ndarray, np.ndarray]:
    angulo = np.asarray(angulo, dtype=float)
    return np.cos(angulo), np.sin(angulo)


# [F64] Profundidade de um ponto (x, y), medida a partir do vértice (0, 0) na direção normal à linha neutra
#   ([F63]): u = n_b x + n_h y. No vértice oposto (b_w, h), u = H(alfa), a altura projetada usada em [F18];
#   nos vértices (b_w, 0) e (0, h), u = n_b b_w e u = n_h h (mudanças da expressão da corda de [F65]).
#   Variáveis: x, y (cm), n_b, n_h (aceitam vetores).  Fonte: NBR 6118:2023, 17.2.2 a) (seções planas).
#   Fórmula: u = n_{b} · x + n_{h} · y
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_concreto_obliquo (l. 1312, 1317); esforcos_resistentes_obliquos (l. 2193);
#       envoltoria_resistente (l. 2211, 2212)
#   Usada também em: validacao.py: nivel_1_flexao_obliqua (l. 1518, 1519, 1544, 1545, 1560, 1561, 1574, 1575)
def profundidade_obliqua(x, y, n_b, n_h) -> np.ndarray:
    return n_b * x + n_h * y


# [F65] Corda da seção b_w x h na profundidade u (reta n_b x + n_h y = u): y entre y_inf = max(0; (u - n_b b_w)/n_h)
#   e y_sup = min(h; u/n_h); x entre x_inf = max(0; (u - n_h h)/n_b) e x_sup = min(b_w; u/n_b). Comprimento:
#   c = (y_sup - y_inf)/n_b se n_b >= n_h, senão c = (x_sup - x_inf)/n_h (divide-se pela maior componente, para não
#   perder precisão); centro da corda: x_m = (x_inf + x_sup)/2 e y_m = (y_inf + y_sup)/2. Vale para 0 < alfa < 90°.
#   Variáveis: profundidade (u, aceita vetores), largura (b_w), altura (h), n_b, n_h.
#   Fonte: geometria da seção retangular.
#   Fórmula: y_{inf} = máx(0; \frac{u − n_{b} · b_{w}}{n_{h}});  y_{sup} = mín(h; \frac{u}{n_{h}})
#   Fórmula: x_{inf} = máx(0; \frac{u − n_{h} · h}{n_{b}});  x_{sup} = mín(b_{w}; \frac{u}{n_{b}})
#   Fórmula: c = \frac{y_{sup} − y_{inf}}{n_{b}} (n_{b} ≥ n_{h});  senão c = \frac{x_{sup} − x_{inf}}{n_{h}}
#   Fórmula: x_{m} = \frac{x_{inf} + x_{sup}}{2};  y_{m} = \frac{y_{inf} + y_{sup}}{2}
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_concreto_obliquo (l. 1324)
def corda_retangulo(profundidade, largura: float, altura: float, n_b, n_h) -> tuple[np.ndarray, np.ndarray,
                                                                                         np.ndarray]:
    y_inferior = np.maximum(0.0, (profundidade - n_b * largura) / n_h)
    y_superior = np.minimum(altura, profundidade / n_h)
    x_inferior = np.maximum(0.0, (profundidade - n_h * altura) / n_b)
    x_superior = np.minimum(largura, profundidade / n_b)
    comprimento = np.where(n_b >= n_h, (y_superior - y_inferior) / n_b, (x_superior - x_inferior) / n_h)
    return np.maximum(comprimento, 0.0), (x_inferior + x_superior) / 2.0, (y_inferior + y_superior) / 2.0


# [F66] Resultantes do concreto na flexão oblíqua, por faixas paralelas à linha neutra e quadratura de
#   Gauss-Legendre por trechos: [0, H] ([F64]) é dividido nas profundidades em que muda a expressão da corda
#   (u = n_b b_w e u = n_h h, [F65]) ou do diagrama ([F10]: eps(u) = 0 e eps(u) = eps_c2, com eps(u) de [F15]);
#   em cada trecho [u_a, u_b], com u_j = (u_a + u_b)/2 + t_j (u_b - u_a)/2 e pesos w_j (u_b - u_a)/2:
#   N_c = soma sigma_c(u_j) c(u_j) w_j;  M_h = soma sigma_c c (h/2 - y_m) w_j;  M_b = soma sigma_c c (b_w/2 - x_m) w_j.
#   Em cada trecho, c, x_m e y_m são lineares em u e, com n = 2, sigma_c é polinômio de grau 2: o integrando tem
#   grau até 4, e a quadratura com PONTOS_GAUSS pontos ([N03]) é exata. Curvatura nula (deformação uniforme, reta b)
#   é tratada sem divisão: os cortes do diagrama vão para u = H.
#   Variáveis: deformacao_topo, curvatura, n_b, n_h (vetores de mesmo tamanho), largura, altura, tensao_pico,
#   eps_c2, expoente; NOS_GAUSS e PESOS_GAUSS de [N03].  Fonte: NBR 6118:2023, 17.2.2 e 8.2.10.1.
#   Fórmula: N_{c} = ∑ σ_{c}(u_{j}) · c(u_{j}) · w_{j}
#   Fórmula: M_{h} = ∑ σ_{c} · c · (\frac{h}{2} − y_{m}) · w_{j}
#   Fórmula: M_{b} = ∑ σ_{c} · c · (\frac{b_{w}}{2} − x_{m}) · w_{j}
#   Fórmula: u_{j} = \frac{u_{a} + u_{b}}{2} + t_{j} · \frac{u_{b} − u_{a}}{2}
#   Fórmula: peso = w_{j} · \frac{u_{b} − u_{a}}{2}
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_resistentes_obliquos (l. 2189)
def esforcos_concreto_obliquo(deformacao_topo, curvatura, n_b, n_h, largura: float, altura: float,
                              tensao_pico: float, eps_c2: float, expoente: float) -> tuple[np.ndarray, np.ndarray,
                                                                                           np.ndarray]:
    deformacao_topo, curvatura, n_b, n_h = np.broadcast_arrays(*(np.asarray(valor, dtype=float) for valor in
                                                                 (deformacao_topo, curvatura, n_b, n_h)))
    altura_projetada = profundidade_obliqua(largura, altura, n_b, n_h)
    tem_curvatura = curvatura != 0.0
    curvatura_segura = np.where(tem_curvatura, curvatura, 1.0)
    corte_zero = np.where(tem_curvatura, deformacao_topo / curvatura_segura, altura_projetada)
    corte_pico = np.where(tem_curvatura, (deformacao_topo - eps_c2) / curvatura_segura, altura_projetada)
    cortes = np.stack([profundidade_obliqua(largura, 0.0, n_b, n_h), profundidade_obliqua(0.0, altura, n_b, n_h),
                       corte_zero, corte_pico], axis=-1)
    cortes = np.sort(np.clip(cortes, 0.0, altura_projetada[..., None]), axis=-1)
    limites = np.concatenate([np.zeros(cortes.shape[:-1] + (1,)), cortes, altura_projetada[..., None]], axis=-1)
    centro_trecho = (limites[..., 1:] + limites[..., :-1]) / 2.0
    meia_altura_trecho = (limites[..., 1:] - limites[..., :-1]) / 2.0
    profundidades = centro_trecho[..., None] + meia_altura_trecho[..., None] * NOS_GAUSS
    comprimento, x_medio, y_medio = corda_retangulo(profundidades, largura, altura, n_b[..., None, None],
                                                    n_h[..., None, None])
    tensoes = tensao_concreto(deformacao_na_profundidade(deformacao_topo[..., None, None], curvatura[..., None, None],
                                                         profundidades), tensao_pico, eps_c2, expoente)
    forcas_faixas = tensoes * comprimento * meia_altura_trecho[..., None] * PESOS_GAUSS
    return (forcas_faixas.sum(axis=(-1, -2)), (forcas_faixas * (altura / 2.0 - y_medio)).sum(axis=(-1, -2)),
            (forcas_faixas * (largura / 2.0 - x_medio)).sum(axis=(-1, -2)))


# [F67] Força em cada barra na flexão oblíqua: F_i = A_si sigma_s(eps_i) - delta A_si sigma_c(eps_i), com
#   eps_i = eps_topo - k u_i ([F15], u_i de [F64]), sigma_s de [F13] e sigma_c de [F10] (nula na tração); delta = 1
#   com a seção líquida ([H05], parcela de concreto deslocada pela barra, como em [F61] e [F62]), 0 com a bruta.
#   Variáveis: deformacao_topo, curvatura (com eixo final de tamanho 1), profundidades (u_i), areas (A_si), fyd,
#   tensao_pico, eps_c2, expoente, descontar (bool).  Fonte: NBR 6118:2023, 17.2.2 b) e 8.3.6.
#   Fórmula: F_{i} = A_{si} · σ_{s}(ε_{i}) − δ · A_{si} · σ_{c}(ε_{i}),  ε_{i} = ε_{topo} − k · u_{i}
#   Fórmula: δ = 1 na seção líquida;  δ = 0 na seção bruta
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_resistentes_obliquos (l. 2194)
def forcas_barras_obliquo(deformacao_topo, curvatura, profundidades, areas: np.ndarray, fyd: float,
                          tensao_pico: float, eps_c2: float, expoente: float, descontar: bool) -> np.ndarray:
    deformacoes = deformacao_na_profundidade(deformacao_topo, curvatura, profundidades)
    forcas_aco = areas * tensao_aco(deformacoes, fyd)
    if not descontar:
        return forcas_aco
    return descontar_parcela(forcas_aco, areas * tensao_concreto(deformacoes, tensao_pico, eps_c2, expoente))


# [F68] Resultantes das forças nas barras em relação ao centro da seção: N_s = soma F_i;
#   M_h = soma F_i (h/2 - y_i);  M_b = soma F_i (b_w/2 - x_i).
#   Variáveis: forcas (F_i, eixo final = barras), x, y (cm), largura (b_w), altura (h).
#   Fonte: NBR 6118:2023, 17.2.2 (equilíbrio da seção).
#   Fórmula: N_{s} = ∑ F_{i};  M_{h} = ∑ F_{i} · (\frac{h}{2} − y_{i});  M_{b} = ∑ F_{i} · (\frac{b_{w}}{2} − x_{i})
#   TCC: sem equação numerada no texto
#   Usada em: esforcos_resistentes_obliquos (l. 2197)
def resultantes_barras_obliquo(forcas: np.ndarray, x: np.ndarray, y: np.ndarray, largura: float,
                               altura: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return forcas.sum(axis=-1), (forcas * (altura / 2.0 - y)).sum(axis=-1), (forcas * (largura / 2.0 - x)).sum(axis=-1)


# [F69] Raio da envoltória mínima (elipse) na direção psi do vetor momento (psi = 0: direção de b_w):
#   (M_b / M_b,mín)² + (M_h / M_h,mín)² = 1  =>  r_e(psi) = 1 / raiz[(cos psi / M_b,mín)² + (sen psi / M_h,mín)²].
#   Com 2ª ordem, M_b,mín = M_d,tot,mín,yy e M_h,mín = M_d,tot,mín,xx (Figura 15.2, seção b x h, com b = b_w);
#   sem 2ª ordem, M1d,mín (Figura 11.3).
#   Variáveis: direcao (psi, rad; aceita vetores), momento_b_minimo, momento_h_minimo (kN·cm).
#   Fonte: NBR 6118:2023, 11.3.3.4.3 (Figura 11.3) e 15.3.2 (Figura 15.2).
#   Fórmula: (\frac{M_{b}}{M_{b,mín}})^{2} + (\frac{M_{h}}{M_{h,mín}})^{2} = 1
#   Fórmula: r_{e}(ψ) = \frac{1}{\sqrt{(cos ψ / M_{b,mín})^{2} + (sen ψ / M_{h,mín})^{2}}}
#   TCC: sem equação numerada no texto
#   Usada em: verificar_envoltoria_minima (l. 2283)
#   Usada também em: validacao.py: nivel_1_flexao_obliqua (l. 1599, 1602);
#       gerar_auditor.py: graficos_do_pilar (l. 395, 398)
def raio_envoltoria_minima(direcao, momento_b_minimo: float, momento_h_minimo: float) -> np.ndarray:
    direcao = np.asarray(direcao, dtype=float)
    return 1.0 / np.sqrt((np.cos(direcao) / momento_b_minimo) ** 2 + (np.sin(direcao) / momento_h_minimo) ** 2)


# [F70] Direção do vetor momento (M_b, M_h): psi = atan2(M_h, M_b) (0 na direção de b_w, 90° na de h).
#   Variáveis: momento_b, momento_h (aceitam vetores).  Fonte: geometria do plano (M_b, M_h).
#   Fórmula: ψ = atan2(M_{h}; M_{b})
#   TCC: sem equação numerada no texto
#   Usada em: verificar_envoltoria_minima (l. 2272)
def direcao_momento(momento_b, momento_h) -> np.ndarray:
    return np.arctan2(momento_h, momento_b)


# [F71] Raio da envoltória resistente na direção psi: interseção da semirreta de direção d = (cos psi, sen psi),
#   com componentes (b, h), com o segmento entre pontos consecutivos P_k e P_k+1 do contorno ([M08]): de
#   r d = P_k + s (P_k+1 - P_k), r = (P_k x P_k+1) / [d x (P_k+1 - P_k)], com a x c = a_b c_h - a_h c_b. Como o
#   contorno é convexo, a corda fica do lado de dentro e o raio obtido não supera o exato (a favor da segurança).
#   Variáveis: momentos_b_k, momentos_h_k (P_k), momentos_b_k1, momentos_h_k1 (P_k+1), direcao (psi, rad).
#   Fonte: geometria analítica do plano.
#   Fórmula: r = \frac{P_{k} × P_{k+1}}{d × (P_{k+1} − P_{k})},  d = (cos ψ; sen ψ)
#   Fórmula: a × c = a_{b} · c_{h} − a_{h} · c_{b}
#   TCC: sem equação numerada no texto
#   Usada em: verificar_envoltoria_minima (l. 2281)
#   Usada também em: validacao.py: nivel_1_flexao_obliqua (l. 1583, 1586)
def raio_contorno(momentos_b_k, momentos_h_k, momentos_b_k1, momentos_h_k1, direcao) -> np.ndarray:
    produto_pontos = momentos_b_k * momentos_h_k1 - momentos_h_k * momentos_b_k1
    produto_direcao = (np.cos(direcao) * (momentos_h_k1 - momentos_h_k)
                       - np.sin(direcao) * (momentos_b_k1 - momentos_b_k))
    return produto_pontos / produto_direcao


# [F72] Fator de atendimento da envoltória mínima numa direção psi: eta(psi) = r_R(psi) / r_e(psi) ([F71] / [F69]);
#   a envoltória resistente engloba a mínima quando min eta >= FATOR_MINIMO_ENVOLTORIA ([R12]).
#   Variáveis: raio_resistente, raio_minimo.  Fonte: NBR 6118:2023, 15.3.2.
#   Fórmula: η(ψ) = \frac{r_{R}(ψ)}{r_{e}(ψ)};  atende quando mín η ≥ 1
#   TCC: sem equação numerada no texto
#   Usada em: verificar_envoltoria_minima (l. 2281)
def fator_envoltoria(raio_resistente, raio_minimo) -> np.ndarray:
    return raio_resistente / raio_minimo


# [F73] Conversão de ângulo: rad -> graus (relatórios).  Variáveis: angulo (rad).
#   Fórmula: graus = rad · \frac{180}{π}
#   TCC: sem equação numerada no texto
#   Usada em: verificar_envoltoria_minima (l. 2286)
def radianos_para_graus(angulo) -> float:
    return float(np.degrees(angulo))


# ───────────────────────────────────────────────────────────── E) REGRAS DE NEGÓCIO (R)
# [R01] Escolha do procedimento de 2ª ordem pela faixa de esbeltez da direção analisada:
#   lambda < lambda_1: efeitos locais de 2ª ordem dispensados (15.8.2: "menor que o valor-limite lambda_1");
#   lambda_1 <= lambda <= 90: pilar-padrão com curvatura aproximada;
#   90 < lambda <= 140: pilar-padrão acoplado a diagramas M, N, 1/r, com fluência;
#   lambda > 140: fora do escopo (método geral obrigatório).
#   Comparações com a tolerância [N05] (o lambda igual ao limite, a menos do erro de ponto flutuante, conta como igual).
#   Variáveis: esbeltez, esbeltez_limite_1.  Fonte: NBR 6118:2023, 15.8.2, 15.8.3.3 e 15.8.4; Seção 3.5 do TCC.
#   Fórmula: λ < λ_{1}: 2ª ordem dispensada;  λ_{1} ≤ λ ≤ 90: curvatura aproximada
#   Fórmula: 90 < λ ≤ 140: M, N, 1/r com fluência
#   Usada em: metodo_segunda_ordem (l. 1455, 1457, 1459); solicitacao_direcao (l. 2111, 2115, 2118);
#       armadura_direcao (l. 2142); area_atende_direcao (l. 2159); momento_total_minimo (l. 2248)
#   Usada também em: validacao.py: nível do módulo (l. 525, 531, 537, 543, 548);
#       validacao.py: metodo_bastos (l. 1100); validacao.py: cadeia_exata_bastos (l. 1127);
#       validacao.py: cadeia_autor_bastos (l. 1149); validacao.py: nivel_1_flexao_obliqua (l. 1623);
#       validacao.py: nivel_2 (l. 1880, 1912); validacao.py: conferir_mn1r_independente (l. 2083);
#       verificar_matriz.py: metodo_esperado (l. 114, 116, 117);
#       verificar_matriz.py: conferir_resultados (l. 379, 492);
#       sensibilidade_secao_liquida.py: comparacao_diagramas (l. 268); gerar_auditor.py: graficos_do_pilar (l. 408);
#       validacao_literatura.py: executar_carvalho (l. 2110); memorial_pilares.py: dados_direcao (l. 425, 429, 445);
#       memorial_pilares.py: texto_metodo (l. 603, 604, 605);
#       memorial_pilares.py: blocos_direcao (l. 660, 662, 672, 705); memorial_pilares.py: blocos_verificacao (l. 714);
#       memorial_pilares.py: blocos_pilar (l. 849); memorial_pilares.py: conferir_memoriais (l. 1035)
METODO_DISPENSADO = "2ª ordem dispensada"
METODO_CURVATURA = "curvatura aproximada"
METODO_MN1R = "M, N, 1/r com fluência"


def metodo_segunda_ordem(esbeltez: float, esbeltez_limite_1: float) -> str:
    if esbeltez < esbeltez_limite_1 - TOLERANCIA_FRONTEIRA_ESBELTEZ:
        return METODO_DISPENSADO
    if esbeltez <= LAMBDA_LIMITE_CURVATURA + TOLERANCIA_FRONTEIRA_ESBELTEZ:
        return METODO_CURVATURA
    if esbeltez <= LAMBDA_LIMITE_PILAR_PADRAO + TOLERANCIA_FRONTEIRA_ESBELTEZ:
        return METODO_MN1R
    raise ValueError("lambda = %.1f > 140: fora do escopo (método geral obrigatório)" % esbeltez)


# [F51] Barras intermediárias em cada face paralela a b_w no arranjo perimetral, para espaçamento
#   aproximadamente igual ao das faces paralelas a h: n_b = max(1, arred(L_b / s_h) - 1), com
#   arredondamento ao inteiro mais próximo.  Variáveis: comprimento_b (L_b), espacamento_h (s_h).
#   Fonte: definição do arranjo perimetral (Seção 3.6 do TCC).
#   Fórmula: n_{b} = máx(1; arred(\frac{L_{b}}{s_{h}}) − 1)
#   TCC: sem equação numerada no texto
#   Usada em: posicoes_barras (l. 1506)
def barras_intermediarias_perimetro(comprimento_b: float, espacamento_h: float) -> int:
    return max(1, math.floor(comprimento_b / espacamento_h + 0.5) - 1)


# [R02] Arranjos de armadura simétrica (escolhidos pela regra [R07]), barras de mesmo diâmetro e
#   n_h = n_h,min ([F49]) + nivel - 1 barras em cada face paralela a h (cantos incluídos):
#   "duas_faces": barras só nas duas faces paralelas a h (perpendiculares a b_w, as mais eficientes para a
#     flexão na direção de b_w, a de e1), sem barras intermediárias nas faces paralelas a b_w;
#   "perimetro": acrescenta n_b ([F51]) barras intermediárias em cada face paralela a b_w, de modo que as
#     barras fiquem distribuídas com espaçamento aproximadamente uniforme ao longo do perímetro.
#   Retorna as coordenadas (x ao longo de b_w, y ao longo de h, medidas a partir de um vértice) e, para
#   cada orientação de face, (comprimento entre eixos extremos, número de barras na face), primeiro as faces
#   paralelas a h e depois as paralelas a b_w.
#   Variáveis: largura (b_w), altura (h), distancia_eixo (d'), arranjo, nivel, espacamento_maximo.
#   Fonte: Seção 3.6 do TCC; NBR 6118:2023, 18.4.2.2 (barra em cada vértice; espaçamento máximo).
#   Fórmula: n_{h} = n_{h,mín} + nível − 1 (barras por face paralela a h)
#   Usada em: posicoes_barras (l. 1503, 1505); nível do módulo (l. 1596, 1658); montar_candidato (l. 2324)
#   Usada também em: validacao.py: nível do módulo (l. 862, 863, 864); validacao.py: nivel_1_flexao_obliqua (l. 1510);
#       processar_matriz.py: linha_inviavel (l. 726, 727); verificar_matriz.py: duas_faces_isoladas (l. 223);
#       verificar_matriz.py: conferir_resultados (l. 396, 397, 413);
#       memorial_pilares.py: nível do módulo (l. 108, 109); memorial_pilares.py: dados_detalhamento (l. 483)
ARRANJO_DUAS_FACES = "duas_faces"
ARRANJO_PERIMETRO = "perimetro"


def posicoes_barras(largura: float, altura: float, distancia_eixo: float, arranjo: str, nivel: int,
                    espacamento_maximo: float) -> tuple[np.ndarray, np.ndarray, tuple]:
    comprimento_h = altura - 2.0 * distancia_eixo
    comprimento_b = largura - 2.0 * distancia_eixo
    barras_h = barras_minimas_face(comprimento_h, espacamento_maximo) + nivel - 1
    y_face = np.linspace(distancia_eixo, altura - distancia_eixo, barras_h)
    y_faces_h = np.concatenate([y_face, y_face])
    x_faces_h = np.concatenate([np.full(barras_h, distancia_eixo), np.full(barras_h, largura - distancia_eixo)])
    if arranjo == ARRANJO_DUAS_FACES:
        return x_faces_h, y_faces_h, ((comprimento_h, barras_h), (comprimento_b, 2))
    if arranjo == ARRANJO_PERIMETRO:
        intermediarias_b = barras_intermediarias_perimetro(comprimento_b,
                                                           espacamento_eixos_face(comprimento_h, barras_h))
        x_intermediarias = np.linspace(distancia_eixo, largura - distancia_eixo, intermediarias_b + 2)[1:-1]
        y = np.concatenate([y_faces_h, np.full(intermediarias_b, distancia_eixo),
                            np.full(intermediarias_b, altura - distancia_eixo)])
        x = np.concatenate([x_faces_h, x_intermediarias, x_intermediarias])
        return x, y, ((comprimento_h, barras_h), (comprimento_b, intermediarias_b + 2))
    raise ValueError("arranjo desconhecido: %s" % arranjo)


# [R03] Verificação dos espaçamentos de um arranjo em todas as faces: s <= s_max ([F45], [F47]) e
#   espaçamento livre na emenda >= a_min ([F44], [F48]).
#   Variáveis: faces (comprimento, n), diametro_longitudinal, espacamento_maximo, espacamento_minimo_livre.
#   Fonte: NBR 6118:2023, 18.4.2.2.
#   Fórmula: s ≤ s_{máx};  a ≥ a_{mín}
#   Usada em: gerar_candidatos (l. 2341)
def espacamentos_atendem(faces: tuple, diametro_longitudinal: float, espacamento_maximo: float,
                         espacamento_minimo_livre: float) -> bool:
    for comprimento_face, numero_barras_face in faces:
        espacamento = espacamento_eixos_face(comprimento_face, numero_barras_face)
        if espacamento > espacamento_maximo + 1e-9:
            return False
        if espacamento_livre_emenda(espacamento, diametro_longitudinal) < espacamento_minimo_livre - 1e-9:
            return False
    return True


# [R04] Critério de escolha do detalhamento: menor área adotada; em empate, menor número de barras;
#   persistindo, menor diâmetro.  Variáveis: area_adotada, numero_barras, diametro.  Fonte: Seção 3.6 do TCC.
#   Usada em: gerar_candidatos (l. 2346)
def chave_escolha_detalhamento(area_adotada: float, numero_barras: int, diametro: float) -> tuple:
    return round(area_adotada, 6), numero_barras, diametro


# [R05] Classificação do resultado do dimensionamento: viável quando algum arranjo candidato atende a [R09];
#   sem arranjo que atenda, a causa é dada pelo As,calc da geometria de nível [R11] do arranjo analisado:
#   sem equilíbrio ou acima de 8 % Ac (infinito) -> seção insuficiente; acima de As,máx -> acima da máxima;
#   caso contrário -> nenhum arranjo atende ao detalhamento.
#   Variáveis: area_calculada, area_maxima, possui_arranjo.  Fonte: NBR 6118:2023, 17.3.5.3; Seção 3.6 do TCC.
#   Fórmula: A_{s,calc} = ∞ (sem equilíbrio ou > 8 % A_{c}): seção insuficiente
#   Fórmula: A_{s,calc} > A_{s,máx}: acima da máxima;  sem arranjo que atenda: detalhamento
#   Usada em: classificar_dimensionamento (l. 1567, 1569, 1571, 1572); dimensionar_arranjo (l. 2471);
#       dimensionar_pilar (l. 2516, 2518)
#   Usada também em: processar_matriz.py: modelar_nos_softwares (l. 200);
#       processar_matriz.py: nível do módulo (l. 330, 331); processar_matriz.py: escrever_resumo (l. 852, 853);
#       verificar_matriz.py: conferir_resultados (l. 398, 402, 468, 471, 479, 488, 526, 539, 566);
#       sensibilidade_secao_liquida.py: nível do módulo (l. 57);
#       conferir_busca_exaustiva.py: escolha_exaustiva (l. 96);
#       conferir_busca_exaustiva.py: main (l. 118, 119, 133, 134); figuras_tcc.py: nível do módulo (l. 227, 228);
#       figuras_tcc.py: contagens_viabilidade (l. 698); figuras_tcc.py: linhas_viaveis (l. 753);
#       gerar_auditor.py: main (l. 454, 455); figuras_complementares.py: figura_taxa (l. 574);
#       validacao_literatura.py: executar_carvalho (l. 2481); memorial_pilares.py: blocos_pilar (l. 804);
#       memorial_pilares.py: escrever_documento (l. 948); memorial_pilares.py: conferir_memoriais (l. 1014)
STATUS_OK = "ok"
STATUS_SECAO_INSUFICIENTE = "inviável: As,calc > 8 % Ac ou sem equilíbrio estável"
STATUS_ACIMA_MAXIMA = "inviável: As,calc > As,máx = 4 % Ac"
STATUS_SEM_ARRANJO = "inviável: nenhum arranjo com As,adot ≤ As,máx atende ao detalhamento"


def classificar_dimensionamento(area_calculada: float, area_maxima: float, possui_arranjo: bool) -> str:
    if math.isinf(area_calculada):
        return STATUS_SECAO_INSUFICIENTE
    if area_calculada > area_maxima:
        return STATUS_ACIMA_MAXIMA
    if not possui_arranjo:
        return STATUS_SEM_ARRANJO
    return STATUS_OK


# [R06] Momentos de 1ª ordem menores que o momento mínimo (aciona alfa_b = 1,0, alínea d de 15.8.2):
#   verdadeiro quando Nd e1 < M1d,min.  Variáveis: momento_aplicado ([F53]), momento_minimo ([F21]).
#   Fonte: NBR 6118:2023, 15.8.2 d).
#   Fórmula: N_{d} · e_{1} < M_{1d,mín}  ⇒  α_{b} = 1,0
#   Usada em: coeficiente_alfa_b (l. 798); solicitacao_direcao (l. 2109)
#   Usada também em: validacao.py: nivel_2 (l. 1900); validacao_literatura.py: executar_kimura (l. 1209);
#       validacao_literatura.py: executar_scadelai (l. 1391); memorial_pilares.py: dados_direcao (l. 416)
def momentos_menores_que_minimo(momento_aplicado: float, momento_minimo: float) -> bool:
    return momento_aplicado < momento_minimo


# [R07] Regra do arranjo da armadura longitudinal: prioritariamente em duas faces opostas (A's = As);
#   distribuída no perímetro somente quando a disposição em duas camadas for inviável (As,calc > As,max,
#   seção insuficiente ou nenhum detalhamento possível).  ARRANJO_REGRA indica a aplicação desta regra.
#   Variáveis: ARRANJO_REGRA, ORDEM_ARRANJOS_REGRA.  Fonte: Seção 3.6 do PP/TCC (parâmetros fixos);
#   anotação do coorientador no PP (pilares que não podem ser armados só com duas camadas).
#   Usada em: PilarTeste (l. 2303); dimensionar_pilar (l. 2510, 2513)
#   Usada também em: processar_matriz.py: criar_pilar (l. 375); verificar_matriz.py: conferir_entradas (l. 265, 266);
#       verificar_matriz.py: conferir_resultados (l. 403); conferir_busca_exaustiva.py: escolha_exaustiva (l. 84, 85);
#       conferir_busca_exaustiva.py: main (l. 127)
ARRANJO_REGRA = "regra_pp"
ORDEM_ARRANJOS_REGRA = (ARRANJO_DUAS_FACES, ARRANJO_PERIMETRO)


# [R08] Armadura calculada do pilar (arranjo fixado): As,calc = max(As,b ; As,h ; As,min) — a armadura
#   deve atender às duas direções principais, analisadas isoladamente, e à armadura mínima.
#   Variáveis: area_direcao_b, area_direcao_h, area_minima.  Fonte: NBR 6118:2023, 15.3.2 e 17.3.5.3.1.
#   Fórmula: A_{s,calc} = máx(A_{s,b}; A_{s,h}; A_{s,mín})
#   Usada em: dimensionar_arranjo.area_necessaria (l. 2433)
#   Usada também em: sensibilidade_secao_liquida.py: area_calculada_na_geometria (l. 192);
#       conferir_busca_exaustiva.py: area_calculada_candidato (l. 77)
def area_calculada_pilar(area_direcao_b: float, area_direcao_h: float, area_minima: float) -> float:
    return max(area_direcao_b, area_direcao_h, area_minima)


# [R09] Um detalhamento atende quando a área adotada (barras comerciais) não é menor que a calculada
#   para a geometria desse mesmo detalhamento: As,adot >= As,calc. A busca (dimensionar_arranjo) percorre
#   todos os arranjos candidatos na ordem de [R04], sem pular nenhum, e verifica cada um diretamente na área
#   adotada (area_atende_direcao): As,adot >= As,mín e folga de [M04] não negativa nas duas direções; o
#   primeiro que atende é o escolhido, e só para ele se calcula As,calc pela busca de raiz. A versão anterior
#   pulava os candidatos com área menor que o As,calc da geometria mais eficiente do arranjo, o que se
#   mostrou incorreto no arranjo no perímetro (conferir_busca_exaustiva.py).
#   Variáveis: area_adotada, area_calculada.  Fonte: Seção 3.6 do TCC.
#   Fórmula: A_{s,adot} ≥ A_{s,calc}
#   Usada em: dimensionar_arranjo.candidato_atende (l. 2438); dimensionar_arranjo (l. 2464)
#   Usada também em: conferir_busca_exaustiva.py: escolha_exaustiva (l. 95);
#       memorial_pilares.py: dados_detalhamento (l. 515)
def detalhamento_atende(area_adotada: float, area_calculada: float) -> bool:
    return area_adotada >= area_calculada


# [R10] Validação das entradas (Módulo 1 da Figura 3 do TCC, Figura 3.1 do PP): dimensões mínimas ([C13]), força
#   normal de compressão positiva, e1 >= 0, vinculação biapoiada ([H04]) e lambda <= 140 nas duas direções
#   principais ([C08]; acima disso o método geral é obrigatório e o caso sai do escopo).
#   Variáveis: largura, altura, forca_normal_calculo, excentricidade_primeira_ordem, comprimento_equivalente,
#   vinculacao.  Fonte: NBR 6118:2023, 13.2.3 e 15.8.3.2; delimitação do tema (Seção 1.1 do PP; Introdução do TCC).
#   Usada em: dimensionar_pilar (l. 2507)
def validar_entradas(largura: float, altura: float, forca_normal_calculo: float,
                     excentricidade_primeira_ordem: float, comprimento_equivalente: float, vinculacao: str) -> None:
    if min(largura, altura) < DIMENSAO_MINIMA_PILAR:
        raise ValueError("menor dimensão < %.0f cm (NBR 6118:2023, 13.2.3)" % DIMENSAO_MINIMA_PILAR)
    if largura * altura < AREA_MINIMA_PILAR:
        raise ValueError("área da seção < %.0f cm² (NBR 6118:2023, 13.2.3)" % AREA_MINIMA_PILAR)
    if forca_normal_calculo <= 0.0:
        raise ValueError("Nd deve ser de compressão (> 0)")
    if excentricidade_primeira_ordem < 0.0:
        raise ValueError("e1 deve ser >= 0")
    if vinculacao != VINCULACAO_BIAPOIADO:
        raise ValueError("vinculação fora do escopo: %s" % vinculacao)
    for dimensao in (altura, largura):
        if indice_esbeltez(comprimento_equivalente, dimensao) > LAMBDA_LIMITE_PILAR_PADRAO + TOLERANCIA_FRONTEIRA_ESBELTEZ:
            raise ValueError("lambda > 140 numa direção principal: método geral obrigatório (fora do escopo)")


# [R11] Geometria de referência, que separa o efeito do detalhamento do efeito do método: armadura em duas
#   faces opostas (duas camadas na direção b_w), menor bitola longitudinal permitida ([F42]), número mínimo
#   de barras por face (nível 1 de [R02]), estribo de [F43] e d' = c + φt + φl/2 ([F46]); com c = 2,5 cm,
#   φt = 5 mm e φl = 10 mm, d' = 3,5 cm em todos os pilares. As,calc,ref = max(As,b,ref; As,h,ref; As,mín)
#   ([R08]) não depende das barras escolhidas; a diferença entre As,calc e As,calc,ref mede o efeito da
#   posição das barras adotadas (Seção 3.9 do TCC). Nos pilares inviáveis, o As,calc relatado é o da
#   geometria de nível [R11] do arranjo analisado, que define a causa da inviabilidade ([R05]).
#   Variáveis: ARRANJO_REFERENCIA, NIVEL_REFERENCIA.  Fonte: hipótese do trabalho (Seção 3.9 do TCC).
#   Usada em: candidato_referencia (l. 2356); dimensionar_arranjo (l. 2455)
ARRANJO_REFERENCIA = ARRANJO_DUAS_FACES
NIVEL_REFERENCIA = 1

# [R12] Verificação da envoltória mínima com 2ª ordem ([H06]; NBR 6118:2023, 11.3.3.4.3 e 15.3.2), com a
#   armadura adotada:
#   (a) semieixos: M_d,tot,mín em cada direção principal, calculados só com o momento mínimo de 1ª ordem
#       (e1 = EXCENTRICIDADE_ENVOLTORIA_MINIMA = 0, logo M1d,A = M1d,mín de [F55] e alfa_b = 1, 15.8.2 d)) e pelo
#       procedimento de 2ª ordem da faixa de esbeltez da direção ([R01], 15.8.3), como na direção de h; no
#       pilar-padrão acoplado a M, N, 1/r, com a armadura adotada (equilíbrio estável; sem equilíbrio, o
#       semieixo é infinito e a verificação não é atendida); onde a 2ª ordem é dispensada, o semieixo é M1d,mín
#       (envoltória de 1ª ordem, Figura 11.3);
#   (b) envoltória resistente a N = Nd: M_Rd das duas direções principais (extremos) e os pontos de
#       ANGULOS_LINHA_NEUTRA ([N04]) na flexão oblíqua ([F63] a [F68], domínios de [F18] com H(alfa));
#   (c) fator eta = min eta(psi) ([F72]) nas DIRECOES_ENVOLTORIA; atende se eta >= FATOR_MINIMO_ENVOLTORIA;
#   (d) arranjo pela elipse (informativo, [H06]): o adotado, se atende; senão, o primeiro candidato seguinte na
#       ordem de [R04], do mesmo arranjo, que atende a [R08], [R09] e (c).
#   Variáveis: EXCENTRICIDADE_ENVOLTORIA_MINIMA (cm), FATOR_MINIMO_ENVOLTORIA.
#   Fonte: NBR 6118:2023, 11.3.3.4.3, 15.3.2, 15.8.2 d) e 15.8.3; Seção 3.5 do TCC.
#   Fórmula: η = mín_{ψ} η(ψ) ≥ 1
#   Usada em: verificar_envoltoria_minima (l. 2287); dimensionar_arranjo (l. 2483, 2485)
#   Usada também em: validacao.py: nivel_1_flexao_obliqua (l. 1626);
#       verificar_matriz.py: conferir_resultados (l. 495, 549); gerar_auditor.py: main (l. 453)
EXCENTRICIDADE_ENVOLTORIA_MINIMA = 0.0
FATOR_MINIMO_ENVOLTORIA = 1.0


# [R13] Barras de uma face que exigem estribo suplementar (grampo) contra a flambagem: o estribo poligonal protege
#   as barras de canto e as situadas a no máximo 20 φt de cada canto ([F75]), se nesse trecho não houver mais de
#   duas barras além da de canto ([C14]); exigem estribo suplementar as barras fora dos dois trechos e, num trecho
#   com mais de duas barras, as excedentes (as mais afastadas do canto). As distâncias são medidas ao longo da
#   face, a partir do eixo da barra de canto (como em Scadelai e Pinheiro, 2007, item 16.10.5), com as barras
#   igualmente espaçadas entre as de canto ([F47]).
#   Variáveis: comprimento_face (entre os eixos das barras de canto, cm), numero_barras_face (cantos incluídos),
#   diametro_estribo_cm.  Fonte: NBR 6118:2023, 18.2.4 e Figura 18.2.
#   Fórmula: barra i (0 < i < n − 1) protegida se mín(y_{i}; L − y_{i}) ≤ 20 · φ_{t} e estiver entre as 2 mais
#   próximas do canto no trecho;  y_{i} = i · L/(n − 1)
#   Usada em: (sem uso neste script)
#   Usada também em: validacao_literatura.py: executar_scadelai (l. 1556);
#       memorial_pilares.py: dados_detalhamento (l. 491)
def barras_sem_protecao_face(comprimento_face: float, numero_barras_face: int, diametro_estribo_cm: float) -> int:
    if numero_barras_face <= 2:
        return 0
    espacamento = espacamento_eixos_face(comprimento_face, numero_barras_face)
    trecho = trecho_protegido_estribo(diametro_estribo_cm)
    sem_protecao = 0
    for indice in range(1, numero_barras_face - 1):
        ordem_a_partir_do_canto = min(indice, numero_barras_face - 1 - indice)
        distancia_ao_canto = ordem_a_partir_do_canto * espacamento
        dentro_do_trecho = distancia_ao_canto <= trecho + 1e-9
        if not dentro_do_trecho or ordem_a_partir_do_canto > BARRAS_MAXIMAS_TRECHO_PROTEGIDO:
            sem_protecao += 1
    return sem_protecao


# ───────────────────────────────────────────────────────────── F) MÉTODOS NUMÉRICOS (M)
# [M01] Bisseção vetorizada: para cada posição do vetor, encontra x em [inferior, superior] com
#   funcao(x) = 0, supondo funcao crescente e funcao(inferior) < 0 <= funcao(superior). Posições em que
#   funcao(superior) < 0 (sem raiz) retornam NaN. Precisão: (superior - inferior) / 2^iteracoes.
#   Variáveis: funcao (vetorizada), inferior, superior (vetores), iteracoes.
#   Fórmula: precisão = \frac{superior − inferior}{2^{iterações}}
#   Usada em: deformacoes_topo_para_forca (l. 1985); envoltoria_resistente (l. 2222)
def bissecao_vetorial(funcao, inferior: np.ndarray, superior: np.ndarray, iteracoes: int) -> np.ndarray:
    inferior = np.array(inferior, dtype=float)
    superior = np.array(superior, dtype=float)
    sem_raiz = funcao(superior) < 0.0
    for _ in range(iteracoes):
        meio = (inferior + superior) / 2.0
        positivo = funcao(meio) >= 0.0
        superior = np.where(positivo, meio, superior)
        inferior = np.where(positivo, inferior, meio)
    return np.where(sem_raiz, np.nan, (inferior + superior) / 2.0)


# [M02] Menor valor x em [inferior, superior] que atende folga(x) >= 0, com folga crescente e
#   folga(inferior) < 0 <= folga(superior): raiz pelo método de Brent e avanço de 'tolerancia' até atender.
#   Variáveis: folga (função), inferior, superior, tolerancia.
#   Usada em: armadura_flexo_compressao (l. 1971); armadura_pilar_padrao_acoplado (l. 2071)
#   Usada também em: sensibilidade_secao_liquida.py: area_necessaria_modelo (l. 259)
def menor_valor_que_atende(folga, inferior: float, superior: float, tolerancia: float) -> float:
    candidato = brentq(folga, inferior, superior, xtol=tolerancia)
    while folga(candidato) < 0.0:
        candidato = min(candidato + tolerancia, superior)
    return candidato


# [M03] Resíduo de uma equação de equilíbrio f(x) = alvo, usado nas buscas de raiz: r = f(x) - alvo.
#   Variáveis: valor (f(x)), alvo.
#   Fórmula: r = f(x) − alvo
#   Usada em: parametro_ultimo_para_forca (l. 1924); limites_busca_armadura (l. 1954);
#       deformacoes_topo_para_forca.residuo_forca (l. 1982); envoltoria_resistente.residuo_forca (l. 2219)
#   Usada também em: validacao.py: momento_fibras_com_forca.residuo_forca (l. 2065);
#       sensibilidade_secao_liquida.py: momento_resistente_modelo (l. 230);
#       sensibilidade_secao_liquida.py: area_necessaria_modelo (l. 247)
def residuo(valor, alvo):
    return valor - alvo


# [M04] Folga de momento de uma armadura candidata: M_Rd - M_Sd; quando M_Rd não existe (a força normal
#   não é resistida, M_Rd = NaN), a folga é -M_Sd (negativa), o que leva a busca a aumentar a armadura.
#   Variáveis: momento_resistente_calculo (M_Rd ou NaN), momento_solicitante (M_Sd).
#   Fórmula: folga = M_{Rd} − M_{Sd}  (folga = −M_{Sd} quando M_{Rd} não existe)
#   Usada em: armadura_flexo_compressao.folga (l. 1965); armadura_pilar_padrao_acoplado.avaliar (l. 2056, 2059);
#       area_atende_direcao (l. 2160, 2170)
#   Usada também em: sensibilidade_secao_liquida.py: area_necessaria_modelo.folga (l. 252)
def folga_momento(momento_resistente_calculo: float, momento_solicitante: float) -> float:
    if math.isnan(momento_resistente_calculo):
        return -momento_solicitante
    return momento_resistente_calculo - momento_solicitante


# [M05] Intervalo da bisseção da deformação do topo que equilibra N para cada curvatura k:
#   inferior = -(eps_su + folga) (seção toda tracionada além do alongamento último) e
#   superior = eps_c2 + k h + folga (toda a seção acima de eps_c2, compressão máxima).
#   Variáveis: curvaturas, altura, eps_c2; DEFORMACAO_ULTIMA_ACO, FOLGA_INTERVALO_DEFORMACAO.
#   Fórmula: inferior = −(ε_{su} + folga);  superior = ε_{c2} + k · h + folga
#   Usada em: deformacoes_topo_para_forca (l. 1979)
def intervalo_deformacao_topo(curvaturas: np.ndarray, altura: float, eps_c2: float) -> tuple[np.ndarray, np.ndarray]:
    inferior = np.full(curvaturas.shape, -(DEFORMACAO_ULTIMA_ACO + FOLGA_INTERVALO_DEFORMACAO))
    superior = eps_c2 + curvaturas * altura + FOLGA_INTERVALO_DEFORMACAO
    return inferior, superior


# [M06] Tolerâncias derivadas das buscas: da área na reta b (centésimo de TOLERANCIA_AREA) e da curvatura de
#   equilíbrio (fração TOLERANCIA_RELATIVA_CURVATURA da maior curvatura da curva M, N, 1/r).
#   Variáveis: TOLERANCIA_AREA, TOLERANCIA_RELATIVA_CURVATURA, curvatura_maxima.
#   Fórmula: tol_{A, reta b} = \frac{tol_{A}}{100};  tol_{1/r} = 10^{−9} · (1/r)_{máx}
#   Usada em: tolerancia_curvatura (l. 1795); limites_busca_armadura (l. 1955);
#       equilibrio_pilar_padrao_acoplado (l. 2038)
#   Usada também em: sensibilidade_secao_liquida.py: area_necessaria_modelo (l. 249)
TOLERANCIA_RELATIVA_CURVATURA = 1e-9


def tolerancia_area_reta_b() -> float:
    return TOLERANCIA_AREA / 100.0


def tolerancia_curvatura(curvatura_maxima: float) -> float:
    return TOLERANCIA_RELATIVA_CURVATURA * curvatura_maxima


# [M07] Área adotada acrescida da tolerância da busca da armadura (TOLERANCIA_AREA): As,adot + tol, usada para
#   conferir que o As,calc obtido pela busca de raiz ([M02]) não excede a área adotada que atendeu à verificação
#   direta de [R09] (o As,calc pode ficar até uma tolerância acima da raiz exata).
#   Variáveis: area_adotada; TOLERANCIA_AREA.
#   Fórmula: A_{s,adot} + tol_{A}
#   Usada em: dimensionar_arranjo (l. 2464)
def folga_tolerancia_area(area_adotada: float) -> float:
    return area_adotada + TOLERANCIA_AREA


# [M08] Segmento do contorno resistente que contém cada direção psi: o contorno (pontos em ordem de alfa) deve ter
#   direções psi_k ([F70]) estritamente crescentes (contorno convexo, percorrido no sentido anti-horário de
#   psi = 0 a 90°); para cada psi, k = (busca binária de psi em psi_k) - 1, limitado a [0, n - 2], de modo que
#   psi_k <= psi <= psi_k+1; retorna os índices k e k + 1 dos extremos do segmento. Se as direções não forem
#   crescentes, a verificação é interrompida com erro.
#   Variáveis: direcoes_contorno (psi_k), direcoes (psi).
#   Usada em: verificar_envoltoria_minima (l. 2273, 2280)
#   Usada também em: validacao.py: nivel_1_flexao_obliqua (l. 1582, 1585, 1595)
def contorno_ordenado(direcoes_contorno: np.ndarray) -> bool:
    return bool(np.all(np.diff(direcoes_contorno) > 0.0))


def segmento_do_contorno(direcoes_contorno: np.ndarray, direcoes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inicio = np.clip(np.searchsorted(direcoes_contorno, direcoes) - 1, 0, direcoes_contorno.size - 2)
    return inicio, inicio + 1


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM.  Abaixo, apenas o motor de cálculo (sem fórmulas escritas diretamente).
# ═════════════════════════════════════════════════════════════════════════════════════════════════


class Material:
    """Propriedades de cálculo derivadas de fck e fyk (MPa)."""

    def __init__(self, fck_mpa: float, fyk_mpa: float = RESISTENCIA_ACO_MPA, alfa_e: float = ALFA_E_AGREGADO):
        self.fck_mpa = fck_mpa
        self.fyk_mpa = fyk_mpa
        self.fcd = resistencia_calculo_concreto(mpa_para_kn_cm2(fck_mpa))
        self.fyd = resistencia_calculo_aco(mpa_para_kn_cm2(fyk_mpa))
        self.eta_c = coeficiente_eta_c(fck_mpa)
        self.eps_c2, self.eps_cu, self.expoente = parametros_parabola_retangulo(fck_mpa)
        self.modulo_inicial = modulo_elasticidade_inicial(fck_mpa, alfa_e)
        self.tensao_pico_resistencia = tensao_pico_concreto(self.fcd, self.eta_c, False)
        self.tensao_pico_deformacao = tensao_pico_concreto(self.fcd, self.eta_c, True)

    def tensao_pico(self, curva_deformacao: bool) -> float:
        return self.tensao_pico_deformacao if curva_deformacao else self.tensao_pico_resistencia


@dataclass
class SecaoDirecao:
    """Seção 'vista' numa direção de flexão: altura na direção do momento, largura e barras."""
    altura: float
    largura: float
    profundidades: np.ndarray      # y_i medidos a partir da face comprimida
    fracoes_area: np.ndarray       # parcela de As em cada barra (soma = 1)

    @property
    def profundidade_util(self) -> float:
        return float(self.profundidades.max())

    @property
    def chave(self) -> tuple:
        return self.altura, self.largura, tuple(np.round(np.sort(self.profundidades), 6))


DIRECAO_B = "b"    # direção de b_w: flexão de 1ª ordem (e1) e lambda da matriz ([H04], [H07])
DIRECAO_H = "h"    # direção de h: só o momento mínimo
ORDEM_DIRECOES = (DIRECAO_B, DIRECAO_H)


def secao_na_direcao(largura_bw: float, altura_h: float, x: np.ndarray, y: np.ndarray, direcao: str) -> SecaoDirecao:
    """Direção 'h': momento na direção de h (profundidades = y). Direção 'b': na de b_w (profundidades = x)."""
    fracoes = fracao_area_barras_iguais(x.size)
    if direcao == DIRECAO_H:
        return SecaoDirecao(altura_h, largura_bw, np.asarray(y, dtype=float), fracoes)
    return SecaoDirecao(largura_bw, altura_h, np.asarray(x, dtype=float), fracoes)


def secao_por_camadas(altura: float, largura: float, profundidades) -> SecaoDirecao:
    """Seção com barras iguais nas profundidades y_i dadas (medidas a partir da face comprimida)."""
    profundidades = np.asarray(profundidades, dtype=float)
    return SecaoDirecao(altura, largura, profundidades, fracao_area_barras_iguais(profundidades.size))


# ----------------------------------------------------------------------------- equilíbrio da seção
def esforcos_resistentes(secao: SecaoDirecao, material: Material, area_aco: float, deformacao_topo, curvatura,
                         curva_deformacao: bool = False):
    """(N, M) internos para uma configuração de deformações (aceita vetores de eps_topo e 1/r)."""
    forca_c, momento_c = esforcos_concreto_retangular(deformacao_topo, curvatura, secao.altura, secao.largura,
                                                      material.tensao_pico(curva_deformacao), material.eps_c2,
                                                      material.expoente)
    areas = areas_barras_secao(area_aco, secao.fracoes_area)
    if DESCONTAR_AREA_BARRAS:
        forca_deslocada, momento_deslocado = esforcos_concreto_deslocado(deformacao_topo, curvatura, secao.altura,
                                                                         secao.profundidades, areas,
                                                                         material.tensao_pico(curva_deformacao),
                                                                         material.eps_c2, material.expoente)
        forca_c = descontar_parcela(forca_c, forca_deslocada)
        momento_c = descontar_parcela(momento_c, momento_deslocado)
    forca_s, momento_s = esforcos_armadura(deformacao_topo, curvatura, secao.altura, secao.profundidades, areas,
                                           material.fyd)
    return soma_concreto_aco(forca_c, forca_s), soma_concreto_aco(momento_c, momento_s)


def configuracao_no_parametro(secao: SecaoDirecao, material: Material, parametro: float) -> tuple[float, float]:
    return configuracao_estado_limite(parametro, secao.altura, secao.profundidade_util, material.eps_c2,
                                      material.eps_cu)


def forca_no_parametro(secao: SecaoDirecao, material: Material, area_aco: float, parametro: float,
                       curva_deformacao: bool = False) -> float:
    deformacao_topo, curvatura = configuracao_no_parametro(secao, material, parametro)
    return float(esforcos_resistentes(secao, material, area_aco, deformacao_topo, curvatura, curva_deformacao)[0])


def parametro_ultimo_para_forca(secao: SecaoDirecao, material: Material, area_aco: float, forca_normal: float,
                                curva_deformacao: bool = False) -> float:
    """Parâmetro t do ELU [F18] com N_R(t) = N; NaN se N está fora do intervalo resistente."""
    forca_minima = forca_no_parametro(secao, material, area_aco, 0.0, curva_deformacao)
    forca_maxima = forca_no_parametro(secao, material, area_aco, 3.0, curva_deformacao)
    if forca_normal < forca_minima or forca_normal > forca_maxima:
        return math.nan
    if forca_normal == forca_maxima:
        return 3.0
    return brentq(lambda parametro: residuo(forca_no_parametro(secao, material, area_aco, parametro,
                                                               curva_deformacao), forca_normal),
                  0.0, 3.0, xtol=TOLERANCIA_PARAMETRO)


def momento_resistente(secao: SecaoDirecao, material: Material, area_aco: float, forca_normal: float) -> float:
    """Momento resistente de cálculo M_Rd para a força normal dada (NaN se a força não é resistida)."""
    parametro = parametro_ultimo_para_forca(secao, material, area_aco, forca_normal)
    if math.isnan(parametro):
        return math.nan
    deformacao_topo, curvatura = configuracao_no_parametro(secao, material, parametro)
    return float(esforcos_resistentes(secao, material, area_aco, deformacao_topo, curvatura)[1])


def diagrama_interacao(secao: SecaoDirecao, material: Material, area_aco: float,
                       pontos: int = 301) -> tuple[np.ndarray, np.ndarray]:
    """Pares (N_Rd, M_Rd) varrendo os domínios de deformação (gráficos e verificações)."""
    configuracoes = [configuracao_no_parametro(secao, material, parametro) for parametro in np.linspace(0, 3, pontos)]
    deformacoes_topo = np.array([configuracao[0] for configuracao in configuracoes])
    curvaturas = np.array([configuracao[1] for configuracao in configuracoes])
    return esforcos_resistentes(secao, material, area_aco, deformacoes_topo, curvaturas)


def limites_busca_armadura(secao: SecaoDirecao, material: Material, forca_normal: float) -> tuple[float, float]:
    """Área para a seção resistir só à força normal (reta b) e o limite superior da busca ([F39])."""
    area_superior = armadura_limite_busca(area_secao_retangular(secao.largura, secao.altura))
    if forca_no_parametro(secao, material, 0.0, 3.0) >= forca_normal:
        return 0.0, area_superior
    if forca_no_parametro(secao, material, area_superior, 3.0) < forca_normal:
        return area_superior, area_superior
    area_inferior = brentq(lambda area: residuo(forca_no_parametro(secao, material, area, 3.0), forca_normal),
                           0.0, area_superior, xtol=tolerancia_area_reta_b())
    return area_inferior, area_superior


def armadura_flexo_compressao(secao: SecaoDirecao, material: Material, forca_normal: float,
                              momento_solicitante: float) -> float:
    """Menor As (cm²) com M_Rd(Nd; As) >= Md (sem As,min). Retorna inf se nem 8 % de Ac atende."""
    area_inferior, area_superior = limites_busca_armadura(secao, material, forca_normal)

    def folga(area: float) -> float:
        return folga_momento(momento_resistente(secao, material, area, forca_normal), momento_solicitante)

    if folga(area_superior) < 0.0:
        return math.inf
    if folga(area_inferior) >= 0.0:
        return area_inferior
    return menor_valor_que_atende(folga, area_inferior, area_superior, TOLERANCIA_AREA)


# ----------------------------------------------------------------------------- relação momento-curvatura
def deformacoes_topo_para_forca(secao: SecaoDirecao, material: Material, area_aco: float, curvaturas: np.ndarray,
                                forca_normal: float, curva_deformacao: bool) -> np.ndarray:
    """eps_topo que equilibra N para cada curvatura (bisseção vetorizada [M01]); NaN se N > N_máx."""
    curvaturas = np.asarray(curvaturas, dtype=float)
    inferior, superior = intervalo_deformacao_topo(curvaturas, secao.altura, material.eps_c2)

    def residuo_forca(deformacao_topo: np.ndarray) -> np.ndarray:
        return residuo(esforcos_resistentes(secao, material, area_aco, deformacao_topo, curvaturas,
                                            curva_deformacao)[0], forca_normal)

    return bissecao_vetorial(residuo_forca, inferior, superior, ITERACOES_BISSECAO)


def curvatura_ultima(secao: SecaoDirecao, material: Material, area_aco: float, forca_normal: float,
                     curva_deformacao: bool) -> float:
    """Curvatura da configuração de ELU [F18] com N_R = N (maior curvatura admissível a N constante)."""
    parametro = parametro_ultimo_para_forca(secao, material, area_aco, forca_normal, curva_deformacao)
    if math.isnan(parametro):
        return 0.0
    return configuracao_no_parametro(secao, material, parametro)[1]


def curva_momento_curvatura(secao: SecaoDirecao, material: Material, area_aco: float, forca_normal: float,
                            curva_deformacao: bool, pontos: int = PONTOS_CURVA) -> tuple[np.ndarray, np.ndarray]:
    curvaturas = np.linspace(0.0, curvatura_ultima(secao, material, area_aco, forca_normal, curva_deformacao), pontos)
    deformacoes_topo = deformacoes_topo_para_forca(secao, material, area_aco, curvaturas, forca_normal,
                                                   curva_deformacao)
    momentos = esforcos_resistentes(secao, material, area_aco, deformacoes_topo, curvaturas, curva_deformacao)[1]
    return curvaturas, np.nan_to_num(momentos, nan=0.0)


def momento_curva_deformacao(secao: SecaoDirecao, material: Material, area_aco: float, forca_deformacao: float,
                             curvatura: float) -> float:
    curvaturas = np.array([curvatura])
    deformacao_topo = deformacoes_topo_para_forca(secao, material, area_aco, curvaturas, forca_deformacao, True)
    return float(esforcos_resistentes(secao, material, area_aco, deformacao_topo, curvaturas, True)[1][0])


# ----------------------------------------------------------------------------- pilar-padrão acoplado a M, N, 1/r
def equilibrio_pilar_padrao_acoplado(secao: SecaoDirecao, material: Material, area_aco: float, forca_normal: float,
                                     momento_primeira_ordem: float, alfa_b: float,
                                     comprimento_equivalente: float) -> tuple[float, float] | None:
    """Menor curvatura de equilíbrio estável ([F30] = 0) e M_d,tot correspondente ([F28]); None se instável."""
    forca_deformacao = forca_normal_curva_deformacao(forca_normal)
    curvaturas, momentos = curva_momento_curvatura(secao, material, area_aco, forca_deformacao, True)
    if curvaturas[-1] <= 0.0:
        return None
    residuos = residuo_equilibrio_pilar_padrao(momentos, alfa_b, momento_primeira_ordem, forca_normal,
                                               comprimento_equivalente, curvaturas)
    indices_equilibrio = np.nonzero(residuos >= 0.0)[0]
    if indices_equilibrio.size == 0:
        return None
    primeiro = int(indices_equilibrio[0])
    if primeiro == 0:
        return 0.0, momento_total_pilar_padrao(alfa_b, momento_primeira_ordem, forca_normal,
                                               comprimento_equivalente, 0.0)

    def residuo_equilibrio(curvatura: float) -> float:
        momento = momento_curva_deformacao(secao, material, area_aco, forca_deformacao, curvatura)
        return float(residuo_equilibrio_pilar_padrao(momento, alfa_b, momento_primeira_ordem, forca_normal,
                                                     comprimento_equivalente, curvatura))

    curvatura = brentq(residuo_equilibrio, curvaturas[primeiro - 1], curvaturas[primeiro],
                       xtol=tolerancia_curvatura(curvaturas[-1]))
    return curvatura, momento_total_pilar_padrao(alfa_b, momento_primeira_ordem, forca_normal,
                                                 comprimento_equivalente, curvatura)


def armadura_pilar_padrao_acoplado(secao: SecaoDirecao, material: Material, forca_normal: float,
                                   momento_primeira_ordem: float, alfa_b: float,
                                   comprimento_equivalente: float) -> tuple[float, float, float]:
    """Menor As com equilíbrio estável e M_d,tot <= M_Rd. Retorna (As, M_d,tot, 1/r); As = inf se inviável."""
    area_inferior, area_superior = limites_busca_armadura(secao, material, forca_normal)
    memoria: dict[float, tuple[float, float, float]] = {}

    def avaliar(area: float) -> tuple[float, float, float]:
        if area not in memoria:
            equilibrio = equilibrio_pilar_padrao_acoplado(secao, material, area, forca_normal, momento_primeira_ordem,
                                                          alfa_b, comprimento_equivalente)
            momento = momento_resistente(secao, material, area, forca_normal)
            if equilibrio is None or math.isnan(momento):
                memoria[area] = (folga_momento(math.nan, momento_primeira_ordem), math.inf, math.nan)
            else:
                curvatura, momento_total = equilibrio
                memoria[area] = (folga_momento(momento, momento_total), momento_total, curvatura)
        return memoria[area]

    def folga(area: float) -> float:
        return avaliar(area)[0]

    if folga(area_superior) < 0.0:
        _, momento_total, curvatura = avaliar(area_superior)
        return math.inf, momento_total, curvatura
    if folga(area_inferior) >= 0.0:
        area = area_inferior
    else:
        area = menor_valor_que_atende(folga, area_inferior, area_superior, TOLERANCIA_AREA)
    _, momento_total, curvatura = avaliar(area)
    return area, momento_total, curvatura


# ----------------------------------------------------------------------------- análise de uma direção
@dataclass
class ResultadoDirecao:
    direcao: str
    altura: float                         # cm (dimensão na direção da flexão)
    esbeltez: float
    esbeltez_limite: float
    metodo: str
    excentricidade_primeira_ordem: float  # cm (e1 aplicado nesta direção; 0 na direção de h)
    momento_aplicado: float               # kN·cm (Nd e1, [F53])
    momento_minimo: float                 # kN·cm ([F21])
    alfa_b: float
    excentricidade_acidental: float       # cm (só com fluência)
    forca_quase_permanente: float         # kN (só com fluência)
    momento_quase_permanente: float       # kN·cm (só com fluência; N_sg e1)
    forca_euler: float                    # kN (só com fluência)
    excentricidade_fluencia: float        # cm
    momento_primeira_ordem: float         # kN·cm (M1d,A de [F55]; com fluência, [F56])
    momento_total: float                  # kN·cm
    curvatura: float                      # 1/cm
    area_necessaria: float                # cm² (equilíbrio, sem As,min)


def solicitacao_direcao(altura: float, largura: float, material: Material, forca_normal: float,
                        comprimento_equivalente: float, direcao: str, excentricidade_1: float) -> ResultadoDirecao:
    """Esforços de cálculo numa direção principal, que não dependem das barras: lambda, lambda1, método, M1d,A
    (com a fluência, se for o caso) e, nos métodos sem acoplamento, a curvatura e M_d,tot. No pilar-padrão
    acoplado a M, N, 1/r, M_d,tot e 1/r dependem da armadura e ficam como NaN. A área fica como NaN."""
    esbeltez = indice_esbeltez(comprimento_equivalente, altura)
    momento_minimo = momento_minimo_primeira_ordem(forca_normal, altura)
    momento_aplicado = momento_excentricidade(forca_normal, excentricidade_1)
    momento_a = momento_primeira_ordem_adotado(momento_aplicado, momento_minimo)
    alfa_b = coeficiente_alfa_b(momento_aplicado, momento_extremidade_b(momento_aplicado, RAZAO_MOMENTOS_EXTREMIDADE),
                                momentos_menores_que_minimo(momento_aplicado, momento_minimo))
    limite = esbeltez_limite(excentricidade_relativa(momento_aplicado, forca_normal, altura), alfa_b)
    metodo = metodo_segunda_ordem(esbeltez, limite)
    excentricidade_a = forca_sg = momento_sg = forca_e = math.nan
    excentricidade_cc = 0.0
    momento_primeira_ordem = momento_a
    if metodo == METODO_DISPENSADO:
        curvatura = 0.0
        momento_total = momento_a
    elif metodo == METODO_CURVATURA:
        nu = forca_normal_adimensional(forca_normal, area_secao_retangular(largura, altura), material.fcd)
        curvatura = curvatura_aproximada(nu, altura)
        momento_total = momento_total_pilar_padrao(alfa_b, momento_a, forca_normal, comprimento_equivalente,
                                                   curvatura)
    else:
        _, forca_gk, forca_qk = parcelas_forca_caracteristica(forca_normal, FRACAO_PERMANENTE)
        forca_sg = forca_quase_permanente(forca_gk, forca_qk, PSI2)
        momento_sg = momento_excentricidade(forca_sg, excentricidade_1)
        forca_e = forca_euler_fluencia(material.modulo_inicial, momento_inercia_retangulo(largura, altura),
                                       comprimento_equivalente)
        excentricidade_a = excentricidade_acidental(desaprumo_local(comprimento_equivalente), comprimento_equivalente)
        excentricidade_cc = excentricidade_fluencia(momento_sg, forca_sg, excentricidade_a, COEFICIENTE_FLUENCIA,
                                                    forca_e)
        momento_primeira_ordem = momento_primeira_ordem_com_fluencia(momento_a, forca_normal, excentricidade_cc)
        momento_total = curvatura = math.nan
    return ResultadoDirecao(direcao, altura, esbeltez, limite, metodo, excentricidade_1, momento_aplicado,
                            momento_minimo, alfa_b, excentricidade_a, forca_sg, momento_sg, forca_e, excentricidade_cc,
                            momento_primeira_ordem, momento_total, curvatura, math.nan)


def armadura_direcao(secao: SecaoDirecao, material: Material, forca_normal: float, comprimento_equivalente: float,
                     solicitacao: ResultadoDirecao) -> ResultadoDirecao:
    """Armadura necessária (sem As,min) da geometria de barras da seção para a solicitação da direção."""
    if solicitacao.metodo != METODO_MN1R:
        return replace(solicitacao, area_necessaria=armadura_flexo_compressao(secao, material, forca_normal,
                                                                              solicitacao.momento_total))
    if math.isinf(solicitacao.momento_primeira_ordem):
        return replace(solicitacao, momento_total=math.inf, area_necessaria=math.inf)
    area, momento_total, curvatura = armadura_pilar_padrao_acoplado(secao, material, forca_normal,
                                                                    solicitacao.momento_primeira_ordem,
                                                                    solicitacao.alfa_b, comprimento_equivalente)
    return replace(solicitacao, momento_total=momento_total, curvatura=curvatura, area_necessaria=area)


def area_atende_direcao(secao: SecaoDirecao, material: Material, forca_normal: float, comprimento_equivalente: float,
                        solicitacao: ResultadoDirecao, area_aco: float) -> bool:
    """[R09]: a área adotada, na própria geometria de barras, resiste à solicitação da direção, isto é, a folga
    de [M04] na área adotada não é negativa (M_Rd >= M_d,tot e, no método M, N, 1/r, equilíbrio estável).
    Como a folga cresce com a área (hipótese das buscas de [M02]), equivale a As,adot >= As,nec da geometria,
    com uma única avaliação da folga no lugar da busca de raiz."""
    if solicitacao.metodo != METODO_MN1R:
        return folga_momento(momento_resistente(secao, material, area_aco, forca_normal),
                             solicitacao.momento_total) >= 0.0
    if math.isinf(solicitacao.momento_primeira_ordem):
        return False
    equilibrio = equilibrio_pilar_padrao_acoplado(secao, material, area_aco, forca_normal,
                                                  solicitacao.momento_primeira_ordem, solicitacao.alfa_b,
                                                  comprimento_equivalente)
    momento = momento_resistente(secao, material, area_aco, forca_normal)
    if equilibrio is None or math.isnan(momento):
        return False
    return folga_momento(momento, equilibrio[1]) >= 0.0


def analisar_direcao(secao: SecaoDirecao, material: Material, forca_normal: float, comprimento_equivalente: float,
                     direcao: str, excentricidade_1: float) -> ResultadoDirecao:
    """M_d,tot e armadura necessária numa direção principal, com e1 aplicado no topo e na base ([H04])."""
    solicitacao = solicitacao_direcao(secao.altura, secao.largura, material, forca_normal, comprimento_equivalente,
                                      direcao, excentricidade_1)
    return armadura_direcao(secao, material, forca_normal, comprimento_equivalente, solicitacao)


# ----------------------------------------------------------------------------- envoltória mínima com 2ª ordem
def esforcos_resistentes_obliquos(largura: float, altura: float, x: np.ndarray, y: np.ndarray, area_aco: float,
                                  material: Material, deformacao_topo, curvatura,
                                  angulo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, M_h, M_b) internos na flexão oblíqua para configurações (eps_topo, 1/r) com linha neutra de
    inclinação alfa ([F63] a [F68]); aceita vetores de mesmo tamanho."""
    n_b, n_h = direcao_normal_linha_neutra(angulo)
    tensao_pico = material.tensao_pico(False)
    forca_c, momento_h_c, momento_b_c = esforcos_concreto_obliquo(deformacao_topo, curvatura, n_b, n_h, largura,
                                                                  altura, tensao_pico, material.eps_c2,
                                                                  material.expoente)
    areas = areas_barras_secao(area_aco, fracao_area_barras_iguais(x.size))
    profundidades = profundidade_obliqua(x, y, np.asarray(n_b)[..., None], np.asarray(n_h)[..., None])
    forcas = forcas_barras_obliquo(np.asarray(deformacao_topo, dtype=float)[..., None],
                                   np.asarray(curvatura, dtype=float)[..., None], profundidades, areas, material.fyd,
                                   tensao_pico, material.eps_c2, material.expoente, DESCONTAR_AREA_BARRAS)
    forca_s, momento_h_s, momento_b_s = resultantes_barras_obliquo(forcas, x, y, largura, altura)
    return (soma_concreto_aco(forca_c, forca_s), soma_concreto_aco(momento_h_c, momento_h_s),
            soma_concreto_aco(momento_b_c, momento_b_s))


def envoltoria_resistente(largura: float, altura: float, x: np.ndarray, y: np.ndarray, area_aco: float,
                          material: Material, forca_normal: float, momento_resistente_b: float,
                          momento_resistente_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Contorno (M_b, M_h) da envoltória resistente a N = Nd no primeiro quadrante ([R12] b): M_Rd,b (psi = 0), os
    pontos de ANGULOS_LINHA_NEUTRA ([N04]) e M_Rd,h (psi = 90°). Para cada alfa, o parâmetro de domínio t de [F18]
    (com a altura projetada H e a profundidade da barra mais tracionada) que dá N_R = Nd é obtido pela bisseção
    vetorizada [M01]."""
    angulos = ANGULOS_LINHA_NEUTRA
    n_b, n_h = direcao_normal_linha_neutra(angulos)
    altura_projetada = profundidade_obliqua(largura, altura, n_b, n_h)
    profundidade_util = profundidade_obliqua(x, y, n_b[:, None], n_h[:, None]).max(axis=1)

    def configuracao(parametros: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return configuracao_estado_limite(parametros, altura_projetada, profundidade_util, material.eps_c2,
                                          material.eps_cu)

    def residuo_forca(parametros: np.ndarray) -> np.ndarray:
        return residuo(esforcos_resistentes_obliquos(largura, altura, x, y, area_aco, material,
                                                     *configuracao(parametros), angulos)[0], forca_normal)

    parametros = bissecao_vetorial(residuo_forca, np.zeros(angulos.size), np.full(angulos.size, 3.0),
                                   ITERACOES_BISSECAO)
    if np.any(np.isnan(parametros)):
        raise RuntimeError("envoltória resistente: N = %.2f kN não é resistida em alguma inclinação" % forca_normal)
    _, momentos_h, momentos_b = esforcos_resistentes_obliquos(largura, altura, x, y, area_aco, material,
                                                              *configuracao(parametros), angulos)
    return (np.concatenate([[momento_resistente_b], momentos_b, [0.0]]),
            np.concatenate([[0.0], momentos_h, [momento_resistente_h]]))


@dataclass
class ResultadoEnvoltoria:
    """Verificação da envoltória mínima com 2ª ordem ([R12]) para uma armadura (arranjo e área)."""
    direcao_b: ResultadoDirecao        # solicitação só com M1d,mín na direção de b_w, com M_d,tot,mín
    direcao_h: ResultadoDirecao        # idem na direção de h
    momentos_b: np.ndarray             # contorno resistente a Nd (kN·cm), de psi = 0 (b_w) a 90° (h)
    momentos_h: np.ndarray
    fator: float                       # eta = min r_R / r_e ([F72])
    direcao_critica: float             # psi do mínimo (graus)
    atende: bool


def momento_total_minimo(secao: SecaoDirecao, material: Material, forca_normal: float, comprimento_equivalente: float,
                         solicitacao: ResultadoDirecao, area_aco: float) -> ResultadoDirecao:
    """M_d,tot,mín da direção ([R12] a): sem acoplamento, o da solicitação; no M, N, 1/r, o do equilíbrio estável
    com a armadura dada (infinito se não houver equilíbrio)."""
    if solicitacao.metodo != METODO_MN1R:
        return solicitacao
    if math.isinf(solicitacao.momento_primeira_ordem):
        return replace(solicitacao, momento_total=math.inf)
    equilibrio = equilibrio_pilar_padrao_acoplado(secao, material, area_aco, forca_normal,
                                                  solicitacao.momento_primeira_ordem, solicitacao.alfa_b,
                                                  comprimento_equivalente)
    if equilibrio is None:
        return replace(solicitacao, momento_total=math.inf)
    return replace(solicitacao, curvatura=equilibrio[0], momento_total=equilibrio[1])


def verificar_envoltoria_minima(pilar: "PilarTeste", material: Material, candidato: "Candidato",
                                solicitacoes_minimas: dict, momentos_resistentes: dict) -> ResultadoEnvoltoria:
    """[R12] (a) a (c) para a armadura do candidato."""
    direcoes = {}
    for direcao in ORDEM_DIRECOES:
        secao = secao_na_direcao(pilar.largura_bw, pilar.altura_h, candidato.x, candidato.y, direcao)
        direcoes[direcao] = momento_total_minimo(secao, material, pilar.forca_normal_calculo,
                                                 pilar.comprimento_equivalente, solicitacoes_minimas[direcao],
                                                 candidato.area)
    momentos_b, momentos_h = envoltoria_resistente(pilar.largura_bw, pilar.altura_h, candidato.x, candidato.y,
                                                   candidato.area, material, pilar.forca_normal_calculo,
                                                   momentos_resistentes[DIRECAO_B], momentos_resistentes[DIRECAO_H])
    direcoes_contorno = direcao_momento(momentos_b, momentos_h)
    if not contorno_ordenado(direcoes_contorno):
        raise RuntimeError("%s: contorno resistente com direções não crescentes ([M08])" % pilar.identificador)
    semieixo_b = direcoes[DIRECAO_B].momento_total
    semieixo_h = direcoes[DIRECAO_H].momento_total
    if not (math.isfinite(semieixo_b) and math.isfinite(semieixo_h)):
        return ResultadoEnvoltoria(direcoes[DIRECAO_B], direcoes[DIRECAO_H], momentos_b, momentos_h, 0.0, math.nan,
                                   False)
    segmentos, seguintes = segmento_do_contorno(direcoes_contorno, DIRECOES_ENVOLTORIA)
    fatores = fator_envoltoria(raio_contorno(momentos_b[segmentos], momentos_h[segmentos], momentos_b[seguintes],
                                             momentos_h[seguintes], DIRECOES_ENVOLTORIA),
                               raio_envoltoria_minima(DIRECOES_ENVOLTORIA, semieixo_b, semieixo_h))
    critico = int(np.argmin(fatores))
    return ResultadoEnvoltoria(direcoes[DIRECAO_B], direcoes[DIRECAO_H], momentos_b, momentos_h,
                               float(fatores[critico]), radianos_para_graus(DIRECOES_ENVOLTORIA[critico]),
                               bool(fatores[critico] >= FATOR_MINIMO_ENVOLTORIA))


# ----------------------------------------------------------------------------- detalhamento e dimensionamento
@dataclass(frozen=True)
class PilarTeste:
    """Entradas do Módulo 1 (Figura 3 do TCC; Figura 3.1 do PP): b_w, h, fck, fyk, Nd, e1, l_e e vinculação ([H07])."""
    identificador: str
    largura_bw: float                      # cm (dimensão na direção da flexão de 1ª ordem; na matriz, a menor)
    altura_h: float                        # cm (dimensão perpendicular à flexão de 1ª ordem; na matriz, a maior)
    fck_mpa: float
    fyk_mpa: float                         # MPa (CA-50: 500)
    forca_normal_calculo: float            # kN (Nd)
    excentricidade_primeira_ordem: float   # cm (e1, na direção de b_w, igual no topo e na base)
    comprimento_equivalente: float         # cm (l_e)
    vinculacao: str = VINCULACAO_BIAPOIADO
    arranjo: str = ARRANJO_REGRA           # ARRANJO_REGRA ([R07]) ou um arranjo fixo de [R02]
    cobrimento: float = COBRIMENTO_NOMINAL
    diametro_maximo_agregado: float = DIAMETRO_MAXIMO_AGREGADO
    taxa_armadura_maxima: float = TAXA_MAXIMA_FORA_EMENDA   # [C09] e [F38]; 8 % só na sensibilidade dos inviáveis


@dataclass
class Candidato:
    diametro: float
    diametro_estribo: float
    distancia_eixo: float
    nivel: int
    x: np.ndarray
    y: np.ndarray
    numero_barras: int
    area: float


def montar_candidato(pilar: PilarTeste, diametro: float, nivel: int) -> tuple[Candidato, tuple]:
    estribo = diametro_estribo(diametro)
    distancia = distancia_eixo_barra(pilar.cobrimento, estribo, diametro)
    x, y, faces = posicoes_barras(pilar.largura_bw, pilar.altura_h, distancia, pilar.arranjo, nivel,
                                  espacamento_maximo_eixos(min(pilar.largura_bw, pilar.altura_h)))
    return Candidato(diametro, estribo, distancia, nivel, x, y, x.size, area_barras(diametro, x.size)), faces


def gerar_candidatos(pilar: PilarTeste, area_limite: float) -> list[Candidato]:
    """Arranjos que atendem ao detalhamento ([F42]-[F49], [R02], [R03]) com área <= area_limite, ordenados por [R04]."""
    menor_dimensao = min(pilar.largura_bw, pilar.altura_h)
    espacamento_maximo = espacamento_maximo_eixos(menor_dimensao)
    candidatos = []
    for diametro in BITOLAS_LONGITUDINAIS:
        if not diametro_longitudinal_permitido(diametro, menor_dimensao):
            continue
        espacamento_livre = espacamento_livre_minimo(diametro, pilar.diametro_maximo_agregado)
        nivel = 1
        while True:
            candidato, faces = montar_candidato(pilar, diametro, nivel)
            if candidato.area > area_limite or not espacamentos_atendem(faces, diametro, espacamento_maximo,
                                                                        espacamento_livre):
                break
            candidatos.append(candidato)
            nivel += 1
    candidatos.sort(key=lambda candidato: chave_escolha_detalhamento(candidato.area, candidato.numero_barras,
                                                                     candidato.diametro))
    return candidatos


def candidato_referencia(pilar: PilarTeste) -> Candidato:
    """Geometria de nível [R11] do arranjo de pilar.arranjo (menor bitola permitida, número mínimo de barras por
    face); com ARRANJO_REFERENCIA, é a geometria de referência de [R11]."""
    menor_dimensao = min(pilar.largura_bw, pilar.altura_h)
    diametro = next(bitola for bitola in BITOLAS_LONGITUDINAIS if diametro_longitudinal_permitido(bitola, menor_dimensao))
    return montar_candidato(pilar, diametro, NIVEL_REFERENCIA)[0]


@dataclass
class ResumoArranjo:
    """Resultado de um arranjo de [R07], com a prova do detalhamento ([R05] e [R09])."""
    arranjo: str
    status: str
    area_calculada: float                   # cm² ([R08]: do adotado; sem arranjo, da geometria de nível [R11])
    candidatos: int                         # arranjos com As,adot <= As,max que atendem ao espaçamento ([R02], [R03])
    candidato_maior_area: Candidato | None  # sem arranjo que atenda: o candidato de maior As,adot
    area_calculada_maior_area: float        # cm² ([R08] na geometria desse candidato; > As,adot, por [R09])


@dataclass
class ResultadoPilar:
    pilar: PilarTeste                       # entradas com o arranjo efetivamente analisado
    forca_normal: float                     # kN
    comprimento_equivalente: float          # cm
    area_concreto: float                    # cm²
    area_minima: float                      # cm²
    area_maxima: float                      # cm²
    esbeltez_b: float                       # lambda na direção de b_w ([F19]; a da matriz)
    esbeltez_h: float                       # lambda na direção de h ([F19])
    nu: float                               # [F25]
    direcao_b: ResultadoDirecao | None = None
    direcao_h: ResultadoDirecao | None = None
    area_calculada: float = math.nan        # [R08] para o arranjo adotado
    area_referencia: float = math.nan       # [R08] na geometria de referência de [R11]
    direcao_b_referencia: ResultadoDirecao | None = None   # direção de b_w na geometria de referência de [R11]
    distancia_eixo_referencia: float = math.nan            # d' da geometria de referência (cm)
    candidato: Candidato | None = None
    status: str = ""
    arranjos_avaliados: tuple = ()          # ordem de [R07] efetivamente percorrida
    avaliacoes: int = 0                     # geometrias analisadas (desempenho)
    momentos_resistentes: dict = field(default_factory=dict)
    envoltoria: ResultadoEnvoltoria | None = None          # [R12] com a armadura adotada
    candidato_elipse: Candidato | None = None              # [R12] d): primeiro arranjo que engloba a elipse
    envoltoria_elipse: ResultadoEnvoltoria | None = None   # [R12] para o candidato_elipse
    resumo_arranjo: ResumoArranjo | None = None             # deste arranjo
    resumos_arranjos: tuple = ()                            # dos arranjos percorridos por [R07], na ordem


def dimensionar_arranjo(pilar: PilarTeste, material: Material) -> ResultadoPilar:
    """Dimensiona o pilar com o arranjo fixado em pilar.arranjo: As,calc nas duas direções e detalhamento."""
    area_concreto = area_secao_retangular(pilar.largura_bw, pilar.altura_h)
    forca_normal = pilar.forca_normal_calculo
    comprimento = pilar.comprimento_equivalente
    resultado = ResultadoPilar(pilar, forca_normal, comprimento, area_concreto,
                               armadura_minima(forca_normal, material.fyd, area_concreto),
                               armadura_maxima(area_concreto, pilar.taxa_armadura_maxima),
                               indice_esbeltez(comprimento, pilar.largura_bw),
                               indice_esbeltez(comprimento, pilar.altura_h),
                               forca_normal_adimensional(forca_normal, area_concreto, material.fcd),
                               arranjos_avaliados=(pilar.arranjo,))
    excentricidades = {DIRECAO_B: pilar.excentricidade_primeira_ordem, DIRECAO_H: 0.0}
    dimensoes = {DIRECAO_B: (pilar.largura_bw, pilar.altura_h), DIRECAO_H: (pilar.altura_h, pilar.largura_bw)}
    solicitacoes = {direcao: solicitacao_direcao(*dimensoes[direcao], material, forca_normal, comprimento, direcao,
                                                 excentricidades[direcao]) for direcao in ORDEM_DIRECOES}
    memoria: dict[tuple, ResultadoDirecao] = {}
    memoria_verificacao: dict[tuple, bool] = {}

    def secao(candidato: Candidato, direcao: str) -> SecaoDirecao:
        return secao_na_direcao(pilar.largura_bw, pilar.altura_h, candidato.x, candidato.y, direcao)

    def analisar(candidato: Candidato, direcao: str) -> ResultadoDirecao:
        secao_direcao = secao(candidato, direcao)
        chave = (direcao,) + secao_direcao.chave
        if chave not in memoria:
            memoria[chave] = armadura_direcao(secao_direcao, material, forca_normal, comprimento,
                                              solicitacoes[direcao])
            resultado.avaliacoes += 1
        return memoria[chave]

    def area_necessaria(candidato: Candidato) -> tuple[float, ResultadoDirecao, ResultadoDirecao]:
        direcao_b = analisar(candidato, DIRECAO_B)
        direcao_h = analisar(candidato, DIRECAO_H)
        return (area_calculada_pilar(direcao_b.area_necessaria, direcao_h.area_necessaria, resultado.area_minima),
                direcao_b, direcao_h)

    def candidato_atende(candidato: Candidato) -> bool:
        """[R08] e [R09] na própria geometria: As,adot >= As,mín e a área adotada resiste nas duas direções."""
        if not detalhamento_atende(candidato.area, resultado.area_minima):
            return False
        for direcao in ORDEM_DIRECOES:
            secao_direcao = secao(candidato, direcao)
            chave = (direcao, candidato.area) + secao_direcao.chave
            if chave not in memoria_verificacao:
                memoria_verificacao[chave] = area_atende_direcao(secao_direcao, material, forca_normal, comprimento,
                                                                 solicitacoes[direcao], candidato.area)
                resultado.avaliacoes += 1
            if not memoria_verificacao[chave]:
                return False
        return True

    def momentos_resistentes(candidato: Candidato) -> dict:
        return {direcao: momento_resistente(secao(candidato, direcao), material, candidato.area, forca_normal)
                for direcao in ORDEM_DIRECOES}

    referencia = candidato_referencia(replace(pilar, arranjo=ARRANJO_REFERENCIA))
    resultado.area_referencia, resultado.direcao_b_referencia, _ = area_necessaria(referencia)
    resultado.distancia_eixo_referencia = referencia.distancia_eixo
    resultado.area_calculada, resultado.direcao_b, resultado.direcao_h = area_necessaria(candidato_referencia(pilar))
    candidatos = gerar_candidatos(pilar, resultado.area_maxima)
    posicao_adotada = len(candidatos)
    for posicao, candidato in enumerate(candidatos):
        if candidato_atende(candidato):
            area, direcao_b, direcao_h = area_necessaria(candidato)
            if not detalhamento_atende(folga_tolerancia_area(candidato.area), area):
                raise RuntimeError("%s: a folga não cresce com a área (As,adot = %.4f cm² atende, mas As,calc = "
                                   "%.4f cm²); a hipótese de [M02] falhou" % (pilar.identificador, candidato.area, area))
            resultado.area_calculada, resultado.direcao_b, resultado.direcao_h = area, direcao_b, direcao_h
            resultado.candidato = candidato
            posicao_adotada = posicao
            break
    resultado.status = classificar_dimensionamento(resultado.area_calculada, resultado.area_maxima,
                                                   resultado.candidato is not None)
    if resultado.candidato is None:
        maior = next(reversed(candidatos), None)
        resultado.resumo_arranjo = ResumoArranjo(pilar.arranjo, resultado.status, resultado.area_calculada,
                                                 len(candidatos), maior,
                                                 area_necessaria(maior)[0] if maior is not None else math.nan)
        return resultado
    resultado.resumo_arranjo = ResumoArranjo(pilar.arranjo, resultado.status, resultado.area_calculada,
                                             len(candidatos), None, math.nan)
    resultado.momentos_resistentes = momentos_resistentes(resultado.candidato)
    solicitacoes_minimas = {DIRECAO_B: solicitacao_direcao(pilar.largura_bw, pilar.altura_h, material, forca_normal,
                                                           comprimento, DIRECAO_B, EXCENTRICIDADE_ENVOLTORIA_MINIMA),
                            DIRECAO_H: solicitacao_direcao(pilar.altura_h, pilar.largura_bw, material, forca_normal,
                                                           comprimento, DIRECAO_H, EXCENTRICIDADE_ENVOLTORIA_MINIMA)}
    resultado.envoltoria = verificar_envoltoria_minima(pilar, material, resultado.candidato, solicitacoes_minimas,
                                                       resultado.momentos_resistentes)
    if resultado.envoltoria.atende:
        resultado.candidato_elipse, resultado.envoltoria_elipse = resultado.candidato, resultado.envoltoria
    else:
        for candidato in candidatos[posicao_adotada:]:
            if candidato is resultado.candidato or not candidato_atende(candidato):
                continue
            envoltoria = verificar_envoltoria_minima(pilar, material, candidato, solicitacoes_minimas,
                                                     momentos_resistentes(candidato))
            resultado.avaliacoes += 1
            if envoltoria.atende:
                resultado.candidato_elipse, resultado.envoltoria_elipse = candidato, envoltoria
                break
    return resultado


def dimensionar_pilar(pilar: PilarTeste) -> ResultadoPilar:
    """Módulos 1 a 6 da Figura 3 do TCC (Figura 3.1 do PP): valida as entradas e dimensiona; com ARRANJO_REGRA
    aplica [R07] (duas faces; perímetro só se duas faces for inviável). Se nenhum arranjo for viável, devolve o
    resultado do arranjo prioritário (duas faces), com a situação de inviabilidade."""
    validar_entradas(pilar.largura_bw, pilar.altura_h, pilar.forca_normal_calculo, pilar.excentricidade_primeira_ordem,
                     pilar.comprimento_equivalente, pilar.vinculacao)
    material = Material(pilar.fck_mpa, pilar.fyk_mpa)
    if pilar.arranjo != ARRANJO_REGRA:
        return dimensionar_arranjo(pilar, material)
    resultados = []
    for arranjo in ORDEM_ARRANJOS_REGRA:
        resultado = dimensionar_arranjo(replace(pilar, arranjo=arranjo), material)
        resultados.append(resultado)
        if resultado.status == STATUS_OK:
            break
    escolhido = resultados[-1] if resultados[-1].status == STATUS_OK else resultados[0]
    escolhido.arranjos_avaliados = tuple(resultado.pilar.arranjo for resultado in resultados)
    escolhido.resumos_arranjos = tuple(resultado.resumo_arranjo for resultado in resultados)
    escolhido.avaliacoes = sum(resultado.avaliacoes for resultado in resultados)
    return escolhido
