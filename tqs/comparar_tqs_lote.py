"""Comparação rotina Python × TQS Pilar nos 212 pilares modeláveis (lote TCCL01 a TCCL16, 04/11/2026).

Lê, para cada edifício do lote, as saídas do TQS copiadas pelo script tqs_01_lote.py (pasta resultados_lote) e os
resultados da rotina (saida/resultados_matriz.csv), e grava uma linha por pilar em comparacao_tqs_lote.csv.

Conferências da entrada (o que o TQS recebeu é o que a matriz pede):
  - pé direito duplo desligado (PEDDUP.DAT);
  - cargas no topo iguais às da entrada arredondadas a 0,01 tf e 0,01 tf·m; peso próprio somado no lance igual a
    25·b·h·ℓ; MX, MZ, FX e FY nulos; MY igual no topo e na base (PRJ-0001.TEP);
  - índice de esbeltez do TQS (SELEC.LST) igual ao da matriz.

Camada 1 (resultados finais): armadura adotada e área necessária da rotina contra as do TQS (SELEC.LST, LISPIL.LST),
e esforços de cálculo da rotina na direção de b_w contra os do TQS na seção intermediária (Pilar.rep).

Camada 2 (mesmo esforço, mesma seção): os esforços que o TQS usou em cada caso, com as barras que ele adotou
(posições do Pilar.rep), passados pelo núcleo de seção da rotina (pilares_nbr6118.py). Calcula-se a relação Sd/Rd
pelo raio da origem no espaço (N, M_b, M_h), que é a medida que o TQS grava em Pilar.rep, com seção bruta e com
seção líquida; e, para o caso que governou, a área necessária com o arranjo do TQS, contra o ASnec da LISPIL.LST.

Execução:  python3 comparar_tqs_lote.py
"""
import csv
import json
import math
import re
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "analise"))
import pilares_nbr6118 as rotina  # noqa: E402
from ler_pilar_rep import ler_pilar_rep, pilares_por_nome  # noqa: E402

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — constantes e fórmulas
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [D01] Arquivos e pastas.  Usada em: principal; ler_edificio
PASTA_TQS = Path(__file__).resolve().parent
PASTA_RESULTADOS_LOTE = PASTA_TQS / "resultados_lote"
ARQUIVO_ENTRADA_TQS = PASTA_TQS / "entrada_tqs_lote.csv"
ARQUIVO_MATRIZ = Path(__file__).resolve().parent.parent / "saida" / "resultados_matriz.csv"
ARQUIVO_SAIDA = PASTA_TQS / "comparacao_tqs_lote.csv"
ARQUIVO_CAMADA_2 = PASTA_TQS / "comparacao_tqs_lote_camada2.json"
PREFIXO_EDIFICIO_LOTE = "TCCL"

# [D02] Aço (critérios do TQS conferidos na LISPIL.LST: CA-50).  Usada em: avaliar_pilar (Material)
FYK_MPA = 500.0

# [D03] Nomes dos casos de combinação do TQS (PRJ-0001.TEP e Pilar.rep): só carga permanente e permanente mais
#   variável; os casos "_V" repetem os mesmos esforços para o vento, que não existe no modelo.
#   Usada em: ler_tep; casos_distintos; avaliar_pilar
CASO_PERMANENTE = "ELU1/PERM/PP+PERM"
CASO_PERMANENTE_VARIAVEL = "ELU1/PERMACID/PP+PERM+ACID"

# [D08] Código do SELEC.LST para os pilares sem 2ª ordem (λ ≤ λ1), nos quais só a seção do topo é verificada.
#   Usada em: principal (seção comparada)
METODO_TQS_SEM_SEGUNDA_ORDEM = "----"

# [D09] Código do SELEC.LST para o pilar-padrão acoplado a diagramas N, M, 1/r. Nesses pilares, o momento da seção
#   intermediária gravado no Pilar.rep é o prévio, pelo pilar-padrão com curvatura aproximada (conferido nos 62
#   pilares), e não o da verificação final, que não fica nos arquivos: a coluna de diferença do momento total fica
#   vazia, e a comparação desses pilares está em camada2_nm1r_tqs.py.  Usada em: principal
METODO_TQS_ACOPLADO = "N,M,1/r"

# [D04] Tolerâncias das conferências da entrada: o TEP imprime 3 casas em tf e tf·m.
#   Usada em: conferir_tep
TOLERANCIA_TEP_TF = 0.0015

# [D05] Busca do fator de escala do raio (N, M_b, M_h) até a superfície resistente: varredura geométrica entre os
#   limites, parando no primeiro ponto já fora da superfície, e refinamento de Brent nesse intervalo.
#   Usada em: relacao_sd_rd
FATOR_ESCALA_INFERIOR = 0.05
FATOR_ESCALA_SUPERIOR = 200.0
PONTOS_VARREDURA_ESCALA = 30
TOLERANCIA_FATOR_ESCALA = 1.0e-7

# [D06] Intervalo e tolerância da busca da área de aço (cm²).  Usada em: area_necessaria_obliqua
AREA_BUSCA_INFERIOR_CM2 = 0.0
AREA_BUSCA_SUPERIOR_CM2 = 200.0
TOLERANCIA_AREA_CM2 = 1.0e-4

# [D07] Processos paralelos.  Usada em: principal
PROCESSOS = 2


# [C01] Conversão de força do Pilar.rep (N) para a rotina (kN).  Variáveis: forca_n.  Fórmula: F_kN = F_N / 1000
#   Usada em: esforcos_caso
def newton_para_kn(forca_n: float) -> float:
    return forca_n / 1000.0


# [C02] Conversão de momento do Pilar.rep (N·m) para a rotina (kN·cm).  Variáveis: momento_n_m.
#   Fórmula: M_kN·cm = M_N·m / 1000 · 100 = M_N·m / 10
#   Usada em: esforcos_caso
def newton_metro_para_kn_cm(momento_n_m: float) -> float:
    return momento_n_m / 10.0


# [C03] Conversão de comprimento do Pilar.rep (m) para a rotina (cm).  Variáveis: valor_m.  Fórmula: v_cm = 100 · v_m
#   Usada em: barras_tqs
def metro_para_cm(valor_m: float) -> float:
    return 100.0 * valor_m


# [C04] Conversão de força do TQS (tf) para kN e de kN para tf (unidade de projeto kgf = 10 N; 1 tf = 10 kN).
#   Variáveis: valor_tf, valor_kn.  Fórmulas: F_kN = 10 · F_tf;  F_tf = F_kN / 10
#   Usada em: conferir_tep (peso próprio esperado em tf); esforcos_lispil
def tf_para_kn(valor_tf: float) -> float:
    return 10.0 * valor_tf


def kn_para_tf(valor_kn: float) -> float:
    return valor_kn / 10.0


# [C05] Área de n barras de mesmo diâmetro (cm²).  Variáveis: numero_barras, diametro_mm.
#   Fórmula: A_s = n · π · (φ/10)² / 4
#   Usada em: barras_tqs; ler_selec (área exata do arranjo adotado)
def area_barras_cm2(numero_barras: int, diametro_mm: float) -> float:
    return numero_barras * math.pi * (diametro_mm / 10.0) ** 2 / 4.0


# [C06] Arredondamento da carga lançada, como o TQS Formas faz ao ler o Modelador (0,01 tf; 0,01 tf·m), com
#   meia unidade arredondada para cima.  Variáveis: valor.  Fórmula: v_r = ⌊100 · v + 0,5⌋ / 100
#   Usada em: conferir_tep
def arredondar_centesimo(valor: float) -> float:
    return math.floor(100.0 * valor + 0.5) / 100.0


# [C07] Módulo do momento no plano (M_b, M_h).  Variáveis: momento_b, momento_h.  Fórmula: r_S = √(M_b² + M_h²)
#   Usada em: relacao_sd_rd; area_necessaria_obliqua
def raio_solicitante(momento_b: float, momento_h: float) -> float:
    return math.hypot(momento_b, momento_h)


# [C08] Folga do raio resistente na direção do momento, com N fixo.  Variáveis: raio_resistente, raio_alvo.
#   Fórmula: folga = r_R(ψ, N) − r_S
#   Usada em: area_necessaria_obliqua
def folga_raio(raio_resistente: float, raio_alvo: float) -> float:
    return raio_resistente - raio_alvo


# [C09] Resíduo do fator de escala k do raio (N, M_b, M_h): o ponto k·(N, M_b, M_h) está sobre a superfície
#   resistente quando k · r_S = r_R(ψ, k·N).  Variáveis: fator, raio_alvo, raio_resistente_escalado.
#   Fórmula: g(k) = k · r_S − r_R(ψ, k · N)
#   Usada em: relacao_sd_rd
def residuo_escala(fator: float, raio_alvo: float, raio_resistente_escalado: float) -> float:
    return fator * raio_alvo - raio_resistente_escalado


# [C10] Relação Sd/Rd pelo raio da origem: inverso do fator de escala que leva o esforço até a superfície.
#   Variáveis: fator_escala (k*).  Fórmula: Sd/Rd = 1 / k*
#   Usada em: relacao_sd_rd
def sd_rd_do_fator(fator_escala: float) -> float:
    return 1.0 / fator_escala


# [C11] Diferença percentual com sinal, da rotina em relação ao TQS.  Variáveis: valor_rotina, valor_tqs.
#   Fórmula: Δ = 100 · (X_rotina − X_TQS) / X_TQS
#   Usada em: avaliar_pilar (todas as colunas "dif.")
def diferenca_percentual(valor_rotina: float, valor_tqs: float) -> float:
    return 100.0 * (valor_rotina - valor_tqs) / valor_tqs


# [C12] Área necessária comparável ao ASnec do TQS: a da flexão composta, sem ser menor que a armadura mínima
#   (NBR 6118, 17.3.5.3.1; o TQS imprime no ASnec o maior dos dois).  Variáveis: area_calculada, area_minima.
#   Fórmula: A_s,nec = max(A_s,calc ; A_s,mín)
#   Usada em: principal (coluna "As,nec rotina arranjo TQS bruta")
def area_com_minima(area_calculada: float, area_minima: float) -> float:
    return max(area_calculada, area_minima)


# [C13] Momento de cálculo do TQS na direção de b_w comparável ao M_d,tot da rotina: na seção intermediária, o
#   total gravado (BDM_MDTOTx = MIDDLE_MSDX, já com o maior entre M1d e M1d,mín e com a 2ª ordem); nos pilares sem
#   2ª ordem, só há a seção do topo, e o comparável é o maior entre o momento do pórtico e o mínimo de 1ª ordem
#   (TOP_MSDX e TOP_MSD_BDMIMX).  Variáveis: caso (Pilar.rep, N·m), secao.
#   Fórmula: M_b = |MIDDLE_MSDX| (com 2ª ordem);  M_b = max(|TOP_MSDX| ; TOP_MSD_BDMIMX) (sem 2ª ordem)
#   Usada em: principal (coluna "Msd,b TQS")
def momento_b_comparavel_tqs(caso: dict, secao: str) -> float:
    if secao == "MIDDLE":
        return abs(newton_metro_para_kn_cm(caso["MIDDLE_MSDX"]))
    return max(abs(newton_metro_para_kn_cm(caso["TOP_MSDX"])), abs(newton_metro_para_kn_cm(caso["TOP_MSD_BDMIMX"])))


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════

PADRAO_NUMERO = re.compile(r"-?(?:\d+\.\d*|\.\d+|\d+)(?:E[+-]?\d+)?", re.IGNORECASE)
# Nome do pilar no início de cada bloco das listagens: P001 a P324 (matriz) e M01 a M09 (teste do momento mínimo).
PADRAO_NOME_PILAR = re.compile(r"PILAR:([PM]\d+)")


def ler_texto(caminho: Path) -> str:
    return caminho.read_bytes().decode("latin-1")


def blocos_por_pilar(texto: str) -> dict[str, list[str]]:
    """Divide uma listagem do TQS em blocos que começam na linha 'PILAR:Pnnn' (ou 'PILAR:Mnn')."""
    blocos: dict[str, list[str]] = {}
    atual = None
    for linha in texto.splitlines():
        achado = PADRAO_NOME_PILAR.search(linha)
        if achado:
            atual = achado.group(1)
            blocos[atual] = []
            continue
        if atual is not None:
            blocos[atual].append(linha)
    return blocos


def ler_selec(caminho: Path) -> dict[str, dict]:
    """Armadura adotada, índice de esbeltez e método de 2ª ordem de cada pilar (SELEC.LST)."""
    resultado = {}
    for pilar, linhas in blocos_por_pilar(ler_texto(caminho)).items():
        for linha in linhas:
            campos = linha.split()
            if len(campos) > 18 and campos[1] == "PILARES":
                numero_barras = int(campos[5])
                diametro = float(campos[6])
                resultado[pilar] = {"numero_barras": numero_barras, "diametro_mm": diametro,
                                    "pe_direito_duplo": campos[7] + campos[8],
                                    "area_adotada": area_barras_cm2(numero_barras, diametro),
                                    "lambda": float(campos[17]), "metodo": campos[-1]}
                break
    return resultado


def ler_lispil(caminho: Path) -> dict[str, dict]:
    """Arranjos testados (bitola, barras, ASnec), o adotado (linha 'L.') e o caso impresso de cada pilar."""
    resultado = {}
    for pilar, linhas in blocos_por_pilar(ler_texto(caminho)).items():
        arranjos = []
        adotado = None
        governante = None
        esbeltez = None
        for linha in linhas:
            partes = linha.split("|")
            if len(partes) < 4 or "....." in linha or not linha.startswith("|"):
                continue
            campos = partes[1].split()
            if not campos or campos[0] == "LANCE":
                continue
            marcado = campos[0] == "L."
            numeros = campos[6:] if marcado else campos
            if len(numeros) < 7 or not all(PADRAO_NUMERO.fullmatch(valor) for valor in numeros):
                continue
            arranjo = {"diametro_mm": float(numeros[0]), "numero_barras": int(numeros[2]),
                       "barras_lado_h": int(numeros[3]), "barras_lado_b": int(numeros[4]),
                       "area": float(numeros[5]), "area_necessaria": float(numeros[7]) if len(numeros) > 7 else None}
            arranjos.append(arranjo)
            if marcado:
                adotado = arranjo
            if esbeltez is None:
                valores_esbeltez = PADRAO_NUMERO.findall(partes[2])
                esbeltez = float(valores_esbeltez[1]) if len(valores_esbeltez) > 1 else None
                valores_esforco = PADRAO_NUMERO.findall(partes[3])
                governante = tuple(float(valor) for valor in valores_esforco[:3]) if len(valores_esforco) >= 3 else None
        resultado[pilar] = {"arranjos": arranjos, "adotado": adotado, "governante": governante, "lambda": esbeltez}
    return resultado


def ler_tep(caminho: Path) -> dict[str, list[dict]]:
    """Esforços transferidos do pórtico (PRJ-0001.TEP): por pilar, a lista de casos com as linhas da base e do
    topo (MX, MY, FZ, MZ, FX, FY, em tf e tf·m). A ordem dos casos é a da lista CASOS do arquivo."""
    texto = ler_texto(caminho)
    nomes_casos = re.findall(r"^\s*0\s+'([^']+)'", texto, flags=re.MULTILINE)
    resultado: dict[str, list[dict]] = {}
    atual = None
    linhas_caso: list[list[float]] = []
    for linha in texto.splitlines():
        achado = re.match(r"^P0\s+'([PM]\d+)", linha)
        if achado:
            atual = achado.group(1)
            resultado[atual] = []
            continue
        if atual is None:
            continue
        if linha.strip().startswith("CASO"):
            linhas_caso = []
            continue
        valores = [float(valor) for valor in PADRAO_NUMERO.findall(linha)]
        if len(valores) == 7:
            valores = valores[1:]
        if len(valores) == 6:
            linhas_caso.append(valores)
            if len(linhas_caso) == 2:
                indice = len(resultado[atual])
                resultado[atual].append({"nome": nomes_casos[indice] if indice < len(nomes_casos) else "?",
                                         "base": linhas_caso[0], "topo": linhas_caso[1]})
    return resultado


def ler_peddup(caminho: Path) -> dict[str, str]:
    """Código de pé direito duplo de cada pilar nas direções x e y (último dígito: 0 = não)."""
    resultado: dict[str, str] = {}
    for linha in ler_texto(caminho).splitlines():
        achado = re.match(r"^\s*\d+\s+([PM]\d+)\s+(\d+)\s*$", linha)
        if achado:
            resultado[achado.group(1)] = resultado.get(achado.group(1), "") + achado.group(2)[-1]
    return resultado


def barras_tqs(trecho: dict) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Posições das barras do arranjo adotado pelo TQS no sistema da rotina (x ao longo de b_w, y ao longo de h):
    no Pilar.rep, PTOY corre ao longo de B (b_w) e PTOX ao longo de H."""
    pontos = [valor for chave, valor in sorted(trecho.items()) if chave.startswith("REINFORCEMENT_PTO")]
    x = np.array([metro_para_cm(ponto["ARMD_PTOY"]) for ponto in pontos])
    y = np.array([metro_para_cm(ponto["ARMD_PTOX"]) for ponto in pontos])
    diametro_mm = 1000.0 * pontos[0]["ARMD_PHI"]
    return x, y, diametro_mm, area_barras_cm2(len(pontos), diametro_mm)


def casos_distintos(trecho: dict) -> list[dict]:
    casos = [valor for chave, valor in trecho.items() if chave.startswith("INTERNALFORCES_CASE[")]
    return [caso for caso in casos if caso.get("CASE_NAME") in (CASO_PERMANENTE, CASO_PERMANENTE_VARIAVEL)]


def esforcos_caso(caso: dict, secao: str) -> tuple[float, float, float]:
    """(N, M_b, M_h) em kN e kN·cm; M_b = MSDX (momento que flete na direção de b_w), M_h = MSDY."""
    return (newton_para_kn(caso["TOP_NSD"]), abs(newton_metro_para_kn_cm(caso[secao + "_MSDX"])),
            abs(newton_metro_para_kn_cm(caso[secao + "_MSDY"])))


def raio_resistente_na_direcao(largura, altura, x, y, area, material, forca_normal, direcao) -> float:
    secao_b = rotina.secao_na_direcao(largura, altura, x, y, rotina.DIRECAO_B)
    secao_h = rotina.secao_na_direcao(largura, altura, x, y, rotina.DIRECAO_H)
    momento_b = rotina.momento_resistente(secao_b, material, area, forca_normal)
    momento_h = rotina.momento_resistente(secao_h, material, area, forca_normal)
    if math.isnan(momento_b) or math.isnan(momento_h):
        return -math.inf
    try:
        contorno_b, contorno_h = rotina.envoltoria_resistente(largura, altura, x, y, area, material, forca_normal,
                                                              momento_b, momento_h)
    except RuntimeError:
        return -math.inf
    direcoes = rotina.direcao_momento(contorno_b, contorno_h)
    inicio, fim = rotina.segmento_do_contorno(direcoes, np.array([direcao]))
    return float(rotina.raio_contorno(contorno_b[inicio], contorno_h[inicio], contorno_b[fim], contorno_h[fim],
                                      np.array([direcao]))[0])


def relacao_sd_rd(largura, altura, x, y, area, material, forca_normal, momento_b, momento_h) -> float:
    direcao = float(rotina.direcao_momento(momento_b, momento_h))
    alvo = raio_solicitante(momento_b, momento_h)

    def residuo(fator: float) -> float:
        raio = raio_resistente_na_direcao(largura, altura, x, y, area, material, fator * forca_normal, direcao)
        return residuo_escala(fator, alvo, raio if math.isfinite(raio) else -1.0e12)

    fatores = np.geomspace(FATOR_ESCALA_INFERIOR, FATOR_ESCALA_SUPERIOR, PONTOS_VARREDURA_ESCALA)
    anterior = fatores[0]
    if residuo(anterior) >= 0.0:
        return math.nan
    for fator in fatores[1:]:
        if residuo(fator) >= 0.0:
            return sd_rd_do_fator(brentq(residuo, anterior, fator, xtol=TOLERANCIA_FATOR_ESCALA))
        anterior = fator
    return math.nan


def area_necessaria_obliqua(largura, altura, x, y, material, forca_normal, momento_b, momento_h) -> float:
    direcao = float(rotina.direcao_momento(momento_b, momento_h))
    alvo = raio_solicitante(momento_b, momento_h)

    def folga(area: float) -> float:
        raio = raio_resistente_na_direcao(largura, altura, x, y, area, material, forca_normal, direcao)
        return folga_raio(raio, alvo) if math.isfinite(raio) else -alvo

    if folga(AREA_BUSCA_INFERIOR_CM2) >= 0.0:
        return AREA_BUSCA_INFERIOR_CM2
    return rotina.menor_valor_que_atende(folga, AREA_BUSCA_INFERIOR_CM2, AREA_BUSCA_SUPERIOR_CM2,
                                         TOLERANCIA_AREA_CM2)


def conferir_tep(casos_tep: list[dict], entrada: dict) -> tuple[bool, str]:
    """Confere o que o TQS transferiu ao pilar com o que a entrada pede (arredondado como o TQS Formas)."""
    forca_g = arredondar_centesimo(float(entrada["fz_permanente_tf"]))
    forca_q = arredondar_centesimo(float(entrada["fz_acidental_tf"]))
    momento_g = arredondar_centesimo(float(entrada["my_permanente_tfm"]))
    momento_q = arredondar_centesimo(float(entrada["my_acidental_tfm"]))
    peso_proprio_tf = kn_para_tf(float(entrada["peso_proprio_kn"]))
    esperado = {CASO_PERMANENTE: (forca_g, momento_g), CASO_PERMANENTE_VARIAVEL: (forca_g + forca_q, momento_g + momento_q)}
    problemas = []
    for caso in casos_tep:
        chave = caso["nome"].replace("PP_V+PERM_V+ACID_V", "PP+PERM+ACID").replace("PP_V+PERM_V", "PP+PERM")
        forca_esperada, momento_esperado = esperado[chave]
        momento_x_b, momento_y_b, forca_b, momento_z_b, forca_x_b, forca_y_b = caso["base"]
        momento_x_t, momento_y_t, forca_t, momento_z_t, forca_x_t, forca_y_t = caso["topo"]
        if max(abs(momento_x_b), abs(momento_x_t), abs(momento_z_b), abs(momento_z_t), abs(forca_x_b),
               abs(forca_x_t), abs(forca_y_b), abs(forca_y_t)) > TOLERANCIA_TEP_TF:
            problemas.append("%s: MX/MZ/FX/FY não nulos" % caso["nome"])
        if abs(momento_y_b - momento_y_t) > TOLERANCIA_TEP_TF or abs(momento_y_t - momento_esperado) > TOLERANCIA_TEP_TF:
            problemas.append("%s: MY %.3f/%.3f, esperado %.2f" % (caso["nome"], momento_y_t, momento_y_b, momento_esperado))
        if abs(forca_t - forca_esperada) > TOLERANCIA_TEP_TF:
            problemas.append("%s: FZ topo %.3f, esperado %.2f" % (caso["nome"], forca_t, forca_esperada))
        if abs((forca_b - forca_t) - peso_proprio_tf) > TOLERANCIA_TEP_TF:
            problemas.append("%s: FZ base−topo %.3f, PP %.4f" % (caso["nome"], forca_b - forca_t, peso_proprio_tf))
    return not problemas, "; ".join(problemas)


def ler_edificio(nome: str) -> dict:
    pasta = PASTA_RESULTADOS_LOTE / nome / "PILAR"
    return {"selec": ler_selec(pasta / "SELEC.LST"), "lispil": ler_lispil(pasta / "LISPIL.LST"),
            "tep": ler_tep(pasta / "PRJ-0001.TEP"), "peddup": ler_peddup(pasta / "PEDDUP.DAT"),
            "rep": pilares_por_nome(ler_pilar_rep(pasta / "Pilar.rep"))}


def avaliar_pilar(dados: dict) -> dict:
    """Camada 2 de um pilar (executada em paralelo): Sd/Rd da rotina com os esforços e as barras do TQS."""
    largura, altura, fck = dados["largura"], dados["altura"], dados["fck"]
    material = rotina.Material(fck, FYK_MPA)
    trecho = dados["trecho"]
    x, y, _, area = barras_tqs(trecho)
    linhas = []
    for caso in casos_distintos(trecho):
        for secao in ("MIDDLE", "TOP"):
            chave_sdrd = secao + "_SDRD"
            if chave_sdrd not in caso:
                continue
            forca, momento_b, momento_h = esforcos_caso(caso, secao)
            resultado = {"caso": caso["CASE_NAME"], "secao": secao, "N": forca, "Mb": momento_b, "Mh": momento_h,
                         "sdrd_tqs": caso[chave_sdrd]}
            for descontar, rotulo in ((False, "bruta"), (True, "liquida")):
                rotina.DESCONTAR_AREA_BARRAS = descontar
                resultado["sdrd_" + rotulo] = relacao_sd_rd(largura, altura, x, y, area, material, forca, momento_b,
                                                            momento_h)
            linhas.append(resultado)
    governante = max(linhas, key=lambda item: item["sdrd_tqs"])
    area_rotina_bruta = None
    if dados["asnec_tqs"] is not None:
        rotina.DESCONTAR_AREA_BARRAS = False
        area_rotina_bruta = area_necessaria_obliqua(largura, altura, x, y, material, governante["N"], governante["Mb"],
                                                    governante["Mh"])
    rotina.DESCONTAR_AREA_BARRAS = True
    return {"id": dados["id"], "casos": linhas, "governante": governante, "area_rotina_bruta": area_rotina_bruta}


def numero(valor: str) -> float:
    return float(valor) if valor not in ("", None) else math.nan


def principal() -> None:
    with open(ARQUIVO_MATRIZ, encoding="utf-8-sig") as arquivo:
        matriz = {linha["ID"]: linha for linha in csv.DictReader(arquivo, delimiter=";")}
    with open(ARQUIVO_ENTRADA_TQS, encoding="utf-8") as arquivo:
        entrada = {linha["id"]: linha for linha in csv.DictReader(arquivo, delimiter=";")
                   if linha["edificio"].startswith(PREFIXO_EDIFICIO_LOTE)}
    edificios = sorted({linha["edificio"] for linha in entrada.values()})
    lidos = {nome: ler_edificio(nome) for nome in edificios}

    tarefas = []
    for identificador, linha_entrada in sorted(entrada.items()):
        edificio = lidos[linha_entrada["edificio"]]
        lispil = edificio["lispil"][identificador]
        tarefas.append({"id": identificador, "largura": float(linha_entrada["b_w_cm"]),
                        "altura": float(linha_entrada["h_cm"]), "fck": numero(matriz[identificador]["fck (MPa)"]),
                        "trecho": edificio["rep"][identificador]["SPAN[1]"],
                        "asnec_tqs": lispil["adotado"]["area_necessaria"] if lispil["adotado"] else None})
    if "reusar" in sys.argv[1:] and ARQUIVO_CAMADA_2.exists():
        camada_2 = json.loads(ARQUIVO_CAMADA_2.read_text(encoding="utf-8"))
    else:
        with ProcessPoolExecutor(max_workers=PROCESSOS) as executor:
            camada_2 = {resultado["id"]: resultado for resultado in executor.map(avaliar_pilar, tarefas)}
        ARQUIVO_CAMADA_2.write_text(json.dumps(camada_2, ensure_ascii=False, indent=1), encoding="utf-8")

    linhas_saida = []
    for identificador, linha_entrada in sorted(entrada.items()):
        nome_edificio = linha_entrada["edificio"]
        edificio = lidos[nome_edificio]
        selec = edificio["selec"][identificador]
        lispil = edificio["lispil"][identificador]
        trecho = edificio["rep"][identificador]["SPAN[1]"]
        rotina_linha = matriz[identificador]
        tep_ok, tep_problemas = conferir_tep(edificio["tep"][identificador], linha_entrada)
        resultado_2 = camada_2[identificador]
        governante = resultado_2["governante"]
        secao_comparada = "TOP" if selec["metodo"] == METODO_TQS_SEM_SEGUNDA_ORDEM else "MIDDLE"
        caso_pv_rep = next(caso for caso in casos_distintos(trecho) if caso["CASE_NAME"] == CASO_PERMANENTE_VARIAVEL)
        forca_pv, _, momento_h_pv = esforcos_caso(caso_pv_rep, secao_comparada)
        area_rotina = numero(rotina_linha["As,calc (cm²)"])
        area_minima_rotina = numero(rotina_linha["As,mín (cm²)"])
        area_camada_2 = (None if resultado_2["area_rotina_bruta"] is None
                         else area_com_minima(resultado_2["area_rotina_bruta"], area_minima_rotina))
        area_adotada_rotina = numero(rotina_linha["As,adot (cm²)"])
        asnec_tqs = lispil["adotado"]["area_necessaria"] if lispil["adotado"] else None
        momento_rotina_b = numero(rotina_linha["Md,tot b_w (kN·m)"])
        momento_tqs_b = rotina.kn_cm_para_kn_m(momento_b_comparavel_tqs(caso_pv_rep, secao_comparada))
        linhas_saida.append({
            "edificio": nome_edificio, "id": identificador, "b_w (cm)": linha_entrada["b_w_cm"],
            "h (cm)": linha_entrada["h_cm"], "fck (MPa)": rotina_linha["fck (MPa)"],
            "λ matriz": rotina_linha["λ (dir. b_w)"], "λ TQS": "%.1f" % selec["lambda"],
            "e1 (cm)": rotina_linha["e1 (cm)"], "Nível e1": rotina_linha["Nível e1"], "ν": rotina_linha["ν"],
            "pé direito duplo (x,y)": edificio["peddup"].get(identificador, "?"),
            "TEP confere": "sim" if tep_ok else "não", "TEP problemas": tep_problemas,
            "ISOK TQS": trecho.get("ISOK"),
            "método rotina b_w": rotina_linha["Método b_w"], "método TQS": selec["metodo"],
            "Nd rotina (kN)": rotina_linha["Nd (kN)"], "Nd TQS (kN)": "%.4f" % forca_pv,
            "dif. Nd (%)": "%.4f" % diferenca_percentual(numero(rotina_linha["Nd (kN)"]), forca_pv),
            "Md,tot b_w rotina (kN·m)": "%.4f" % momento_rotina_b, "seção TQS comparada": secao_comparada,
            "Msd,b TQS (kN·m)": "%.4f" % momento_tqs_b,
            "obs. Msd,b TQS": "prévio (curvatura aproximada); o final do N, M, 1/r não é gravado"
            if selec["metodo"] == METODO_TQS_ACOPLADO else "",
            "Msd,h TQS concomitante (kN·m)": "%.4f" % rotina.kn_cm_para_kn_m(momento_h_pv),
            "dif. Md,tot b_w (%)": "" if selec["metodo"] == METODO_TQS_ACOPLADO
            else "%.3f" % diferenca_percentual(momento_rotina_b, momento_tqs_b),
            "barras rotina": rotina_linha["Barras"], "As,adot rotina (cm²)": "%.3f" % area_adotada_rotina,
            "barras TQS": "%d φ %.1f" % (selec["numero_barras"], selec["diametro_mm"]),
            "As,adot TQS (cm²)": "%.3f" % selec["area_adotada"],
            "dif. As,adot (%)": "%.2f" % diferenca_percentual(area_adotada_rotina, selec["area_adotada"]),
            "As,calc rotina (cm²)": "%.3f" % area_rotina,
            "ASnec TQS arranjo adotado (cm²)": "" if asnec_tqs is None else "%.2f" % asnec_tqs,
            "dif. As,nec (%)": "" if asnec_tqs is None else "%.2f" % diferenca_percentual(area_rotina, asnec_tqs),
            "caso governante TQS": "%s/%s" % (governante["caso"].split("/")[1], governante["secao"]),
            "N gov (kN)": "%.4f" % governante["N"], "Mb gov (kN·m)": "%.4f" % rotina.kn_cm_para_kn_m(governante["Mb"]),
            "Mh gov (kN·m)": "%.4f" % rotina.kn_cm_para_kn_m(governante["Mh"]),
            "Sd/Rd TQS": "%.5f" % governante["sdrd_tqs"], "Sd/Rd rotina bruta": "%.5f" % governante["sdrd_bruta"],
            "Sd/Rd rotina líquida": "%.5f" % governante["sdrd_liquida"],
            "dif. Sd/Rd bruta (%)": "%.3f" % diferenca_percentual(governante["sdrd_bruta"], governante["sdrd_tqs"]),
            "dif. Sd/Rd líquida (%)": "%.3f" % diferenca_percentual(governante["sdrd_liquida"], governante["sdrd_tqs"]),
            "As,mín rotina (cm²)": "%.3f" % area_minima_rotina,
            "As,calc rotina arranjo TQS bruta (cm²)": "" if resultado_2["area_rotina_bruta"] is None
            else "%.3f" % resultado_2["area_rotina_bruta"],
            "As,nec rotina arranjo TQS bruta (cm²)": "" if area_camada_2 is None else "%.3f" % area_camada_2,
            "dif. As,nec camada 2 (%)": "" if area_camada_2 is None or not asnec_tqs
            else "%.2f" % diferenca_percentual(area_camada_2, asnec_tqs),
            "casos Sd/Rd (caso/seção: TQS; bruta; líquida)": " | ".join(
                "%s/%s: %.4f; %.4f; %.4f" % (caso["caso"].split("/")[1], caso["secao"], caso["sdrd_tqs"],
                                             caso["sdrd_bruta"], caso["sdrd_liquida"]) for caso in resultado_2["casos"]),
        })
    with open(ARQUIVO_SAIDA, "w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas_saida[0]), delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas_saida)
    print("%d pilares gravados em %s" % (len(linhas_saida), ARQUIVO_SAIDA))


if __name__ == "__main__":
    principal()
