"""Camada 2 dos pilares com 90 < λ ≤ 140 (pilar-padrão acoplado a diagramas M, N, 1/r): o método da rotina aplicado
às barras que o TQS adotou.

Nesses pilares, o TQS testa arranjos até que o equilíbrio do pilar-padrão acoplado seja atendido (METGERAL.LST), e
os resultados dessa verificação (curvatura, excentricidade de fluência, momento total) não ficam nos arquivos: o
momento da seção intermediária gravado no Pilar.rep é o prévio, pelo pilar-padrão com curvatura aproximada
(conferido nos 62 pilares, a menos de 0,4 %). Por isso, a comparação é feita pela armadura: com as posições das
barras do TQS, a rotina calcula a área necessária pelo seu pilar-padrão acoplado (com a fluência da Seção 15.8.4) e,
com a área adotada pelo TQS, o momento total e a razão M_d,tot / M_Rd. A diferença que resta é a do método de 2ª
ordem (registro calibracao_tqs_P165_2026-11-04: e_a fora da fluência, e_cc sem amplificação, N_sg no topo e rigidez
secante constante no TQS).

Lê comparacao_tqs_lote.csv (pilares e método), entrada_tqs_lote.csv, resultados_matriz.csv e os Pilar.rep do lote.
Grava camada2_nm1r_tqs.csv.  Execução:  python3 camada2_nm1r_tqs.py
"""
import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "analise"))
import pilares_nbr6118 as rotina  # noqa: E402
import comparar_tqs_lote as comparacao  # noqa: E402
from ler_pilar_rep import ler_pilar_rep, pilares_por_nome  # noqa: E402

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — constantes e fórmulas
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [D01] Arquivos.  Usada em: principal
PASTA_TQS = Path(__file__).resolve().parent
ARQUIVO_COMPARACAO = PASTA_TQS / "comparacao_tqs_lote.csv"
ARQUIVO_SAIDA = PASTA_TQS / "camada2_nm1r_tqs.csv"

# [D02] Código do método no SELEC.LST e aço.  Usada em: principal; avaliar
METODO_TQS_ACOPLADO = "N,M,1/r"
FYK_MPA = 500.0

# [D03] Processos paralelos.  Usada em: principal
PROCESSOS = 2


# [N01] Diferença percentual com sinal, da rotina em relação ao TQS (a mesma [C11] de comparar_tqs_lote.py).
#   Variáveis: valor_rotina, valor_tqs.  Fórmula: Δ = 100 · (X_rotina − X_TQS) / X_TQS
#   Usada em: principal (colunas "dif.")
def diferenca_percentual(valor_rotina: float, valor_tqs: float) -> float:
    return 100.0 * (valor_rotina - valor_tqs) / valor_tqs


# [N03] Razão entre a área adotada pelo TQS e a necessária pela rotina com as mesmas posições de barras.
#   Variáveis: area_tqs, area_rotina (cm²).  Fórmula: ρ = A_s,adot,TQS / A_s,nec,rotina
#   Usada em: principal (coluna "As,adot TQS / As,nec rotina")
def razao_areas(area_tqs: float, area_rotina: float) -> float:
    return area_tqs / area_rotina


# [N02] Razão entre o momento total e o momento resistente da seção na direção de b_w, com N fixo.
#   Variáveis: momento_total, momento_resistente (kN·cm).  Fórmula: η = M_d,tot / M_Rd
#   Usada em: avaliar
def razao_momento(momento_total: float, momento_resistente: float) -> float:
    return momento_total / momento_resistente


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


def avaliar(dados: dict) -> dict:
    """Equilíbrio do pilar-padrão acoplado da rotina com as barras do TQS, seção líquida (padrão da rotina) e bruta."""
    secao = rotina.secao_na_direcao(dados["largura"], dados["altura"], dados["x"], dados["y"], rotina.DIRECAO_B)
    material = rotina.Material(dados["fck"], FYK_MPA)
    resultado = {"id": dados["id"]}
    for descontar, rotulo in ((True, "liquida"), (False, "bruta")):
        rotina.DESCONTAR_AREA_BARRAS = descontar
        area_necessaria, _, _ = rotina.armadura_pilar_padrao_acoplado(secao, material, dados["forca_normal"],
                                                                       dados["momento_primeira_ordem"],
                                                                       dados["alfa_b"], dados["comprimento"])
        resultado["area_" + rotulo] = area_necessaria
        equilibrio = rotina.equilibrio_pilar_padrao_acoplado(secao, material, dados["area"], dados["forca_normal"],
                                                             dados["momento_primeira_ordem"], dados["alfa_b"],
                                                             dados["comprimento"])
        momento_resistente = rotina.momento_resistente(secao, material, dados["area"], dados["forca_normal"])
        if equilibrio is None:
            resultado[rotulo] = (math.nan, math.nan, momento_resistente, math.nan)
        else:
            curvatura, momento_total = equilibrio
            resultado[rotulo] = (momento_total, curvatura, momento_resistente,
                                 razao_momento(momento_total, momento_resistente))
    rotina.DESCONTAR_AREA_BARRAS = True
    return resultado


def principal() -> None:
    with open(ARQUIVO_COMPARACAO, encoding="utf-8-sig") as arquivo:
        comparados = [linha for linha in csv.DictReader(arquivo, delimiter=";")
                      if linha["método TQS"] == METODO_TQS_ACOPLADO]
    with open(comparacao.ARQUIVO_MATRIZ, encoding="utf-8-sig") as arquivo:
        matriz = {linha["ID"]: linha for linha in csv.DictReader(arquivo, delimiter=";")}
    repositorios = {}
    tarefas = []
    for linha in comparados:
        edificio = linha["edificio"]
        if edificio not in repositorios:
            repositorios[edificio] = pilares_por_nome(ler_pilar_rep(comparacao.PASTA_RESULTADOS_LOTE / edificio
                                                                    / "PILAR" / "Pilar.rep"))
        trecho = repositorios[edificio][linha["id"]]["SPAN[1]"]
        x, y, _, area = comparacao.barras_tqs(trecho)
        rotina_linha = matriz[linha["id"]]
        tarefas.append({"id": linha["id"], "largura": float(linha["b_w (cm)"]), "altura": float(linha["h (cm)"]),
                        "fck": float(rotina_linha["fck (MPa)"]), "x": x, "y": y, "area": area,
                        "forca_normal": float(rotina_linha["Nd (kN)"]),
                        "momento_primeira_ordem": 100.0 * float(rotina_linha["M1d,A b_w (kN·m)"]),
                        "alfa_b": float(rotina_linha["αb b_w"]), "comprimento": float(rotina_linha["ℓe (cm)"])})
    with ProcessPoolExecutor(max_workers=PROCESSOS) as executor:
        resultados = {resultado["id"]: resultado for resultado in executor.map(avaliar, tarefas)}
    linhas_saida = []
    for linha in comparados:
        resultado = resultados[linha["id"]]
        area_tqs = float(linha["As,adot TQS (cm²)"])
        saida = {"edificio": linha["edificio"], "id": linha["id"], "b_w (cm)": linha["b_w (cm)"],
                 "h (cm)": linha["h (cm)"], "fck (MPa)": linha["fck (MPa)"], "λ matriz": linha["λ matriz"],
                 "ν": linha["ν"], "e1 (cm)": linha["e1 (cm)"], "barras TQS": linha["barras TQS"],
                 "barras rotina": linha["barras rotina"], "dif. As,adot (%)": linha["dif. As,adot (%)"],
                 "As,adot TQS (cm²)": "%.3f" % area_tqs, "As,calc rotina arranjo próprio (cm²)":
                 linha["As,calc rotina (cm²)"], "Md,tot rotina barras próprias (kN·m)": linha["Md,tot b_w rotina (kN·m)"]}
        for rotulo, nome in (("liquida", "líquida"), ("bruta", "bruta")):
            momento_total, curvatura, momento_resistente, razao = resultado[rotulo]
            area_rotina = resultado["area_" + rotulo]
            saida["As,nec rotina arranjo TQS %s (cm²)" % nome] = "%.3f" % area_rotina
            saida["As,adot TQS / As,nec rotina %s" % nome] = "%.4f" % razao_areas(area_tqs, area_rotina)
            saida["dif. As,nec rotina arranjo TQS %s × As,adot TQS (%%)" % nome] = "%.2f" % diferenca_percentual(
                area_rotina, area_tqs)
            saida["Md,tot rotina barras TQS %s (kN·m)" % nome] = "%.4f" % rotina.kn_cm_para_kn_m(momento_total)
            saida["MRd b_w barras TQS %s (kN·m)" % nome] = "%.4f" % rotina.kn_cm_para_kn_m(momento_resistente)
            saida["Md,tot/MRd %s" % nome] = "%.4f" % razao
        linhas_saida.append(saida)
    with open(ARQUIVO_SAIDA, "w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas_saida[0]), delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas_saida)
    print("%d pilares gravados em %s" % (len(linhas_saida), ARQUIVO_SAIDA))


if __name__ == "__main__":
    principal()
