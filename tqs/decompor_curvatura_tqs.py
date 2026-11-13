"""Decomposição da diferença de armadura entre a rotina e o TQS nos pilares com curvatura aproximada (λ ≤ 90).

Para cada pilar com método CURV no TQS, a partir do caso que governou no programa (seção intermediária):
  A_TQS   : ASnec do TQS para o arranjo adotado (LISPIL.LST);
  A_2     : área da rotina com o mesmo arranjo e os mesmos esforços (N, M_b, M_h), seção bruta (camada 2);
  A_sem_h : área da rotina com o mesmo arranjo, N e M_b, mas sem o momento concomitante na direção de h (M_h = 0),
            que no TQS é o momento de 2ª ordem dessa direção (NBR 6118:2023, 15.8.3.3.5);
  A_rot   : As,calc da rotina (arranjo próprio, seção líquida, direções verificadas separadamente e envoltória
            mínima, Seção 3.6 do TCC).
Parcelas: efeito de M_h = A_2 − A_sem_h; arranjo e seção = A_rot − A_sem_h; seção (camada 2) = A_2 − A_TQS.
Todas as áreas comparadas respeitam a armadura mínima, como o ASnec do TQS.

Lê comparacao_tqs_lote.csv e os Pilar.rep do lote; grava decomposicao_curvatura_tqs.csv.
Execução:  python3 decompor_curvatura_tqs.py
"""
import csv
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

# [D01] Arquivos e método.  Usada em: principal
PASTA_TQS = Path(__file__).resolve().parent
ARQUIVO_COMPARACAO = PASTA_TQS / "comparacao_tqs_lote.csv"
ARQUIVO_SAIDA = PASTA_TQS / "decomposicao_curvatura_tqs.csv"
METODO_TQS_CURVATURA = "CURV"
FYK_MPA = 500.0

# [D02] Momento nulo na direção de h (kN·cm) e processos paralelos.  Usada em: avaliar; principal
MOMENTO_H_NULO = 0.0
PROCESSOS = 2


# [P01] Diferença percentual com sinal em relação ao ASnec do TQS.  Variáveis: valor, referencia.
#   Fórmula: Δ = 100 · (A − A_TQS) / A_TQS
#   Usada em: principal (colunas "dif.")
def diferenca_percentual(valor: float, referencia: float) -> float:
    return 100.0 * (valor - referencia) / referencia


# [P02] Parcela de uma diferença de área, em porcentagem do ASnec do TQS.  Variáveis: area_maior, area_menor,
#   referencia.  Fórmula: p = 100 · (A_1 − A_2) / A_TQS
#   Usada em: principal (colunas "parcela")
def parcela_percentual(area_1: float, area_2: float, referencia: float) -> float:
    return 100.0 * (area_1 - area_2) / referencia


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


def avaliar(dados: dict) -> dict:
    material = rotina.Material(dados["fck"], FYK_MPA)
    rotina.DESCONTAR_AREA_BARRAS = False
    area_sem_h = comparacao.area_necessaria_obliqua(dados["largura"], dados["altura"], dados["x"], dados["y"],
                                                    material, dados["forca_normal"], dados["momento_b"],
                                                    MOMENTO_H_NULO)
    rotina.DESCONTAR_AREA_BARRAS = True
    return {"id": dados["id"], "area_sem_h": comparacao.area_com_minima(area_sem_h, dados["area_minima"])}


def principal() -> None:
    with open(ARQUIVO_COMPARACAO, encoding="utf-8-sig") as arquivo:
        linhas = [linha for linha in csv.DictReader(arquivo, delimiter=";")
                  if linha["método TQS"] == METODO_TQS_CURVATURA]
    repositorios = {}
    tarefas = []
    for linha in linhas:
        edificio = linha["edificio"]
        if edificio not in repositorios:
            repositorios[edificio] = pilares_por_nome(ler_pilar_rep(comparacao.PASTA_RESULTADOS_LOTE / edificio
                                                                    / "PILAR" / "Pilar.rep"))
        x, y, _, _ = comparacao.barras_tqs(repositorios[edificio][linha["id"]]["SPAN[1]"])
        tarefas.append({"id": linha["id"], "largura": float(linha["b_w (cm)"]), "altura": float(linha["h (cm)"]),
                        "fck": float(linha["fck (MPa)"]), "x": x, "y": y, "forca_normal": float(linha["N gov (kN)"]),
                        "momento_b": 100.0 * float(linha["Mb gov (kN·m)"]),
                        "area_minima": float(linha["As,mín rotina (cm²)"])})
    with ProcessPoolExecutor(max_workers=PROCESSOS) as executor:
        resultados = {resultado["id"]: resultado for resultado in executor.map(avaliar, tarefas)}
    saida = []
    for linha in linhas:
        area_tqs = float(linha["ASnec TQS arranjo adotado (cm²)"])
        area_2 = float(linha["As,nec rotina arranjo TQS bruta (cm²)"])
        area_sem_h = resultados[linha["id"]]["area_sem_h"]
        area_rotina = float(linha["As,calc rotina (cm²)"])
        saida.append({"id": linha["id"], "b_w (cm)": linha["b_w (cm)"], "h (cm)": linha["h (cm)"],
                      "λ matriz": linha["λ matriz"], "ν": linha["ν"], "e1 (cm)": linha["e1 (cm)"],
                      "Mb gov (kN·m)": linha["Mb gov (kN·m)"], "Mh gov (kN·m)": linha["Mh gov (kN·m)"],
                      "A_TQS (cm²)": "%.2f" % area_tqs, "A_2 (cm²)": "%.3f" % area_2,
                      "A_sem_h (cm²)": "%.3f" % area_sem_h, "A_rot (cm²)": "%.3f" % area_rotina,
                      "dif. total (%)": "%.2f" % diferenca_percentual(area_rotina, area_tqs),
                      "parcela M_h (%)": "%.2f" % parcela_percentual(area_sem_h, area_2, area_tqs),
                      "parcela arranjo e seção (%)": "%.2f" % parcela_percentual(area_rotina, area_sem_h, area_tqs),
                      "parcela camada 2 (%)": "%.2f" % parcela_percentual(area_2, area_tqs, area_tqs)})
    with open(ARQUIVO_SAIDA, "w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(saida[0]), delimiter=";")
        escritor.writeheader()
        escritor.writerows(saida)
    print("%d pilares gravados em %s" % (len(saida), ARQUIVO_SAIDA))


if __name__ == "__main__":
    principal()
