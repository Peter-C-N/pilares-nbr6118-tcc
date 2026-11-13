"""Resumo estatístico da comparação rotina × TQS (lê comparacao_tqs_lote.csv e grava comparacao_tqs_lote_resumo.txt).

Agrupa os 212 pilares pelo método de 2ª ordem do TQS (sem 2ª ordem, curvatura aproximada, N,M,1/r) e, para cada
grandeza comparada, informa quantidade, média, mínimo, máximo e média dos valores absolutos das diferenças.

Execução:  python3 resumir_comparacao_tqs.py
"""
import csv
import math
import statistics
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — constantes e fórmulas
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [D01] Arquivos.  Usada em: principal
PASTA_TQS = Path(__file__).resolve().parent
ARQUIVO_COMPARACAO = PASTA_TQS / "comparacao_tqs_lote.csv"
ARQUIVO_RESUMO = PASTA_TQS / "comparacao_tqs_lote_resumo.txt"

# [D02] Grupos pelo método do TQS (coluna "método TQS" do SELEC.LST) e rótulos.  Usada em: principal
GRUPOS = (("----", "sem 2ª ordem (λ ≤ λ1)"), ("CURV", "pilar-padrão com curvatura aproximada"),
          ("N,M,1/r", "pilar-padrão acoplado a diagramas N, M, 1/r"))

# [D03] Colunas de diferença resumidas (em %).  Usada em: principal
COLUNAS_DIFERENCA = ("dif. Nd (%)", "dif. Md,tot b_w (%)", "dif. As,adot (%)", "dif. As,nec (%)",
                     "dif. Sd/Rd bruta (%)", "dif. Sd/Rd líquida (%)", "dif. As,nec camada 2 (%)")

# [D04] Faixas de |diferença| para a contagem (%).  Usada em: resumo_coluna
FAIXAS_ABSOLUTAS = (0.5, 1.0, 2.0, 5.0)


# [S01] Média aritmética das diferenças.  Variáveis: valores (lista).  Fórmula: x̄ = Σx / n
#   Usada em: resumo_coluna
def media(valores: list[float]) -> float:
    return statistics.fmean(valores)


# [S02] Média dos valores absolutos.  Variáveis: valores.  Fórmula: |x|‾ = Σ|x| / n
#   Usada em: resumo_coluna
def media_absoluta(valores: list[float]) -> float:
    return statistics.fmean(abs(valor) for valor in valores)


# [S03] Quantidade de valores com |x| ≤ limite.  Variáveis: valores, limite.  Fórmula: n(|x| ≤ limite)
#   Usada em: resumo_coluna
def quantidade_ate(valores: list[float], limite: float) -> int:
    return sum(1 for valor in valores if abs(valor) <= limite)


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


def valores_coluna(linhas: list[dict], coluna: str) -> list[float]:
    valores = []
    for linha in linhas:
        texto = linha.get(coluna, "")
        if texto not in ("", None):
            valor = float(texto)
            if math.isfinite(valor):
                valores.append(valor)
    return valores


def resumo_coluna(linhas: list[dict], coluna: str) -> str:
    valores = valores_coluna(linhas, coluna)
    if not valores:
        return "  %-28s sem valores" % coluna
    faixas = "  ".join("≤%.1f%%: %d" % (limite, quantidade_ate(valores, limite)) for limite in FAIXAS_ABSOLUTAS)
    return ("  %-28s n=%3d  média %+8.3f  mín %+8.3f  máx %+8.3f  |média| %7.3f  %s"
            % (coluna, len(valores), media(valores), min(valores), max(valores), media_absoluta(valores), faixas))


def principal() -> None:
    with open(ARQUIVO_COMPARACAO, encoding="utf-8-sig") as arquivo:
        linhas = list(csv.DictReader(arquivo, delimiter=";"))
    saida = ["Comparação rotina × TQS: %d pilares" % len(linhas), ""]
    for codigo, rotulo in GRUPOS:
        grupo = [linha for linha in linhas if linha["método TQS"] == codigo]
        metodos_rotina = sorted({linha["método rotina b_w"] for linha in grupo})
        saida.append("%s (TQS %s): %d pilares; método da rotina na direção de b_w: %s"
                     % (rotulo, codigo, len(grupo), ", ".join(metodos_rotina)))
        for coluna in COLUNAS_DIFERENCA:
            saida.append(resumo_coluna(grupo, coluna))
        saida.append("")
    saida.append("Todos os pilares:")
    for coluna in COLUNAS_DIFERENCA:
        saida.append(resumo_coluna(linhas, coluna))
    texto = "\n".join(saida) + "\n"
    ARQUIVO_RESUMO.write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    principal()
