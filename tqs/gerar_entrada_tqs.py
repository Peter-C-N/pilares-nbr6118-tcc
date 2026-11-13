"""Gera a tabela de entrada do TQS para os 212 pilares modeláveis (entrada_tqs_lote.csv).

Lê saida/resultados_matriz.csv (rotina) e grava, para cada pilar, os dados que o script do TQS (tqs_01_lote.py)
lança no Modelador: edifício, pé direito, seção, classe do concreto, cargas no topo (com o peso próprio do lance
descontado da carga permanente) e posição em planta. Todas as contas estão na zona de auditoria.

Decisões da calibração (04/11/2026; registros calibracao_tqs_P015 e calibracao_tqs_P165):
  - o TQS guarda o pé direito com 3 casas (m): arredonda-se para baixo, ao milímetro, para que λ não passe dos
    limites de método (90 e 140) nos pilares com λ exato;
  - o TQS soma ao pilar o peso próprio do lance (massa específica 25 kN/m³, na base, em todas as seções); a força
    normal da matriz já é a total, então o peso próprio sai da carga permanente aplicada no topo;
  - um edifício por pé direito (16 edifícios) e dois edifícios de validação, com P015 e P165 isolados;
  - teste do momento mínimo (Seção 3.6 do TCC, M01 a M09, e1 = 0): um edifício por pilar (TCCM01 a TCCM09),
    acrescentado em 04/11/2026, 13h30, com as mesmas contas dos demais.

Execução:  python3 gerar_entrada_tqs.py
"""
import csv
import math
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — constantes e fórmulas
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [D01] Arquivos.  Usada em: principal
PASTA_ROTINA = Path(__file__).resolve().parent.parent
ARQUIVO_MATRIZ = PASTA_ROTINA / "saida" / "resultados_matriz.csv"
ARQUIVO_ENTRADA_TQS = PASTA_ROTINA / "tqs" / "entrada_tqs_lote.csv"
ARQUIVO_TESTE_MINIMO = PASTA_ROTINA / "saida" / "resultados_teste_minimo.csv"

# [D02] Massa específica do concreto usada pelo TQS no peso próprio (kN/m³; critério do pórtico, 2,5 tf/m³).
#   Usada em: peso_proprio_lance
MASSA_ESPECIFICA_CONCRETO_KN_M3 = 25.0

# [D03] Fatores de unidade: a unidade de projeto do TQS é kgf = 10 N, logo 1 tf = 10 kN; cm → m; m → mm.
#   Usada em: kn_para_tf; cm_para_m; pe_direito_tqs
KN_POR_TF = 10.0
CM_POR_M = 100.0
MM_POR_CM = 10.0
MM_POR_M = 1000.0

# [D04] Folga numérica para o arredondamento para baixo do pé direito (mm): evita perder 1 mm quando o produto
#   em ponto flutuante fica um pouco abaixo de um inteiro exato.  Usada em: pe_direito_tqs
FOLGA_ARREDONDAMENTO_MM = 1.0e-6

# [D05] Disposição em planta: pilares alinhados no eixo X, com 10 m entre eixos.  Usada em: posicao_x
ESPACAMENTO_PILARES_CM = 1000.0

# [D06] Prefixos dos edifícios (nomes curtos, até 8 caracteres) e pilares isolados da validação.
#   Usada em: nome_edificio_lote; nome_edificio_validacao; nome_edificio_minimo; principal
PREFIXO_EDIFICIO_LOTE = "TCCL"
PREFIXO_EDIFICIO_VALIDACAO = "TCCV"
PREFIXO_EDIFICIO_MINIMO = "TCCM"
PILARES_VALIDACAO = ("P015", "P165")


# [F01] Pé direito lançado no TQS (m): ℓe da matriz (cm, vinculação biapoiada, fator 1) arredondado para baixo
#   ao milímetro.  Variáveis: comprimento_cm.
#   Fórmula: ℓ_TQS = ⌊ℓe · 10 + folga⌋ / 1000
#   Usada em: principal (laço dos pilares)
def pe_direito_tqs(comprimento_cm: float) -> float:
    return math.floor(comprimento_cm * MM_POR_CM + FOLGA_ARREDONDAMENTO_MM) / MM_POR_M


# [F02] Conversão cm → m.  Variáveis: valor_cm.  Fórmula: v_m = v_cm / 100
#   Usada em: peso_proprio_lance; momento_caracteristico
def cm_para_m(valor_cm: float) -> float:
    return valor_cm / CM_POR_M


# [F03] Peso próprio do lance que o TQS soma ao pilar (kN).
#   Variáveis: largura_cm (b_w), altura_cm (h), pe_direito_m (ℓ_TQS).
#   Fórmula: PP = γ_c · b_w · h · ℓ_TQS
#   Usada em: principal (laço dos pilares)
def peso_proprio_lance(largura_cm: float, altura_cm: float, pe_direito_m: float) -> float:
    return MASSA_ESPECIFICA_CONCRETO_KN_M3 * cm_para_m(largura_cm) * cm_para_m(altura_cm) * pe_direito_m


# [F04] Carga permanente aplicada no topo (kN): a da matriz menos o peso próprio que o TQS acrescenta.
#   Variáveis: forca_permanente_kn (N_gk), peso_proprio_kn (PP).  Fórmula: F_z,g = N_gk − PP
#   Usada em: principal (laço dos pilares)
def carga_permanente_topo(forca_permanente_kn: float, peso_proprio_kn: float) -> float:
    return forca_permanente_kn - peso_proprio_kn


# [F05] Momento característico aplicado no topo (kN·m), na direção de b_w.
#   Variáveis: forca_kn (N_gk ou N_qk), excentricidade_cm (e1).  Fórmula: M = N · e1
#   Usada em: principal (laço dos pilares, parcelas permanente e variável)
def momento_caracteristico(forca_kn: float, excentricidade_cm: float) -> float:
    return forca_kn * cm_para_m(excentricidade_cm)


# [F06] Conversão para as unidades da API do TQS: kN → tf e kN·m → tf·m.
#   Variáveis: valor_kn.  Fórmula: v_tf = v_kN / 10
#   Usada em: principal (colunas em tf e tf·m)
def kn_para_tf(valor_kn: float) -> float:
    return valor_kn / KN_POR_TF


# [F07] Índice de esbeltez com o pé direito do TQS (conferência dos limites de método).
#   Variáveis: pe_direito_m, dimensao_cm.  Fórmula: λ = ℓ_TQS · √12 / dimensão
#   Usada em: principal (colunas lambda_b_tqs e lambda_h_tqs)
def indice_esbeltez_tqs(pe_direito_m: float, dimensao_cm: float) -> float:
    return pe_direito_m * CM_POR_M * math.sqrt(12.0) / dimensao_cm


# [F08] Posição X do pilar de ordem i no edifício (cm).  Variáveis: ordem.  Fórmula: x = i · espaçamento
#   Usada em: principal (laço dos pilares)
def posicao_x(ordem: int) -> float:
    return ordem * ESPACAMENTO_PILARES_CM


# [F09] Nome do edifício do lote para o índice do pé direito (1 a 16).  Variáveis: indice.
#   Fórmula: "TCCL" + índice com dois dígitos
#   Usada em: principal (agrupamento por pé direito)
def nome_edificio_lote(indice: int) -> str:
    return "%s%02d" % (PREFIXO_EDIFICIO_LOTE, indice)


# [F10] Nome do edifício de validação de um pilar.  Variáveis: identificador ("P015").
#   Fórmula: "TCCV" + número do pilar
#   Usada em: principal (pilares de validação)
def nome_edificio_validacao(identificador: str) -> str:
    return PREFIXO_EDIFICIO_VALIDACAO + identificador[1:]


# [F11] Nome do edifício de um pilar do teste do momento mínimo.  Variáveis: identificador ("M01").
#   Fórmula: "TCCM" + número do pilar
#   Usada em: principal (pilares do teste do momento mínimo)
def nome_edificio_minimo(identificador: str) -> str:
    return PREFIXO_EDIFICIO_MINIMO + identificador[1:]


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════

COLUNAS_SAIDA = ("edificio", "id", "ordem", "x_cm", "y_cm", "pe_direito_m", "b_w_cm", "h_cm", "classe_concreto",
                 "fz_permanente_tf", "fz_acidental_tf", "my_permanente_tfm", "my_acidental_tfm",
                 "peso_proprio_kn", "ngk_kn", "nqk_kn", "e1_cm", "le_matriz_cm", "lambda_b_tqs", "lambda_h_tqs")


def ler_pilares_modelaveis(caminho: Path = ARQUIVO_MATRIZ) -> list[dict]:
    with open(caminho, encoding="utf-8-sig") as arquivo:
        linhas = list(csv.DictReader(arquivo, delimiter=";"))
    return [linha for linha in linhas if linha["Modelar nos softwares"] == "sim"]


def linha_saida(edificio: str, ordem: int, pilar: dict) -> dict:
    largura = float(pilar["b_w (cm)"])
    altura = float(pilar["h (cm)"])
    comprimento = float(pilar["ℓe (cm)"])
    excentricidade = float(pilar["e1 (cm)"])
    forca_permanente = float(pilar["Ngk (kN)"])
    forca_variavel = float(pilar["Nqk (kN)"])
    pe_direito = pe_direito_tqs(comprimento)
    peso_proprio = peso_proprio_lance(largura, altura, pe_direito)
    return {
        "edificio": edificio, "id": pilar["ID"], "ordem": ordem, "x_cm": repr(posicao_x(ordem)), "y_cm": "0.0",
        "pe_direito_m": "%.3f" % pe_direito, "b_w_cm": repr(largura), "h_cm": repr(altura),
        "classe_concreto": "C%d" % round(float(pilar["fck (MPa)"])),
        "fz_permanente_tf": repr(kn_para_tf(carga_permanente_topo(forca_permanente, peso_proprio))),
        "fz_acidental_tf": repr(kn_para_tf(forca_variavel)),
        "my_permanente_tfm": repr(kn_para_tf(momento_caracteristico(forca_permanente, excentricidade))),
        "my_acidental_tfm": repr(kn_para_tf(momento_caracteristico(forca_variavel, excentricidade))),
        "peso_proprio_kn": repr(peso_proprio), "ngk_kn": repr(forca_permanente), "nqk_kn": repr(forca_variavel),
        "e1_cm": repr(excentricidade), "le_matriz_cm": repr(comprimento),
        "lambda_b_tqs": "%.4f" % indice_esbeltez_tqs(pe_direito, largura),
        "lambda_h_tqs": "%.4f" % indice_esbeltez_tqs(pe_direito, altura),
    }


def principal() -> None:
    pilares = ler_pilares_modelaveis()
    comprimentos = sorted({pe_direito_tqs(float(pilar["ℓe (cm)"])) for pilar in pilares})
    indice_por_comprimento = {comprimento: indice + 1 for indice, comprimento in enumerate(comprimentos)}
    linhas = []
    contagem_por_edificio: dict[str, int] = {}
    for pilar in sorted(pilares, key=lambda item: item["ID"]):
        edificio = nome_edificio_lote(indice_por_comprimento[pe_direito_tqs(float(pilar["ℓe (cm)"]))])
        ordem = contagem_por_edificio.get(edificio, 0)
        contagem_por_edificio[edificio] = ordem + 1
        linhas.append(linha_saida(edificio, ordem, pilar))
    por_id = {pilar["ID"]: pilar for pilar in pilares}
    for identificador in PILARES_VALIDACAO:
        linhas.append(linha_saida(nome_edificio_validacao(identificador), 0, por_id[identificador]))
    pilares_minimo = ler_pilares_modelaveis(ARQUIVO_TESTE_MINIMO)
    for pilar in sorted(pilares_minimo, key=lambda item: item["ID"]):
        linhas.append(linha_saida(nome_edificio_minimo(pilar["ID"]), 0, pilar))
    ARQUIVO_ENTRADA_TQS.parent.mkdir(parents=True, exist_ok=True)
    with open(ARQUIVO_ENTRADA_TQS, "w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_SAIDA, delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas)
    print("%d linhas (%d pilares em %d edifícios + %d de validação + %d do teste do momento mínimo) gravadas em %s"
          % (len(linhas), len(pilares), len(contagem_por_edificio), len(PILARES_VALIDACAO), len(pilares_minimo),
             ARQUIVO_ENTRADA_TQS))
    for edificio in sorted(contagem_por_edificio):
        print("  %s: %2d pilares" % (edificio, contagem_por_edificio[edificio]))


if __name__ == "__main__":
    principal()
