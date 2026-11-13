# -*- coding: utf-8 -*-
"""
gerar_auditor.py
================
Gera o auditor visual da rotina: um único arquivo HTML, que funciona sem internet, com
  1) "Pilar a pilar": todas as etapas do cálculo de cada um dos 324 pilares-teste e dos 9 pilares do teste do
     momento mínimo ([R13] de processar_matriz), cada valor ligado à fórmula da zona de auditoria que o produz;
     diagramas de interação N-M nas duas direções com o ponto de cálculo, a seção com as barras adotadas, a
     envoltória resistente com a envoltória mínima com 2ª ordem ([R12] do núcleo) e, no método M, N, 1/r, a curva
     de deformação com a reta do momento externo;
  2) "Fórmulas": todos os itens da ZONA DE AUDITORIA de pilares_nbr6118.py (comentário, código e "Usada em"),
     com botões que abrem o PDF local da norma ou do livro NA PÁGINA da fonte citada;
  3) "Matriz": mapa de viabilidade e gráfico da taxa de armadura em função de λ;
  4) "Validação": os testes de validacao.py, com filtro.

Os botões abrem os PDFs da pasta bibliografia (fontes/paginas_bibliografia.json; ver [R01a]); nenhuma página da norma é
copiada para o HTML (exemplar licenciado). O mapa seção -> página do PDF da norma está em
fontes/paginas_nbr6118.json (só números de seção e de página).

Uso:  python3 gerar_auditor.py      (grava saida/auditor_pilares.html; requer processar_matriz.py e validacao.py)
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import pilares_nbr6118 as nucleo
import processar_matriz
import rastreabilidade

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — INÍCIO
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [R01] Arquivos de entrada e de saída.
#   Variáveis: PASTA, ARQUIVO_RESULTADOS, ARQUIVO_TESTE_MINIMO, ARQUIVO_INVIAVEIS, ARQUIVO_VALIDACAO, ARQUIVO_PAGINAS_NBR,
#   ARQUIVO_BIBLIOGRAFIA, ARQUIVO_MODELO, ARQUIVO_SAIDA, SCRIPT_NUCLEO.
#   Usada em: nível do módulo (l. 51, 52, 53, 54, 55, 56, 57, 58, 59); blocos_do_nucleo (l. 273, 277, 278);
#       endereco_bibliografia (l. 418); main (l. 422, 423, 424, 431, 432, 434, 458, 459, 460)
PASTA = Path(__file__).resolve().parent
ARQUIVO_RESULTADOS = PASTA / "saida" / "resultados_matriz.csv"
ARQUIVO_TESTE_MINIMO = PASTA / "saida" / "resultados_teste_minimo.csv"
ARQUIVO_INVIAVEIS = PASTA / "saida" / "resultados_inviaveis.csv"
ARQUIVO_VALIDACAO = PASTA / "saida" / "validacao.json"
ARQUIVO_PAGINAS_NBR = PASTA / "fontes" / "paginas_nbr6118.json"
ARQUIVO_BIBLIOGRAFIA = PASTA / "fontes" / "paginas_bibliografia.json"
ARQUIVO_MODELO = PASTA / "auditor_modelo.html"
ARQUIVO_SAIDA = PASTA / "saida" / "auditor_pilares.html"
SCRIPT_NUCLEO = PASTA / "pilares_nbr6118.py"

# [R01a] Pasta dos PDFs das obras citadas, que os botões de fonte abrem (as obras não fazem parte do material). Por
#   padrão, a pasta bibliografia ao lado da pasta saida, em endereço relativo ao auditor gerado (pasta_local de
#   fontes/paginas_bibliografia.json); a variável de ambiente BIBLIOGRAFIA_TCC, se definida, indica outra pasta do
#   computador de quem gera o auditor. Nenhum caminho absoluto fica gravado no código (alteração de 13/11/2026).
#   Variáveis: VARIAVEL_BIBLIOGRAFIA.
#   Usada em: endereco_bibliografia (main)
VARIAVEL_BIBLIOGRAFIA = "BIBLIOGRAFIA_TCC"

# [R02] Discretização dos gráficos: pontos do diagrama de interação e da curva momento-curvatura.
#   Variáveis: PONTOS_INTERACAO, PONTOS_CURVATURA, CASAS_GRAFICO.
#   Usada em: interacao (l. 352, 353, 354); curva_mn1r (l. 361, 364, 365, 366);
#       graficos_do_pilar (l. 388, 389, 390, 391, 402, 403, 404)
PONTOS_INTERACAO = 181
PONTOS_CURVATURA = 61
CASAS_GRAFICO = 4

# [R03] Ligação entre as colunas de resultados_matriz.csv e os itens da zona de auditoria que as produzem
#   ({d} = direção b_w ou h). A primeira etiqueta é a principal; as demais são as fórmulas auxiliares.
#   Variáveis: ETIQUETAS_COLUNAS.
#   Usada em: main (l. 442)
ETIQUETAS_COLUNAS = {
    "ℓe (cm)": ("F20", "R10"), "Nd (kN)": ("F26",), "e1 (cm)": ("F53",), "fcd (kN/cm²)": ("F05", "C01"),
    "Nk (kN)": ("F35",), "Ngk (kN)": ("F35", "H03"), "Nqk (kN)": ("F35", "H03"), "ν calc": ("F25",),
    "λ b_w calc": ("F19",), "λ h calc": ("F19",),
    "λ1 {d}": ("F24", "F22", "C08"), "αb {d}": ("F23", "R06", "F54"), "Método {d}": ("R01", "C08"),
    "Nd·e1 {d} (kN·m)": ("F53",), "M1d,mín {d} (kN·m)": ("F21",), "e_a {d} (cm)": ("F32", "F31", "C11"),
    "N_sg {d} (kN)": ("F36", "F35", "H03"), "M_sg {d} (kN·m)": ("F53",), "N_e {d} (kN)": ("F33", "F14", "F04"),
    "e_cc {d} (cm)": ("F34", "H03"), "M1d,A {d} (kN·m)": ("F55", "F56"), "Md,tot {d} (kN·m)": ("F28", "F30", "C07"),
    "1/r {d} (1/m)": ("F27", "F30", "F29", "C03", "C04"), "As,nec {d} (cm²)": ("M02", "M04", "F18", "F16", "F17"),
    "As,mín (cm²)": ("F37", "C09"), "As,máx (cm²)": ("F38", "C09"), "As,calc (cm²)": ("R08",),
    "As,nec b_w ref (cm²)": ("R11",), "Md,tot b_w ref (kN·m)": ("R11", "F28", "F30"), "As,calc ref (cm²)": ("R11", "R08"),
    "d' ref (cm)": ("R11", "F46"), "Governa": ("R08",), "Arranjo adotado": ("R07", "R02"),
    "Arranjos avaliados": ("R07",), "Barras": ("R04", "R09", "R03", "F42", "C12"), "n barras": ("R02", "F49"),
    "φl (mm)": ("F42", "C12"), "φt (mm)": ("F43", "C12"), "d' (cm)": ("F46", "H02"), "As,adot (cm²)": ("F40",),
    "ρ adot": ("F41",), "MRd b_w (kN·m)": ("F18", "F16", "F17"), "MRd h (kN·m)": ("F18", "F16", "F17"),
    "Situação": ("R05",),
    "Md,tot,mín b_w (kN·m)": ("R12", "F55", "F28", "F30"), "Md,tot,mín h (kN·m)": ("R12", "F55", "F28", "F30"),
    "η envoltória mínima": ("F72", "F71", "F69", "F66", "F67", "F68", "M08", "N04"),
    "ψ crítico (°)": ("F72", "F70", "F73"), "Envoltória mínima (15.3.2)": ("R12", "H06"),
    "As,adot elipse (cm²)": ("R12",), "Barras elipse": ("R12", "R04"),
}

# [R04] Padrões de citação de fontes nos comentários e deslocamentos de página (página do PDF = página impressa
#   + deslocamento; conferidos em fontes/paginas_bibliografia.json). Número de seção da norma: não pode ser
#   seguido de dígito nem de vírgula decimal com dígito ("0,85"); uma vírgula seguida de espaço, como em
#   "11.3.3.4.3, 15.3.2", separa itens (a versão de 21/10/2026 recusava qualquer vírgula e truncava o número
#   anterior a ela, por exemplo 17.2.2 lido como 17.2).
#   Variáveis: PADRAO_SECAO_NBR, PADRAO_TABELA_FIGURA, PADRAO_BASTOS, PADRAO_FUSCO, PADRAO_PINHEIRO.
#   Usada em: fontes_do_comentario (l. 323, 328, 329, 334, 337, 339)
PADRAO_SECAO_NBR = re.compile(r"(?<![\d.,])(\d{1,2}(?:\.\d{1,2}){1,5})(?!\d|,\d)")
PADRAO_TABELA_FIGURA = re.compile(r"(Tabela|Figura)\s+(\d{1,2}\.\d{1,2})")
PADRAO_BASTOS = re.compile(r"Bastos\s*\(2017,\s*p\.\s*(\d+)")
PADRAO_FUSCO = re.compile(r"Fusco\s*\(1981,\s*p\.\s*(\d+)")
PADRAO_PINHEIRO = re.compile(r"(?:Pinheiro|PINHEIRO|Scadelai)[^;]*?item\s+(\d{1,2}\.\d{1,2}(?:\.\d)?)")


# [F01] Página do PDF a partir da página impressa: página impressa + deslocamento.
#   Variáveis: pagina_impressa, deslocamento.
#   Usada em: fontes_engenharia (l. 260, 264, 267); fontes_do_comentario (l. 336, 338)
def pagina_pdf(pagina_impressa: int, deslocamento: int) -> int:
    return pagina_impressa + deslocamento


# [F02] Página impressa da norma a partir da página do PDF (para o rótulo do botão): PDF - 18.
#   Variáveis: pagina_do_pdf.
#   Usada em: pagina_impressa_nbr (l. 131); fontes_do_comentario (l. 327, 332)
DESLOCAMENTO_NBR = 18


def pagina_impressa_nbr(pagina_do_pdf: int) -> int:
    return pagina_do_pdf - DESLOCAMENTO_NBR


# [F03] Arredondamento dos valores dos gráficos (tamanho do arquivo): round(x, casas).
#   Variáveis: valores, casas.
#   Usada em: interacao (l. 353); curva_mn1r (l. 364, 365, 366);
#       graficos_do_pilar (l. 383, 388, 389, 390, 391, 402, 403, 404)
def arredondar(valores, casas: int) -> list:
    return [None if not math.isfinite(valor) else round(float(valor), casas) for valor in np.asarray(valores).ravel()]


# [F04] Momento externo do pilar-padrão dividido por gama_f3, para comparação com a curva de deformação:
#   M_ext/gama_f3 = [alfa_b M1d,A + Nd (l_e²/10)(1/r)] / gama_f3 (mesma expressão de [F28] e [F30] do núcleo).
#   Variáveis: alfa_b, momento_primeira_ordem, forca_normal, comprimento, curvaturas.
#   Usada em: curva_mn1r (l. 362)
def reta_momento_externo(alfa_b: float, momento_primeira_ordem: float, forca_normal: float, comprimento: float,
                         curvaturas: np.ndarray) -> np.ndarray:
    return (alfa_b * momento_primeira_ordem + forca_normal * comprimento ** 2 / nucleo.DIVISOR_PILAR_PADRAO
            * curvaturas) / nucleo.GAMA_F3


# [F05] Conversão de unidades dos gráficos: kN·cm -> kN·m (/100) e 1/cm -> 1/km (x 10^5).
#   Variáveis: valores.
#   Usada em: interacao (l. 353); curva_mn1r (l. 364, 365, 366, 367, 369);
#       graficos_do_pilar (l. 388, 389, 390, 391, 402, 403, 404)
def para_kn_m(valores) -> np.ndarray:
    return np.asarray(valores, dtype=float) / 100.0


def para_por_km(valores) -> np.ndarray:
    return np.asarray(valores, dtype=float) * 1.0e5


# [F06] Momento da curva de deformação no ponto de equilíbrio: M_d,tot / gama_f3 ([F30] do núcleo com R = 0).
#   Variáveis: momento_total_kn_m.
#   Usada em: curva_mn1r (l. 369)
def momento_reduzido(momento_total_kn_m: float) -> float:
    return momento_total_kn_m / nucleo.GAMA_F3


# [F07] Tamanho do arquivo em MB: bytes / 10^6.  Variáveis: tamanho_bytes.
#   Usada em: main (l. 460)
def megabytes(tamanho_bytes: int) -> float:
    return tamanho_bytes / 1.0e6


# [F08] Coordenadas no plano (M_h, M_b) de pontos dados em forma polar (raio r, direção psi):
#   M_b = r cos psi;  M_h = r sen psi (psi = 0 na direção de b_w; elipse de [F69] do núcleo e ponto crítico da
#   envoltória).
#   Variáveis: raios, direcoes (rad).
#   Usada em: graficos_do_pilar (l. 395, 397)
def coordenadas_polares(raios, direcoes) -> tuple[np.ndarray, np.ndarray]:
    raios = np.asarray(raios, dtype=float)
    direcoes = np.asarray(direcoes, dtype=float)
    return raios * np.cos(direcoes), raios * np.sin(direcoes)


# [F09] Raio da envoltória resistente na direção crítica: r_R = eta r_e ([F72] do núcleo).
#   Variáveis: fator (eta), raio_minimo (r_e).
#   Usada em: graficos_do_pilar (l. 398)
def raio_resistente_critico(fator: float, raio_minimo: float) -> float:
    return fator * raio_minimo


# [R06] Pontos da elipse nos gráficos do auditor: uma a cada PASSO_PONTOS_ELIPSE direções de DIRECOES_ENVOLTORIA
#   ([N04] do núcleo; 721 direções -> 91 pontos, de 1° em 1°).  Variáveis: PASSO_PONTOS_ELIPSE.
#   Usada em: graficos_do_pilar (l. 394)
PASSO_PONTOS_ELIPSE = 8


# [R07] Linhas do comentário de cada item da zona de auditoria do núcleo: "Usada em:" e "Usada também em:" (com as
#   linhas de continuação, indentadas com seis espaços depois do "# ") vão para o campo de uso; "Fórmula:" traz a
#   expressão em notação matemática (marcação: _{...} índice, ^{...} expoente, \frac{num}{den} fração e
#   \sqrt{...} raiz), que o auditor exibe com símbolos, índices e frações (item 7 do autor, 23/10/2026).
#   Variáveis: PREFIXOS_USO, INDENTACAO_CONTINUACAO_USO, PREFIXO_FORMULA.
#   Usada em: separar_comentario (l. 241, 244, 246, 248)
PREFIXOS_USO = ("Usada em:", "Usada também em:")
INDENTACAO_CONTINUACAO_USO = "      "
PREFIXO_FORMULA = "Fórmula:"

# [R08] Botões de fonte da tabela da engenharia reversa de Bastos (aba Validação): a página inicial do exemplo em
#   Bastos (2017) e, nas linhas de ω, a página do ábaco usado pelo autor e a das equações de equilíbrio da apostila de
#   Venturini e Rodrigues (1987, eq. 4.6, p. 14: seção bruta), pela regra de páginas de paginas_bibliografia.json.
#   Variáveis: PAGINA_EQUILIBRIO_VENTURINI, PADRAO_PAGINA_EXEMPLO, PADRAO_NUMERO_ABACO.
#   Usada em: fontes_engenharia (l. 258, 262, 267)
PAGINA_EQUILIBRIO_VENTURINI = 14
PADRAO_PAGINA_EXEMPLO = re.compile(r"p\. (\d+)")
PADRAO_NUMERO_ABACO = re.compile(r"A-(\d+)")

# [R05] Valores não finitos (NaN, infinito) dos testes e resultados são gravados como null no JSON do HTML.
#   Variáveis: valor.
#   Usada em: main (l. 443, 444, 446)
def valor_json(valor):
    if isinstance(valor, float) and not math.isfinite(valor):
        return None
    return valor

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


# ----------------------------------------------------------------------------- fórmulas da zona de auditoria
def separar_comentario(comentario: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Separa o comentário de um item da zona de auditoria em corpo, linhas "Usada em" (com as de continuação, que
    começam com seis espaços) e expressões das linhas "Fórmula:" (marcação exibida pelo auditor com símbolos)."""
    corpo, uso, formulas = [], [], []
    em_uso = False
    for linha in comentario:
        conteudo = linha.strip()
        if conteudo.startswith(PREFIXOS_USO):
            em_uso = True
            uso.append(linha)
        elif em_uso and linha.startswith(INDENTACAO_CONTINUACAO_USO):
            uso.append(linha)
        elif conteudo.startswith(PREFIXO_FORMULA):
            em_uso = False
            formulas.append(conteudo[len(PREFIXO_FORMULA):].strip())
        else:
            em_uso = False
            corpo.append(linha)
    return corpo, uso, formulas


def fontes_engenharia(linha: dict, documentos: dict) -> list[dict]:
    """Botões de fonte de uma linha da engenharia reversa de Bastos ([R08])."""
    bastos, venturini = documentos["BASTOS"], documentos["VENTURINI"]
    primeira = int(PADRAO_PAGINA_EXEMPLO.match(linha["paginas"]).group(1))
    fontes = [{"rotulo": "Bastos (2017), %s" % linha["paginas"], "arquivo": bastos["arquivo"],
               "pagina": pagina_pdf(primeira, bastos["deslocamento"]), "titulo": "exemplo %s" % linha["exemplo"]}]
    if linha["criterio"] == "leitura":
        numero = int(PADRAO_NUMERO_ABACO.search(linha["abaco"]).group(1))
        fontes.append({"rotulo": "Ábaco %s" % linha["abaco"], "arquivo": venturini["arquivo"],
                       "pagina": pagina_pdf(numero, venturini["deslocamento_abaco"]),
                       "titulo": "Venturini e Rodrigues (1987)"})
        fontes.append({"rotulo": "Venturini e Rodrigues (1987), eq. 4.6", "arquivo": venturini["arquivo"],
                       "pagina": pagina_pdf(PAGINA_EQUILIBRIO_VENTURINI, venturini["deslocamento"]),
                       "titulo": "equações de equilíbrio: seção bruta"})
    return fontes


def blocos_do_nucleo() -> list[dict]:
    texto = SCRIPT_NUCLEO.read_text(encoding="utf-8")
    linhas = texto.split("\n")
    blocos = rastreabilidade.blocos_da_zona(linhas, ast.parse(texto))
    fim_zona = rastreabilidade.limites_zona(linhas)[1]
    paginas_nbr = json.loads(ARQUIVO_PAGINAS_NBR.read_text(encoding="utf-8"))["itens"]
    bibliografia = json.loads(ARQUIVO_BIBLIOGRAFIA.read_text(encoding="utf-8"))
    itens = []
    for posicao, bloco in enumerate(blocos):
        comentario = [linha[2:] if linha.startswith("# ") else linha[1:] for linha in
                      linhas[bloco["primeira"]:bloco["ultima"] + 1]]
        fim_codigo = blocos[posicao + 1]["primeira"] if posicao + 1 < len(blocos) else fim_zona
        codigo = [linha for linha in linhas[bloco["ultima"] + 1:fim_codigo]
                  if not linha.startswith("# ─") and not linha.startswith("# ═")]
        while codigo and not codigo[-1].strip():
            codigo.pop()
        while codigo and not codigo[0].strip():
            codigo.pop(0)
        corpo, uso, formulas = separar_comentario(comentario)
        titulo = re.sub(r"^\[[A-Z]\d{2}\]\s*", "", corpo[0]).strip()
        texto_comentario = " ".join(linha.strip() for linha in corpo)
        itens.append({"etiqueta": bloco["etiqueta"], "titulo": titulo, "comentario": "\n".join(corpo),
                      "formulas": formulas,
                      "codigo": "\n".join(codigo), "uso": "\n".join(linha.strip() for linha in uso),
                      "linha": bloco["primeira"] + 1, "nomes": bloco["nomes"],
                      "fontes": fontes_do_comentario(texto_comentario, paginas_nbr, bibliografia)})
    return itens


def trecho_das_fontes(texto: str) -> str:
    encontro = re.search(r"Fonte[s]?:\s*(.*?)(?:\s+(?:TCC:|Variáveis:|Usada)|$)", texto)
    return encontro.group(1) if encontro else ""


def fontes_do_comentario(texto: str, paginas_nbr: dict, bibliografia: dict) -> list[dict]:
    """Botões de fonte: NBR (seções, tabelas e figuras citadas no trecho "Fonte:" ou em "NBR 6118:2023, x"),
    Bastos, Fusco e Pinheiro/Scadelai com a página do PDF local."""
    documentos = bibliografia["documentos"]
    fontes, vistos = [], set()

    def incluir(documento: str, rotulo: str, pagina: int, titulo: str = "") -> None:
        if (documento, pagina, rotulo) in vistos:
            return
        vistos.add((documento, pagina, rotulo))
        fontes.append({"documento": documento, "rotulo": rotulo, "arquivo": documentos[documento]["arquivo"],
                       "pagina": pagina, "titulo": titulo})

    trechos_nbr = [*re.findall(r"NBR 6118(?::2023)?,?\s*([^;()]*)", texto),
                   *[parte for parte in trecho_das_fontes(texto).split(";")
                     if not re.search(r"Bastos|Fusco|Pinheiro|PINHEIRO|Scadelai|PP|TCC|Venturini|Kimura", parte)]]
    for trecho in trechos_nbr:
        for tipo, numero in PADRAO_TABELA_FIGURA.findall(trecho):
            chave = "%s %s" % (tipo, numero)
            if chave in paginas_nbr:
                pagina = paginas_nbr[chave]["pagina_pdf"]
                incluir("NBR", "NBR %s (p. %d)" % (chave, pagina_impressa_nbr(pagina)), pagina, paginas_nbr[chave]["titulo"])
        sem_tabelas = PADRAO_TABELA_FIGURA.sub("", trecho)
        for numero in PADRAO_SECAO_NBR.findall(sem_tabelas):
            if numero in paginas_nbr:
                pagina = paginas_nbr[numero]["pagina_pdf"]
                incluir("NBR", "NBR %s (p. %d)" % (numero, pagina_impressa_nbr(pagina)), pagina,
                        paginas_nbr[numero]["titulo"])
    for pagina in PADRAO_BASTOS.findall(texto):
        incluir("BASTOS", "Bastos (2017), p. %s" % pagina,
                pagina_pdf(int(pagina), documentos["BASTOS"]["deslocamento"]))
    for pagina in PADRAO_FUSCO.findall(texto):
        incluir("FUSCO", "Fusco (1981), p. %s" % pagina, pagina_pdf(int(pagina), documentos["FUSCO"]["deslocamento"]))
    for item in PADRAO_PINHEIRO.findall(texto):
        if item in documentos["PINHEIRO"]["itens"]:
            incluir("PINHEIRO", "Scadelai e Pinheiro (2007), item %s" % item, documentos["PINHEIRO"]["itens"][item])
    return fontes


# ----------------------------------------------------------------------------- pilares
def ler_resultados(arquivo_resultados: Path) -> list[dict]:
    with arquivo_resultados.open(encoding="utf-8") as arquivo:
        return list(csv.DictReader(arquivo, delimiter=";"))


def interacao(secao: nucleo.SecaoDirecao, material: nucleo.Material, area: float) -> dict:
    forcas, momentos = nucleo.diagrama_interacao(secao, material, area, PONTOS_INTERACAO)
    return {"N": arredondar(forcas, CASAS_GRAFICO), "M": arredondar(para_kn_m(momentos), CASAS_GRAFICO),
            "area": round(area, CASAS_GRAFICO)}


def curva_mn1r(secao: nucleo.SecaoDirecao, material: nucleo.Material, direcao: nucleo.ResultadoDirecao,
               area: float, forca_normal: float, comprimento: float) -> dict:
    forca_deformacao = nucleo.forca_normal_curva_deformacao(forca_normal)
    curvaturas, momentos = nucleo.curva_momento_curvatura(secao, material, area, forca_deformacao, True,
                                                          PONTOS_CURVATURA)
    externo = reta_momento_externo(direcao.alfa_b, direcao.momento_primeira_ordem, forca_normal, comprimento,
                                   curvaturas)
    return {"k": arredondar(para_por_km(curvaturas), CASAS_GRAFICO),
            "Mdef": arredondar(para_kn_m(momentos), CASAS_GRAFICO),
            "Mext": arredondar(para_kn_m(externo), CASAS_GRAFICO),
            "k_eq": None if not math.isfinite(direcao.curvatura) else round(float(para_por_km(direcao.curvatura)), 6),
            "M_eq": None if not math.isfinite(direcao.momento_total) else round(
                momento_reduzido(float(para_kn_m(direcao.momento_total))), 6)}


def graficos_do_pilar(caso: processar_matriz.CasoMatriz) -> dict:
    """Recalcula o pilar com o módulo de cálculo para gerar os gráficos (mesmos valores do CSV)."""
    resultado = nucleo.dimensionar_pilar(caso.pilar)
    pilar = resultado.pilar
    material = nucleo.Material(pilar.fck_mpa, pilar.fyk_mpa)
    candidato = resultado.candidato
    if candidato is not None:
        geometria, area = candidato, candidato.area
    else:
        geometria = nucleo.candidato_referencia(pilar)
        area = resultado.area_maxima
    graficos = {"barras": {"x": arredondar(geometria.x, 3), "y": arredondar(geometria.y, 3),
                           "diametro": geometria.diametro, "adotadas": candidato is not None}}
    envoltoria = resultado.envoltoria
    if envoltoria is not None:
        graficos["envoltoria"] = {
            "Mb": arredondar(para_kn_m(envoltoria.momentos_b), CASAS_GRAFICO),
            "Mh": arredondar(para_kn_m(envoltoria.momentos_h), CASAS_GRAFICO),
            "semieixo_b": arredondar(para_kn_m(envoltoria.direcao_b.momento_total), CASAS_GRAFICO)[0],
            "semieixo_h": arredondar(para_kn_m(envoltoria.direcao_h.momento_total), CASAS_GRAFICO)[0]}
        semieixos = (envoltoria.direcao_b.momento_total, envoltoria.direcao_h.momento_total)
        if all(math.isfinite(semieixo) for semieixo in semieixos):
            direcoes = nucleo.DIRECOES_ENVOLTORIA[::PASSO_PONTOS_ELIPSE]
            elipse_b, elipse_h = coordenadas_polares(nucleo.raio_envoltoria_minima(direcoes, *semieixos), direcoes)
            direcao_critica = np.radians(envoltoria.direcao_critica)
            critico_b, critico_h = coordenadas_polares(
                raio_resistente_critico(envoltoria.fator, float(nucleo.raio_envoltoria_minima(direcao_critica,
                                                                                              *semieixos))),
                direcao_critica)
            graficos["envoltoria"].update({
                "elipse_b": arredondar(para_kn_m(elipse_b), CASAS_GRAFICO),
                "elipse_h": arredondar(para_kn_m(elipse_h), CASAS_GRAFICO),
                "critico": arredondar(para_kn_m([critico_b, critico_h]), CASAS_GRAFICO)})
    for direcao, rotulo in ((resultado.direcao_b, nucleo.DIRECAO_B), (resultado.direcao_h, nucleo.DIRECAO_H)):
        secao = nucleo.secao_na_direcao(pilar.largura_bw, pilar.altura_h, geometria.x, geometria.y, rotulo)
        graficos["interacao_" + rotulo] = interacao(secao, material, area)
        if direcao.metodo == nucleo.METODO_MN1R and candidato is not None:
            graficos["mn1r_" + rotulo] = curva_mn1r(secao, material, direcao, direcao.area_necessaria,
                                                    resultado.forca_normal, resultado.comprimento_equivalente)
    return graficos


# ----------------------------------------------------------------------------- montagem do HTML

def endereco_bibliografia(padrao: str) -> str:
    """Endereço da pasta dos PDFs citados ([R01a]): a da variável de ambiente, se definida, ou o padrão relativo."""
    pasta = os.environ.get(VARIAVEL_BIBLIOGRAFIA, "").strip()
    return Path(pasta).resolve().as_uri() + "/" if pasta else padrao

def main() -> None:
    linhas_matriz = ler_resultados(ARQUIVO_RESULTADOS)
    linhas_teste = ler_resultados(ARQUIVO_TESTE_MINIMO)
    linhas_inviaveis = ler_resultados(ARQUIVO_INVIAVEIS)
    linhas = [*linhas_matriz, *linhas_teste]
    casos = {caso.identificador: caso for caso in [*processar_matriz.montar_matriz(),
                                                   *processar_matriz.montar_teste_minimo()]}
    with ProcessPoolExecutor() as executor:
        graficos = dict(zip([linha["ID"] for linha in linhas],
                            executor.map(graficos_do_pilar, [casos[linha["ID"]] for linha in linhas], chunksize=8)))
    validacao = json.loads(ARQUIVO_VALIDACAO.read_text(encoding="utf-8"))
    bibliografia = json.loads(ARQUIVO_BIBLIOGRAFIA.read_text(encoding="utf-8"))
    dados = {
        "versao": {"md5_resultados": hashlib.md5(ARQUIVO_RESULTADOS.read_bytes()).hexdigest(),
                   "python": sys.version.split()[0], "pilares": len(linhas_matriz),
                   "pilares_teste": len(linhas_teste)},
        "pasta_local": endereco_bibliografia(bibliografia["pasta_local"]),
        "documentos": {chave: {"arquivo": valor["arquivo"], "regra": valor["regra"]}
                       for chave, valor in bibliografia["documentos"].items()},
        "colunas": list(linhas[0].keys()), "pilares": linhas, "graficos": graficos,
        "inviaveis": {linha["ID"]: linha for linha in linhas_inviaveis},
        "formulas": blocos_do_nucleo(), "etiquetas_colunas": ETIQUETAS_COLUNAS,
        "testes": [{chave: valor_json(valor) for chave, valor in teste.items()} for teste in validacao["testes"]],
        "tolerancias": [{chave: valor_json(valor) for chave, valor in item.items()} for item in validacao["tolerancias"]],
        "explicacao_niveis": validacao["explicacao_niveis"],
        "engenharia_reversa": [{**{chave: valor_json(valor) for chave, valor in linha.items()},
                                "fontes": fontes_engenharia(linha, bibliografia["documentos"])}
                               for linha in validacao["engenharia_reversa_bastos"]],
        "niveis": {"lambda": processar_matriz.ESBELTEZES, "nu": processar_matriz.FORCAS_ADIMENSIONAIS,
                   "e1bw": processar_matriz.EXCENTRICIDADES_RELATIVAS,
                   "e1bw_teste": processar_matriz.EXCENTRICIDADE_TESTE_MINIMO},
        "notacao": nucleo.NOTACAO_DIMENSOES,
        "leitura_envoltoria": nucleo.LEITURA_ENVOLTORIA_MINIMA, "fator_minimo": nucleo.FATOR_MINIMO_ENVOLTORIA,
        "status": {"ok": nucleo.STATUS_OK, "A": nucleo.STATUS_ACIMA_MAXIMA, "E": nucleo.STATUS_SECAO_INSUFICIENTE,
                   "D": nucleo.STATUS_SEM_ARRANJO},
    }
    conteudo = json.dumps(dados, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    modelo = ARQUIVO_MODELO.read_text(encoding="utf-8")
    ARQUIVO_SAIDA.write_text(modelo.replace("/*__DADOS__*/null", conteudo.replace("</", "<\\/")), encoding="utf-8")
    print("auditor gravado em %s (%.1f MB)" % (ARQUIVO_SAIDA, megabytes(ARQUIVO_SAIDA.stat().st_size)))


if __name__ == "__main__":
    main()
