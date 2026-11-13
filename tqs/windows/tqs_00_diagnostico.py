# -*- coding: utf-8 -*-
"""Diagnóstico (somente leitura) do modelo de um edifício do TQS, para calibrar o script do lote.

Lê, pela TQSPythonInterface, os dados do edifício (plantas, pés direitos, concreto dos pilares) e de cada seção de
pilar do modelo (título, número, "nasce em", inserção, geometria, ângulo, polígono, planta final e dados por
planta: fck, cobrimento, articulações, pé direito duplo e coeficientes de flambagem). Não grava nada no edifício.
O resultado vai para o arquivo diagnostico_<EDIFICIO>.txt, na pasta deste script.

Uso (Prompt de Comando, com o TQS fechado):
    py -3.13 tqs_00_diagnostico.py TCCPILAR
"""
import sys
import traceback
from pathlib import Path

from TQS import TQSBuild, TQSModel

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — constantes (este script não tem fórmulas de cálculo)
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [D01] Edifício padrão, se nenhum for passado na linha de comando.  Usada em: principal
EDIFICIO_PADRAO = "TCCPILAR"

# [D02] Pasta e nome do relatório de saída.  Usada em: principal
PASTA_SAIDA = Path(__file__).resolve().parent
PREFIXO_RELATORIO = "diagnostico_"

# [D03] Tipo de objeto do modelo lido: seções de pilar (TQSModel.TYPE_SECPIL = 13).  Usada em: listar_planta
TIPO_SECAO_PILAR = TQSModel.TYPE_SECPIL

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


class Relatorio:
    def __init__(self) -> None:
        self.linhas: list[str] = []

    def escrever(self, texto: str) -> None:
        self.linhas.append(texto)
        print(texto)

    def gravar(self, caminho: Path) -> None:
        caminho.write_text("\n".join(self.linhas) + "\n", encoding="utf-8")


def ler_atributo(objeto, nome: str):
    """Lê um atributo da API sem interromper o diagnóstico se ele não existir nesta versão."""
    try:
        return getattr(objeto, nome)
    except Exception as erro:  # noqa: BLE001 (diagnóstico: registra e segue)
        return "ERRO(%s)" % erro


def listar_dados_edificio(relatorio: Relatorio, edificio) -> list[str]:
    plantas = edificio.floorsplan
    quantidade = ler_atributo(plantas, "floorsPlanNumber")
    relatorio.escrever("Plantas: %s" % quantidade)
    nomes = []
    for indice in range(int(quantidade)):
        nome = plantas.GetName(indice) if hasattr(plantas, "GetName") else "?"
        pe_direito = plantas.GetFloorHeight(indice)
        relatorio.escrever("  planta %d: nome=%r pe_direito_m=%r" % (indice, nome, pe_direito))
        nomes.append(nome)
    materiais = edificio.materials
    relatorio.escrever("Concreto dos pilares (edifício): %r" % ler_atributo(materiais, "columnConcreteStrength"))
    relatorio.escrever("Cobrimento de pilares (edifício): %r" % ler_atributo(edificio.covers, "columnCover"))
    return nomes


def listar_secao(relatorio: Relatorio, planta, secao) -> None:
    pilar = secao.column
    relatorio.escrever("  Pilar titulo=%r numero=%r nasce_em=%r" % (ler_atributo(pilar, "title"),
                                                                     ler_atributo(pilar, "number"),
                                                                     ler_atributo(pilar, "columnStarts")))
    relatorio.escrever("    insercao x=%r y=%r ate_planta=%r" % (ler_atributo(secao, "insertionX"),
                                                                ler_atributo(secao, "insertionY"),
                                                                ler_atributo(secao, "sectionLastFloor")))
    geometria = secao.columnGeometry
    relatorio.escrever("    geometria tipo=%r B1=%r H1=%r angulo=%r" % (ler_atributo(geometria, "sectionType"),
                                                                      ler_atributo(geometria, "sectionB1"),
                                                                      ler_atributo(geometria, "sectionH1"),
                                                                      ler_atributo(geometria, "sectionAngle")))
    insercao = secao.columnInsertion
    relatorio.escrever("    insercao tipo=%r canto=%r" % (ler_atributo(insercao, "insertionType"),
                                                         ler_atributo(insercao, "insertionCorner")))
    try:
        poligono = secao.sectionPolig
        pontos = [poligono.GetPt(indice) for indice in range(poligono.NumPts())]
        relatorio.escrever("    poligono=%r" % (pontos,))
    except Exception as erro:  # noqa: BLE001
        relatorio.escrever("    poligono ERRO(%s)" % erro)
    try:
        dados_planta = secao.GetColumnFloorData(planta)
        relatorio.escrever("    dados_planta fck=%r cobrimento=%r articulacoes=%r pe_direito_duplo_x=%r "
                           "pe_direito_duplo_y=%r flambagem_x=%r flambagem_y=%r por_planta=%r"
                           % (ler_atributo(dados_planta, "columnConcreteFc"),
                              ler_atributo(dados_planta, "columnCover"),
                              ler_atributo(dados_planta, "columnHinges"),
                              ler_atributo(dados_planta, "columnDoubleStoryX"),
                              ler_atributo(dados_planta, "columnDoubleStoryY"),
                              ler_atributo(dados_planta, "columnBucklingX"),
                              ler_atributo(dados_planta, "columnBucklingY"),
                              ler_atributo(dados_planta, "infoByFloor")))
    except Exception as erro:  # noqa: BLE001
        relatorio.escrever("    dados_planta ERRO(%s)" % erro)


def listar_planta(relatorio: Relatorio, modelo, nome_planta: str) -> None:
    planta = modelo.floors.GetFloor(nome_planta)
    quantidade = planta.iterator.GetNumObjects(TIPO_SECAO_PILAR)
    relatorio.escrever("Planta %r: %d seções de pilar" % (nome_planta, quantidade))
    for indice in range(quantidade):
        secao = planta.iterator.GetObject(TIPO_SECAO_PILAR, indice)
        if secao is not None:
            listar_secao(relatorio, planta, secao)


def principal() -> None:
    nome_edificio = sys.argv[1] if len(sys.argv) > 1 else EDIFICIO_PADRAO
    relatorio = Relatorio()
    caminho_saida = PASTA_SAIDA / (PREFIXO_RELATORIO + nome_edificio + ".txt")
    try:
        relatorio.escrever("Diagnóstico do edifício %s" % nome_edificio)
        edificio = TQSBuild.Building()
        if edificio.file.Open(nome_edificio) != 0:
            relatorio.escrever("ERRO: não abri os dados do edifício %s" % nome_edificio)
            return
        nomes_plantas = listar_dados_edificio(relatorio, edificio)
        edificio_modelo = TQSBuild.Building()
        if edificio_modelo.RootFolder(nome_edificio) != 0:
            relatorio.escrever("ERRO: RootFolder(%s) falhou" % nome_edificio)
            return
        modelo = TQSModel.Model()
        if modelo.file.OpenModel() != 0:
            relatorio.escrever("ERRO: não abri o modelo")
            return
        lista = modelo.floors
        for indice in range(1, lista.GetNumFloors() + 1):
            listar_planta(relatorio, modelo, lista.GetFloorName(indice))
        modelo.file.Close()
        relatorio.escrever("Plantas dos dados do edifício: %r" % nomes_plantas)
        relatorio.escrever("FIM")
    except Exception:  # noqa: BLE001
        relatorio.escrever("EXCEÇÃO:\n" + traceback.format_exc())
    finally:
        relatorio.gravar(caminho_saida)
        print("Relatório gravado em %s" % caminho_saida)


if __name__ == "__main__":
    principal()
