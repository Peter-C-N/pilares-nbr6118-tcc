# -*- coding: utf-8 -*-
"""Lançamento e processamento dos pilares-teste no TQS, pela TQSPythonInterface.

Para cada edifício da tabela entrada_tqs_lote.csv (gerada por gerar_entrada_tqs.py, com as contas na zona de
auditoria daquele script):
  1. copia o edifício base (TCCPILAR, já configurado: critérios, unidades, cobrimento, fluência) para o edifício
     do lote e ajusta o pé direito da planta PILARES;
  2. apaga o modelo copiado e lança, na planta PILARES, um pilar por linha da tabela (nasce no solo; seção,
     classe do concreto, pé direito duplo "Não", coeficiente de flambagem 1) e a carga concentrada no topo
     (Fz e My, parcelas permanente e acidental, exatamente sobre o eixo do pilar);
  3. processa todos os edifícios em uma fila do gerenciador (processamento global com pórtico e pilares);
  4. copia as listagens de cada edifício para a pasta lote, ao lado deste script, para leitura posterior.

Uso (Prompt de Comando na pasta deste script, com o TQS fechado):
    py -3.13 tqs_01_lote.py validacao          -> TCCV015 e TCCV165 (comparação com os modelos feitos à mão)
    py -3.13 tqs_01_lote.py lote               -> TCCL01 a TCCL16 (212 pilares)
    py -3.13 tqs_01_lote.py minimo             -> TCCM01 a TCCM09 (teste do momento mínimo, M01 a M09)
    py -3.13 tqs_01_lote.py edificio TCCL04    -> um edifício
    Acrescente "sem_processar" para só lançar os modelos, ou "so_copiar" para só copiar as listagens.
"""
import csv
import shutil
import sys
import time
import traceback
from pathlib import Path

from TQS import TQSBuild, TQSExec, TQSModel

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — constantes (as contas de cargas e pés direitos estão em gerar_entrada_tqs.py)
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [D01] Pastas e arquivos.  Usada em: principal; copiar_resultados; Registro
PASTA_SCRIPT = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_SCRIPT / "entrada_tqs_lote.csv"
PASTA_RAIZ_TQS = Path(r"C:\TQS")
PASTA_RESULTADOS = PASTA_SCRIPT / "lote"
ARQUIVO_REGISTRO = PASTA_SCRIPT / "registro_tqs_01_lote.txt"

# [D02] Edifício base e plantas (dados do edifício TCCPILAR: Fundacao = 0, PILARES = 1). O pilar e a carga são
#   lançados na planta PILARES, onde o P165 feito à mão foi desenhado.
#   Usada em: preparar_edificio; lancar_modelo
EDIFICIO_BASE = "TCCPILAR"
NOME_PLANTA_FUNDACAO = "Fundacao"
NOME_PLANTA_PILARES = "PILARES"
INDICE_PLANTA_PILARES = 1

# [D03] Prefixos dos grupos de edifícios na tabela.  Usada em: selecionar_edificios
PREFIXO_VALIDACAO = "TCCV"
PREFIXO_LOTE = "TCCL"
PREFIXO_MINIMO = "TCCM"

# [D04] Dados do pilar no Modelador, iguais aos do P165 feito à mão (diagnostico_TCCPILAR.txt, 04/11/2026):
#   - nasce direto no solo (COLUMNSTART_NASCEDIRT = 0), com a seção nas plantas Fundacao e PILARES;
#   - inserção pelo centro; seção retangular com B1 = b_w ao longo de X, H1 = h ao longo de Y e ângulo 0
#     (o "ANG = 90" das listagens do TQS Pilar é a convenção do próprio TQS Pilar);
#   - pé direito duplo desligado nas duas direções. A docstring da API diz "(0) Geometria (1) Não (2) Sim", mas
#     o campo da DLL é ITRAVADOX/ITRAVADOY ("travado"): na validação de 04/11/2026, 11:55, o valor 1 gerou
#     PEDDUP.DAT com o código 1 (pé direito duplo, λ = 180 no P015). Usa-se 2 ("travado: sim"), a confirmar no
#     PEDDUP.DAT (código 0) e no λ da listagem;
#   - coeficiente de flambagem 1 (ℓe = pé direito).
#   Usada em: definir_dados_pilar
NASCE_EM = TQSModel.COLUMNSTART_NASCEDIRT
INSERCAO_CENTRO = 0
TIPO_SECAO_RETANGULAR = TQSModel.COLUMNTYPE_R
ANGULO_SECAO_GRAUS = 0.0
PE_DIREITO_DUPLO_NAO = 2
COEFICIENTE_FLAMBAGEM = 1.0

# [D05] Cargas: caso 1 ("TODAS"), componentes da API em tf e tf·m; o momento é o My (calibrado no P015 e P165:
#   flete a face de b_w).  Usada em: lancar_carga
CASO_CARGA = 1
COMPONENTES_ZERADAS = (TQSModel.TPLOAD_FORCFX, TQSModel.TPLOAD_FORCFY, TQSModel.TPLOAD_FORCMX)
COMPONENTE_FORCA = TQSModel.TPLOAD_FORCFZ
COMPONENTE_MOMENTO = TQSModel.TPLOAD_FORCMY
VALOR_ZERO = 0.0

# [D06] Processamento global (mesmas opções da tela usada nos modelos feitos à mão em 04/11/2026).
#   Usada em: processar
OPCOES_PROCESSAMENTO = dict(floorPlan=2, floorDraw=0, slabs=0, beams=1, columnsData=1, columns=2,
                            columnsReport=1, gridModel=1, gridDraw=1, gridExtr=1, gridAnalysis=1,
                            gridBeamsTrnsf=0, gridSlabsTrnsf=1, gridNonLinear=0, frameModel=1, frameAnalysis=1,
                            frameBeamsTrnsf=1, frameColumnsTrnsf=1, foundations=0, stairs=0, fire=0)

# [D07] Arquivos copiados de cada edifício (pasta, padrão).  Usada em: copiar_resultados
ARQUIVOS_RESULTADO = (("PILAR", "*.LST"), ("PILAR", "*.PIL"), ("PILAR", "*.TEP"), ("PILAR", "*.DAT"),
                      ("PILAR", "Pilar.rep"), ("PILAR/0001", "*.LST"), ("ESPACIAL", "FOR0001.POR"),
                      ("ESPACIAL", "FOR0001.Lst"), ("ESPACIAL", "CARPIL.LST"), ("ESPACIAL", "CRITPOR.DAT"),
                      ("ESPACIAL", "PGLOERR.HTM"))

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


class Registro:
    def __init__(self) -> None:
        self.arquivo = open(ARQUIVO_REGISTRO, "a", encoding="utf-8")

    def escrever(self, texto: str) -> None:
        linha = "%s  %s" % (time.strftime("%d/%m/%Y %H:%M:%S"), texto)
        print(linha)
        self.arquivo.write(linha + "\n")
        self.arquivo.flush()

    def fechar(self) -> None:
        self.arquivo.close()


def ler_entrada() -> dict[str, list[dict]]:
    with open(ARQUIVO_ENTRADA, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo, delimiter=";"))
    por_edificio: dict[str, list[dict]] = {}
    for linha in linhas:
        por_edificio.setdefault(linha["edificio"], []).append(linha)
    for pilares in por_edificio.values():
        pilares.sort(key=lambda item: int(item["ordem"]))
    return por_edificio


def selecionar_edificios(por_edificio: dict[str, list[dict]], argumentos: list[str]) -> list[str]:
    if argumentos[0] == "validacao":
        return sorted(nome for nome in por_edificio if nome.startswith(PREFIXO_VALIDACAO))
    if argumentos[0] == "lote":
        return sorted(nome for nome in por_edificio if nome.startswith(PREFIXO_LOTE))
    if argumentos[0] == "minimo":
        return sorted(nome for nome in por_edificio if nome.startswith(PREFIXO_MINIMO))
    if argumentos[0] == "edificio" and len(argumentos) > 1 and argumentos[1] in por_edificio:
        return [argumentos[1]]
    raise SystemExit("Uso: py -3.13 tqs_01_lote.py validacao | lote | minimo | edificio NOME [sem_processar | so_copiar]")


def preparar_edificio(registro: Registro, nome: str, pe_direito_m: float) -> None:
    """Copia o edifício base com o pé direito do grupo (ou reabre a cópia, se já existir)."""
    edificio = TQSBuild.Building()
    origem = nome if (PASTA_RAIZ_TQS / nome).is_dir() else EDIFICIO_BASE
    if edificio.file.Open(origem) != 0:
        raise RuntimeError("não abri o edifício %s" % origem)
    edificio.floorsplan.SetFloorHeight(INDICE_PLANTA_PILARES, pe_direito_m)
    edificio.file.SuppressMessages(1)
    gravou = edificio.file.Save() if origem == nome else edificio.file.SaveAs(nome)
    if gravou != 0:
        raise RuntimeError("não gravei o edifício %s" % nome)
    lido = edificio.floorsplan.GetFloorHeight(INDICE_PLANTA_PILARES)
    registro.escrever("%s: edifício gravado a partir de %s; pé direito pedido %.3f m, lido %r m"
                      % (nome, origem, pe_direito_m, lido))


def definir_dados_pilar(modelo, pilar: dict) -> None:
    dados = modelo.current.globalColumnData.columnData
    dados.columnStarts = NASCE_EM
    dados.columnInsertion.insertionType = INSERCAO_CENTRO
    dados.columnConcreteFc = pilar["classe_concreto"]
    dados.columnBucklingX = COEFICIENTE_FLAMBAGEM
    dados.columnBucklingY = COEFICIENTE_FLAMBAGEM
    dados.columnDoubleStoryX = PE_DIREITO_DUPLO_NAO
    dados.columnDoubleStoryY = PE_DIREITO_DUPLO_NAO
    geometria = dados.columnGeometry
    geometria.sectionType = TIPO_SECAO_RETANGULAR
    geometria.sectionB1 = float(pilar["b_w_cm"])
    geometria.sectionH1 = float(pilar["h_cm"])
    geometria.sectionAngle = ANGULO_SECAO_GRAUS


def lancar_carga(planta_topo, pilar: dict, x_cm: float, y_cm: float):
    dados_carga = planta_topo.current.floorLoadData
    for componente in COMPONENTES_ZERADAS:
        carga_zerada = dados_carga.GetLoad(componente)
        carga_zerada.SetMainLoad(CASO_CARGA, VALOR_ZERO)
        carga_zerada.SetLiveLoad(CASO_CARGA, VALOR_ZERO)
    carga_forca = dados_carga.GetLoad(COMPONENTE_FORCA)
    carga_forca.SetLoad(TQSModel.LOADTYPE_CONCEN)
    carga_forca.SetMainLoad(CASO_CARGA, float(pilar["fz_permanente_tf"]))
    carga_forca.SetLiveLoad(CASO_CARGA, float(pilar["fz_acidental_tf"]))
    carga_momento = dados_carga.GetLoad(COMPONENTE_MOMENTO)
    carga_momento.SetLoad(TQSModel.LOADTYPE_MOMCON)
    carga_momento.SetMainLoad(CASO_CARGA, float(pilar["my_permanente_tfm"]))
    carga_momento.SetLiveLoad(CASO_CARGA, float(pilar["my_acidental_tfm"]))
    return planta_topo.create.CreateConcentratedLoad(x_cm, y_cm)


def lancar_modelo(registro: Registro, nome: str, pilares: list[dict]) -> None:
    edificio = TQSBuild.Building()
    if edificio.RootFolder(nome) != 0:
        raise RuntimeError("RootFolder(%s) falhou" % nome)
    modelo = TQSModel.Model()
    if modelo.file.OpenNewModel() != 0:
        raise RuntimeError("não abri modelo novo em %s" % nome)
    modelo.floors.GetFloor(NOME_PLANTA_FUNDACAO)
    planta_topo = modelo.floors.GetFloor(NOME_PLANTA_PILARES)
    for pilar in pilares:
        x_cm = float(pilar["x_cm"])
        y_cm = float(pilar["y_cm"])
        definir_dados_pilar(modelo, pilar)
        objeto_pilar = planta_topo.create.CreateColumn(x_cm, y_cm)
        objeto_pilar.title = pilar["id"]
        lancar_carga(planta_topo, pilar, x_cm, y_cm)
        registro.escrever("%s: pilar %s lançado (x = %s cm; %s x %s cm; %s; Fz %s / %s tf; My %s / %s tf·m)"
                          % (nome, pilar["id"], pilar["x_cm"], pilar["b_w_cm"], pilar["h_cm"],
                             pilar["classe_concreto"], pilar["fz_permanente_tf"], pilar["fz_acidental_tf"],
                             pilar["my_permanente_tfm"], pilar["my_acidental_tfm"]))
    if modelo.file.Save() != 0:
        raise RuntimeError("não salvei o modelo de %s" % nome)
    modelo.file.Close()


def processar(registro: Registro, nomes: list[str]) -> float:
    fila = TQSExec.Job()
    for nome in nomes:
        fila.EnterTask(TQSExec.TaskFolder(nome, TQSExec.TaskFolder.FOLDER_FRAMES))
        fila.EnterTask(TQSExec.TaskGlobalProc(**OPCOES_PROCESSAMENTO))
    inicio = time.time()
    registro.escrever("processamento de %d edifício(s) iniciado" % len(nomes))
    fila.Execute()
    registro.escrever("processamento terminado em %.0f s" % (time.time() - inicio))
    return inicio


def copiar_resultados(registro: Registro, nome: str, inicio_processamento: float) -> None:
    destino_edificio = PASTA_RESULTADOS / nome
    if destino_edificio.is_dir():
        shutil.rmtree(destino_edificio)
    copiados = 0
    for subpasta, padrao in ARQUIVOS_RESULTADO:
        origem = PASTA_RAIZ_TQS / nome / subpasta
        destino = destino_edificio / subpasta
        destino.mkdir(parents=True, exist_ok=True)
        for arquivo in origem.glob(padrao):
            shutil.copy2(arquivo, destino / arquivo.name)
            copiados += 1
    listagem = PASTA_RAIZ_TQS / nome / "PILAR" / "LISPIL.LST"
    atual = listagem.is_file() and listagem.stat().st_mtime >= inicio_processamento
    registro.escrever("%s: %d arquivos copiados para %s; LISPIL.LST %s"
                      % (nome, copiados, destino_edificio, "NOVA" if atual else "ANTIGA OU AUSENTE (conferir)"))


def principal() -> None:
    argumentos = sys.argv[1:] or ["validacao"]
    registro = Registro()
    try:
        por_edificio = ler_entrada()
        nomes = selecionar_edificios(por_edificio, argumentos)
        registro.escrever("=== tqs_01_lote %s: %s" % (" ".join(argumentos), ", ".join(nomes)))
        inicio = 0.0
        if "so_copiar" not in argumentos:
            for nome in nomes:
                pilares = por_edificio[nome]
                preparar_edificio(registro, nome, float(pilares[0]["pe_direito_m"]))
                lancar_modelo(registro, nome, pilares)
            if "sem_processar" in argumentos:
                registro.escrever("modelos lançados; processamento não pedido")
                return
            inicio = processar(registro, nomes)
        for nome in nomes:
            copiar_resultados(registro, nome, inicio)
        registro.escrever("FIM")
    except Exception:  # noqa: BLE001
        registro.escrever("EXCEÇÃO:\n" + traceback.format_exc())
    finally:
        registro.fechar()


if __name__ == "__main__":
    principal()
