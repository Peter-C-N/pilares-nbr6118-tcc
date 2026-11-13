"""Programa principal da rotina: executa, com um único comando, todos os programas do TCC na ordem correta.

Uso (na pasta da rotina):
    python executar_rotina.py              rotina completa (sem a busca exaustiva, que é opcional e demorada)
    python executar_rotina.py --completo   rotina completa, com a busca exaustiva
    python executar_rotina.py --tqs        só a comparação com o TQS (exige as listagens em tqs/resultados_lote)
    python executar_rotina.py --lista      só mostra as etapas, o que cada uma faz e onde o TCC a descreve

Cada etapa é um programa desta pasta (ou da pasta tqs), executado em sequência com o mesmo Python; a execução para na
primeira etapa que falhar. Os programas que lançam e processam os pilares no TQS (pasta tqs/windows) não fazem parte
da sequência: rodam no computador com o TQS, com a interface de programação do programa (ver o LEIAME.txt).
"""
import subprocess
import sys
import time
from pathlib import Path

# ===================================================================================================================
# ZONA DE AUDITORIA
# ===================================================================================================================

# [E01] Etapas da rotina, na ordem de execução do LEIAME.txt: (programa, argumentos, o que faz, onde o TCC descreve,
#   opcional). A busca exaustiva só roda com --completo.
#   Variáveis: ETAPAS_ROTINA.
#   Usada em: etapas_escolhidas, listar
ETAPAS_ROTINA = [
    ("processar_matriz.py", [], "monta os 324 pilares-teste e os 9 do teste do momento mínimo e dimensiona todos",
     "Seções 3.6 e 4.2; Apêndice B", False),
    ("verificar_matriz.py", [], "confere as entradas e os resultados (40 requisitos)", "Seções 3.6 e 4.2", False),
    ("validacao.py", [], "validação interna em dois níveis (450 testes)", "Seções 3.7 e 4.1; Apêndice D", False),
    ("sensibilidade_secao_liquida.py", [], "sensibilidade ao diagrama de tensões e à seção de integração",
     "Seção 4.2 (Tabela 5)", False),
    ("conferir_busca_exaustiva.py", [], "conferência independente da escolha das barras (11 548 arranjos)",
     "Seção 4.2", True),
    ("figuras_tcc.py", [], "figuras calculadas pela rotina (domínios, interação, viabilidade e diagramas)",
     "Capítulos 2 e 4 e Apêndice G", False),
    ("gerar_auditor.py", [], "auditor visual (saida/auditor_pilares.html)", "Seção 4.5 (Figura 27)", False),
    ("figuras_complementares.py", [], "figuras da seção do V1, das curvas M, N, 1/r, da taxa e do auditor",
     "Seções 4.1, 4.2 e 4.5", False),
    ("validacao_literatura.py", [], "engenharia reversa dos exemplos resolvidos da literatura",
     "Seções 3.7 e 4.1; Apêndice F", False),
    ("memorial_pilares.py", [], "memorial de cálculo dos 333 pilares (736 páginas)", "Seção 4.5; Apêndices D e E",
     False),
    ("tqs/gerar_entrada_tqs.py", [], "monta a tabela dos pilares a lançar no TQS", "Seção 3.8.1", False),
    ("rastreabilidade.py", ["--conferir"], "confere a zona de auditoria e a ausência de fórmulas no corpo",
     "Seção 4.5", False),
]

# [E02] Etapas da comparação com o TQS (pasta tqs), depois do lançamento no computador com o TQS (a tabela de entrada
#   do TQS é montada na sequência da rotina, [E01]). Exigem as listagens e
#   os relatórios gravados pelo TQS em tqs/resultados_lote, que não fazem parte da versão pública.
#   Variáveis: ETAPAS_TQS, PASTA_LISTAGENS_TQS.
#   Usada em: etapas_escolhidas, listar, executar
ETAPAS_TQS = [
    ("tqs/comparar_tqs_lote.py", [], "lê os resultados do TQS e faz a comparação em duas camadas",
     "Seções 3.9 e 4.3", False),
    ("tqs/resumir_comparacao_tqs.py", [], "resumo estatístico da comparação", "Seção 4.3", False),
    ("tqs/decompor_curvatura_tqs.py", [], "decompõe a diferença nos pilares com curvatura aproximada", "Seção 4.4",
     False),
    ("tqs/camada2_nm1r_tqs.py", [], "aplica o método da rotina às barras do TQS", "Seção 4.4", False),
    ("tqs/replica_tqs_nm1r.py", [], "refaz o cálculo do TQS nos pilares com pilar-padrão acoplado", "Seção 4.4",
     False),
    ("tqs/teste_minimo_tqs.py", [], "analisa o teste do momento mínimo no TQS", "Seção 4.4", False),
]
PASTA_LISTAGENS_TQS = Path("tqs") / "resultados_lote"

# [E03] Programas executados fora desta sequência, no computador com o TQS (Windows e interface de programação do TQS).
#   Variáveis: PROGRAMAS_COMPUTADOR_TQS.
#   Usada em: listar
PROGRAMAS_COMPUTADOR_TQS = [
    ("tqs/windows/tqs_00_diagnostico.py", "lê o edifício-modelo", "Seção 3.8.1"),
    ("tqs/windows/tqs_01_lote.py", "lança e processa os pilares e copia as listagens", "Seção 3.8.1"),
]


# [E04] Duração de uma etapa, em segundos.
#   Variáveis: inicio, fim (s, relógio monotônico).
#   Usada em: executar (registro do tempo de cada etapa)
def duracao(inicio: float, fim: float) -> float:
    return fim - inicio


# ===================================================================================================================
# EXECUÇÃO
# ===================================================================================================================

PASTA = Path(__file__).resolve().parent


def etapas_escolhidas(argumentos: list[str]) -> list:
    if "--tqs" in argumentos:
        return ETAPAS_TQS
    return [etapa for etapa in ETAPAS_ROTINA if not etapa[4] or "--completo" in argumentos]


def listar() -> None:
    print("Rotina (python executar_rotina.py; a busca exaustiva só com --completo):")
    for numero, (programa, argumentos, descricao, secao, opcional) in enumerate(ETAPAS_ROTINA, start=1):
        print("  %2d. %-32s %s%s [TCC: %s]" % (numero, " ".join([programa] + argumentos), descricao,
                                                " (opcional)" if opcional else "", secao))
    print("No computador com o TQS (fora da sequência):")
    for programa, descricao, secao in PROGRAMAS_COMPUTADOR_TQS:
        print("      %-32s %s [TCC: %s]" % (programa, descricao, secao))
    print("Comparação com o TQS (python executar_rotina.py --tqs; exige %s):" % PASTA_LISTAGENS_TQS.as_posix())
    for numero, (programa, argumentos, descricao, secao, _) in enumerate(ETAPAS_TQS, start=1):
        print("  %2d. %-32s %s [TCC: %s]" % (numero, programa, descricao, secao))


def executar(etapas: list) -> int:
    for numero, (programa, argumentos, descricao, _, _) in enumerate(etapas, start=1):
        caminho = PASTA / programa
        print("[%d/%d] %s: %s" % (numero, len(etapas), programa, descricao), flush=True)
        inicio = time.monotonic()
        retorno = subprocess.run([sys.executable, caminho.name] + argumentos, cwd=caminho.parent).returncode
        tempo = duracao(inicio, time.monotonic())
        if retorno != 0:
            print("ERRO: %s terminou com código %d (%.0f s); execução interrompida" % (programa, retorno, tempo))
            return retorno
        print("      concluído em %.0f s" % tempo, flush=True)
    print("Todas as %d etapas concluídas." % len(etapas))
    return 0


def principal(argumentos: list[str]) -> int:
    if "--lista" in argumentos:
        listar()
        return 0
    if "--tqs" in argumentos and not (PASTA / PASTA_LISTAGENS_TQS).is_dir():
        print("A comparação com o TQS exige as listagens e os relatórios do TQS em %s, que não fazem parte da versão "
              "pública; ver o LEIAME.txt." % PASTA_LISTAGENS_TQS.as_posix())
        return 1
    return executar(etapas_escolhidas(argumentos))


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))
