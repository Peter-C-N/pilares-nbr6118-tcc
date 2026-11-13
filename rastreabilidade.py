# -*- coding: utf-8 -*-
"""
rastreabilidade.py
==================
Gera e confere os comentários de rastreabilidade da ZONA DE AUDITORIA dos scripts do TCC.

Para cada item da zona de auditoria (constante, fórmula, regra ou método numérico identificado por uma
etiqueta como [F21]), o programa:
    1) localiza, pela árvore sintática (módulo ast), todas as linhas em que o item é invocado, nos
       próprios scripts e nos demais scripts do projeto, e escreve a linha "Usada em:" com a função
       e a linha de cada uso;
    2) escreve a linha "TCC:" com o número da equação correspondente no texto do TCC, lido do arquivo
       gerado pelo montador do documento (equacoes.json);
    3) verifica que, abaixo da zona de auditoria (motor de cálculo), não há operações aritméticas,
       isto é, que nenhuma fórmula foi escrita diretamente no corpo do programa.

Uso:
    python3 rastreabilidade.py            -> atualiza os comentários (repete até estabilizar as linhas)
    python3 rastreabilidade.py --conferir -> apenas confere; termina com erro se algo estiver desatualizado
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — INÍCIO
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [R01] Scripts que compõem a rotina e são conferidos (na ordem em que aparecem nas listagens).
#   Variáveis: PASTA_SCRIPTS, SCRIPTS.
#   Usada em: <AUTO>
PASTA_SCRIPTS = Path(__file__).resolve().parent
SCRIPTS = ("pilares_nbr6118.py", "validacao.py", "processar_matriz.py", "verificar_matriz.py",
           "sensibilidade_secao_liquida.py", "conferir_busca_exaustiva.py", "figuras_tcc.py", "gerar_auditor.py",
           "figuras_complementares.py", "validacao_literatura.py", "memorial_pilares.py")

# [R02] Marcadores que delimitam a zona de auditoria e padrões das linhas geradas.
#   Variáveis: MARCADOR_INICIO, MARCADOR_FIM, PADRAO_ETIQUETA, PREFIXO_USO, PREFIXO_USO_EXTERNO,
#   PREFIXO_TCC, PREFIXO_CONTINUACAO, LARGURA_MAXIMA_LINHA.
#   Usada em: <AUTO>
MARCADOR_INICIO = "ZONA DE AUDITORIA — INÍCIO"
MARCADOR_FIM = "ZONA DE AUDITORIA — FIM"
PADRAO_ETIQUETA = re.compile(r"^# \[([A-Z]\d{2})\]")
PREFIXO_USO = "#   Usada em: "
PREFIXO_USO_EXTERNO = "#   Usada também em: "
PREFIXO_TCC = "#   TCC: "
PREFIXO_CONTINUACAO = "#       "
LARGURA_MAXIMA_LINHA = 118

# [R03] Mapa "script:etiqueta" -> número(s) da equação no texto do TCC, gravado pelo montador do documento
#   (montar_tcc_v2.py) ao identificar cada equação numerada do texto.
#   Variáveis: ARQUIVO_EQUACOES.
#   Usada em: <AUTO>
ARQUIVO_EQUACOES = PASTA_SCRIPTS / "equacoes_tcc.json"

# [R04] Operações aritméticas proibidas no motor de cálculo (fórmulas devem estar na zona de auditoria).
#   Exceções admitidas (não são fórmulas de cálculo): soma ou subtração de 1 a contadores e índices;
#   concatenação de tuplas; formatação de texto com '%'; concatenação de textos (um dos operandos é
#   texto literal) e junção de caminhos de arquivo (operador '/' com texto literal à direita).
#   Variáveis: OPERACOES_PROIBIDAS.
#   Usada em: <AUTO>
OPERACOES_PROIBIDAS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.FloorDiv, ast.Mod, ast.MatMult)

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


def ler_mapa_equacoes() -> dict[str, list[int]]:
    if not ARQUIVO_EQUACOES.exists():
        return {}
    return json.loads(ARQUIVO_EQUACOES.read_text(encoding="utf8"))


def limites_zona(linhas: list[str]) -> tuple[int, int]:
    """Índices (base 0) das linhas dos marcadores de início e fim da zona de auditoria."""
    inicio = next(indice for indice, linha in enumerate(linhas) if MARCADOR_INICIO in linha)
    fim = next(indice for indice, linha in enumerate(linhas) if MARCADOR_FIM in linha)
    return inicio, fim


def blocos_da_zona(linhas: list[str], arvore: ast.Module) -> list[dict]:
    """Blocos de comentário etiquetados e os nomes definidos pelos comandos que os seguem."""
    inicio, fim = limites_zona(linhas)
    blocos = []
    indice = inicio
    while indice < fim:
        etiqueta = PADRAO_ETIQUETA.match(linhas[indice])
        if not etiqueta:
            indice += 1
            continue
        primeira = indice
        while indice < fim and linhas[indice].startswith("#"):
            indice += 1
        blocos.append({"etiqueta": etiqueta.group(1), "primeira": primeira, "ultima": indice - 1, "nomes": []})
    for posicao, bloco in enumerate(blocos):
        linha_inicial = bloco["ultima"] + 2
        linha_final = blocos[posicao + 1]["primeira"] if posicao + 1 < len(blocos) else fim
        for comando in arvore.body:
            if not linha_inicial <= comando.lineno <= linha_final:
                continue
            if isinstance(comando, (ast.FunctionDef, ast.ClassDef)):
                bloco["nomes"].append(comando.name)
            elif isinstance(comando, ast.Assign):
                bloco["nomes"].extend(alvo.id for alvo in comando.targets if isinstance(alvo, ast.Name))
            elif isinstance(comando, ast.AnnAssign) and isinstance(comando.target, ast.Name):
                bloco["nomes"].append(comando.target.id)
    return blocos


def apelidos_de_modulos(arvore: ast.Module) -> tuple[dict[str, str], dict[str, str]]:
    """(apelido -> módulo) dos 'import x as y' e (nome -> módulo) dos 'from x import nome' de scripts do projeto."""
    modulos_do_projeto = {Path(script).stem for script in SCRIPTS}
    apelidos, importados = {}, {}
    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            for nome in no.names:
                if nome.name in modulos_do_projeto:
                    apelidos[nome.asname or nome.name] = nome.name
        elif isinstance(no, ast.ImportFrom) and no.module in modulos_do_projeto:
            for nome in no.names:
                importados[nome.asname or nome.name] = no.module
    return apelidos, importados


def funcao_envolvente(no: ast.AST, pais: dict[ast.AST, ast.AST]) -> str:
    nomes = []
    atual = pais.get(no)
    while atual is not None:
        if isinstance(atual, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nomes.append(atual.name)
        atual = pais.get(atual)
    return ".".join(reversed(nomes)) if nomes else "nível do módulo"


def coletar_usos(arvores: dict[str, ast.Module], definicoes: dict[str, dict[str, str]]) -> dict:
    """usos[(modulo_definidor, nome)] = lista de (script, função, linha)."""
    usos: dict[tuple[str, str], list[tuple[str, str, int]]] = {}
    for script, arvore in arvores.items():
        modulo = Path(script).stem
        apelidos, importados = apelidos_de_modulos(arvore)
        pais = {filho: pai for pai in ast.walk(arvore) for filho in ast.iter_child_nodes(pai)}
        for no in ast.walk(arvore):
            chave = None
            if isinstance(no, ast.Name) and isinstance(no.ctx, ast.Load):
                if no.id in definicoes.get(modulo, {}):
                    chave = (modulo, no.id)
                elif no.id in importados and no.id in definicoes.get(importados[no.id], {}):
                    chave = (importados[no.id], no.id)
            elif isinstance(no, ast.Attribute) and isinstance(no.value, ast.Name) and no.value.id in apelidos:
                modulo_alvo = apelidos[no.value.id]
                if no.attr in definicoes.get(modulo_alvo, {}):
                    chave = (modulo_alvo, no.attr)
            if chave:
                usos.setdefault(chave, []).append((script, funcao_envolvente(no, pais), no.lineno))
    return usos


def formatar_lista(prefixo: str, itens: list[str]) -> list[str]:
    """Quebra a lista em linhas de comentário de até LARGURA_MAXIMA_LINHA caracteres."""
    linhas, atual = [], prefixo
    for posicao, item in enumerate(itens):
        texto = item + ("; " if posicao + 1 < len(itens) else "")
        if len(atual) + len(texto.rstrip()) > LARGURA_MAXIMA_LINHA and atual.strip() not in ("#", prefixo.strip()):
            linhas.append(atual.rstrip())
            atual = PREFIXO_CONTINUACAO
        atual += texto
    linhas.append(atual.rstrip())
    return linhas


def descrever_usos(usos: list[tuple[str, str, int]], script_definidor: str) -> tuple[list[str], list[str]]:
    agrupado: dict[tuple[str, str], list[int]] = {}
    for script, funcao, linha in sorted(usos, key=lambda uso: (SCRIPTS.index(uso[0]), uso[2])):
        agrupado.setdefault((script, funcao), []).append(linha)
    internos, externos = [], []
    for (script, funcao), numeros in agrupado.items():
        texto = "%s (l. %s)" % (funcao, ", ".join(str(numero) for numero in sorted(set(numeros))))
        if script == script_definidor:
            internos.append(texto)
        else:
            externos.append("%s: %s" % (script, texto))
    return internos, externos


def texto_tcc(script: str, etiqueta: str, mapa_equacoes: dict[str, list[int]]) -> str:
    if not ARQUIVO_EQUACOES.exists():
        return PREFIXO_TCC + "a vincular ao número da equação quando o texto do TCC for montado"
    numeros = mapa_equacoes.get("%s:%s" % (script, etiqueta), [])
    if not numeros:
        return PREFIXO_TCC + "sem equação numerada no texto"
    if len(numeros) == 1:
        return PREFIXO_TCC + "Eq. (%d)" % numeros[0]
    return PREFIXO_TCC + "Eqs. " + ", ".join("(%d)" % numero for numero in numeros[:-1]) + " e (%d)" % numeros[-1]


def usos_externos_existentes(trecho: list[str]) -> list[str]:
    """Entradas "script: função (l. ...)" já escritas na linha "Usada também em:" do bloco."""
    inicio = next((indice for indice, linha in enumerate(trecho) if linha.startswith(PREFIXO_USO_EXTERNO)), None)
    if inicio is None:
        return []
    partes = [trecho[inicio][len(PREFIXO_USO_EXTERNO):]]
    partes += [linha[len(PREFIXO_CONTINUACAO):] for linha in trecho[inicio + 1:] if linha.startswith(PREFIXO_CONTINUACAO)]
    return [entrada.strip() for entrada in " ".join(partes).split("; ") if entrada.strip()]


def ordem_do_script(entrada: str) -> int:
    return SCRIPTS.index(entrada.split(":", 1)[0])


def reescrever(linhas: list[str], blocos: list[dict], usos: dict, script: str, mapa_equacoes,
               ausentes: tuple = ()) -> list[str]:
    modulo = Path(script).stem
    novas = list(linhas)
    for bloco in reversed(blocos):
        trecho = novas[bloco["primeira"]:bloco["ultima"] + 1]
        indice_uso = next(indice for indice, linha in enumerate(trecho) if linha.startswith(PREFIXO_USO))
        usos_bloco = [uso for nome in bloco["nomes"] for uso in usos.get((modulo, nome), [])]
        internos, externos = descrever_usos(usos_bloco, script)
        preservados = [entrada for entrada in usos_externos_existentes(trecho) if entrada.split(":", 1)[0] in ausentes]
        externos = sorted(externos + preservados, key=ordem_do_script)
        geradas = formatar_lista(PREFIXO_USO, internos or ["(sem uso neste script)"])
        if externos:
            geradas += formatar_lista(PREFIXO_USO_EXTERNO, externos)
        cabecalho = [linha for linha in trecho[:indice_uso] if not linha.startswith(PREFIXO_TCC)]
        if any(linha.startswith(PREFIXO_TCC) for linha in trecho[:indice_uso]):
            cabecalho.append(texto_tcc(script, bloco["etiqueta"], mapa_equacoes))
        novas[bloco["primeira"]:bloco["ultima"] + 1] = cabecalho + geradas
    return novas


def violacoes_do_motor(arvore: ast.Module, linha_fim_zona: int) -> list[str]:
    """Operações aritméticas abaixo da zona de auditoria, exceto as exceções de [R04]."""
    violacoes = []
    for no in ast.walk(arvore):
        if getattr(no, "lineno", 0) <= linha_fim_zona:
            continue
        if isinstance(no, ast.BinOp) and isinstance(no.op, OPERACOES_PROIBIDAS):
            contador = (isinstance(no.op, (ast.Add, ast.Sub)) and isinstance(no.right, ast.Constant)
                        and no.right.value == 1)
            tuplas = isinstance(no.op, ast.Add) and (isinstance(no.left, ast.Tuple) or isinstance(no.right, ast.Tuple))
            texto = isinstance(no.op, ast.Mod) and isinstance(no.left, ast.Constant) and isinstance(no.left.value, str)
            literal_texto = lambda operando: isinstance(operando, ast.Constant) and isinstance(operando.value, str)
            concatenacao = isinstance(no.op, ast.Add) and (literal_texto(no.left) or literal_texto(no.right))
            caminho = isinstance(no.op, ast.Div) and (literal_texto(no.right) or (
                isinstance(no.right, ast.BinOp) and isinstance(no.right.op, ast.Add)
                and (literal_texto(no.right.left) or literal_texto(no.right.right))))
            if not (contador or tuplas or texto or concatenacao or caminho):
                violacoes.append("linha %d: %s" % (no.lineno, ast.unparse(no)))
        elif isinstance(no, ast.UnaryOp) and isinstance(no.op, ast.USub) and not isinstance(no.operand, ast.Constant):
            violacoes.append("linha %d: %s" % (no.lineno, ast.unparse(no)))
    return violacoes


def processar(conferir: bool) -> int:
    mapa_equacoes = ler_mapa_equacoes()
    caminhos = {script: PASTA_SCRIPTS / script for script in SCRIPTS if (PASTA_SCRIPTS / script).exists()}
    ausentes = tuple(script for script in SCRIPTS if script not in caminhos)
    if ausentes:
        print("scripts ausentes (as menções a eles são mantidas, sem conferência):", ", ".join(ausentes))
    alterado_em_alguma_rodada = False
    for rodada in range(8):
        textos = {script: caminho.read_text(encoding="utf8") for script, caminho in caminhos.items()}
        arvores = {script: ast.parse(texto) for script, texto in textos.items()}
        linhas = {script: texto.split("\n") for script, texto in textos.items()}
        blocos = {script: blocos_da_zona(linhas[script], arvores[script]) for script in caminhos}
        definicoes = {Path(script).stem: {nome: bloco["etiqueta"] for bloco in blocos[script] for nome in bloco["nomes"]}
                      for script in caminhos}
        usos = coletar_usos(arvores, definicoes)
        alterados = []
        for script in caminhos:
            novas = reescrever(linhas[script], blocos[script], usos, script, mapa_equacoes, ausentes)
            if novas != linhas[script]:
                alterados.append(script)
                if not conferir:
                    caminhos[script].write_text("\n".join(novas), encoding="utf8")
        if conferir or not alterados:
            break
        alterado_em_alguma_rodada = True
    falhas = []
    if conferir and alterados:
        falhas.append("comentários de rastreabilidade desatualizados em: " + ", ".join(alterados))
    for script in caminhos:
        sem_uso = [bloco["etiqueta"] for bloco in blocos[script]
                   if not any(usos.get((Path(script).stem, nome)) for nome in bloco["nomes"])]
        if sem_uso:
            falhas.append("%s: itens sem nenhum uso: %s" % (script, ", ".join(sem_uso)))
        sem_nome = [bloco["etiqueta"] for bloco in blocos[script] if not bloco["nomes"]]
        if sem_nome:
            falhas.append("%s: blocos sem definição associada: %s" % (script, ", ".join(sem_nome)))
        linha_fim = limites_zona(linhas[script])[1] + 1
        violacoes = violacoes_do_motor(arvores[script], linha_fim)
        if violacoes:
            falhas.append("%s: operações aritméticas no motor de cálculo:\n    " % script + "\n    ".join(violacoes))
    for falha in falhas:
        print("FALHA —", falha)
    resumo = {script: len(blocos[script]) for script in caminhos}
    print("itens rastreados por script:", resumo, "| atualizado:", alterado_em_alguma_rodada and not conferir)
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(processar("--conferir" in sys.argv))
