# -*- coding: utf-8 -*-
"""
conferir_busca_exaustiva.py
===========================
Conferência independente da escolha do detalhamento feita pelo módulo de cálculo (dimensionar_pilar).

A busca do módulo verifica cada arranjo candidato diretamente na área adotada (uma avaliação da folga por
direção, [R09]). Este script refaz a escolha pela definição literal de [R09], sem nenhum atalho: para TODOS
os candidatos, na ordem de [R04], calcula As,calc da própria geometria pela busca de raiz ([R08]) e escolhe o
primeiro com As,adot >= As,calc; percorre os arranjos na ordem da regra [R07]. Depois compara, pilar a pilar,
a situação, o arranjo e as barras com os gravados em saida/resultados_matriz.csv.

Registra também, para documentação, quantos candidatos teriam As,calc menor que o da geometria de nível [R11]
do próprio arranjo (motivo da troca do critério de busca em 22/10/2026).

Uso:  python3 conferir_busca_exaustiva.py     (grava saida/conferencia_busca_exaustiva.txt; leva cerca de 20 min)
"""
from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pilares_nbr6118 as nucleo
import processar_matriz

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — INÍCIO
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [R01] Arquivos: resultados gravados por processar_matriz.py e relatório desta conferência.
#   Variáveis: ARQUIVO_RESULTADOS, ARQUIVO_RELATORIO.
#   Usada em: ler_resultados (l. 105); main (l. 143)
ARQUIVO_RESULTADOS = Path(__file__).resolve().parent / "saida" / "resultados_matriz.csv"
ARQUIVO_RELATORIO = Path(__file__).resolve().parent / "saida" / "conferencia_busca_exaustiva.txt"

# [R02] Folga numérica para dizer que um candidato tem As,calc menor que o da geometria de nível [R11]
#   (cm²): a tolerância da busca da armadura (TOLERANCIA_AREA do núcleo).
#   Variáveis: FOLGA_COMPARACAO.
#   Usada em: abaixo_do_nivel (l. 52)
FOLGA_COMPARACAO = nucleo.TOLERANCIA_AREA


# [F01] Candidato com As,calc abaixo do da geometria de nível [R11]: As,calc,cand < As,calc,nível - folga.
#   Variáveis: area_candidato, area_nivel.
#   Usada em: escolha_exaustiva (l. 93)
def abaixo_do_nivel(area_candidato: float, area_nivel: float) -> bool:
    return area_candidato < area_nivel - FOLGA_COMPARACAO


# [F02] Tempo decorrido desde um instante de referência (s): t = t_atual - t_inicio.
#   Variáveis: inicio.
#   Usada em: main (l. 141)
def tempo_decorrido(inicio: float) -> float:
    return time.perf_counter() - inicio

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


def area_calculada_candidato(pilar: nucleo.PilarTeste, material: nucleo.Material, candidato) -> float:
    """As,calc ([R08] do módulo) da geometria de um candidato, pela busca de raiz, sem memória compartilhada."""
    areas = []
    excentricidades = {nucleo.DIRECAO_B: pilar.excentricidade_primeira_ordem, nucleo.DIRECAO_H: 0.0}
    for direcao in nucleo.ORDEM_DIRECOES:
        secao = nucleo.secao_na_direcao(pilar.largura_bw, pilar.altura_h, candidato.x, candidato.y, direcao)
        areas.append(nucleo.analisar_direcao(secao, material, pilar.forca_normal_calculo,
                                             pilar.comprimento_equivalente, direcao,
                                             excentricidades[direcao]).area_necessaria)
    area_minima = nucleo.armadura_minima(pilar.forca_normal_calculo, material.fyd,
                                         nucleo.area_secao_retangular(pilar.largura_bw, pilar.altura_h))
    return nucleo.area_calculada_pilar(areas[0], areas[1], area_minima)


def escolha_exaustiva(caso: processar_matriz.CasoMatriz) -> dict:
    pilar_regra = caso.pilar
    material = nucleo.Material(pilar_regra.fck_mpa, pilar_regra.fyk_mpa)
    registro = {"ID": caso.identificador, "arranjo": "", "barras": "", "situacao": "", "candidatos": 0,
                "abaixo_do_nivel": {arranjo: 0 for arranjo in nucleo.ORDEM_ARRANJOS_REGRA}}
    for arranjo in nucleo.ORDEM_ARRANJOS_REGRA:
        pilar = replace(pilar_regra, arranjo=arranjo)
        area_maxima = nucleo.armadura_maxima(nucleo.area_secao_retangular(pilar.largura_bw, pilar.altura_h),
                                             pilar.taxa_armadura_maxima)
        area_nivel = area_calculada_candidato(pilar, material, nucleo.candidato_referencia(pilar))
        for candidato in nucleo.gerar_candidatos(pilar, area_maxima):
            registro["candidatos"] += 1
            area = area_calculada_candidato(pilar, material, candidato)
            if abaixo_do_nivel(area, area_nivel):
                registro["abaixo_do_nivel"][arranjo] += 1
            if nucleo.detalhamento_atende(candidato.area, area):
                registro.update(arranjo=arranjo, situacao=nucleo.STATUS_OK,
                                barras="%d φ %.1f" % (candidato.numero_barras,
                                                      processar_matriz.cm_para_mm(candidato.diametro)))
                return registro
    registro["situacao"] = "inviável"
    return registro


def ler_resultados() -> dict:
    with ARQUIVO_RESULTADOS.open(encoding="utf-8") as arquivo:
        return {linha["ID"]: linha for linha in csv.DictReader(arquivo, delimiter=";")}


def main() -> None:
    inicio = time.perf_counter()
    casos = processar_matriz.montar_matriz()
    gravados = ler_resultados()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        registros = list(executor.map(escolha_exaustiva, casos, chunksize=2))
    divergencias = []
    for registro in registros:
        linha = gravados[registro["ID"]]
        viavel_rotina = linha["Situação"] == nucleo.STATUS_OK
        viavel_exaustiva = registro["situacao"] == nucleo.STATUS_OK
        mesma = viavel_rotina == viavel_exaustiva and (not viavel_rotina or (
            linha["Arranjo adotado"] == registro["arranjo"] and linha["Barras"] == registro["barras"]))
        if not mesma:
            divergencias.append("  %s: rotina %s %s %s | exaustiva %s %s %s" % (
                registro["ID"], linha["Situação"], linha["Arranjo adotado"], linha["Barras"],
                registro["situacao"], registro["arranjo"], registro["barras"]))
    abaixo = {arranjo: sum(registro["abaixo_do_nivel"][arranjo] for registro in registros)
              for arranjo in nucleo.ORDEM_ARRANJOS_REGRA}
    linhas = [
        "Conferência exaustiva da escolha do detalhamento (definição literal de [R09], sem atalhos)",
        "Pilares: %d | candidatos calculados pela busca de raiz: %d" % (
            len(registros), sum(registro["candidatos"] for registro in registros)),
        "Viáveis: rotina %d | exaustiva %d" % (
            sum(linha["Situação"] == nucleo.STATUS_OK for linha in gravados.values()),
            sum(registro["situacao"] == nucleo.STATUS_OK for registro in registros)),
        "Pilares com situação, arranjo ou barras diferentes: %d" % len(divergencias),
        *divergencias,
        "Candidatos com As,calc menor que o da geometria de nível [R11] do próprio arranjo: %s" % ", ".join(
            "%s %d" % (arranjo, quantidade) for arranjo, quantidade in abaixo.items()),
        "Conclusão: %s" % ("a escolha da rotina é a mesma da busca exaustiva em todos os pilares"
                           if not divergencias else "VERIFICAR: há divergências"),
        "Tempo: %.1f s" % tempo_decorrido(inicio),
    ]
    ARQUIVO_RELATORIO.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))
    sys.exit(0 if not divergencias else 1)


if __name__ == "__main__":
    main()
