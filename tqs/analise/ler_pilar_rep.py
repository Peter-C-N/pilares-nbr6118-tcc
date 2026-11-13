"""Leitura do Pilar.rep do TQS Pilar (XML compactado em gzip, DynamicTreeDataRepository) em dicionários aninhados.

Uso como módulo: ler_pilar_rep(caminho) -> dict.  Uso direto: python3 ler_pilar_rep.py <Pilar.rep> [pilar]
(imprime as chaves e valores do pilar, achatados).
"""
import gzip
import sys
import xml.etree.ElementTree as ET

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — constantes (este leitor não tem fórmulas de cálculo)
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# [D01] Espaços de nomes do XML gravado pelo TQS.  Usada em: no_para_dict
ESPACO_ARRAYS = "{http://schemas.microsoft.com/2003/10/Serialization/Arrays}"
ESPACO_DADOS = "{http://schemas.datacontract.org/2004/07/TQS.Data}"
ESPACO_INSTANCIA = "{http://www.w3.org/2001/XMLSchema-instance}"

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ZONA DE AUDITORIA — FIM
# ═════════════════════════════════════════════════════════════════════════════════════════════════


def converter_valor(elemento):
    tipo = elemento.get(ESPACO_INSTANCIA + "type", "")
    if tipo == "TreeNode":
        dados = elemento.find(ESPACO_DADOS + "data")
        return no_para_dict(dados) if dados is not None else {}
    texto = elemento.text or ""
    if tipo.endswith(":double"):
        return float(texto)
    if tipo.endswith(":int"):
        return int(texto)
    if tipo.endswith(":boolean"):
        return texto == "true"
    return texto


def no_para_dict(elemento_dados) -> dict:
    resultado = {}
    for par in elemento_dados.findall(ESPACO_ARRAYS + "KeyValueOfstringanyType"):
        chave = par.find(ESPACO_ARRAYS + "Key").text
        resultado[chave] = converter_valor(par.find(ESPACO_ARRAYS + "Value"))
    return resultado


def ler_pilar_rep(caminho) -> dict:
    bruto = open(caminho, "rb").read()
    texto = gzip.decompress(bruto)
    raiz = ET.fromstring(texto)
    dados_externos = raiz.find(ESPACO_DADOS + "data")
    dados_internos = dados_externos.find(ESPACO_DADOS + "data")
    return no_para_dict(dados_internos if dados_internos is not None else dados_externos)


def pilares_por_nome(repositorio: dict) -> dict:
    pilares = {}
    for chave, valor in repositorio.items():
        if chave.startswith("COLUMN[") and isinstance(valor, dict):
            pilares[valor.get("NAME")] = valor
    return pilares


def achatar(no, prefixo=""):
    if isinstance(no, dict):
        for chave, valor in no.items():
            yield from achatar(valor, prefixo + "/" + chave if prefixo else chave)
    else:
        yield prefixo, no


if __name__ == "__main__":
    repositorio = ler_pilar_rep(sys.argv[1])
    if len(sys.argv) > 2:
        for chave, valor in achatar(pilares_por_nome(repositorio)[sys.argv[2]]):
            print(chave, "=", valor)
    else:
        print([chave for chave in repositorio if not chave.startswith("COLUMN[")])
        print(sorted(pilares_por_nome(repositorio)))
