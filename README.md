# Rotina em Python para o dimensionamento de pilares de concreto armado (ABNT NBR 6118:2023)

Material digital do Trabalho de Conclusão de Curso *Desenvolvimento de programa em Python para dimensionamento de
pilares de concreto armado à flexão normal composta reta com λ ≤ 140 e comparação com o software comercial TQS*,
de Pedro Costaneski Neto (Engenharia Civil, Faculdade de Arquitetura, Engenharia e Tecnologia, Universidade Federal
de Mato Grosso, 2026), sob orientação do Prof. Dr. Milton Soares Filho e coorientação do Prof. Dr. Alberto Rodrigues
Dalmaso.

O repositório reúne o código-fonte da rotina, os programas usados para lançar e processar os pilares no TQS 25.14.6
e para comparar os resultados, o memorial de cálculo dos 333 pilares e os principais resultados. O arquivo
`LEIAME.txt` descreve cada programa, a ordem de execução, o tempo medido, o ambiente usado e as alterações de cada
versão.

## Programas e onde o TCC descreve cada um

O TCC descreve os programas pela função (Seção 4.5 e Apêndice A), sem os nomes dos arquivos; a tabela abaixo liga
cada arquivo à seção correspondente. Todos são executados, na ordem certa, pelo programa principal.

| Programa | O que faz | Onde o TCC descreve |
|---|---|---|
| `executar_rotina.py` | **Programa principal**: executa todos os programas abaixo, na ordem, com um único comando | Apêndice A |
| `pilares_nbr6118.py` | Núcleo de cálculo; fórmulas, constantes e regras reunidas na zona de auditoria, no início do arquivo | Seção 3.5 |
| `processar_matriz.py` | Monta os 324 pilares-teste e os 9 do teste do momento mínimo e dimensiona todos | Seções 3.6 e 4.2; Apêndice B |
| `verificar_matriz.py` | Confere as entradas e os resultados (40 requisitos) | Seções 3.6 e 4.2 |
| `validacao.py` | Validação interna em dois níveis (450 testes) | Seções 3.7 e 4.1; Apêndice D |
| `validacao_literatura.py` | Engenharia reversa dos exemplos resolvidos da literatura | Seções 3.7 e 4.1; Apêndice F |
| `sensibilidade_secao_liquida.py` | Sensibilidade ao diagrama de tensões e à seção de integração | Seção 4.2 (Tabela 5) |
| `conferir_busca_exaustiva.py` | Conferência independente da escolha das barras (11 548 arranjos; opcional) | Seção 4.2 |
| `figuras_tcc.py`, `figuras_complementares.py` | Figuras calculadas pela rotina (as imagens estão no próprio TCC) | Capítulos 2 e 4; Apêndice G |
| `memorial_pilares.py` | Memorial de cálculo dos 333 pilares | Seção 4.5; Apêndices D e E |
| `gerar_auditor.py` | Auditor visual (lê `auditor_modelo.html` e `fontes/`) | Seção 4.5 (Figura 27) |
| `rastreabilidade.py` | Confere a zona de auditoria e a ausência de fórmulas no corpo dos 11 programas da rotina | Seção 4.5 |
| `tqs/gerar_entrada_tqs.py` | Monta a tabela dos pilares a lançar no TQS | Seção 3.8.1 |
| `tqs/windows/tqs_00_diagnostico.py`, `tqs/windows/tqs_01_lote.py` | Executados no computador com o TQS: leem o edifício-modelo, lançam e processam os pilares | Seção 3.8.1 |
| `tqs/comparar_tqs_lote.py` (com `tqs/analise/ler_pilar_rep.py`) | Lê os resultados do TQS e faz a comparação em duas camadas | Seções 3.9 e 4.3 |
| `tqs/resumir_comparacao_tqs.py`, `tqs/decompor_curvatura_tqs.py`, `tqs/camada2_nm1r_tqs.py`, `tqs/replica_tqs_nm1r.py`, `tqs/teste_minimo_tqs.py` | Resumo da comparação e análise das divergências | Seções 4.3 e 4.4 |

## Resultados no repositório

| Arquivo | Conteúdo |
|---|---|
| `saida/memorial/memorial_pilares.pdf` | Memorial de cálculo dos 333 pilares (736 páginas) |
| `saida/auditor_pilares.html` | Auditor visual: cada pilar ligado às fórmulas e às fontes; abre no navegador, sem internet |
| `saida/resultados_matriz.xlsx` | Entradas, resultados, pilares inviáveis, teste do momento mínimo, entrada para o TQS e conferência das orientações |
| `saida/validacao.txt`, `saida/validacao_literatura.txt` | Relatórios da validação (450 testes) e da engenharia reversa dos exemplos da literatura |
| `tqs/entrada_tqs_lote.csv` | Pilares lançados no TQS, lida por `windows/tqs_01_lote.py` |
| `tqs/comparacao_tqs_lote.csv` | Comparação rotina × TQS, uma linha por pilar |

Os demais arquivos citados no `LEIAME.txt` (tabelas em CSV e JSON, figuras e relatórios auxiliares) são gravados
pelos programas quando executados e não ficam no repositório.

## Execução

Requer Python 3.13 e as bibliotecas de `requirements.txt`; o memorial em PDF usa também o LibreOffice.

```
pip install -r requirements.txt
python executar_rotina.py              # rotina completa, na ordem certa (cerca de 30 min)
python executar_rotina.py --completo   # inclui a busca exaustiva (mais cerca de 17 min)
python executar_rotina.py --lista      # mostra as etapas e a seção do TCC que descreve cada uma
```

A comparação com o TQS (`python executar_rotina.py --tqs`) exige as listagens gravadas pelo TQS, que não estão no
repositório. O `LEIAME.txt` traz a ordem das etapas, o tempo de cada uma e o que cada programa grava.

## Nota de uso

A rotina foi desenvolvida e validada para as condições estudadas no TCC: seções retangulares de 20 × 30 cm a
40 × 80 cm, fck de 25 a 35 MPa, ν de 0,2 a 0,6, e1/b_w de 0,15 ou 0,30, pilares biapoiados com momento de 1ª ordem
constante, aço CA-50, armadura em duas faces ou no perímetro e classe de agressividade I. Ela é uma ferramenta de
estudo e de conferência: a validação feita não comprova a aptidão para uso irrestrito em projetos reais, e os
resultados devem ser verificados pelo engenheiro responsável pelo projeto.

O texto da ABNT NBR 6118:2023 e as obras citadas não fazem parte deste material. As listagens e os relatórios
gravados pelo TQS (pasta `tqs/resultados_lote`, lida por `comparar_tqs_lote.py` e pelos programas de análise da
pasta `tqs`) também não estão no repositório; por isso, esses programas só podem ser executados com uma nova rodada
no TQS. As regras do TQS descritas nos programas da pasta `tqs` foram inferidas dos resultados do programa
(engenharia reversa) e valem para a versão 25.14.6, nos pilares estudados.

## Licença

O código e os resultados deste repositório são distribuídos sob a licença de código aberto MIT (arquivo
[`LICENSE`](LICENSE)). A licença não se estende à ABNT NBR 6118:2023, às obras citadas nem ao TQS e aos arquivos
gravados por ele, que não fazem parte do repositório.
