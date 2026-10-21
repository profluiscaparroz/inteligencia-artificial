# ⚙️ Aula 07 — Sistemas de Produção

**Disciplina:** Inteligência Artificial · **Curso:** Ciência da Computação

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/profluiscaparroz/inteligencia-artificial/blob/main/07-sistemas-producao/07_Sistemas_Producao.ipynb)

---

## 📂 Conteúdo da pasta

| Arquivo | Descrição |
|---|---|
| [`07_Sistemas_Producao.ipynb`](07_Sistemas_Producao.ipynb) | Aula completa: teoria, implementações em Python, **animações**, **controles interativos** e experimentos |
| [`Exercicios_Sistemas_Producao.ipynb`](Exercicios_Sistemas_Producao.ipynb) | Lista de exercícios |

### Como executar

- **Google Colab (recomendado):** clique em *Abrir no Colab* e execute as células em ordem (`Shift + Enter`).
- **Localmente:**
  ```bash
  pip install numpy matplotlib ipywidgets jupyter
  jupyter notebook 07_Sistemas_Producao.ipynb
  ```

Células com 🎬 geram **animações** (botão ▶) e com 🎛️ têm **controles interativos**. Depois de **cada resultado** há uma seção **🔎 Como interpretar**, que explica o que aconteceu, por que acontece e como ler a saída.

---

## 🎯 Ideia central

Um **sistema de produção** representa conhecimento como regras **SE condição ENTÃO ação** e as executa em um ciclo:

```
┌──────────────┐   ┌──────────────────┐   ┌──────────────┐
│ 1. MATCH     │──►│ 2. SELECT        │──►│ 3. EXECUTE   │──┐
│ quais regras │   │ resolução de     │   │ altera a     │  │
│ se aplicam?  │   │ conflito         │   │ memória      │  │
└──────────────┘   └──────────────────┘   └──────────────┘  │
       ▲                                                    │
       └────────────────────────────────────────────────────┘
```

Os três componentes são a **base de regras** (o conhecimento), a **memória de trabalho** (os fatos atuais) e o **motor de inferência** (o controle). Separar conhecimento de controle é o que torna esses sistemas **modulares** e **explicáveis**.

---

## 👨‍🏫 Autores e pioneiros

| Autor(es) | Ano | Contribuição | Onde aparece na aula |
|---|:-:|---|---|
| **Emil Post** | 1943 | **Sistemas canônicos de Post**: regras de reescrita de cadeias, tão poderosas quanto a máquina de Turing. Origem do termo "regra de produção" | Seção 1.5 (animação de reescrita) |
| **Andrey A. Markov (filho)** | 1954 | **Algoritmos normais de Markov**: regras de reescrita **ordenadas**, aplicadas à primeira ocorrência (o modelo usado no exemplo de ordenação de letras) | Seção 1.5 |
| **Noam Chomsky** | 1956–57 | **Gramáticas gerativas** e a hierarquia de Chomsky: regras de produção para descrever linguagens | Base das gramáticas formais (Aula 09) |
| **Allen Newell & Herbert A. Simon** | 1956–1972 | *Logic Theorist*, **GPS** e o livro *Human Problem Solving* (1972), que propôs os sistemas de produção como **modelo da cognição humana**. Prêmio Turing de 1975 | Partes 1 e 2 |
| **Randall Davis & Jonathan King** | 1977 | *An Overview of Production Systems*: síntese clássica da arquitetura, dos ciclos e das estratégias de controle | Parte 2 |
| **Charles L. Forgy** | 1979–1982 | Linguagem **OPS5** e o algoritmo **RETE** (tese de 1979, artigo de 1982), que tornou viáveis sistemas com milhares de regras | Seções 6.1 e 6.1.7 (experimento) |
| **John McDermott** | 1980–1982 | **R1/XCON**, o configurador de computadores VAX da DEC escrito em OPS5: primeiro grande sucesso comercial de sistemas de regras | Parte 7 |
| **Edward Shortliffe & Bruce Buchanan** | 1972–1984 | **MYCIN**: regras de produção com fatores de certeza e *backward chaining* | Parte 4 (e Aula 08) |
| **John R. Anderson** | 1976–1993 | Arquiteturas cognitivas **ACT\*** e **ACT-R**, baseadas em regras de produção | Parte 7 |
| **John Laird, Allen Newell & Paul Rosenbloom** | 1987 | Arquitetura cognitiva **SOAR** (resolução de impasses e aprendizado por *chunking*) | Parte 7 |
| **Gary Riley e equipe da NASA** | 1985 | **CLIPS**, *shell* de sistemas de regras em C desenvolvida no Johnson Space Center, ainda usada hoje | Parte 7 |
| **Jeff Orkin** | 2005 | **GOAP** (*Goal-Oriented Action Planning*) na IA do jogo *F.E.A.R.*, evolução das regras reativas para planejamento | Seção 7.5 (NPC) |
| **Nils J. Nilsson** | 1980/1998 | Livros-texto que formalizaram sistemas de produção como modelo geral de busca em IA | Definição formal (Seção 1.1) |

---

## 🗺️ Roteiro do notebook

| Parte | Tema | Destaques práticos |
|:-:|---|---|
| 0 | Preparação | Funções auxiliares de animação |
| 1 | Conceitos, história e características | Termostato em 20 linhas; Sistemas de Post com animação 🎬 |
| 2 | Arquitetura e ciclo reconhecer–agir | — |
| 3 | *Forward chaining* | Implementação corrigida (refratariedade por instanciação); animação do ciclo 🎬; mesmas regras com fatos diferentes |
| 4 | *Backward chaining* | Árvore de prova com *backtracking* 🎬 |
| 5 | *Forward* × *backward* | Experimento com 240 regras geradas automaticamente |
| 6 | Tópicos avançados | RETE (ingênuo × indexado, até ~100× mais rápido); estratégias de conflito 🎛️; refratariedade e *loops* |
| 7 | Aplicações | NPC de jogo controlado por regras 🎬 |
| 8–10 | Vantagens/desvantagens, exercícios e referências | — |

---

## 📚 Referências

- POST, E. L. Formal reductions of the general combinatorial decision problem. *American Journal of Mathematics*, 65(2), 1943.
- NEWELL, A.; SIMON, H. A. *Human Problem Solving*. Prentice-Hall, 1972.
- DAVIS, R.; KING, J. An overview of production systems. In: *Machine Intelligence 8*, 1977.
- FORGY, C. L. Rete: a fast algorithm for the many pattern/many object pattern match problem. *Artificial Intelligence*, 19(1), 17–37, 1982.
- McDERMOTT, J. R1: a rule-based configurer of computer systems. *Artificial Intelligence*, 19(1), 39–88, 1982.
- BROWNSTON, L. et al. *Programming Expert Systems in OPS5*. Addison-Wesley, 1985.
- LAIRD, J. E.; NEWELL, A.; ROSENBLOOM, P. S. SOAR: an architecture for general intelligence. *Artificial Intelligence*, 33(1), 1987.
- NILSSON, N. J. *Artificial Intelligence: A New Synthesis*. Morgan Kaufmann, 1998.
- RUSSELL, S.; NORVIG, P. *Inteligência Artificial: uma abordagem moderna*. 4. ed. (capítulos sobre inferência lógica e encadeamento).
- GIARRATANO, J.; RILEY, G. *Expert Systems: Principles and Programming*. 4. ed. Thomson, 2005.
- CLIPS: <https://www.clipsrules.net/>
