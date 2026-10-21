# ⚕️ Aula 08 — Sistemas Especialistas

**Disciplina:** Inteligência Artificial · **Curso:** Ciência da Computação

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/profluiscaparroz/inteligencia-artificial/blob/main/08-sistemas-especialistas/08_Sistemas_Especialistas.ipynb)

---

## 📂 Conteúdo da pasta

| Arquivo | Descrição |
|---|---|
| [`08_Sistemas_Especialistas.ipynb`](08_Sistemas_Especialistas.ipynb) | Aula completa: arquitetura, fatores de certeza, sistema médico, explicação POR QUÊ/COMO, **animações**, **controles interativos** e verificação de bases de regras |
| [`Exercicios_Sistemas_Especialistas.ipynb`](Exercicios_Sistemas_Especialistas.ipynb) | Lista de exercícios |

### Como executar

- **Google Colab (recomendado):** clique em *Abrir no Colab* e execute as células em ordem.
- **Localmente:**
  ```bash
  pip install numpy matplotlib ipywidgets jupyter
  jupyter notebook 08_Sistemas_Especialistas.ipynb
  ```

Células com 🎬 geram **animações** e com 🎛️ têm **controles interativos**. Depois de **cada resultado** há uma seção **🔎 Como interpretar**.

---

## 🎯 Ideia central

Um **sistema especialista** é um programa que resolve problemas que exigiriam um **especialista humano**, usando conhecimento **explícito** (regras) separado de um **motor de inferência**. Além de um sistema de produção, ele tem:

- tratamento de **incerteza** (fatores de certeza, probabilidades ou lógica *fuzzy*);
- um **módulo de explicação** que responde **POR QUÊ?** (por que esta pergunta?) e **COMO?** (como chegou a esta conclusão?);
- um processo de **engenharia do conhecimento** para extrair e manter as regras com especialistas.

**Fatores de certeza (MYCIN):**

| Operação | Fórmula |
|---|---|
| Premissas ligadas por E | `CF = min(CF₁, CF₂, ...)` |
| Propagação por uma regra | `CF(conclusão) = CF(regra) × CF(premissas)` |
| Duas evidências a favor | `CF = CF₁ + CF₂ − CF₁·CF₂` |
| Evidências conflitantes | `CF = (CF₁ + CF₂) / (1 − min(|CF₁|, |CF₂|))` |

---

## 👨‍🏫 Autores e pioneiros

| Autor(es) | Ano | Contribuição | Onde aparece na aula |
|---|:-:|---|---|
| **Edward Feigenbaum** | 1965–1977 | Considerado o "pai dos sistemas especialistas". Liderou o DENDRAL, cunhou o termo **engenharia do conhecimento** e formulou o princípio de que **o poder está no conhecimento**. Prêmio Turing de 1994 | Partes 1 e 5 |
| **Joshua Lederberg** (Nobel de Medicina, 1958), **Bruce Buchanan** e **Carl Djerassi** | 1965 | **DENDRAL**: inferência de estruturas moleculares a partir de espectrometria de massa, o primeiro sistema especialista | Parte 1 |
| **Edward H. Shortliffe** | 1972–1976 | **MYCIN**: diagnóstico de infecções e recomendação de antibióticos, com *backward chaining* e explicações | Partes 2, 4 e 5 |
| **Edward Shortliffe & Bruce Buchanan** | 1975 | Modelo de **fatores de certeza** (*A Model of Inexact Reasoning in Medicine*) | Seções 3.2–3.6 e 4 |
| **Victor Yu e colaboradores** | 1979 | Avaliação cega do MYCIN publicada no *JAMA*: desempenho igual ou superior ao dos especialistas avaliados | Seção 5.5 |
| **Randall Davis** | 1976–1979 | **TEIRESIAS**: aquisição interativa de conhecimento e meta-regras | Parte 5 |
| **William Clancey** | 1979–1987 | **GUIDON** (tutor baseado no MYCIN) e **NEOMYCIN** | Parte 5 |
| **Richard Duda, Peter Hart & John Gaschnig** | 1976–1979 | **PROSPECTOR** (SRI): exploração mineral com raciocínio bayesiano | Parte 1 |
| **Harry Pople & Jack Myers** | 1974–1982 | **INTERNIST-I/CADUCEUS**: diagnóstico em medicina interna | Parte 8 |
| **John McDermott** | 1980 | **R1/XCON** (DEC): configuração de computadores, o primeiro grande sucesso comercial | Parte 1 |
| **Marvin Minsky** | 1974 | **Frames**: estruturas para representar objetos e estereótipos | Seção 3.4 |
| **Lotfi Zadeh** | 1965 | **Lógica *fuzzy***: graus de pertinência | Seção 3.5 (e Aula 19) |
| **Arthur Dempster & Glenn Shafer** | 1967/1976 | **Teoria da evidência** (Dempster–Shafer), alternativa para raciocínio com incerteza | Referências |
| **David Heckerman** | 1986 | Interpretação probabilística dos CFs: mostrou que a combinação do MYCIN pressupõe **independência condicional** entre evidências | Seção 4.7 |
| **Judea Pearl** | 1988 | **Redes bayesianas**, que substituíram os CFs ao modelar dependências explicitamente. Prêmio Turing de 2011 | Seção 4.7 (e Aula 16) |
| **Hubert & Stuart Dreyfus** | 1986 | *Mind over Machine*: crítica sobre os limites de capturar a perícia humana em regras | Parte 7 |
| **Douglas Lenat** | 1984– | Projeto **Cyc**: base de conhecimento de senso comum, resposta à fragilidade dos sistemas especialistas | Referências |

---

## 🗺️ Roteiro do notebook

| Parte | Tema | Destaques práticos |
|:-:|---|---|
| 0 | Preparação | — |
| 1 | O que são sistemas especialistas e história | — |
| 2 | Arquitetura | Mini sistema que **pergunta** e explica **POR QUÊ/COMO** |
| 3 | Representação do conhecimento e fatores de certeza | Gráficos e calculadora interativa de CF 🎛️ |
| 4 | Sistema médico completo | 3 casos clínicos; animação da certeza crescendo 🎬; rede de conhecimento; consultório virtual 🎛️; limitações dos CFs (dupla contagem e dependência) |
| 5 | Estudo de caso: MYCIN | Avaliação de Yu et al. (1979) |
| 6 | Sistemas modernos | Verificação automática de bases de regras (redundância, conflito, subsunção, ciclos) |
| 7–10 | Comparações, aplicações, exercícios e referências | — |

---

## 📚 Referências

- BUCHANAN, B. G.; SHORTLIFFE, E. H. *Rule-Based Expert Systems: The MYCIN Experiments of the Stanford Heuristic Programming Project*. Addison-Wesley, 1984.
- SHORTLIFFE, E. H.; BUCHANAN, B. G. A model of inexact reasoning in medicine. *Mathematical Biosciences*, 23(3–4), 351–379, 1975.
- YU, V. L. et al. Antimicrobial selection by a computer: a blinded evaluation by infectious diseases experts. *JAMA*, 242(12), 1279–1282, 1979.
- FEIGENBAUM, E. A. The art of artificial intelligence: themes and case studies of knowledge engineering. *IJCAI*, 1977.
- LINDSAY, R. K. et al. DENDRAL: a case study of the first expert system for scientific hypothesis formation. *Artificial Intelligence*, 61(2), 1993.
- HECKERMAN, D. Probabilistic interpretations for MYCIN's certainty factors. In: *Uncertainty in Artificial Intelligence*, 1986.
- PEARL, J. *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann, 1988.
- JACKSON, P. *Introduction to Expert Systems*. 3. ed. Addison-Wesley, 1999.
- GIARRATANO, J.; RILEY, G. *Expert Systems: Principles and Programming*. 4. ed. Thomson, 2005.
- RUSSELL, S.; NORVIG, P. *Inteligência Artificial: uma abordagem moderna*. 4. ed.
- experta (sucessor do PyKnow): <https://github.com/nilp0inter/experta>
