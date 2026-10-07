# 🤖 Aula 06 — Machine Learning (Aprendizado de Máquina)

**Disciplina:** Inteligência Artificial · **Curso:** Ciência da Computação

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/profluiscaparroz/inteligencia-artificial/blob/main/06-machine-learning/06_Machine_Learning.ipynb)

---

## 📂 Conteúdo da pasta

| Arquivo | Descrição |
|---|---|
| [`06_Machine_Learning.ipynb`](06_Machine_Learning.ipynb) | Aula completa: teoria, implementações do zero em Python, scikit-learn, **animações** e **gráficos interativos** |
| [`Exercicios_Machine_Learning.ipynb`](Exercicios_Machine_Learning.ipynb) | Lista de exercícios |

### Como executar

- **Google Colab (recomendado):** clique no botão *Abrir no Colab* acima e execute as células em ordem (`Shift + Enter`).
- **Localmente:**
  ```bash
  pip install numpy pandas matplotlib scikit-learn ipywidgets jupyter
  jupyter notebook 06_Machine_Learning.ipynb
  ```

As células marcadas com 🎬 geram **animações** (use o botão ▶ do player) e as marcadas com 🎛️ possuem **controles interativos** (*sliders*).

Logo após **cada resultado, gráfico ou animação** há uma seção **🔎 Como interpretar**, que explica o que aconteceu, **por que** acontece, como ler o gráfico ou a saída e qual é o conceito-chave envolvido.

---

## 🎯 O que é Machine Learning?

**Machine Learning** (Aprendizado de Máquina) é a área da Inteligência Artificial que estuda algoritmos capazes de **melhorar seu desempenho a partir da experiência**, sem que todas as regras sejam escritas explicitamente pelo programador.

```
Programação tradicional:   Dados + Regras     ──►  Respostas
Machine Learning:          Dados + Respostas  ──►  Regras (modelo)
```

### A definição de Tom M. Mitchell (1997)

A definição mais citada da área está no primeiro capítulo do livro *Machine Learning*, de **Tom M. Mitchell**:

> Diz-se que um programa de computador **aprende** com a experiência **E**, em relação a uma classe de tarefas **T** e a uma medida de desempenho **P**, se o seu desempenho nas tarefas de **T**, medido por **P**, melhora com a experiência **E**.

A força dessa definição é ser **operacional**: ela transforma a ideia vaga de "aprender" em algo **verificável**. Para afirmar que um sistema aprende, precisamos especificar:

| Elemento | Pergunta | Exemplo (jogo de damas) | Exemplo (filtro de spam) |
|:---:|---|---|---|
| **T** — Tarefa | O que o sistema faz? | Jogar damas | Classificar e-mails |
| **P** — Desempenho | Como medimos? | % de partidas vencidas | % de e-mails classificados corretamente |
| **E** — Experiência | De onde aprende? | Partidas jogadas contra si mesmo | E-mails marcados pelos usuários como spam |

Mitchell também descreve o **projeto de um sistema de aprendizado** em quatro escolhas — a experiência de treinamento, a função-alvo, a representação dessa função e o algoritmo que a aproxima — e trata o aprendizado como uma **busca em um espaço de hipóteses**, guiada por um **viés indutivo**. Essas ideias atravessam toda a aula.

---

## 👨‍🏫 Autores e pioneiros

### ⭐ Tom M. Mitchell — o autor de referência desta aula

| | |
|---|---|
| **Instituição** | Carnegie Mellon University (CMU), Pittsburgh, EUA — *E. Fredkin University Professor* |
| **Formação** | Doutorado em Engenharia Elétrica pela Stanford University (1979) |
| **Obra principal** | ***Machine Learning*** (McGraw-Hill, 1997) — um dos primeiros livros-texto da área e referência em cursos de graduação e pós-graduação no mundo todo |

**Principais contribuições:**

- **Definição formal de aprendizado (T, P, E):** a definição operacional usada até hoje para delimitar o que significa "aprender".
- **Espaço de Versões e Eliminação de Candidatos (1977–1982):** em sua tese e no artigo *Generalization as Search* (1982), mostrou que o aprendizado de conceitos pode ser visto como uma **busca** em um espaço de hipóteses ordenado do mais geral ao mais específico, representado de forma compacta pelas fronteiras **S** e **G**.
- **Viés indutivo (1980):** no relatório *The Need for Biases in Learning Generalizations*, argumentou que um aprendiz **sem nenhuma suposição prévia** não tem base racional para generalizar — um dos fundamentos teóricos da área.
- **Aprendizado baseado em explicações (1986):** com Keller e Kedar-Cabelli, propôs a *Explanation-Based Generalization*, que combina conhecimento prévio e exemplos.
- **Primeiro departamento de Machine Learning do mundo (2006):** foi o chefe fundador do *Machine Learning Department* da CMU.
- **NELL — *Never-Ending Language Learning* (2010):** sistema que lia a web continuamente para aprender fatos e melhorar a própria capacidade de leitura, levando a ideia de "melhorar com a experiência" ao limite.
- **ML aplicado à neurociência:** trabalhos pioneiros usando aprendizado de máquina para decodificar e prever padrões de atividade cerebral (fMRI) associados ao significado de palavras.
- **Reconhecimento:** membro da *National Academy of Engineering* dos EUA, *fellow* da AAAI e ex-presidente da AAAI (*Association for the Advancement of Artificial Intelligence*).

**Capítulos do livro usados nesta aula:**

| Capítulo (Mitchell, 1997) | Parte da aula |
|---|---|
| 1 — Introdução (T/P/E, projeto de um sistema de aprendizado, regra LMS) | Partes 1 e 4 |
| 2 — Aprendizado de conceitos e ordenação geral-para-específico (Find-S, Espaço de Versões, viés indutivo) | Parte 2 |
| 3 — Aprendizado de árvores de decisão (ID3, entropia, ganho de informação, *PlayTennis*) | Parte 9 |
| 4 — Redes neurais artificiais (Perceptron, gradiente descendente) | Partes 4 e 6 |
| 5 — Avaliação de hipóteses | Partes 5 e 11 |
| 6 — Aprendizado bayesiano (MAP, Naive Bayes) | Parte 8 |
| 8 — Aprendizado baseado em instâncias (KNN) | Parte 7 |
| 13 — Aprendizado por reforço (Q-Learning) | Parte 13 |

---

### Outros autores fundamentais

| Autor(es) | Ano | Contribuição | Onde aparece na aula |
|---|:-:|---|---|
| **Warren McCulloch & Walter Pitts** | 1943 | Primeiro modelo matemático de neurônio artificial | Parte 1 (história) |
| **Alan Turing** | 1950 | Propôs em *Computing Machinery and Intelligence* a ideia de "máquinas que aprendem" | Parte 1 |
| **Arthur Samuel** (IBM) | 1959 | Programa de damas que aprendia jogando contra si mesmo; popularizou o termo *machine learning* | Partes 1 e 13 |
| **Frank Rosenblatt** | 1957–58 | **Perceptron**, primeiro algoritmo de aprendizado de uma rede neural | Parte 6 |
| **Marvin Minsky & Seymour Papert** | 1969 | Livro *Perceptrons*: mostrou os limites do Perceptron (ex.: XOR) | Parte 6 |
| **Thomas Cover & Peter Hart** | 1967 | Análise teórica do classificador do **vizinho mais próximo** (KNN) | Parte 7 |
| **Stuart Lloyd** | 1957/1982 | Algoritmo de **K-Means** (publicado em 1982) | Parte 12 |
| **Leslie Valiant** | 1984 | Teoria **PAC** (*Probably Approximately Correct*) do aprendizado; Prêmio Turing 2010 | Parte 1 |
| **J. Ross Quinlan** | 1986 / 1993 | Árvores de decisão **ID3** e **C4.5** | Parte 9 |
| **Leo Breiman** | 1984 / 2001 | **CART** (com Friedman, Olshen e Stone) e **Random Forests** | Partes 9 e 10 |
| **David Rumelhart, Geoffrey Hinton & Ronald Williams** | 1986 | Popularizaram o **backpropagation** para redes multicamadas | Parte 6 / Aula 11 |
| **Chris Watkins** | 1989 | Algoritmo **Q-Learning** | Parte 13 |
| **Corinna Cortes & Vladimir Vapnik** | 1995 | **Máquinas de Vetores de Suporte** (SVM); Vapnik também criou a teoria VC | Parte 10 |
| **Richard Sutton & Andrew Barto** | 1998 / 2018 | Livro de referência em **aprendizado por reforço**; Prêmio Turing 2024 | Parte 13 |
| **Trevor Hastie, Robert Tibshirani & Jerome Friedman** | 2001 | *The Elements of Statistical Learning*: ML sob a ótica estatística | — |
| **Christopher Bishop** | 2006 | *Pattern Recognition and Machine Learning*: abordagem probabilística/bayesiana | Parte 8 |
| **Pedro Domingos** | 2012 / 2015 | *A Few Useful Things to Know About Machine Learning* e *O Algoritmo Mestre* (as "cinco tribos" do ML) | Partes 5 e 15 |
| **Geoffrey Hinton, Yann LeCun & Yoshua Bengio** | 2006–2012 | Fundamentos do *deep learning*; Prêmio Turing 2018 | Parte 1 / Aula 11 |
| **Katti Faceli, Ana Carolina Lorena, João Gama, Tiago Almeida & André de Carvalho** | 2011 / 2021 | *Inteligência Artificial: uma abordagem de aprendizado de máquina* — principal livro-texto em português | Referência |

---

## 🗺️ Roteiro do notebook

| Parte | Tema | Destaques práticos |
|:-:|---|---|
| 0 | Preparação do ambiente | Imports e funções auxiliares de visualização |
| 1 | O que é aprender? T/P/E, história, terminologia, paradigmas | 🎬 Animação "P melhora com E"; linha do tempo; Celsius → Fahrenheit aprendido |
| 2 | Aprendizado de conceitos (Mitchell, cap. 2) | Find-S do zero 🎛️; Espaço de Versões por força bruta 🎬; viés indutivo |
| 3 | Dados e pré-processamento | Valores ausentes, *one-hot*, normalização, treino/validação/teste, vazamento de dados |
| 4 | Regressão linear e gradiente descendente | Ajuste manual 🎛️; GD em 1D 🎬; reta + superfície de custo 🎬; efeito de α 🎛️; equação normal |
| 5 | *Overfitting*, *underfitting* e viés–variância | Regressão polinomial 🎛️🎬; 25 modelos para visualizar variância; Ridge 🎛️ |
| 6 | Regressão logística e Perceptron | Sigmoide; fronteira durante o treino 🎬; Perceptron atualizando 🎬; problema XOR |
| 7 | KNN | Exemplo à mão (filmes); KNN do zero; busca de vizinhos 🎬; efeito de *k* 🎛️ |
| 8 | Naive Bayes | Exemplo *PlayTennis* passo a passo; filtro de spam com texto |
| 9 | Árvores de decisão (ID3) | Entropia e ganho; ID3 do zero; árvore crescendo 🎬; `plot_tree` |
| 10 | SVM e *ensembles* | Margem e vetores de suporte; kernel RBF 🎛️; Random Forest crescendo 🎬 |
| 11 | Avaliação de modelos | Matriz de confusão; precisão/revocação/F1; ROC varrendo o limiar 🎬; validação cruzada; *Grid Search* |
| 12 | Não supervisionado | K-Means do zero 🎬; cotovelo e silhueta; DBSCAN; segmentação de clientes; PCA 🎬 |
| 13 | Aprendizado por reforço | Q-Learning em labirinto; política aprendida ao longo dos episódios 🎬 |
| 14 | Projeto completo | *Pipeline* de *churn*: EDA, `ColumnTransformer`, comparação de modelos, ajuste, avaliação e uso |
| 15 | Ética, resumo, exercícios e referências | *Checklist* de boas práticas e 14 exercícios |

---

## 📚 Referências

- MITCHELL, Tom M. ***Machine Learning***. New York: McGraw-Hill, 1997.
- MITCHELL, Tom M. Generalization as search. *Artificial Intelligence*, v. 18, n. 2, p. 203–226, 1982.
- MITCHELL, Tom M. *The Need for Biases in Learning Generalizations*. Technical Report CBM-TR-117, Rutgers University, 1980.
- SAMUEL, Arthur L. Some studies in machine learning using the game of checkers. *IBM Journal of Research and Development*, v. 3, n. 3, 1959.
- ROSENBLATT, Frank. The perceptron: a probabilistic model for information storage and organization in the brain. *Psychological Review*, v. 65, n. 6, 1958.
- QUINLAN, J. Ross. Induction of decision trees. *Machine Learning*, v. 1, n. 1, 1986.
- CORTES, Corinna; VAPNIK, Vladimir. Support-vector networks. *Machine Learning*, v. 20, n. 3, 1995.
- BREIMAN, Leo. Random forests. *Machine Learning*, v. 45, n. 1, 2001.
- DOMINGOS, Pedro. A few useful things to know about machine learning. *Communications of the ACM*, v. 55, n. 10, 2012.
- RUSSELL, Stuart; NORVIG, Peter. *Inteligência Artificial: uma abordagem moderna*. 4. ed.
- FACELI, Katti et al. *Inteligência Artificial: uma abordagem de aprendizado de máquina*. 2. ed. Rio de Janeiro: LTC, 2021.
- HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. *The Elements of Statistical Learning*. 2. ed. Springer, 2009.
- SUTTON, Richard S.; BARTO, Andrew G. *Reinforcement Learning: An Introduction*. 2. ed. MIT Press, 2018.
- Documentação do scikit-learn: <https://scikit-learn.org/stable/user_guide.html>
