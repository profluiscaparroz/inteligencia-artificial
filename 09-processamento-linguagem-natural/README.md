# 🗣️ Aula 09 — Processamento de Linguagem Natural (PLN)

**Disciplina:** Inteligência Artificial · **Curso:** Ciência da Computação

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/profluiscaparroz/inteligencia-artificial/blob/main/09-processamento-linguagem-natural/09_Processamento_Linguagem_Natural.ipynb)

---

## 📂 Conteúdo da pasta

| Arquivo | Descrição |
|---|---|
| [`09_Processamento_Linguagem_Natural.ipynb`](09_Processamento_Linguagem_Natural.ipynb) | Aula completa: ambiguidade e mini-ELIZA, pré-processamento (RSLP × lematização), Lei de Zipf com Machado de Assis, BPE, distância de edição, BoW/TF-IDF (com conta à mão), buscador, sentimentos (Naive Bayes à mão, precisão/revocação/F1), modelo de linguagem e perplexidade, *embeddings*, atenção e LLMs, com **animações** e **controles interativos** |
| [`Exercicios_PLN.ipynb`](Exercicios_PLN.ipynb) | Lista de exercícios |

### Como executar

- **Google Colab (recomendado):** clique em *Abrir no Colab* e execute as células em ordem. A seção da Lei de Zipf baixa automaticamente os romances de **Machado de Assis** (domínio público) pelo NLTK.
- **Localmente:**
  ```bash
  pip install numpy matplotlib scikit-learn nltk ipywidgets jupyter
  jupyter notebook 09_Processamento_Linguagem_Natural.ipynb
  ```
  > No macOS, se o download do NLTK falhar com erro de certificado SSL, execute o `Install Certificates.command` da sua instalação do Python.

Células com 🎬 geram **animações** e com 🎛️ têm **controles interativos**. Depois de **cada resultado** há uma seção **🔎 Como interpretar**.

---

## 🎯 Ideia central

PLN é a área da IA que permite a computadores **processar, compreender e gerar** linguagem humana. O desafio é que a linguagem é **ambígua**, depende de **contexto** e de **conhecimento de mundo**. A aula percorre a evolução das abordagens:

```
Regras escritas à mão  ──►  Estatística e contagens  ──►  Aprendizado de máquina  ──►  Embeddings e Transformers
(regex, léxicos)            (Zipf, n-gramas, TF-IDF)       (Naive Bayes)                 (Word2Vec, BERT, GPT)
```

Dois princípios atravessam toda a aula:
1. **Texto precisa virar números** (vetores) para ser processado por algoritmos;
2. **O significado de uma palavra está no seu uso**: palavras em contextos parecidos têm significados parecidos (hipótese distribucional).

---

## 👨‍🏫 Autores e pioneiros

| Autor(es) | Ano | Contribuição | Onde aparece na aula |
|---|:-:|---|---|
| **Alan Turing** | 1950 | Propôs a conversa em linguagem natural como teste de inteligência (Teste de Turing) | Parte 1 |
| **Claude Shannon** | 1948 | Teoria da informação e as primeiras aproximações da língua inglesa por **n-gramas** | Seção 4.5 (modelo de linguagem) |
| **George Kingsley Zipf** | 1935/1949 | **Lei de Zipf**: a frequência de uma palavra é inversamente proporcional ao seu ranking | Seção 2.7 |
| **Zellig Harris & J. R. Firth** | 1954/1957 | **Hipótese distribucional**: o significado de uma palavra é dado pelos contextos em que ela ocorre | Seção 4.6 (*embeddings*) |
| **Noam Chomsky** | 1957 | *Syntactic Structures*: gramáticas gerativas e a hierarquia de linguagens formais | Parte 1; regex (Seção 2.6) |
| **Vladimir Levenshtein** | 1965 | **Distância de edição** entre cadeias | Seção 2.9 (animação e corretor) |
| **Joseph Weizenbaum** | 1966 | **ELIZA**, o primeiro *chatbot*, baseado em casamento de padrões | Seção 1.6 (mini-ELIZA) |
| **Terry Winograd** | 1968–1972 | **SHRDLU** e os esquemas de Winograd (ambiguidade referencial) | Seções 1.2 e 1.5 |
| **Karen Spärck Jones** | 1972 | **IDF** (*inverse document frequency*): termos raros discriminam mais | Seções 3.3–3.5 |
| **Gerard Salton** | 1970s | **Modelo vetorial** de recuperação de informação e o sistema SMART | Seção 3.5 (buscador) |
| **Martin Porter** | 1980 | Algoritmo de ***stemming*** de Porter (inglês) | Seção 2 |
| **Viviane Orengo & Christian Huyck** | 2001 | ***Stemmer* RSLP** para o português (disponível no NLTK) | Seção 2 |
| **Frederick Jelinek** e equipe da IBM | 1970s–1990s | Reconhecimento de fala e tradução **estatísticos**, a virada das regras para os dados | Parte 1 |
| **Philip Gage / Rico Sennrich et al.** | 1994/2016 | **BPE**: algoritmo de compressão adaptado para tokenização por subpalavras | Seção 2.8 (animação) |
| **Tomas Mikolov et al.** | 2013 | **Word2Vec**: *embeddings* de palavras aprendidos por redes neurais | Seção 4.6 |
| **Ashish Vaswani et al.** | 2017 | **Transformer** (*Attention Is All You Need*) | Seções 4.7–4.8 (atenção e LLMs) |
| **Jacob Devlin et al.** | 2018 | **BERT**: modelos pré-treinados bidirecionais | Parte 5 |
| **Fábio Souza, Rodrigo Nogueira & Roberto Lotufo** | 2020 | **BERTimbau**, BERT para o português brasileiro | Parte 5 |
| **Dan Jurafsky & James H. Martin** | 2000– | *Speech and Language Processing*, o principal livro-texto da área | Toda a aula |
| **Christopher Manning & Hinrich Schütze** | 1999 | *Foundations of Statistical NLP* | Partes 2–4 |

---

## 🗺️ Roteiro do notebook

| Parte | Tema | Destaques práticos |
|:-:|---|---|
| 0 | Preparação | — |
| 1 | O que é PLN, história, níveis de análise e tipos de ambiguidade | Mini-ELIZA com regras de regex |
| 2 | Pré-processamento | Pipeline interativo 🎛️; *stemming* RSLP × lematização; regex; Lei de Zipf com 540 mil palavras de Machado de Assis; BPE 🎬; distância de edição 🎬; corretor ortográfico |
| 3 | Representação vetorial | Matriz BoW e similaridades; n-gramas; TF-IDF passo a passo; mini buscador 🎛️ |
| 4 | Sentimentos e além | Léxico × Naive Bayes (e a armadilha da mudança de distribuição); precisão, revocação e F1; Naive Bayes calculado à mão; modelo de linguagem de bigramas 🎬; perplexidade e suavização; *embeddings* por coocorrência (PPMI + SVD); autoatenção de brinquedo; como funciona um LLM |
| 5–10 | Bibliotecas (NLTK, spaCy, Hugging Face), aplicações, comparações, resumo da aula, exercícios e referências | Perguntas para discussão |

---

## 📚 Referências

- JURAFSKY, D.; MARTIN, J. H. *Speech and Language Processing*. 3. ed. (rascunho on-line): <https://web.stanford.edu/~jurafsky/slp3/>
- MANNING, C. D.; SCHÜTZE, H. *Foundations of Statistical Natural Language Processing*. MIT Press, 1999.
- BIRD, S.; KLEIN, E.; LOPER, E. *Natural Language Processing with Python*. O'Reilly, 2009.
- SHANNON, C. E. A mathematical theory of communication. *Bell System Technical Journal*, 27, 1948.
- ZIPF, G. K. *Human Behavior and the Principle of Least Effort*. Addison-Wesley, 1949.
- SPÄRCK JONES, K. A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1), 1972.
- SALTON, G.; WONG, A.; YANG, C. S. A vector space model for automatic indexing. *Communications of the ACM*, 18(11), 1975.
- LEVENSHTEIN, V. I. Binary codes capable of correcting deletions, insertions, and reversals. *Soviet Physics Doklady*, 10(8), 1966.
- ORENGO, V. M.; HUYCK, C. A stemming algorithm for the Portuguese language. *SPIRE*, 2001.
- SENNRICH, R.; HADDOW, B.; BIRCH, A. Neural machine translation of rare words with subword units. *ACL*, 2016.
- MIKOLOV, T. et al. Efficient estimation of word representations in vector space. *arXiv:1301.3781*, 2013.
- VASWANI, A. et al. Attention is all you need. *NeurIPS*, 2017.
- DEVLIN, J. et al. BERT: pre-training of deep bidirectional transformers for language understanding. *NAACL*, 2019.
- SOUZA, F.; NOGUEIRA, R.; LOTUFO, R. BERTimbau: pretrained BERT models for Brazilian Portuguese. *BRACIS*, 2020.
- Corpus Machado de Assis (NLTK): <https://www.nltk.org/nltk_data/>
