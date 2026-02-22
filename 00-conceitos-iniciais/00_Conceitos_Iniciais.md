# 00 — Conceitos Iniciais para Inteligência Artificial

**Disciplina:** Inteligência Artificial — Ciência da Computação

**Objetivo desta aula:** Revisar e consolidar os pré-requisitos matemáticos, lógicos e de programação que todo estudante deve dominar antes de iniciar o estudo formal de Inteligência Artificial.

**Duração estimada:** 2 horas de aula

---

## Sumário

1. [Por que pré-requisitos importam em IA?](#1-por-que-pré-requisitos-importam-em-ia)
2. [Matemática Essencial](#2-matemática-essencial)
   - 2.1 [Álgebra Linear](#21-álgebra-linear)
   - 2.2 [Cálculo Diferencial](#22-cálculo-diferencial)
   - 2.3 [Probabilidade e Estatística](#23-probabilidade-e-estatística)
3. [Lógica e Raciocínio Formal](#3-lógica-e-raciocínio-formal)
4. [Fundamentos de Programação em Python](#4-fundamentos-de-programação-em-python)
5. [Algoritmos e Estruturas de Dados](#5-algoritmos-e-estruturas-de-dados)
6. [Panorama da Inteligência Artificial](#6-panorama-da-inteligência-artificial)
7. [Perguntas e Respostas](#7-perguntas-e-respostas)
8. [Exercícios Práticos](#8-exercícios-práticos)
9. [Referências Bibliográficas](#9-referências-bibliográficas)

---

## 1. Por que pré-requisitos importam em IA?

A Inteligência Artificial é uma disciplina **interdisciplinar**: ela combina fundamentos de matemática, estatística, ciência da computação e domínio de aplicação. Antes de criar sistemas inteligentes, precisamos entender a linguagem usada para descrevê-los.

> *"Para construir sistemas que imitam a inteligência, é preciso primeiro entender os alicerces sobre os quais eles se apoiam."*

| Área | Por que é importante em IA? |
|------|--------------------------|
| Álgebra Linear | Representação de dados, transformações, redes neurais |
| Cálculo | Otimização de modelos (gradiente descendente) |
| Probabilidade & Estatística | Incerteza, aprendizado, inferência |
| Lógica | Raciocínio simbólico, regras, sistemas especialistas |
| Programação Python | Implementação prática de algoritmos e modelos |
| Algoritmos e Estruturas de Dados | Eficiência computacional |

---

## 2. Matemática Essencial

### 2.1 Álgebra Linear

Álgebra Linear é a **linguagem nativa** da IA moderna. Dados são representados como vetores e matrizes; transformações são operações matriciais.

#### Escalares, Vetores e Matrizes

- **Escalar:** Um único número. Ex.: `5`, `3.14`, `-2`.
- **Vetor:** Uma sequência ordenada de números.
  - Vetor coluna: **v** = [1, 2, 3]ᵀ
  - Representa um ponto no espaço ou um conjunto de atributos de um exemplo.
- **Matriz:** Uma grade retangular de números com _m_ linhas e _n_ colunas.

```
A = | 1  2  3 |
    | 4  5  6 |
    | 7  8  9 |
```

#### Operações Fundamentais

| Operação | Notação | Uso em IA |
|----------|---------|-----------|
| Soma de vetores | **u** + **v** | Combinação de features |
| Produto escalar | **u** · **v** = Σ uᵢvᵢ | Similaridade, ativações neurais |
| Multiplicação matricial | **A** × **B** | Transformações lineares, camadas de redes neurais |
| Transposta | **A**ᵀ | Reorganização de dados |
| Determinante | det(**A**) | Verificação de inversibilidade |
| Matriz inversa | **A**⁻¹ | Solução de sistemas lineares |

#### Exemplo: produto escalar

```
u = [1, 2, 3]
v = [4, 5, 6]

u · v = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```

O produto escalar mede o **quanto dois vetores "apontam na mesma direção"** — conceito central em classificadores lineares e em similaridade de cosseno em NLP.

#### Autovetores e Autovalores

Se **A** **v** = λ **v**, então:
- **v** é um **autovetor** de **A**
- λ é o **autovalor** correspondente

Fundamentais em:
- Análise de Componentes Principais (PCA)
- Algoritmos de recomendação (SVD)
- PageRank (Google)

#### Normas de Vetores

A norma mede o "tamanho" de um vetor:
- **L1 (Manhattan):** ‖v‖₁ = Σ|vᵢ|
- **L2 (Euclidiana):** ‖v‖₂ = √(Σvᵢ²)

Usadas para regularização de modelos (evitar overfitting).

---

### 2.2 Cálculo Diferencial

O **gradiente descendente** — o motor de aprendizado por trás de praticamente todos os modelos modernos — é puro cálculo.

#### Derivadas

A derivada de f(x) em relação a x mede a **taxa de variação instantânea** de f:

```
f'(x) = df/dx = lim(h→0) [f(x+h) - f(x)] / h
```

**Exemplos básicos:**

| Função f(x) | Derivada f'(x) |
|------------|---------------|
| x² | 2x |
| x³ | 3x² |
| e^x | e^x |
| ln(x) | 1/x |
| sen(x) | cos(x) |

#### Regra da Cadeia

Fundamental para **backpropagation** em redes neurais:

```
Se y = f(g(x)), então:
dy/dx = (df/dg) × (dg/dx)
```

#### Gradiente

Para funções de múltiplas variáveis f(x₁, x₂, ..., xₙ), o gradiente é o vetor das derivadas parciais:

```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

O gradiente **aponta na direção de maior crescimento** da função. Para minimizar (ex.: uma função de custo), andamos na direção **oposta** ao gradiente.

#### Gradiente Descendente (visão intuitiva)

```
θ ← θ - α × ∇L(θ)
```

Onde:
- θ = parâmetros do modelo
- α = taxa de aprendizado (learning rate)
- L(θ) = função de perda (loss function)

> **Analogia:** Imagine estar em um terreno acidentado com os olhos vendados e querer chegar ao ponto mais baixo do vale. A cada passo, você toca o chão e caminha na direção que desce mais — isso é o gradiente descendente.

---

### 2.3 Probabilidade e Estatística

Muitos problemas de IA são, na essência, problemas de **raciocínio sob incerteza**.

#### Conceitos Fundamentais de Probabilidade

- **Espaço amostral (Ω):** Conjunto de todos os resultados possíveis.
- **Evento:** Subconjunto do espaço amostral.
- **Probabilidade P(A):** Número entre 0 e 1 que representa a chance de A ocorrer.

**Axiomas de Kolmogorov:**
1. P(A) ≥ 0 para todo evento A
2. P(Ω) = 1
3. Se A e B são mutuamente exclusivos: P(A ∪ B) = P(A) + P(B)

#### Probabilidade Condicional e Regra de Bayes

```
P(A|B) = P(A ∩ B) / P(B)
```

**Teorema de Bayes:**

```
P(A|B) = P(B|A) × P(A) / P(B)
```

Este teorema é a base de:
- Classificadores Naïve Bayes
- Inferência Bayesiana
- Redes Bayesianas

**Exemplo:** Um médico suspeita de uma doença rara. O exame tem 95% de acurácia. A doença afeta 1% da população. Se o paciente testou positivo, qual a chance real de ter a doença?

```
P(doença|positivo) = P(positivo|doença) × P(doença) / P(positivo)
= 0.95 × 0.01 / [(0.95 × 0.01) + (0.05 × 0.99)]
≈ 0.161 (16.1%)
```

Apesar do exame ser 95% preciso, a probabilidade real é de apenas ~16% — isso é o paradoxo de Bayes, e é fundamental para entender classificadores.

#### Estatística Descritiva

| Medida | Fórmula | Significado |
|--------|---------|-------------|
| Média (μ) | Σxᵢ/n | Centro dos dados |
| Variância (σ²) | Σ(xᵢ-μ)²/n | Dispersão |
| Desvio-padrão (σ) | √σ² | Dispersão na mesma unidade dos dados |
| Mediana | Valor central | Robusta a outliers |

#### Distribuições Importantes

- **Bernoulli:** Experimento com dois resultados (sucesso/falha). Base para classificação binária.
- **Binomial:** k sucessos em n tentativas.
- **Gaussiana (Normal):** Distribuição "em sino" — regra dos 68-95-99.7%.
- **Uniforme:** Todos os valores igualmente prováveis.

#### Correlação vs. Causalidade

> **Atenção:** Correlação **não implica** causalidade!

Em IA, é fácil encontrar correlações espúrias nos dados. Um modelo bem treinado aprende padrões, mas não necessariamente causas.

---

## 3. Lógica e Raciocínio Formal

### Lógica Proposicional

Uma **proposição** é uma afirmação que pode ser verdadeira (V) ou falsa (F).

**Conectivos lógicos:**

| Símbolo | Nome | Significado |
|---------|------|-------------|
| ¬ | Negação (NOT) | Inverte o valor |
| ∧ | Conjunção (AND) | Verdade se ambos verdadeiros |
| ∨ | Disjunção (OR) | Verdade se pelo menos um verdadeiro |
| → | Implicação (IF...THEN) | Falso apenas se antecedente V e consequente F |
| ↔ | Bicondicional (IFF) | Verdade se ambos iguais |

**Tabela-verdade do AND e OR:**

| P | Q | P ∧ Q | P ∨ Q | P → Q |
|---|---|-------|-------|-------|
| V | V |   V   |   V   |   V   |
| V | F |   F   |   V   |   F   |
| F | V |   F   |   V   |   V   |
| F | F |   F   |   F   |   V   |

### Leis de De Morgan

```
¬(P ∧ Q) ≡ ¬P ∨ ¬Q
¬(P ∨ Q) ≡ ¬P ∧ ¬Q
```

Fundamentais em programação e na definição de condições complexas.

### Raciocínio Dedutivo vs. Indutivo

| Tipo | Descrição | Exemplo em IA |
|------|-----------|--------------|
| **Dedutivo** | Da regra geral para o caso específico | Sistemas especialistas com regras |
| **Indutivo** | Do caso específico para a regra geral | Aprendizado de máquina (aprende padrões de exemplos) |
| **Abdutivo** | Infere a melhor explicação | Diagnóstico médico, detecção de anomalias |

---

## 4. Fundamentos de Programação em Python

Python é a linguagem dominante em IA. Antes do curso, você deve estar confortável com os conceitos abaixo.

### 4.1 Tipos de Dados e Variáveis

```python
# Tipos básicos
inteiro = 42
flutuante = 3.14
booleano = True
texto = "Inteligência Artificial"

# Coleções
lista = [1, 2, 3, 4, 5]
tupla = (10, 20, 30)         # imutável
dicionario = {"chave": "valor", "nome": "IA"}
conjunto = {1, 2, 3, 3}     # {1, 2, 3} — sem repetição
```

### 4.2 Estruturas de Controle

```python
# Condicional
if temperatura > 37.5:
    print("Febre!")
elif temperatura > 36:
    print("Normal")
else:
    print("Hipotermia")

# Laço for
for i in range(5):
    print(i)

# Laço while
contador = 0
while contador < 10:
    contador += 1

# Compreensão de lista (list comprehension)
quadrados = [x**2 for x in range(10)]
pares = [x for x in range(20) if x % 2 == 0]
```

### 4.3 Funções

```python
# Definição básica
def saudacao(nome):
    return f"Olá, {nome}!"

# Parâmetros padrão
def potencia(base, expoente=2):
    return base ** expoente

# Funções lambda (anônimas)
dobro = lambda x: x * 2

# Funções de ordem superior
numeros = [1, 2, 3, 4, 5]
quadrados = list(map(lambda x: x**2, numeros))
pares = list(filter(lambda x: x % 2 == 0, numeros))
```

### 4.4 Bibliotecas Essenciais para IA

#### NumPy — Computação Numérica

```python
import numpy as np

# Criando arrays
v = np.array([1, 2, 3])
A = np.array([[1, 2], [3, 4]])

# Operações vetorizadas (muito mais rápidas que loops!)
resultado = v * 2          # [2, 4, 6]
produto = np.dot(v, v)     # produto escalar = 14

# Funções úteis
np.zeros((3, 3))           # matriz 3×3 de zeros
np.ones((2, 4))            # matriz 2×4 de uns
np.random.rand(3, 3)       # matriz aleatória uniforme
np.random.randn(3, 3)      # matriz aleatória normal
A.shape                    # (2, 2) — dimensões
A.T                        # transposta
np.linalg.inv(A)           # inversa
np.linalg.det(A)           # determinante
```

#### Pandas — Manipulação de Dados

```python
import pandas as pd

# Criando DataFrame
df = pd.DataFrame({
    "nome": ["Ana", "Bruno", "Carlos"],
    "idade": [25, 30, 22],
    "nota": [8.5, 7.0, 9.2]
})

# Explorando os dados
df.head()            # primeiras linhas
df.describe()        # estatísticas descritivas
df.info()            # tipos e valores nulos
df["nota"].mean()    # média de uma coluna

# Filtragem
aprovados = df[df["nota"] >= 7.0]

# Leitura de arquivo
df = pd.read_csv("dados.csv")
```

#### Matplotlib — Visualização de Dados

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, label="seno", color="blue")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Função Seno")
plt.legend()
plt.grid(True)
plt.show()
```

> **Dica:** A visualização de dados é fundamental em IA. Antes de treinar qualquer modelo, **sempre** explore visualmente seus dados!

---

## 5. Algoritmos e Estruturas de Dados

### Complexidade Algorítmica (Notação Big-O)

Mede o **comportamento do tempo de execução** em função do tamanho da entrada n.

| Notação | Nome | Exemplo |
|---------|------|---------|
| O(1) | Constante | Acesso a elemento de array |
| O(log n) | Logarítmica | Busca binária |
| O(n) | Linear | Percorrer uma lista |
| O(n log n) | Linearítmica | Merge sort, heapsort |
| O(n²) | Quadrática | Bubble sort, comparações par a par |
| O(2ⁿ) | Exponencial | Força bruta em subconjuntos |

> Em IA, preferimos sempre algoritmos com complexidade mais baixa, especialmente quando trabalhamos com grandes volumes de dados.

### Estruturas de Dados Relevantes

| Estrutura | Acesso | Inserção | Uso em IA |
|-----------|--------|----------|-----------|
| **Array/Lista** | O(1) | O(n) | Vetores de features |
| **Dicionário (Hash Map)** | O(1) | O(1) | Vocabulários em NLP |
| **Fila (Queue)** | O(n) | O(1) | BFS em busca em grafos |
| **Pilha (Stack)** | O(n) | O(1) | DFS, parsers |
| **Grafo** | — | — | Redes bayesianas, agentes |
| **Árvore** | O(log n) | O(log n) | Árvores de decisão |

### Recursão

Muitos algoritmos de IA (busca em árvore, divisão e conquista) são naturalmente recursivos:

```python
def fatorial(n):
    if n == 0:     # caso base
        return 1
    return n * fatorial(n - 1)   # chamada recursiva

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

> **Atenção:** Recursão sem caso base bem definido gera *stack overflow*. Lembre-se sempre de definir o caso base!

---

## 6. Panorama da Inteligência Artificial

### O que é Inteligência Artificial?

IA é o campo da ciência da computação que estuda como criar sistemas capazes de realizar tarefas que normalmente requerem **inteligência humana**: raciocinar, aprender, perceber, comunicar e agir no mundo.

### Principais Subáreas da IA

```
Inteligência Artificial
│
├── Representação do Conhecimento
│   ├── Lógica (proposicional, primeira ordem)
│   └── Ontologias
│
├── Busca e Planejamento
│   ├── Busca cega (BFS, DFS)
│   └── Busca heurística (A*)
│
├── Aprendizado de Máquina (Machine Learning)
│   ├── Aprendizado supervisionado
│   ├── Aprendizado não supervisionado
│   └── Aprendizado por reforço
│
├── Redes Neurais Artificiais
│   └── Deep Learning
│
├── Processamento de Linguagem Natural (NLP)
│
├── Visão Computacional
│
└── Sistemas Multiagentes
```

### IA Fraca vs. IA Forte

| Tipo | Definição | Status atual |
|------|-----------|--------------|
| **IA Fraca (Narrow AI)** | Especializada em uma tarefa específica | Existente (ChatGPT, AlphaGo, reconhecimento de imagens) |
| **IA Forte (General AI)** | Capacidade cognitiva geral como humanos | Hipotética, não alcançada |
| **Super IA** | Supera a inteligência humana em todos os aspectos | Teórica |

### O Teste de Turing

Proposto por **Alan Turing** em 1950, o teste avalia se uma máquina pode exibir comportamento inteligente indistinguível do humano. Um avaliador humano conversa (via texto) com um humano e uma máquina; se não conseguir distinguir qual é qual, a máquina "passou" no teste.

### Aplicações de IA no Mundo Real

| Área | Exemplo |
|------|---------|
| Saúde | Diagnóstico por imagem, descoberta de medicamentos |
| Finanças | Detecção de fraudes, trading algorítmico |
| Transporte | Veículos autônomos |
| Entretenimento | Recomendação (Netflix, Spotify) |
| Linguagem | Tradução automática, assistentes virtuais |
| Segurança | Reconhecimento facial, detecção de intrusão |

---

## 7. Perguntas e Respostas

> Esta seção simula uma sessão de revisão com perguntas que tipicamente surgem antes de começar a disciplina.

---

### 📐 Bloco 1: Matemática

**P1. Por que preciso de álgebra linear para aprender IA?**

> **R:** Dados são representados como vetores (cada exemplo é um ponto num espaço n-dimensional). Modelos de IA aplicam **transformações lineares** a esses vetores. Redes neurais são, matematicamente, composições de multiplicações matriciais e funções de ativação. Sem álgebra linear, é impossível entender o que acontece "por baixo do capô" de qualquer modelo.

---

**P2. O que é o gradiente e por que ele é importante?**

> **R:** O gradiente é o vetor de derivadas parciais de uma função em relação a cada parâmetro. Em IA, usamos funções de perda (loss functions) que medem o erro do modelo. O gradiente indica a direção de maior crescimento dessa função; ao andamos na direção **oposta** (gradiente descendente), reduzimos o erro e o modelo "aprende". É o mecanismo central de treinamento de quase todos os modelos modernos.

---

**P3. Por que probabilidade é fundamental em IA?**

> **R:** O mundo real é incerto. Dados são ruidosos, observações são incompletas, e modelos nunca são 100% certos. Probabilidade nos dá uma linguagem rigorosa para raciocinar sobre essa incerteza. Classificadores como Naïve Bayes, modelos generativos e redes bayesianas são diretamente baseados em teoria das probabilidades. Mesmo métricas de avaliação (como a função de custo de entropia cruzada) têm interpretação probabilística.

---

**P4. Qual a diferença entre variância e desvio-padrão?**

> **R:** A **variância** (σ²) mede a dispersão dos dados ao quadrado da unidade original. O **desvio-padrão** (σ) é a raiz quadrada da variância, portanto está na mesma unidade dos dados e é mais intuitivo para interpretação. Por exemplo, se os dados são em centímetros, a variância é em cm² e o desvio-padrão em cm. Em IA, o desvio-padrão é muito usado para normalizar features (padronização Z-score: z = (x - μ) / σ).

---

**P5. O que é correlação? Ela implica causalidade?**

> **R:** Correlação mede a força e direção da relação linear entre duas variáveis (varia de -1 a +1). **Não**, correlação não implica causalidade! Um exemplo famoso: o número de afogamentos em piscinas aumenta junto com o consumo de sorvete — mas sorvete não causa afogamentos. Ambos aumentam no verão (variável confundidora). Em IA, modelos podem aprender correlações espúrias, o que pode levar a conclusões erradas ou modelos injustos.

---

### 🧠 Bloco 2: Lógica

**P6. O que é uma tautologia?**

> **R:** Uma tautologia é uma fórmula lógica que é **sempre verdadeira**, independentemente dos valores de suas variáveis. Exemplo: P ∨ ¬P (ou P é verdadeiro, ou sua negação é verdadeira — sempre verdade). Tautologias são usadas para validar sistemas de raciocínio em IA simbólica.

---

**P7. Por que a lei de De Morgan é útil na programação?**

> **R:** A lei de De Morgan afirma que `¬(A ∧ B) ≡ ¬A ∨ ¬B` e `¬(A ∨ B) ≡ ¬A ∧ ¬B`. Na prática, é frequentemente necessário negar condições compostas. Por exemplo, em vez de `not (x > 0 and y > 0)`, podemos escrever `x <= 0 or y <= 0`. Isso é essencial para escrever condições de parada claras em algoritmos de busca e em sistemas de regras.

---

**P8. Qual a diferença entre raciocínio dedutivo e indutivo?**

> **R:** No raciocínio **dedutivo**, partimos de regras gerais para chegar a conclusões específicas (garantidas). Ex.: "Todos os humanos são mortais; Sócrates é humano; logo, Sócrates é mortal." No raciocínio **indutivo**, partimos de casos específicos para formular hipóteses gerais (prováveis, mas não garantidas). Ex.: "Observei 1.000 cisnes brancos; logo, provavelmente todos os cisnes são brancos" — mas um único cisne negro invalida isso. O aprendizado de máquina é essencialmente **indutivo**: aprende padrões de exemplos.

---

### 🐍 Bloco 3: Python

**P9. Por que Python domina a área de IA?**

> **R:** Python combina **simplicidade de sintaxe** com um **ecossistema excepcional** de bibliotecas científicas: NumPy (álgebra linear), Pandas (dados tabulares), Matplotlib/Seaborn (visualização), Scikit-learn (machine learning), TensorFlow e PyTorch (deep learning). Além disso, tem enorme comunidade, é multiplataforma e integra bem com C/C++ para partes críticas de performance. Tudo isso torna a prototipagem rápida e o compartilhamento de código muito eficiente.

---

**P10. O que é vetorização e por que ela importa em NumPy?**

> **R:** Vetorização é a técnica de aplicar operações a arrays inteiros em vez de usar laços explícitos. Em NumPy, operações vetorizadas são implementadas em C internamente, sendo **dezenas a centenas de vezes mais rápidas** que loops Python. Em IA, trabalhamos frequentemente com milhões de dados; a diferença pode ser entre um código que roda em segundos versus horas.
> ```python
> # Lento (loop Python)
> resultado = [x**2 for x in range(1_000_000)]
> 
> # Rápido (vetorizado NumPy)
> import numpy as np
> resultado = np.arange(1_000_000) ** 2
> ```

---

**P11. O que é overfitting e como ele se relaciona com os conceitos que estamos estudando?**

> **R:** **Overfitting** ocorre quando um modelo aprende os dados de treinamento tão bem que "memoriza" ruídos e deixa de generalizar para dados novos. Matematicamente, um modelo com muitos parâmetros pode ter variância alta (responde excessivamente a flutuações nos dados). Isso está diretamente relacionado ao **trade-off viés-variância**, conceito estatístico fundamental. Técnicas como regularização (usando normas L1/L2 de álgebra linear) e validação cruzada (estatística) são usadas para combater o overfitting.

---

**P12. Qual a complexidade de busca em uma lista vs. em um dicionário Python?**

> **R:** A busca em uma **lista** tem complexidade **O(n)** no pior caso (precisa verificar elemento por elemento). A busca em um **dicionário** tem complexidade **O(1)** em média, pois usa uma tabela hash internamente. Em IA, isso é crítico: ao construir índices invertidos para NLP ou mapas de estados em problemas de busca, usar dicionários em vez de listas pode reduzir o tempo de execução drasticamente.

---

### 🤖 Bloco 4: Panorama de IA

**P13. Qual a diferença entre IA, Machine Learning e Deep Learning?**

> **R:**
> - **IA** é o campo mais amplo: qualquer técnica que permite a máquinas realizarem tarefas inteligentes.
> - **Machine Learning (ML)** é uma subárea da IA que foca em sistemas que **aprendem com dados** sem serem explicitamente programados.
> - **Deep Learning (DL)** é uma subárea do ML que usa **redes neurais com muitas camadas** para aprender representações hierárquicas de dados. O Deep Learning é responsável por avanços recentes em visão computacional, NLP e geração de conteúdo.

---

**P14. O que significa dizer que um modelo é "supervisionado"?**

> **R:** Um modelo **supervisionado** aprende a partir de pares de (entrada, saída esperada), chamados de dados rotulados. O modelo ajusta seus parâmetros para minimizar o erro entre suas previsões e as saídas esperadas. Ex.: treinar um classificador de spam com e-mails rotulados como "spam" ou "não spam". No aprendizado **não supervisionado**, não há rótulos — o modelo busca padrões ocultos nos dados (ex.: clustering).

---

**P15. Quais são os riscos éticos da IA que um desenvolvedor deve conhecer?**

> **R:** Um desenvolvedor responsável deve conhecer:
> - **Viés algorítmico:** modelos treinados em dados históricos podem reproduzir e amplificar preconceitos sociais.
> - **Privacidade:** modelos podem memorizar dados sensíveis de treinamento.
> - **Interpretabilidade:** modelos complexos ("caixa-preta") são difíceis de auditar e explicar.
> - **Uso malicioso:** deepfakes, desinformação automatizada, vigilância em massa.
> - **Deslocamento de empregos:** automação de tarefas cognitivas e físicas.
>
> O campo de **IA Responsável (Responsible AI)** aborda esses desafios com técnicas como explicabilidade (XAI), fairness, e privacidade diferencial.

---

## 8. Exercícios Práticos

Os exercícios a seguir devem ser realizados em Python (Jupyter Notebook recomendado).

### Exercício 1 — Álgebra Linear com NumPy

```python
import numpy as np

# a) Crie dois vetores u = [1, 2, 3] e v = [4, 5, 6].
# Calcule: produto escalar, norma L2 de cada um e
# o ângulo entre eles (cos θ = u·v / (‖u‖ × ‖v‖))

u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Seu código aqui...

# b) Crie a matriz identidade 3×3 e verifique que A × I = A
# para qualquer matriz A 3×3 de sua escolha.
```

### Exercício 2 — Probabilidade com Simulação

```python
import numpy as np

# Simule 10.000 lançamentos de um dado de 6 faces.
# a) Calcule a frequência de cada face.
# b) Verifique que a média converge para 3.5.
# c) Calcule a probabilidade experimental de obter
#    um número par em pelo menos 3 de 5 lançamentos.

np.random.seed(42)
# Seu código aqui...
```

### Exercício 3 — Análise Exploratória com Pandas

```python
import pandas as pd
import matplotlib.pyplot as plt

# Crie um DataFrame com dados fictícios de 50 alunos:
# (nome, nota_prova1, nota_prova2, frequencia)
# a) Calcule média, mediana e desvio-padrão das notas.
# b) Crie um histograma das notas.
# c) Encontre alunos com frequência < 75%.
# d) Calcule a correlação entre nota_prova1 e nota_prova2.

import numpy as np
np.random.seed(42)
n = 50
df = pd.DataFrame({
    "nota_prova1": np.random.normal(7, 1.5, n).clip(0, 10),
    "nota_prova2": np.random.normal(6.5, 2, n).clip(0, 10),
    "frequencia": np.random.uniform(50, 100, n)
})

# Seu código aqui...
```

### Exercício 4 — Gradiente Descendente do Zero

```python
import numpy as np
import matplotlib.pyplot as plt

# Implemente o gradiente descendente para minimizar:
# f(x) = x² - 4x + 4   (mínimo em x = 2)
#
# f'(x) = 2x - 4
#
# Parâmetros: x_inicial = 0.0, alpha = 0.1, n_iterações = 50

def f(x):
    return x**2 - 4*x + 4

def df(x):
    return 2*x - 4

# Seu código aqui (implemente o loop de atualização e
# plote a trajetória convergindo para x = 2)
```

---

## 9. Referências Bibliográficas

1. **Russell, S. & Norvig, P.** — *Artificial Intelligence: A Modern Approach* (4ª ed.). Pearson, 2020.
2. **Goodfellow, I., Bengio, Y. & Courville, A.** — *Deep Learning*. MIT Press, 2016. [Disponível em: deeplearningbook.org]
3. **Bishop, C. M.** — *Pattern Recognition and Machine Learning*. Springer, 2006.
4. **Geron, A.** — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3ª ed.). O'Reilly, 2022.
5. **VanderPlas, J.** — *Python Data Science Handbook*. O'Reilly, 2016. [Disponível em: jakevdp.github.io/PythonDataScienceHandbook]
6. **Strang, G.** — *Introduction to Linear Algebra* (5ª ed.). Wellesley-Cambridge Press, 2016.
7. **Wackerly, D., Mendenhall, W. & Scheaffer, R.** — *Mathematical Statistics with Applications* (7ª ed.). Cengage, 2008.

---

> 💡 **Dica final:** Se você se sentiu inseguro em algum dos tópicos desta aula, dedique pelo menos uma semana para revisá-los antes de prosseguir. Uma base sólida em matemática e programação fará toda a diferença na sua jornada pela Inteligência Artificial!

---

*Material preparado para a disciplina de Inteligência Artificial — Ciência da Computação*
