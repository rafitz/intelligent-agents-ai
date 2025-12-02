# ☁️ Lógica Fuzzy - Sistemas de Controle

Este diretório contém exemplos práticos de **Sistemas de Inferência Fuzzy** (Lógica Difusa) aplicados a problemas de controle, utilizando a biblioteca `scikit-fuzzy`.

Ao contrário da lógica booleana tradicional (0 ou 1), a lógica fuzzy permite modelar graus de pertinência, aproximando-se do raciocínio humano.

---

## 🎯 Projetos Incluídos

### 1. 🍽️ Cálculo de Gorjeta (`calculo_gorjeta.py`)
Um sistema clássico para determinar a porcentagem de gorjeta em um restaurante.
* **Antecedentes (Entradas):**
    * Qualidade da comida (0 a 10)
    * Qualidade do serviço (0 a 10)
* **Consequente (Saída):**
    * Valor da gorjeta (0% a 20%)
* **Lógica:** Regras combinam qualidade e serviço para definir se a gorjeta será baixa, média ou alta.

### 2. 🌱 Sistema de Irrigação Inteligente (`sistema_irrigacao.py`)
Um sistema de automação para determinar o tempo de rega baseado em sensores ambientais.
* **Antecedentes (Entradas):**
    * Umidade do solo (0% a 100%)
    * Temperatura (0°C a 40°C)
* **Consequente (Saída):**
    * Tempo de rega (0 a 60 minutos)
* **Lógica:** O tempo de rega aumenta se o solo estiver seco e quente, e diminui se estiver úmido ou frio.

---

## 🛠️ Detalhes da Implementação

O fluxo de ambos os sistemas segue o padrão:
1.  **Fuzzificação:** Definição das funções de pertinência (triangulares, gaussianas, sigmoides).
2.  **Base de Regras:** Criação das regras "SE... ENTÃO".
3.  **Inferência:** Aplicação das regras às entradas.
4.  **Defuzzificação:** Cálculo do valor final (crisp) utilizando o método do centroide.
5.  **Visualização:** Gráficos das funções de pertinência e resultado.

---

## 📂 Arquivos
- `calculo_gorjeta.py`
- `sistema_irrigacao.py`

---

## ⚙️ Tecnologias
- Python
- Scikit-Fuzzy
- NumPy
- Matplotlib

---

## ▶️ Como Executar

Instale a biblioteca `scikit-fuzzy` (necessária para rodar os exemplos):

```bash
pip install scikit-fuzzy numpy matplotlib