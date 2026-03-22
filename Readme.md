# PySDID: Synthetic Difference-in-Differences em Python

Uma implementação leve, rápida e matematicamente rigorosa (ou o mais próximo disso que eu consegui) do estimador **Synthetic Difference-in-Differences (SDID)** em Python, baseado no artigo seminal de [Arkhangelsky et al. (2021)](https://www.aeaweb.org/articles?id=10.1257/aer.20190159).

## Filosofia do Pacote: Simples e Robusto

O objetivo principal do **PySDID** é fornecer um pacote **simples e robusto**, utilizando apenas "o básico do básico" para funcionar com máxima eficiência.

O motor do PySDID é construído 100% sobre **Otimização Convexa (Quadratic Programming)** usando a biblioteca `cvxpy` (com os solvers OSQP/ECOS). 

Convergência global, evitando falhas de *linesearch*, sem gambiarras de limites matemáticos e com uma velocidade impressionante até mesmo nos Testes de Placebo. (Fizemos testes de placebos )

## Principais Funcionalidades

* **Estimação Precisa do ATT:** Cálculo direto dos pesos de tempo ($\lambda$) e de unidade ($\omega$) com penalização Ridge.
* **Inferência Padrão-Ouro:** Implementação nativa de **Testes de Placebo no Espaço** para cálculo exato de P-valor e Intervalo de Confiança (ideal para cenários com poucos tratados, como $N_{tr} = 1$).
* **Gráficos Prontos para Artigos:** Funções embutidas para plotar o Histograma de Placebos e a Evolução Temporal (Tratado vs. Sintético).
* **Tabela de Resultados (Summary):** Saída de dados formatada no estilo `statsmodels`, pronta para relatórios e artigos acadêmicos.

## Instalação

Como o pacote foca no essencial, as dependências são mínimas (`numpy`, `cvxpy`, `matplotlib`, `pandas`, `scipy`).

Instale diretamente via pip:
```bash
# (Em breve no PyPI)
# pip install pysdid

# Para instalar a partir do código fonte localmente:
git clone [https://github.com/seu-usuario/pysdid.git](https://github.com/seu-usuario/pysdid.git)
cd pysdid
pip install -e .