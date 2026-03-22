# 1. Puxa a classe principal para o nível da raiz da biblioteca
from .estimator import SyntheticDID

# 2. Define a versão da sua biblioteca
__version__ = "0.1.0"
__author__ = "Ademir Jose Fazzio Junior"
__description__ = "An attemp of robust implementation of Synthetic Diff-in-Diff in Python."

# 3. Controla o que é exportado se o utilizador fizer "from syndid import *"
__all__ = [
    "SyntheticDID"
]
