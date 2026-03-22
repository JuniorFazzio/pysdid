import numpy as np
from scipy.optimize import minimize
import cvxpy as cp


def compute_zeta(Y_pre_co: np.ndarray, N_tr: int, T_post: int) -> float:
    """
    Calcula o parâmetro de regularização (zeta) usando as primeiras diferenças das unidades de controle.
    """
    # Pegando os calculos de diferença de tempo e fazendo sigma
    diffTime = np.diff(Y_pre_co, axis=0)
    sigma = np.std(diffTime)

    # Aplicando sigma na conta do zeta

    zeta = (N_tr * T_post)**(1/4) * sigma

    return zeta


def solve_weights(X: np.ndarray, y: np.ndarray, penalty: float = 1e-6) -> np.ndarray:
    """
    O motor principal usando Otimização Convexa (CVXPY).
    Garante a convergência global sem falhas de 'linesearch'.

    Retorna:
        np.ndarray: Um vetor contendo os pesos (N_units) e o viés/interceção no último índice.
    """
    N_units = X.shape[1]

    # 1. Definir as Variáveis de Otimização no formato CVXPY
    # w são os pesos, w0 é o viés (interceção)
    w = cp.Variable(N_units)
    w0 = cp.Variable(1)

    # 2. Definir a Função de Custo (Objetivo)
    # Erro Quadrático: sum((X * w + w0 - y)^2)
    # Penalização Ridge: penalty * sum(w^2)
    # Utilizamos cp.sum_squares para respeitar as regras de programação convexa
    residuals = X @ w + w0 - y
    error = cp.sum_squares(residuals)
    ridge = penalty * cp.sum_squares(w)

    objective = cp.Minimize(error + ridge)

    # 3. Definir as Restrições (Constraints)
    # A beleza do CVXPY: podemos usar 0 absoluto sem colapsar o otimizador
    constraints = [
        w >= 0,               # Pesos não negativos
        cp.sum(w) == 1.0      # A soma dos pesos tem de ser exatamente 1
    ]

    # 4. Criar o Problema e Resolver
    prob = cp.Problem(objective, constraints)

    try:
        # O OSQP é o solver padrão para programação quadrática, extremamente rápido e fiável.
        # Pode ajustar verbose=False para não encher o ecrã com logs de otimização.
        prob.solve(solver=cp.OSQP, verbose=False)

        # Mecanismo de fallback: se o OSQP achar a matriz difícil, usamos o ECOS.
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            prob.solve(solver=cp.ECOS, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(
                f"Solver have failed to find optimal solution: {prob.status}")

    except Exception as e:
        raise RuntimeError(f"Critical Error with CVXPY: {e}")

    # 5. Extrair e formatar os resultados
    # value extrai o array numpy resultante do otimizador
    winningWeights = w.value
    winningBias = w0.value[0]  # cvxpy retorna um array de dimensão 1 para w0

    # Retorna exatamente no formato que o seu código antigo esperava: [pesos..., viés]
    resultado_final = np.concatenate([winningWeights, [winningBias]])

    return resultado_final


def estimate_omega_weights(Y_pre_co: np.ndarray, Y_pre_tr: np.ndarray, zeta: float) -> tuple:
    """
    Gera os pesos das unidades (controles sintéticos). Usa penalização zeta.
    """
    # Calculando a penalidade com zeta

    nTimes = Y_pre_co.shape[0]  # quantidade de períodos
    penalty = (zeta ** 2) * nTimes  # penalti para função

    # chamando solver

    omegaWeights = solve_weights(Y_pre_co, Y_pre_tr, penalty)

    return omegaWeights[:-1], omegaWeights[-1]


def estimate_lambda_weights(Y_pre_co: np.ndarray, Y_post_co: np.ndarray) -> tuple:
    """
    Gera os pesos do tempo. Sem penalização, mas exige a transposição da matriz.
    """

    # 1. Tira a média no tempo para criar o vetor alvo (tamanho: N_co)
    alvo_y = np.mean(Y_post_co, axis=0)

    # 2. Chama o motor.
    # X é a matriz pré-tratamento transposta (.T)
    # y é a média que acabámos de calcular
    # penalty é ZERO para os pesos de tempo, segundo o paper matemático.
    lambdaWeights = solve_weights(X=Y_pre_co.T, y=alvo_y, penalty=0)

    return lambdaWeights[:-1], lambdaWeights[-1]

# CLASSE PARA PROBLEMAS GERAIS
# Classe para validar o Erro de otimização


class OptimizationFailed(Exception):
    """Raised when the weight optimization fail."""
    pass
