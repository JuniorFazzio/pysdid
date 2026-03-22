import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_numeric_dtype
import numbers


def validate_panel(df: pd.DataFrame, unit_col: str, time_col: str) -> bool:
    """
    Verifica se o dataframe é um painel balanceado e adequado.
    """

    decision = np.array([])

    # verificando se há valores nulos:
    if df.isnull().sum().sum() > 0:
        raise ValueError("There is null values in data.")

    decision = np.concatenate([[True], decision])

    # verificando se cada unidade tem o mesmo número de períodos de tempo
    ntimes = df[time_col].nunique()

    if np.any(df.groupby(unit_col)[time_col].size() != ntimes):
        raise StructuralPanelError(
            "Different number of time periods for unit column.")

    decision = np.concatenate([[True], decision])

    # Verificar se a coluna de tempo é numérica ou datetime para facilitar na ordenação
    if not is_datetime(df[time_col]) and not is_numeric_dtype(df[time_col]):
        raise ValueError("Time period column not numeric or datetime.")

    decision = np.concatenate([[True], decision])

    # Levantar um ValueError se algo estiver errado, ou retornar True se estiver tudo ok
    # Como checamos que o painel tem números iguais de períodos de tempo para unidades
    # vamos retornar true

    return np.all(decision)


def pivot_to_matrix(df: pd.DataFrame, unit_col: str, time_col: str, outcome_col: str) -> pd.DataFrame:
    """
    Transforma o dado formato longo (tidy) para uma matriz larga (Tempo nas linhas, Unidades nas colunas).
    """
    # Pivoteando a matrix através do método de pivot table do próprio pandas
    pivotedMatrix = pd.pivot_table(df, outcome_col, time_col, unit_col)

    # Ordenando
    pivotedMatrix = pivotedMatrix.sort_index()

    return pivotedMatrix


def slice_matrices(df_pivot: pd.DataFrame, treated_unit: str, treatment_start_time) -> tuple:
    """
    Recebe a matriz pivotada e a corta nos 4 quadrantes clássicos do DiD.

    Retorna:
        Y_pre_co (np.ndarray): Controles antes do tratamento
        Y_post_co (np.ndarray): Controles depois do tratamento
        Y_pre_tr (np.ndarray): Tratado(s) antes do tratamento
        Y_post_tr (np.ndarray): Tratado(s) depois do tratamento
    """
    # Testando se o treatment_start_time é numérico ou datetime
    if not is_datetime(pd.to_datetime(treatment_start_time)) and not isinstance(treatment_start_time, numbers.Number):
        raise ValueError(
            "Treatment start time passed is not numeric or datetime.")

    # Vamos criar 3 matrizes para filtragem, a pivoteada dos valores
    # 3 mais duas, uma para tratamento e outra para estados tratados

    timePeriods = df_pivot.index
    treatedUnits = df_pivot.columns
    Ymatrix = df_pivot.values

    # ------------------------------ #
    # Criando matrizes binárias

    # Tempo
    matrixTreatement = np.zeros_like(Ymatrix)
    matrixTreatement[(timePeriods >= treatment_start_time), :] = 1

    # Unidades
    matrixTreatedUnits = np.zeros_like(Ymatrix)
    matrixTreatedUnits[:, np.isin(treatedUnits, treated_unit)] = 1

    # =========================================================
    # ÁLGEBRA DE MATRIZES PARA OBTER OS QUADRANTES
    # =========================================================

    # Podemos inverter as matrizes com operações algébricas simples:
    M_preTreatment = 1 - matrixTreatement        # 1 onde é pré-tratamento
    M_control = 1 - matrixTreatedUnits  # 1 onde é unidade de controlo

    # Agora extraímos os blocos exatos sem nos preocuparmos com a posição!
    # Lemos apenas as linhas e colunas onde as matrizes binárias são 1 (True)

    # Y_pre_co (Linhas onde M_pre=1 E Colunas onde M_controlo=1)
    Y_pre_co = Ymatrix[M_preTreatment[:, 0] == 1][:, M_control[0, :] == 1]

    # Y_post_co
    Y_post_co = Ymatrix[matrixTreatement[:, 0] == 1][:, M_control[0, :] == 1]

    # Y_pre_tr
    Y_pre_tr = Ymatrix[M_preTreatment[:, 0] ==
                       1][:, matrixTreatedUnits[0, :] == 1]

    # Y_post_tr
    Y_post_tr = Ymatrix[matrixTreatement[:, 0]
                        == 1][:, matrixTreatedUnits[0, :] == 1]

    return Y_pre_co, Y_post_co, Y_pre_tr, Y_post_tr


# Classe para validar o Erro de estrutura do painel
class StructuralPanelError(Exception):
    """Raised when the structure of panel dataset is not correct."""
    pass
