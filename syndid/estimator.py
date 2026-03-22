import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# Aqui você importa as peças que você mesmo construiu!
from syndid.data_utils import validate_panel, pivot_to_matrix, slice_matrices
from syndid.optimize import compute_zeta, estimate_omega_weights, estimate_lambda_weights


class SyntheticDID:
    def __init__(self, data: pd.DataFrame, unit_col: str, time_col: str,
                 treated_unit: list, treatment_start_time, outcome_col: str, confidence: float = 0.05):
        """
        Salva os parâmetros passados pelo usuário.
        """
        self.data = data.copy()
        self.unit_col = unit_col
        self.time_col = time_col
        self.treated_unit = treated_unit
        self.treatment_start_time = treatment_start_time
        self.outcome_col = outcome_col
        self.confidence = confidence

        # Valida os paineis passados
        validate_panel(self.data, self.unit_col, self.time_col)  # validando

        # Pivotea a matrix para separar
        self.dfPivoted = pivot_to_matrix(
            self.data, self.unit_col, self.time_col, self.outcome_col)  # pivotando

        # Separa em diferente matrizes!
        self.Y_pre_co, self.Y_post_co, self.Y_pre_tr, self.Y_post_tr = slice_matrices(
            self.dfPivoted, self.treated_unit, self.treatment_start_time
        )  # quebrando matrix

        # Salvando quantidade de unidades tratadas e tempo de tratamento
        self.Ntreated = self.Y_pre_tr.shape[1]
        self.Ncontrols = self.Y_pre_co.shape[1]
        self.NtimesTreatment = self.Y_post_tr.shape[0]

        # Variáveis que serão preenchidas após o fit()
        self.omega_weights_ = None
        self.lambda_weights_ = None

        # Efeito de Tratamento Médio (Average Treatment Effect on the Treated)
        self.att_ = None

    def sub_compute_sdid_core(
        self,
        Y_pre_co_sub: np.ndarray, Y_post_co_sub: np.ndarray,
        Y_pre_tr_sub: np.ndarray, Y_post_tr_sub: np.ndarray
    ) -> tuple:
        """
        Função pura. Recebe matrizes fatiadas e devolve os pesos e o ATT.
        NÃO salva nada no 'self'.
        """

        # pegando valores passados
        Ntreated_sub = Y_pre_tr_sub.shape[1]
        NtimesTreatment_sub = Y_post_tr_sub.shape[0]

        # Organizando dados para caso de multiplas unidades tratadas
        if Y_pre_tr_sub.shape[1] > 1:  # Fazendo uma super unidade média para ATT
            Y_pre_tr_sub = np.mean(Y_pre_tr_sub, axis=1)
            Y_post_tr_sub = np.mean(Y_post_tr_sub, axis=1)

        else:  # Caso seja apenas 1 unidade, ele vai achatar
            Y_pre_tr_sub = Y_pre_tr_sub.flatten()
            Y_post_tr_sub = Y_post_tr_sub.flatten()

        # Roda os calculos de otimização
        zeta_sub = compute_zeta(
            Y_pre_co_sub, Ntreated_sub, NtimesTreatment_sub)  # computando ZETA
        omega_weights_sub = estimate_omega_weights(
            Y_pre_co_sub, Y_pre_tr_sub, zeta_sub)  # peso omega
        lambda_weights_sub = estimate_lambda_weights(
            Y_pre_co_sub, Y_post_co_sub)  # peso lambda

        # calculando deltas e DIFF in DIFF
        deltaTreat_sub = np.mean(Y_post_tr_sub, axis=0) \
            - lambda_weights_sub[0] @ Y_pre_tr_sub  # Calcula o salto dos tratados

        deltaControl_sub = omega_weights_sub[0] @ np.mean(Y_post_co_sub, axis=0) \
            - lambda_weights_sub[0] @ Y_pre_co_sub @ omega_weights_sub[0]  # calcula o controle sin e
        # tira o efeito do tempo

        # CALCULANDO O DIFF IN DIFF
        att_sub = deltaTreat_sub - deltaControl_sub

        return att_sub, omega_weights_sub, lambda_weights_sub

    def standard_error(self) -> float:
        """
        Calcula o Erro Padrão (SE) do estimador SDID usando o método Jackknife com pesos FIXOS.
        """
        if getattr(self, 'att_', None) is None:
            raise ValueError("The model was not fitted.")

        jackknife_atts = np.zeros(self.Ncontrols)

        # Resgata os pesos do modelo principal (FIXOS)
        # Atenção: pegue apenas os pesos, ignorando o viés se ele estiver no final do vetor
        omega_fixo = self.omega_weights_[0]
        lambda_fixo = self.lambda_weights_[0]

        # O salto dos tratados nunca muda no Jackknife dos controles
        delta_treat = np.mean(self.Y_post_tr, axis=0) - \
            lambda_fixo @ self.Y_pre_tr

        for j in range(self.Ncontrols):
            # 1. Deleta a unidade de controle j das matrizes
            Y_pre_co_j = np.delete(self.Y_pre_co, j, axis=1)
            Y_post_co_j = np.delete(self.Y_post_co, j, axis=1)

            # 2. Deleta o peso omega correspondente à unidade j
            omega_j = np.delete(omega_fixo, j)

            # 3. Re-normaliza os pesos restantes para que a soma volte a ser exatamente 1.0
            # (Se você não fizer isso, a média ponderada fica errada)
            # soma_pesos = np.sum(omega_j)
            # if soma_pesos > 0:
            #     omega_j = omega_j / soma_pesos

            # 4. Calcula o salto do controle sintético usando APENAS ÁLGEBRA LINEAR
            delta_control_j = omega_j @ np.mean(Y_post_co_j,
                                                axis=0) - lambda_fixo @ Y_pre_co_j @ omega_j

            # 5. Salva o ATT do Jackknife
            jackknife_atts[j] = np.mean(delta_treat) - delta_control_j

        # Média dos estimadores parciais
        tau_mean = np.mean(jackknife_atts)

        # Variância de Jackknife
        jack_var = ((self.Ncontrols - 1) / self.Ncontrols) * \
            np.sum((jackknife_atts - tau_mean)**2)

        self.se_ = np.sqrt(jack_var)

        return self.se_

    def placebo_p_value(self) -> float:
        """
        Calcula o P-valor através do Teste de Placebo no Espaço.
        Baseado na Seção 5 de Arkhangelsky et al. (2021).
        Permuta iterativamente cada unidade de controle assumindo que é a tratada,
        roda o modelo completo e extrai o ATT placebo.
        """
        if getattr(self, 'att_', None) is None:
            raise ValueError("The model was not fitted.")

        placebo_atts = np.zeros(self.Ncontrols)

        for j in range(self.Ncontrols):
            # 1. Isola a coluna j (unidade de controle) para ser a falsa tratada
            # Usamos o slice [:, j:j+1] para manter a dimensão como matriz 2D (T_pre, 1)
            Y_pre_tr_placebo = self.Y_pre_co[:, j:j+1]
            Y_post_tr_placebo = self.Y_post_co[:, j:j+1]

            # 2. O novo grupo de controle doador são as unidades restantes
            Y_pre_co_placebo = np.delete(self.Y_pre_co, j, axis=1)
            Y_post_co_placebo = np.delete(self.Y_post_co, j, axis=1)

            # 3. Roda o motor completo para recalcular pesos e o ATT do Placebo
            try:
                att_placebo, _, _ = self.sub_compute_sdid_core(
                    Y_pre_co_placebo, Y_post_co_placebo,
                    Y_pre_tr_placebo, Y_post_tr_placebo
                )
                placebo_atts[j] = att_placebo

            except Exception as e:
                print(
                    f"Optimization Failed for Placebo {j}. Error: {e}")
                placebo_atts[j] = np.nan

        # 4. Cálculo do P-Valor
        # Filtra os NaNs de placebos onde a otimização falhou
        placebo_atts_validos = placebo_atts[~np.isnan(placebo_atts)]

        # Teste Bicaudal (analisa a magnitude do choque em módulo)
        efeito_real_abs = np.abs(self.att_)
        efeitos_placebo_abs = np.abs(placebo_atts_validos)

        # P-valor = proporção de placebos que geraram um efeito maior ou igual ao real
        p_valor = np.sum(efeitos_placebo_abs >=
                         efeito_real_abs) / len(placebo_atts_validos)

        # Salvando atributos para plotagem futura (Histograma)
        self.placebo_atts_ = placebo_atts
        self.p_value_ = p_valor

        print(f"\n--- Placebo Results ---")
        print(f"Original ATT: {self.att_:.4f}")
        print(
            f"Valid Placebos: {len(placebo_atts_validos)}/{self.Ncontrols}")
        print(f"P-Valor: {p_valor:.4f}")

        return None

    def fit(self):
        """
        Orquestra toda a lógica: prepara os dados, acha os pesos e calcula o efeito.
        """

        # Rodando subrotina para calcular os valores estimados para o dataset fornecido
        # Cálculo do Efeito (Matemática Final do SDID)

        self.att_, self.omega_weights_, self.lambda_weights_ = self.sub_compute_sdid_core(
            self.Y_pre_co,
            self.Y_post_co,
            self.Y_pre_tr,
            self.Y_post_tr
        )

        if self.Ntreated > 1:
            # Calculo do intervalo de confiança através do método Jackknive
            self.standard_error()

            # Extrai o Z-score exato do Scipy para nível de confiança
            z_score = st.norm.ppf(1 - self.confidence / 2)

            # Calcula P valor
            z_stat = self.att_ / self.se_
            self.p_value_ = 2 * (1 - st.norm.cdf(np.abs(z_stat)))

            # 3. Calcula os limites inferior e superior
            self.upper_limit = self.att_ - (z_score * self.se_)
            self.lower_limit = self.att_ + (z_score * self.se_)

        else:
            # Calculo do intervalo de confiança através do método Jackknive
            self.placebo_p_value()

            # Extrai o Z-score exato do Scipy para nível de confiança
            z_score = st.norm.ppf(1 - self.confidence / 2)

            # 1. Calcule o desvio padrão dos efeitos placebo (Esse é o seu Erro Padrão real)
            # Vai dar algo em torno de 9.8
            self.se_ = np.std(self.placebo_atts_[
                ~np.isnan(self.placebo_atts_)], ddof=1)

            # 2. Calcule os limites (usando Z = 1.96 para 95% de confiança)
            self.lower_limit = self.att_ - (z_score * self.se_)
            self.upper_limit = self.att_ + (z_score * self.se_)

        return None

    def summary(self) -> dict:
        """
        Devolve os resultados formatados para o usuário.
        """
        # 1. Verifica se o modelo já foi ajustado (fit)
        if getattr(self, 'att_', None) is None:
            raise ValueError(
                "Model not fitted or no result founded."
            )

        # 2. Constrói um dicionário bonitinho e estruturado
        results = {
            "causal_effect": {
                "att": self.att_,
                "se": self.se_,
                "upper_limit": self.upper_limit,
                "lower_limit": self.lower_limit,
                "p_value": self.p_value_,
            },
            "weights": {
                "unit_weights": self.omega_weights_[0],
                "time_weights": self.lambda_weights_[0]
            },
            "intercept_base": {
                "unit_weights_intercept": self.omega_weights_[1],
                "time_weights_intercept": self.lambda_weights_[1]
            }
        }

        # Extraindo as variáveis do dicionário
        att = results.get("causal_effect").get('att', 0.0)
        se = results.get("causal_effect").get('se', 0.0)
        p_val = results.get("causal_effect").get('p_value', 0.0)
        ci_lower = results.get("causal_effect").get('lower_limit', 0.0)
        ci_upper = results.get("causal_effect").get('upper_limit', 0.0)

        n_tr = self.Ntreated
        n_co = self.Ncontrols
        n_obs = n_tr + n_co

        # Calculando a estatística Z (se o Erro Padrão for maior que zero)
        z_stat = att / se if se > 0 else np.nan

        # Configurando as larguras da tabela
        width = 78

        # ================= CABEÇALHO =================
        print("=" * width)
        print(f"{'Synthetic Difference-in-Differences Results':^{width}}")
        print("=" * width)

        # Informações do modelo (dividido em duas colunas)
        print(f"{'Dep. Variable:':<20} {'Y':<20} {'No. Observations:':<20} {n_obs:>15}")
        print(f"{'Estimator:':<20} {'SDID':<20} {'No. Treated:':<20} {n_tr:>15}")
        print(
            f"{'Inference:':<20} {'Placebo Test':<20} {'No. Controls:':<20} {n_co:>15}")
        print("-" * width)

        # ================= CORPO DA TABELA =================
        # Cabeçalho das métricas
        header = f"{'':<12} {'coef':>10} {'std err':>10} {'z':>10} {'P>|z|':>10} {'[0.025':>10} {'0.975]':>10}"
        print(header)
        print("-" * width)

        # Linha do ATT
        row = (f"{'ATT':<12} "
               f"{att:>10.4f} "
               f"{se:>10.4f} "
               f"{z_stat:>10.3f} "
               f"{p_val:>10.3f} "
               f"{ci_lower:>10.4f} "
               f"{ci_upper:>10.4f}")

        print(row)

        # ================= RODAPÉ =================
        print("=" * width)

        return None

    def plot_placebos(self):
        """
        Plota a distribuição dos efeitos placebo e a linha do ATT real.
        Gera a visualização clássica de inferência por permutação.
        """
        if getattr(self, 'att_', None) is None:
            raise ValueError(
                "Model not fitted or no result founded."
            )

        # Filtrar possíveis NaNs (caso tenha sobrado algum no array)
        placebos = self.placebo_atts_[~np.isnan(self.placebo_atts_)]

        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-whitegrid')  # Estilo limpo e acadêmico

        # Histograma dos placebos (a "nuvem" de ruído natural)
        plt.hist(placebos, bins=15, color='gray', alpha=0.7, edgecolor='black',
                 label='Placebo Effect (Controls)')

        # A linha vermelha que representa o que aconteceu de fato na Califórnia
        plt.axvline(x=self.att_, color='red', linestyle='--', linewidth=2.5,
                    label=f'ATT: {self.att_:.2f}')

        plt.title('Distribution of Placebo Tests',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Effect Size Estimation', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(fontsize=11)

        plt.tight_layout()
        plt.show()

    def plot_trends(self, times=None):
        """
        Plota a evolução temporal da unidade tratada vs. o controle sintético ponderado.

        anos: Lista ou array opcional com os rótulos do eixo X (ex: range(1970, 2001)).
            Se None, usará o índice numérico dos períodos.
        """
        if getattr(self, 'att_', None) is None:
            raise ValueError(
                "Model not fitted or no result founded."
            )

        # 1. Reconstruir a série do Tratado e do Controle (Pré + Pós)
        # Assumindo que você tem essas matrizes no formato (T, N)
        Y_tr_full = np.vstack([self.Y_pre_tr, self.Y_post_tr])
        Y_co_full = np.vstack([self.Y_pre_co, self.Y_post_co])

        if self.Ntreated > 1:  # Considerando casos de multiplos tratados
            Y_tr_full = Y_tr_full.mean(axis=1).flatten()

        else:
            Y_tr_full = Y_tr_full.flatten()

        # 2. Aplicar os pesos Ômega aos controles para criar o "Sintético"
        # Lembre-se que solve_weights retorna [pesos..., viés]. Isolamos só os pesos.
        omega = self.omega_weights_[0]

        # Controle Sintético = Produto escalar da matriz de controle pelos pesos
        Y_synthetic = Y_co_full @ omega

        # 3. Definir o eixo do tempo
        T_total = len(Y_tr_full)
        T_pre = len(self.Y_pre_tr)

        if times is None:
            times = np.arange(1, T_total + 1)

        times_post = times[T_pre:]

        # 4. Plotagem
        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-whitegrid')

        # Linha da unidade Tratada (Real)
        plt.plot(times, Y_tr_full, color='black',
                 linewidth=2.5, label='Treated')

        # Linha do Controle Sintético
        plt.plot(times, Y_synthetic, color='blue', linestyle='--', linewidth=2.5,
                 label='Control (SDID)')

        # Linha vertical marcando a intervenção
        x_intervencao = times[T_pre]  # Último ano antes do tratamento
        plt.axvline(x=x_intervencao, color='red', linestyle=':', linewidth=2,
                    label='Intervention')

        if True:
            # Função auxiliar para calcular e plotar a reta (y = ax + b)
            def plot_reta(x, y, cor, estilo, label_nome):
                z = np.polyfit(x, y, 1)  # Regressão linear de grau 1
                p = np.poly1d(z)
                plt.plot(x, p(x), color=cor, linestyle=estilo,
                         linewidth=3, label=label_nome, alpha=0.45)

            # Retas no período PÓS-Tratamento (Aqui você vê o distanciamento/ATT)
            plot_reta(times_post, self.Y_post_tr.flatten()-self.omega_weights_[1], 'black',
                      '-.', 'Trends Post (Treated)')
            plot_reta(times_post, Y_synthetic[T_pre:].flatten(), 'blue',
                      '-.', 'Trends Post (Counterfactual)')

            # Preencher a diferença entre as tendências retas no PÓS (O ATT visual)
            z_tr = np.poly1d(np.polyfit(
                times_post, self.Y_post_tr.flatten() - self.omega_weights_[1], 1))
            z_sy = np.poly1d(np.polyfit(
                times_post, Y_synthetic[T_pre:].flatten(), 1))
            plt.fill_between(times_post, z_tr(times_post), z_sy(
                times_post), color='gray', alpha=0.3, label='Difference (ATT)')

        plt.title('Treated vs. Synthetic Control',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Var. Y', fontsize=12)
        plt.legend(fontsize=11)

        plt.tight_layout()
        plt.show()
