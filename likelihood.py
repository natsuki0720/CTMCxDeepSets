import numpy as np
from scipy.optimize import minimize, differential_evolution
from formate_matrix_toMLData import *
from data_generator import * 
from skopt import gp_minimize
from scipy.linalg import expm

class LikeLihood:
    def __init__(self, data):
        self.data = data
        
    def vec_to_UpperTriMatrix(self,vector):
        count = 0
        i = 1
        while count < len(vector):
            count += i
            i += 1
        if(count != len(vector)):
            raise IndexError("vec length is incorrect!")
        matrix = np.zeros((i,i))
        iterator =iter(vector)
        for j in range(0,i):
            for k in range(j+1, i):
                matrix[j][k] = max(0,next(iterator))
        for j in range(0,j):
            sum = np.sum(matrix[j,:])
            matrix[j][j] = -sum
        return matrix
    
    # L-BFGS-Bの実装
    def optimize_byL_BFGS_B(self, vec=None, bounds  =[(0,None)for _ in range(6)]):
        if vec == None:
            vec = np.random.uniform(0,1,6)

        result = minimize(self.f, vec, method = "L-BFGS-B", bounds= bounds,options={'gtol': 1e-5, 'maxiter': 1000, 'disp': False})
        return result
        
    #差分進化法
    def optimize_by_DifferentialEvolution(self,bounds =[(0,1)for _ in range(6)]):
        result = differential_evolution(self.f, bounds = bounds, maxiter = 1000, tol = 1e-6)
        return result
    
    # マルチスタート戦略（計算量やばそう）
    def multi_start_optimization(self, bounds=[(0,None)for _ in range(6)], num_start = 10, method = "L-BFGS-B"): #できればマジックナンバー解消 
        best_result = None
        all_result = []
        for i in range(num_start):
            initial_guess = np.random.uniform(0, 1, size=6) #できればsizeは一般化
            result = result = minimize(self.f, initial_guess, method=method,
                          bounds=bounds,
                          options={'gtol': 1e-5, 'maxiter': 1000, 'disp': False})
            
            if best_result is None or result.fun < best_result.fun:
                best_result = result
        return best_result


    # ベイズ最適化の実行メソッド
    def bayesian_optimization(self, bounds = [(0,1) for _ in range(6)]):
        def objective(vec):
            value = self.f(vec)
            print("Evaluating at", vec, "->", value)  # 各評価点の結果を表示
            return value
        result = gp_minimize(self.f,bounds, n_calls=50, random_state=0)
        return result
    
    #尤度関数の定義
    def f(self,x):
        """_summary_

        Args:
            x : vector
        """
        sum = 0
        Q = self.vec_to_UpperTriMatrix(x)
        for sample in(self.data):
            pm = CalcProbmatrix(Q,sample[2]).calcProbmatrix()
            sum -= np.log(pm[int(sample[0])-1][int(sample[1])-1]+1e-6)
        return sum
    
class Likelihood_for_diagonal:
    def __init__(self, data):
        self.data = data
        
    def vec_to_TRMatrix(self,vector):
        num_elements = len(vector)
        matrix = np.zeros((num_elements + 1, num_elements + 1))
        for i in range(num_elements):
            matrix[i][i] = -vector[i]
            matrix[i][i+1] = vector[i]
        return matrix
    
    # L-BFGS-Bの実装
    def optimize_byL_BFGS_B(self, vec=None, bounds  =[(0,None)for _ in range(3)]):
        if vec == None:
            vec = np.random.uniform(0,1,3)

        result = minimize(self.f, vec, method = "L-BFGS-B", bounds= bounds,options={'gtol': 1e-5, 'maxiter': 1000, 'disp': False})
        return result
    
    # マルチスタート戦略（計算量やばそう）
    def multi_start_optimization(self, bounds=[(0,None)for _ in range(3)], num_start = 10, method = "L-BFGS-B"): #できればマジックナンバー解消 
        best_result = None
        all_result = []
        for i in range(num_start):
            initial_guess = np.random.uniform(0, 1, size=3) #できればsizeは一般化
            result = result = minimize(self.f, initial_guess, method=method,
                          bounds=bounds,
                          options={'gtol': 1e-5, 'maxiter': 1000, 'disp': False})
            
            if best_result is None or result.fun < best_result.fun:
                best_result = result
        return best_result
    
    #差分進化法
    def optimize_by_DifferentialEvolution(self,bounds =[(0,1)for _ in range(3)]):
        result = differential_evolution(self.f, bounds = bounds, maxiter = 1000, tol = 1e-6)
        return result
    
    def f(self, x):
        sum = 0
        Q = self.vec_to_TRMatrix(x)
        for sample in(self.data):
            pm = CalcProbmatrix(Q,sample[2]).calcProbmatrix()
            sum -= np.log(pm[int(sample[0])-1][int(sample[1])-1]+1e-6)
        return sum
    
class Likelihood_diagonal_exp:
    def __init__(self, data, num_state=4):
        self.data = data
        self.num_state = num_state
    
    def generate_Q_from_r(self, r_vec):
        Q = np.zeros((self.num_state, self.num_state))
        for i in range(len(r_vec)):
            Q[i,i+1]= np.exp(r_vec[i])
            Q[i,i] = -Q[i,i+1]
        return Q
    
    
    def log_likelihood(self, r_vec):
        likelihood = 0
        Q = self.generate_Q_from_r(r_vec)
        for sample in self.data:
            prob_m = expm(Q * sample[2])
            likelihood += np.log(prob_m[int(sample[0] - 1)][int(sample[1])-1] +1e-12)
        return -likelihood

    def optimize(self, vec):
        result = minimize(self.log_likelihood, vec, method="BFGS",options={'gtol': 1e-5, 'maxiter': 1000, 'disp': False})
        return self.generate_Q_from_r(result.x)
    
    def optimize_with_hessian(self, vec):
        result = minimize(self.log_likelihood, vec, method = "trust-krylov",options={'gtol': 1e-5, 'maxiter': 1000, 'disp': False})
        return self.generate_Q_from_r(result.x)
                