import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
#ランダムに推移率行列を生成するクラス(上三角行列)
class TransitionRateMatrixGenerator:
    # コンストラクタ: 推移率行列の型を状態数により指定
    def __init__(self, states):
        self.num_states = states
        self.matrix = np.zeros((states, states))
        
    def generateMatrix(self, func):
        for i in range(self.num_states):
            func(i)  # 引数にインデックスを渡す
        return self.matrix
        
    # 推移率行列を一様分布により生成
    def setTransitionRateByUni(self, row_Index): 
        vector = np.zeros(self.num_states)
        for i in range(row_Index+1, self.num_states):
            vector[i] = np.random.uniform(0,1/self.num_states)
        sum = 0
        for j in range(0, len(vector)):
            sum += vector[j]
        vector[row_Index] = -sum
        self.matrix[row_Index] = vector
    
            
    # 対角成分をランダム生成し、それに合わせて各要素を決定
    def setTransitionRateFromDiagonal(self,row_Index):
        vector = np.zeros(self.num_states)
        for i in range(row_Index + 1, self.num_states):
            vector[i] = np.random.uniform()
        diag_val = 0
        if row_Index+1 != self.num_states:
            vector = self.__normalize(vector)
            diag_val = np.random.uniform(0,1)
             
        for i in range(row_Index + 1, self.num_states):
            vector[i] *= diag_val
        vector[row_Index] = -diag_val
        self.matrix[row_Index] = vector
    
    def __normalize(self, vector):
        sum = 0
        for i in vector:
            sum += i
        for i in range(0, len(vector)):
            vector[i] = vector[i]/sum
        return vector

class DiagonalTransitionRateMatrixGenerator:
    def __init__(self,states):
        self.num_states = states
        self.matrix = np.zeros((self.num_states, self.num_states))
    
    def setDiagonalElement(self, row_index):
        if row_index < self.num_states-1:
            element = np.random.uniform(0.01,1)
            self.matrix[row_index][row_index] = -element
            self.matrix[row_index][row_index+1] = element
    
    def setDiagonalElement_byLifespan(self, row_index, lifespan):
        if row_index < self.num_states - 1:
            element = 1/np.random.uniform(1,lifespan)
            self.matrix[row_index][row_index] = -element
            self.matrix[row_index][row_index + 1] = element
    
    def generateMatrix(self, func,lifespan = 100):
        for i in range(0, self.num_states):
            func(i, lifespan)  
        return self.matrix
    
    
# 推移率行列から指定期間での推移確率行列を生成
class CalcProbmatrix:
    def __init__(self,matrix, delta_time):
        self.transitionRate_matrix = matrix
        self.delta_time = delta_time
        self.dim = len(matrix)
    # 水谷先生の論文中のAの行列を求めるメソッド
    def __calc_A(self, index):
        matrix  = np.eye(self.dim)
        for j in range(0,self.dim):
            if(j != index):
                mulMatrix = (self.transitionRate_matrix - self.transitionRate_matrix[j][j] * np.eye(self.dim))/(self.transitionRate_matrix[index][index] - self.transitionRate_matrix[j][j])
                matrix =  matrix @ mulMatrix 
        return matrix
    #推移確率を生成
    def calcProbmatrix(self):
        preb = np.zeros((self.dim,self.dim))
        for i in range(0, self.dim):
            A = self.__calc_A(i)
            mat = np.exp(self.delta_time * self.transitionRate_matrix[i][i]) * A
            preb += mat
        return preb


        

class DataGenerator:
    def __init__(self, matrix, data_size):
        self.TR_M = matrix
        self.data_size = data_size
        self.num_states = len(matrix)
    
    def set_initialState_ratio(self):
        alpha = [1]*3
        ratio = np.random.dirichlet(alpha)
        return ratio
    
    # とりあえず一様分布
    def set_initialState(self, ratio):
        p = np.random.rand()
        initial_state = None
        sum_ratio = 0
        for idx,r in enumerate(ratio):
            sum_ratio += r
            if p <= sum_ratio:
                initial_state = idx+1
                break
        if initial_state == None:
            initial_state = len(ratio)
        return initial_state
        
    # とりあえず一様分布
    def set_deltaTime(self, max):
        delta_time = np.random.uniform(0,max)
        return delta_time
    
    def set_deltaTime_log_normal(self):
        delta_time = np.random.lognormal(1, 0.5) 
        return delta_time
    
    def generate_sample(self, init_state, delta_t):
        probM = CalcProbmatrix(self.TR_M, delta_t).calcProbmatrix()
        probV = probM[init_state - 1]
        rand_int = np.random.rand()
        next_state = 0
        cumulative = 0
        for i in range(0, len(probV)):
            cumulative += probV[i]
            if(rand_int < cumulative):
                next_state = i+1
                break
        sample = np.array([init_state, next_state, delta_t])
        return sample
    
    def generate_dataFile(self, matrix,file_name,path):
        
        name = file_name +"_"+ str(self.data_size)+"_" + str(self.num_states)+".csv"
        # nested_directory = os.path.join("data","diagonal","nonFix")
        path = self.__setPath(name, path)
        matrix = np.array(matrix)
        np.savetxt(path, matrix,delimiter=",",fmt="%s",encoding='utf-8')
        
    def generate_matrix(self,func):
        data_matrix = []
        ratio = self.set_initialState_ratio()
        for _ in range(self.data_size):
            init_state = self.set_initialState(ratio)
            delta_t = round(func(),1) #一旦これ
            sampleVector = self.generate_sample(init_state, delta_t)
            data_matrix.append(sampleVector)
        data_matrix = np.array(data_matrix)
        data_matrix = self.__pad_matrix(data_matrix)
        matrix = np.vstack((self.TR_M,data_matrix))
        
        return matrix
    
        
    def generate_dataFile_fixTime(self, file_name, delta_t):
        data_matrix = []
        for i in range(0, self.data_size):
            init_state = self.set_initialState()
            sampleVector = self.generate_sample(init_state, delta_t)
            data_matrix.append(sampleVector)
        name = "fix"+ str(delta_t) +"_" + file_name + str(self.data_size)+"_" + str(self.num_states)+".csv"
        nested_directory = os.path.join("data","diagonal","fix"+str(delta_t))
        
        path = self.__setPath(name, nested_directory)
        
        data_matrix = np.array(data_matrix)
        data_matrix = self.__pad_matrix(data_matrix)
        matrix = np.vstack((self.TR_M,data_matrix))
        np.savetxt(path, matrix,delimiter=",",fmt="%s")
    
    def __setPath(self,file_name, directory = "data"):
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, file_name)
        return file_path
        
    def __pad_matrix(self, matrix):
        matrix = np.array(matrix)
        matrix_cols = matrix.shape[1]
        
        if matrix_cols < self.num_states:
            pad_cols = self.num_states - matrix_cols
            padding = np.zeros((matrix.shape[0],pad_cols))
            matrix = np.hstack((matrix,padding))
        return matrix
    

            
        
