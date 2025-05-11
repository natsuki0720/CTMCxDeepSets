import numpy as np
import os

# データのmatrixから所定の行列をトリミング
class matrix_trimer:
    def __init__(self, matrix):
        self.matrix = np.array(matrix)
        self.num_state = self.matrix.shape[1]
        
    # 推移率行列を取得
    def trim_transitionRateMatrix(self):
        return self.matrix[:self.num_state, :self.num_state]
    # データを取得
    def trim_data(self):
        temp_M = self.matrix[self.num_state:]
        data = temp_M[:,:3]
        return data

class formate_dataMatrix:
    def GetOutputVector_byUpperTriangle(self,matrix):
        m = np.array(matrix)
        num_states = m.shape[1]
        vec = np.array([])
        for i in range(0,num_states-1):
            for j in range(i+1, num_states):
                element = m[i][j]
                vec = np.append(vec,element)
        return vec

    def GetOutputVector_byDiagonal(self,matrix):
        m = np.array(matrix)
        num_states = m.shape[1]
        vec = np.array([])
        for i in range(0, num_states-1):
            element = m[i][i+1]
            vec = np.append(vec,element)
        return vec

    def process_all_files_in_directory(self,directory, func, start = 0, end = None):
         # ファイル一覧をソートして取得
        all_files = sorted(os.listdir(directory))
        
        # ファイルの範囲を制限
        selected_files = all_files[start:end]
        
        for filename in selected_files:
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                func(file_path)


        
    