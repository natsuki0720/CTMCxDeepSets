import numpy as np
import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class varSets_Datasets(Dataset):
    def __init__(self, states, del_t, outputs, transform = None):
        self.states = states # リスト: 各要素は (2, num_samples_i)
        self.del_t = del_t # リスト: 各要素は (num_samples_i,)
        self.outputs = outputs # リスト: 各要素は (3,)
        self.transform = transform
        self.num_sumples = len(states)
    
    def __len__(self):
        return  self.num_sumples
    
    def __getitem__(self, idx):
        state = self.states[idx]
        delta_t = self.del_t[idx]
        target = self.outputs[idx]

        if self.transform:
            state, delta_t = self.transform(state, delta_t)
        
        state = torch.tensor(state, dtype=torch.long) # (2, num_samples_i)
        delta_t = torch.tensor(delta_t, dtype=torch.float32) # (num_samples_i,)
        target = torch.tensor(target, dtype=torch.float32) # (3,)

        length = torch.tensor(state.shape[1])  # ここ追加 (num_samples_i)
        
        return state, delta_t, target, length

# あとでやる
def collate_fn(batch):
    state_batch = [item[0] for item in batch]     # 各 state は (2, num_samples_i)
    delta_t_batch = [item[1] for item in batch]   # 各 delta_t は (num_samples_i,)
    target_batch = torch.stack([item[2] for item in batch])  # (batch_size, 3)
    
    # 各サンプルの長さを取得
    lengths = lengths = torch.tensor([s.shape[1] if len(s.shape) > 1 else 0 for s in state_batch]) # (batch_size,)
    
    # 最大長を取得
    max_length = lengths.max() if len(lengths) > 0 else 0
    
    # state のパディング
    state_padded = []
    for s in state_batch:
        if len(s.shape) < 2:  # shape[1]がない場合
            s = s.unsqueeze(0)  # 必要なら2次元に変換
        
        pad_size = max_length - s.shape[1]
        
        # パディング
        s_padded = torch.nn.functional.pad(s, (0, pad_size), mode='constant', value=0)
        state_padded.append(s_padded)
    state_padded = torch.stack(state_padded)  # (batch_size, 2, max_length)
    
    # delta_t のパディング
    delta_t_padded = []
    for dt in delta_t_batch:
        pad_size = max_length - dt.shape[0]
        dt_padded = torch.nn.functional.pad(dt, (0, pad_size), mode='constant', value=0)
        delta_t_padded.append(dt_padded)
    delta_t_padded = torch.stack(delta_t_padded)  # (batch_size, max_length)
    
    return state_padded, delta_t_padded, target_batch, lengths

class DeepSets_varSets_forDiagnel(nn.Module):
    def __init__(self, num_categories = 4, embedding_dim = 2):
        super(DeepSets_varSets_forDiagnel, self).__init__()
        #埋め込み層
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        
        #各サンプルの特徴抽出層
        self.fc1 = nn.Linear(embedding_dim * 2 + 1, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(p=0.1)  # ★低めのドロップアウト率で特徴の欠損を抑える
        
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(p=0.3)  # ★中程度のドロップアウト率で過学習を防ぐ

        # 集約後の層 (高次特徴を扱うため、軽めのドロップアウト率)
        self.fc3 = nn.Linear(16, 64)
        self.dropout3 = nn.Dropout(p=0.3)  # ★中間層での汎化性能向上

        self.fc4 = nn.Linear(64, 32)
        self.dropout4 = nn.Dropout(p=0.1)  # ★最終出力層に近いため控えめに
        self.fc5 = nn.Linear(32, 3)
    
    def forward(self, state, delta_t, lengths):
        """
        Args:
            state: Tensor of shape (batch_size, 2, max_length)
            delta_t: Tensor of shape (batch_size, max_length)
            lengths: Tensor of shape (batch_size,)
        Returns:
            output: Tensor of shape (batch_size, 3)
        """
        batch_size = state.size(0) 
        max_length = state.size(2) 
        
        #状態量の分離(pre, post)
        pre = state[:, 0, :] # (batch_size, max_length)
        post = state[:, 1, :] # (batch_size, max_length)
        
        #埋め込み
        pre_embedded = self.embedding(pre.long() -1) # (batch_size, max_length, embedding_dim)
        post_embedded = self.embedding(post.long() -1) # (batch_size, max_length, embedding_dim)
        
        #delta_tの整形
        delta_t = delta_t.unsqueeze(2) #(batch_size, max_length, 1)
        
        #特徴の結合
        x = torch.cat([pre_embedded, post_embedded, delta_t], dim=2)  #(batch_size, max_length, 2*embedding_dim +1)
        
        #バッチ内の全サンプルについて処理の準備（特徴抽出）
        x = x.view(-1, x.size(2)) #(batch_size * max_length, feature_size)
        
        #特徴抽出層
        x = F.relu(self.bn1(self.fc1(x))) #(batch_size * max_length, 64)
        x = self.dropout1(x)  # ★ドロップアウト適用
        x = self.dropout2(F.relu(self.bn2(self.fc2(x)))) # (batch_size * max_length, 32)
        
        # 元の形状に戻す
        x = x.view(batch_size, max_length, -1)  # (batch_size, max_length, 32)
        
        # マスクの作成
        mask = torch.arange(max_length).unsqueeze(0) < lengths.unsqueeze(1)  # (batch_size, max_length)
        mask = mask.unsqueeze(2).float()  # (batch_size, max_length, 1)
        
        
        # マスクを適用してパディング部分を無視
        x = x * mask  # パディング部分が0になる
        
        # 集約（マスクを考慮して平均化）
        x_sum = x.sum(dim=1)  # (batch_size, 32)
        lengths = lengths.unsqueeze(1).float()  # (batch_size, 1)
        x_mean = x_sum / lengths  # (batch_size, 32)
        
        # 出力層
        x = self.dropout3(F.relu(self.fc3(x_mean)))  # (batch_size, 64)
        x = self.dropout4(F.relu(self.fc4(x)))    # (batch_size, 32)
        output = F.softplus(self.fc5(x))# (batch_size, 3)
        # print(self.fc5(x))
        # print(output)
        return output

def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
        