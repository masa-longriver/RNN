import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.config_model = config['model']
        self.input_dim  = self.config_model['input_dim']
        self.hidden_dim = self.config_model['hidden_dim']
        self.output_dim = self.config_model['output_dim']

        self.input_layer  = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layer = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

        self.act    = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.config_model['dropout_rate'])
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 隠れ状態の初期化
        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        # 各タイムステップでの計算
        for t in range(seq_len):
            h_new = self.act(self.input_layer(x[:, t, :]))   # 入力を隠れ状態に変換
            combined = torch.cat((h_new, h), dim=1)          # 隠れ状態と前の隠れ状態を結合
            combined = self.dropout(combined)
            h = self.act(self.hidden_layer(combined))        # 結合したものを隠れ層に入力
        
        out = self.output_layer(h)                           # 最後に出力層へ

        return out