import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Attention(nn.Module):
    def __init__(self, query_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        # Query = (B, T, Q)
        # Keys = (B, T, K)
        # Values = (B, T, V)
        # Outputs = lin_comb:(B, T, V)

        # Here we assume Q == K (dot product attention)
        keys = keys.transpose(1, 2)  # (B, T, K) -> (B, K, T)
        energy = torch.bmm(query, keys)  # (B, T, Q) x (B, K, T) -> (B, T, T)
        energy = F.softmax(energy.mul_(self.scale), dim=2)  # scale, normalize
        linear_combination = torch.bmm(energy, values)  # (B, T, T) x (B, T, V) -> (B, T, V)
        return linear_combination


class FusionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FusionModule, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)
        input = input.permute(0, 2, 3, 1).contiguous()
        output = self.linear(input)
        output = output.permute(0, 3, 1, 2).contiguous()
        return self.relu(output)


class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=224):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = self.calculate_positional_encodings(d_model, max_seq_len)
        self.register_buffer('pe', pe)

    def calculate_positional_encodings(self, d_model, max_seq_len):
        pe = torch.zeros(1, max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(2)
        pes = self.pe[:, :seq_len, :]
        pes = pes.permute(0, 2, 1).contiguous()
        pes = pes.repeat(batch_size, 1, 1)
        pes = pes.cuda() # comment this line if you are not using GPU
        x = x + pes
        return self.dropout(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, bias=True, attenion_size=224*224):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                              out_channels=4 * self.hidden_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.cross_attention = Attention(query_dim=attenion_size)
        self.fusion = FusionModule(input_dim=hidden_channels + hidden_channels, hidden_dim=hidden_channels)
        self.position_encoding = PositionEncoding(d_model=hidden_channels, max_seq_len=attenion_size)

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        # return (Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])),
        #         Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1]))) # comment this line if you are not using GPU
        return (Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda())

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # step 1: cross attention
        query, key, value = self.get_qkv(input_tensor, cur_state)
        query, key, value = query.view(query.size(0), -1, query.size(2) * query.size(3)), \
                            key.view(key.size(0), -1, key.size(2) * key.size(3)), \
                            value.view(value.size(0), -1, value.size(2) * value.size(3))
        cross_attention = self.cross_attention(query, key, value)
        cross_attention = cross_attention.view(cross_attention.size(0), cross_attention.size(1), input_tensor.size(2), input_tensor.size(3))
        # step 2: feature fusion
        fusion = self.fusion(h_cur, cross_attention)
        # step 3: position encoding
        fusion = fusion.view(fusion.size(0), fusion.size(1), -1)
        position_encoding = self.position_encoding(fusion)
        position_encoding = position_encoding.view(position_encoding.size(0), position_encoding.size(1), input_tensor.size(2), input_tensor.size(3))
        position_encoding = torch.sigmoid(position_encoding)
        # step 4: calculate gates
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        # step 5: calculate next state
        c_next = f * c_cur + position_encoding
        h_next = o * torch.tanh(c_next) + i * g + position_encoding
        return h_next, c_next

    def get_qkv(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        query, key, value = input_tensor, h_cur, input_tensor
        return query, key, value


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True, attenion_size=224*224):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.num_layers = len(hidden_channels)
        self.kernel_size = kernel_size
        self.bias = bias
        self.all_layers = []
        self.attenion_size = attenion_size
        for layer in range(self.num_layers):
            name = 'cell{}'.format(layer)
            cell = ConvLSTMCell(self.input_channels[layer], self.hidden_channels[layer], self.kernel_size, self.bias, self.attenion_size)
            setattr(self, name, cell)
            self.all_layers.append(cell)

    def forward(self, input_tensor):
        steps = input_tensor.size(1)
        batch_size = input_tensor.size(0)
        h, c = ConvLSTMCell.init_hidden(batch_size, self.hidden_channels[0], [input_tensor.size(3), input_tensor.size(4)])
        for layer in range(self.num_layers):
            output_inner = []
            for step in range(steps):
                h, c = getattr(self, 'cell{}'.format(layer))(input_tensor[:, step, :, :, :], (h, c))
                output_inner.append(h)
            input_tensor = torch.stack(output_inner, dim=1)
        # return input_tensor
        return input_tensor[:, -1, :, :, :] # comment this line if you need the whole sequence


class LSTMLayer(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True, num_classes=4, attenion_size=224*224):
        super(LSTMLayer, self).__init__()
        self.forward_net = ConvLSTM(input_channels, hidden_channels, kernel_size, bias, attenion_size)
        self.reverse_net = ConvLSTM(input_channels, hidden_channels, kernel_size, bias, attenion_size)
        self.conv = nn.Conv2d(2 * hidden_channels[-1], num_classes, kernel_size=1)

    def forward(self, x1, x2, x3):
        x1 = torch.unsqueeze(x1, dim=1)
        x2 = torch.unsqueeze(x2, dim=1)
        x3 = torch.unsqueeze(x3, dim=1)

        xforward = torch.cat((x1, x2), dim=1)
        xreverse = torch.cat((x3, x2), dim=1)

        yforward = self.forward_net(xforward)
        yreverse = self.reverse_net(xreverse)

        ycat = torch.cat((yforward, yreverse), dim=1)
        y = self.conv(ycat)
        return y


class LSTMSA(nn.Module):
    def __init__(self, input_channels=64, hidden_channels=[64], kernel_size=5, bias=True, num_classes=4, attenion_size=224*224, encoder=None):
        super(LSTMSA, self).__init__()
        self.encoder = encoder
        self.lstmlayer = LSTMLayer(input_channels, hidden_channels, kernel_size, bias, num_classes, attenion_size)

    def forward(self, x1, x2, x3):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x3 = self.encoder(x3)
        y = self.lstmlayer(x1, x2, x3)
        return y
