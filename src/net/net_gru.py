import torch.nn as nn
import torch
import torch.nn.functional as F


class NetGRU(nn.Module):
    def __init__(self):
        super(NetGRU, self).__init__()
        # GRU
        self.input_layer_L = nn.Linear(in_features=4, out_features=4)
        self.gru= nn.GRU(input_size=4,
                            hidden_size=256,  # hidden cells
                            num_layers=1,  # lstm layers
                            batch_first=True)
        self.FC_L1 = nn.Linear(in_features=256, out_features=64)
        self.FC_L2 = nn.Linear(in_features=64, out_features=4)

        self.input_layer_R = nn.Linear(in_features=8, out_features=32)
        self.FC_R1 = nn.Linear(in_features=32, out_features=64)
        self.FC_R2 = nn.Linear(in_features=320, out_features=64)
        self.FC_R3 = nn.Linear(in_features=64, out_features=2)

    def forward(self, input_seq, candidates):
        # the left side accepts the input of the first four points 
        input_seq = self.input_layer_L(input_seq)
        gru_out,  _=self.gru(input_seq)
        # prepare for concat and FC
        gru_out =  torch.tanh(gru_out[:, -1, :])
        output_left= F.elu(self.FC_L1(gru_out))
        pred_reg= F.elu(self.FC_L2(output_left))


        #  the right side  accepts  two points 
        output_right = self.input_layer_R(candidates)
        output_right = F.elu(self.FC_R1(output_right))
        output_right = torch.cat((gru_out, output_right), 1)
        output_right = F.elu(self.FC_R2(output_right))
        pred_cls = self.FC_R3(output_right)

        return pred_reg,pred_cls
