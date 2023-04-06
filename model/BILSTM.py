import torch
import torch.nn as nn
import torch.nn.functional as F

class BILSTM3(nn.Module): # MAGLSTM
    def __init__(self):
        super(BILSTM3, self).__init__()
        self.lstm1 = nn.LSTM(input_size=161, hidden_size=161, num_layers=3,dropout=0.02,bidirectional=True, batch_first = True)
        self.fc = nn.Linear(161*2, 161)# fc
    def forward(self, X):
        # X = torch.squeeze(X)
        bs,m,z = X.shape
        newshape = (bs,z)
        X = torch.reshape(X,newshape)
        pred_stft = torch.stft(X, n_fft=320,hop_length=160,win_length=320) #(Bs, F, T, 2)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-6)
        batch_size, m, z = pred_mag.shape # (16, 161, 103) (Bs, F, T)
        #print(out.shape)
        input = pred_mag.permute(0,2,1)#指定维度新的位置 (Bs, T, F)
        # h1 = torch.randn(3*2, batch_size, 161).to(device)  # [num_layers(=3) * num_directions(=2), batch_size, n_hidden]
        # c1 = torch.randn(3*2, batch_size, 161).to(device)

        # outputs1, (_, _) = self.lstm1(input, (h1, c1)) # (seq_length,batch_size,num_directions*hidden_size)
        outputs1, (_, _) = self.lstm1(input) # (seq_length,batch_size,num_directions*hidden_size)
        # outputs = outputs1.permute(0,2,1)   # (batch_size,seq_length,num_directions*hidden_size)
        mag=self.fc(outputs1)
        mag=mag.permute(0,2,1)
        pha = torch.atan2(pred_stft_imag, pred_stft_real)  # ([16, 161, 103])
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        model = torch.istft(torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1), n_fft=320, hop_length=160,
                            win_length=320)
        model = torch.unsqueeze(model,1)
        return model


class BLSTM(nn.Module):

    def __init__(self):
        super(BLSTM, self).__init__()
        input_size = 161
        hidden_size = 161
        num_layers = 1
        self.blstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out, _ = self.blstm(x)
        out = self.linear(out)
        out = self.relu(out)
        out = out + residual
        return out

class resBLSTM(nn.Module):

    def __init__(self):
        super(resBLSTM, self).__init__()
        input_size = 161
        hidden_size = 161
        num_layers = 1
        self.blstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(hidden_size * 2, input_size)
        self.relu1 = nn.ReLU()
        self.blstm2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True,
                              batch_first=True)
        self.linear2 = nn.Linear(hidden_size * 2, input_size)
        self.relu2 = nn.ReLU()
        self.blstm3 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True,
                              batch_first=True)
        self.linear3 = nn.Linear(hidden_size * 2, input_size)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        residual = x
        out, _ = self.blstm1(x)
        out = self.linear1(out)
        out = self.relu1(out)
        out += residual
        residual = out
        out, _ = self.blstm2(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out += residual
        residual = out
        out, _ = self.blstm3(out)
        out = self.linear3(out)
        # out = self.relu3(out)
        out += residual
        return out

class RESMAGLSTM(nn.Module):
    def __init__(self):
        super(RESMAGLSTM, self).__init__()
        self.blstm1 = BLSTM()
        self.blstm2 = BLSTM()
        self.blstm3 = BLSTM()

    def forward(self, X):
        # mu = -10.401364  # bone   # air -3.48766 9.9217825
        # sigma = 19.957256
        # calculate maximum and minimum values of X for each batch
        # max_val, _ = torch.max(torch.abs(X), dim=2, keepdim=True)

        # normalize X for each batch
        # X_normalized = X / max_val
        X_normalized = X

        # scale X_normalized to be between -1 and 1

        batch_size, m, z = X_normalized.shape  # (batch,1,16320)
        newshape = (batch_size, z)
        X = torch.reshape(X_normalized, newshape)
        pred_stft = torch.stft(X, n_fft=320, hop_length=160, win_length=320)  # (Bs, F, T, 2)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-6)
        # pred_mag = torch.log(pred_mag ** 2 + 1e-12)  # 将计算幅度谱改成计算对数幅度谱
        # pred_mag = (pred_mag - mu) / sigma

        # batch_size, m, z = pred_mag.shape # (16, 161, 103) (Bs, F, T)
        # print(out.shape)
        input = pred_mag.permute(0, 2, 1)  # 指定维度新的位置 (Bs, T, F)
        # h1 = torch.randn(3*2, batch_size, 161).to(device)  # [num_layers(=3) * num_directions(=2), batch_size, n_hidden]
        # c1 = torch.randn(3*2, batch_size, 161).to(device)
        out = self.blstm1(input)
        out = self.blstm2(out)
        out = self.blstm3(out)

        mag = out.permute(0, 2, 1)
        # mag = mag * sigma + mu
        # mag = torch.exp(mag/2)  # 将对数幅度谱转换成幅度谱
        # if (mag < 0).any():
        #     print("error")
        pha = torch.atan2(pred_stft_imag, pred_stft_real)  # ([16, 161, 103])
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        model = torch.istft(torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1), n_fft=320, hop_length=160,
                            win_length=320)
        if torch.isnan(model).any():
            print("nan error")
            input("press enter ")
        model = torch.unsqueeze(model, 1)
        # model = model * max_val

        return model


class BILSTMModel(nn.Module):
    def __init__(self):
        super(BILSTMModel, self).__init__()
        self.blstm1 = BLSTM()
        self.blstm2 = BLSTM()
        self.blstm3 = BLSTM()

    def forward(self, X):
        # mu = -10.401364  # bone   # air -3.48766 9.9217825
        # sigma = 19.957256
        # # calculate maximum and minimum values of X for each batch
        # max_val, _ = torch.max(torch.abs(X), dim=2, keepdim=True)
        #
        # # normalize X for each batch
        # X_normalized = X / max_val
        #
        # # scale X_normalized to be between -1 and 1
        #
        # batch_size, m, z = X_normalized.shape  # (batch,1,16320)
        # newshape = (batch_size, z)
        # X = torch.reshape(X_normalized, newshape)
        # pred_stft = torch.stft(X, n_fft=320, hop_length=160, win_length=320)  # (Bs, F, T, 2)
        # pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        # pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-12)
        # pred_mag = torch.log(pred_mag ** 2 + 1e-12)  # 将计算幅度谱改成计算对数幅度谱
        # pred_mag = (pred_mag - mu) / sigma
        #
        # # batch_size, m, z = pred_mag.shape # (16, 161, 103) (Bs, F, T)
        # # print(out.shape)
        # input = pred_mag.permute(0, 2, 1)  # 指定维度新的位置 (Bs, T, F)
        # h1 = torch.randn(3*2, batch_size, 161).to(device)  # [num_layers(=3) * num_directions(=2), batch_size, n_hidden]
        # c1 = torch.randn(3*2, batch_size, 161).to(device)
        out = self.blstm1(X)
        out = self.blstm2(out)
        out = self.blstm3(out)
        # mag = out.permute(0, 2, 1)
        # mag = mag * sigma + mu
        # mag = torch.exp(mag/2)  # 将对数幅度谱转换成幅度谱
        # if (mag < 0).any():
        #     print("error")
        # pha = torch.atan2(pred_stft_imag, pred_stft_real)  # ([16, 161, 103])
        # real = mag * torch.cos(pha)
        # imag = mag * torch.sin(pha)
        # model = torch.istft(torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1), n_fft=320, hop_length=160,
        #                     win_length=320)
        # if torch.isnan(model).any():
        #     print("nan error")
        #     input("press enter ")
        # model = torch.unsqueeze(model, 1)
        # model = model * max_val

        return out

class BILSTM4(nn.Module):
    def __init__(self):
        super(BILSTM4, self).__init__()
        self.lstm1 = nn.LSTM(input_size=161, hidden_size=161, num_layers=2,dropout=0.02,bidirectional=True, batch_first = True)
        self.fc = nn.Linear(161*2, 161)# fc
    def forward(self, X):
        # X = torch.squeeze(X)
        # bs,m,z = X.shape
        # newshape = (bs,z)
        # X = torch.reshape(X,newshape)
        # pred_stft = torch.stft(X, n_fft=320,hop_length=160,win_length=320) #(Bs, F, T, 2)
        # pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        # pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-12)
        # batch_size, m, z = pred_mag.shape # (16, 161, 103) (Bs, F, T)
        # #print(out.shape)
        # input = pred_mag.permute(0,2,1)#指定维度新的位置 (Bs, T, F)
        # h1 = torch.randn(3*2, batch_size, 161).to(device)  # [num_layers(=3) * num_directions(=2), batch_size, n_hidden]
        # c1 = torch.randn(3*2, batch_size, 161).to(device)

        # outputs1, (_, _) = self.lstm1(input, (h1, c1)) # (seq_length,batch_size,num_directions*hidden_size)
        outputs1, (_, _) = self.lstm1(X) # (seq_length,batch_size,num_directions*hidden_size)
        # outputs = outputs1.permute(0,2,1)   # (batch_size,seq_length,num_directions*hidden_size)
        out = self.fc(outputs1)
        # mag=mag.permute(0,2,1)
        # pha = torch.atan2(pred_stft_imag, pred_stft_real)  # ([16, 161, 103])
        # real = mag * torch.cos(pha)
        # imag = mag * torch.sin(pha)
        # model = torch.istft(torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1), n_fft=320, hop_length=160,
        #                     win_length=320)
        # model = torch.unsqueeze(model,1)
        return out

class WAVBLSTMModel(nn.Module):
    def __init__(self):
        super(WAVBLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=161, hidden_size=161, num_layers=3, dropout=0.02, bidirectional=True,
                             batch_first=True)
        self.fc = nn.Linear(161 * 2, 161)  # fc
        self.relu = nn.ReLU()

    def forward(self, X):
        # mu = -9.981625  # bone   # air -3.48766 9.9217825
        # sigma = 15.002887
        # calculate maximum and minimum values of X for each batch
        if torch.isnan(X).any():
            print("nan error")
            input("press enter X nan error")
        # max_val = torch.max(torch.abs(X), dim = -1, keepdim=True).values
        # if torch.isnan(max_val).any():
        #     print("nan error")
        #     input("press enter max_val nan error")
        # normalize X for each batch
        # X_normalized = X / (max_val + 1e-6)
        X_normalized = X

        # scale X_normalized to be between -1 and 1

        batch_size, m, z = X_normalized.shape  # (batch,1,16320)
        newshape = (batch_size, z)
        X = torch.reshape(X_normalized, newshape)
        pred_stft = torch.stft(X, n_fft=320, hop_length=160, win_length=320)  # (Bs, F, T, 2)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-6)
        if torch.isnan(pred_stft).any():
            print("nan error")
            input("press enter pred_stft nan error")
        if torch.isnan(pred_stft_real).any():
            print("nan error")
            input("press enter pred_stft_real nan error")
        if torch.isnan(pred_stft_imag).any():
            print("nan error")
            input("press enter pred_stft_imag nan error")
        if torch.isnan(pred_mag).any():
            print("nan error")
            input("press enter pred_mag nan error")

        pred_mag = torch.log(pred_mag ** 2 + 1e-6)  # 将计算幅度谱改成计算对数幅度谱
        # pred_mag = (pred_mag - mu) / sigma

        # batch_size, m, z = pred_mag.shape # (16, 161, 103) (Bs, F, T)
        # print(out.shape)
        input = pred_mag.permute(0, 2, 1)  # 指定维度新的位置 (Bs, T, F)
        # h1 = torch.randn(3*2, batch_size, 161).to(device)  # [num_layers(=3) * num_directions(=2), batch_size, n_hidden]
        # c1 = torch.randn(3*2, batch_size, 161).to(device)
        out, (_, _) = self.lstm1(input)
        out = self.fc(out)
        # out = self.relu(out)
        mag = out.permute(0, 2, 1)
        # mag = mag * sigma + mu
        mag = torch.exp(mag/2)  # 将对数幅度谱转换成幅度谱
        # mag = torch.abs(mag)
        eps = 1e-6
        # assert torch.ge(mag,-eps)
        if torch.any(mag < 0):
            print("error  mag < 0 ")
            input("mag < 0 ")
        if torch.isnan(mag).any():
            print("nan error")
            input("press enter mag nan error")
        pha = torch.atan2(pred_stft_imag, (pred_stft_real + 1e-6))  # ([16, 161, 103])

        if torch.isnan(pha).any():
            print("nan error")
            input("press enter pha nan error")
        mask = torch.isnan(pha) | torch.isinf(pha)  # 找到 pha 中的异常值
        pha = torch.where(mask, torch.zeros_like(pha), pha)  # 将异常值替换为 0
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        model = torch.istft(torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1), n_fft=320, hop_length=160,
                            win_length=320)
        if torch.isnan(model).any():
            print("nan error")
            input("press enter nan error")
        model = torch.unsqueeze(model, 1)
        # model = model * max_val

        return model


class CRNNet(nn.Module):
    def __init__(self):
        super(CRNNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))

        self.lstm = nn.LSTM(256 * 4, 256 * 4, 2, batch_first=True)

        self.conv5_t = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv4_t = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv3_t = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv2_t = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0), output_padding=(0, 1))
        self.conv1_t = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)

        self.bn5_t = nn.BatchNorm2d(128)
        self.bn4_t = nn.BatchNorm2d(64)
        self.bn3_t = nn.BatchNorm2d(32)
        self.bn2_t = nn.BatchNorm2d(16)
        self.bn1_t = nn.BatchNorm2d(1)

        self.elu = nn.ELU(inplace=True)
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = x.unsqueeze(dim=1)
        e1 = self.elu(self.bn1(self.conv1(out)[:, :, :-1, :].contiguous()))
        e2 = self.elu(self.bn2(self.conv2(e1)[:, :, :-1, :].contiguous()))
        e3 = self.elu(self.bn3(self.conv3(e2)[:, :, :-1, :].contiguous()))
        e4 = self.elu(self.bn4(self.conv4(e3)[:, :, :-1, :].contiguous()))
        e5 = self.elu(self.bn5(self.conv5(e4)[:, :, :-1, :].contiguous()))

        out = e5.contiguous().transpose(1, 2)
        q1 = out.size(2)
        q2 = out.size(3)
        out = out.contiguous().view(out.size(0), out.size(1), -1)
        out, _ = self.lstm(out)
        out = out.contiguous().view(out.size(0), out.size(1), q1, q2)
        out = out.contiguous().transpose(1, 2)

        out = torch.cat([out, e5], dim=1)

        d5 = self.elu(torch.cat([self.bn5_t(F.pad(self.conv5_t(out), [0, 0, 1, 0]).contiguous()), e4], dim=1))
        d4 = self.elu(torch.cat([self.bn4_t(F.pad(self.conv4_t(d5), [0, 0, 1, 0]).contiguous()), e3], dim=1))
        d3 = self.elu(torch.cat([self.bn3_t(F.pad(self.conv3_t(d4), [0, 0, 1, 0]).contiguous()), e2], dim=1))
        d2 = self.elu(torch.cat([self.bn2_t(F.pad(self.conv2_t(d3), [0, 0, 1, 0]).contiguous()), e1], dim=1))
        d1 = self.softplus(self.bn1_t(F.pad(self.conv1_t(d2), [0, 0, 1, 0]).contiguous()))

        out = torch.squeeze(d1, dim=1)

        return out

class CRN(nn.Module):
    def __init__(self):
        super(CRN, self).__init__()
        # self.lstm1 = nn.LSTM(input_size=161, hidden_size=161, num_layers=3, dropout=0.02, bidirectional=True,
        #                      batch_first=True)
        # self.fc = nn.Linear(161 * 2, 161)  # fc
        # self.relu = nn.ReLU()
        self.crn = CRNNet()

    def forward(self, X):
        # mu = -9.981625  # bone   # air -3.48766 9.9217825
        # sigma = 15.002887
        # calculate maximum and minimum values of X for each batch
        if torch.isnan(X).any():
            print("nan error")
            input("press enter X nan error")
        # max_val = torch.max(torch.abs(X), dim = -1, keepdim=True).values
        # if torch.isnan(max_val).any():
        #     print("nan error")
        #     input("press enter max_val nan error")
        # normalize X for each batch
        # X_normalized = X / (max_val + 1e-6)
        X_normalized = X

        # scale X_normalized to be between -1 and 1

        batch_size, m, z = X_normalized.shape  # (batch,1,16320)
        newshape = (batch_size, z)
        X = torch.reshape(X_normalized, newshape)
        pred_stft = torch.stft(X, n_fft=320, hop_length=160, win_length=320)  # (Bs, F, T, 2)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-6)
        if torch.isnan(pred_stft).any():
            print("nan error")
            input("press enter pred_stft nan error")
        if torch.isnan(pred_stft_real).any():
            print("nan error")
            input("press enter pred_stft_real nan error")
        if torch.isnan(pred_stft_imag).any():
            print("nan error")
            input("press enter pred_stft_imag nan error")
        if torch.isnan(pred_mag).any():
            print("nan error")
            input("press enter pred_mag nan error")

        # pred_mag = torch.log(pred_mag ** 2 + 1e-6)  # 将计算幅度谱改成计算对数幅度谱
        # pred_mag = (pred_mag - mu) / sigma

        # batch_size, m, z = pred_mag.shape # (16, 161, 103) (Bs, F, T)
        # print(out.shape)
        input = pred_mag.permute(0, 2, 1)  # 指定维度新的位置 (Bs, T, F)
        # h1 = torch.randn(3*2, batch_size, 161).to(device)  # [num_layers(=3) * num_directions(=2), batch_size, n_hidden]
        # c1 = torch.randn(3*2, batch_size, 161).to(device)
        out = self.crn(input)
        # out, (_, _) = self.lstm1(input)
        # out = self.fc(out)
        # out = self.relu(out)
        mag = out.permute(0, 2, 1)
        # mag = mag * sigma + mu
        # mag = torch.exp(mag/2)  # 将对数幅度谱转换成幅度谱
        # mag = torch.abs(mag)
        # eps = 1e-6
        # assert torch.ge(mag,-eps)
        if torch.any(mag < 0):
            print("error  mag < 0 ")
            input("mag < 0 ")
        if torch.isnan(mag).any():
            print("nan error")
            input("press enter mag nan error")
        pha = torch.atan2(pred_stft_imag, (pred_stft_real + 1e-6))  # ([16, 161, 103])

        if torch.isnan(pha).any():
            print("nan error")
            input("press enter pha nan error")
        mask = torch.isnan(pha) | torch.isinf(pha)  # 找到 pha 中的异常值
        pha = torch.where(mask, torch.zeros_like(pha), pha)  # 将异常值替换为 0
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        model = torch.istft(torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1), n_fft=320, hop_length=160,
                            win_length=320)
        if torch.isnan(model).any():
            print("nan error")
            input("press enter nan error")
        model = torch.unsqueeze(model, 1)
        # model = model * max_val

        return model

class COPYBLSTMModel(nn.Module):
    def __init__(self):
        super(COPYBLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=161, hidden_size=161, num_layers=3, dropout=0.02, bidirectional=True,
                             batch_first=True)
        self.fc = nn.Linear(161 * 2, 161)  # fc
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        # mu = -9.981625  # bone   # air -3.48766 9.9217825
        # sigma = 15.002887
        # calculate maximum and minimum values of X for each batch
        if torch.isnan(X).any():
            print("nan error")
            input("press enter X nan error")
        # max_val = torch.max(torch.abs(X), dim = -1, keepdim=True).values
        # if torch.isnan(max_val).any():
        #     print("nan error")
        #     input("press enter max_val nan error")
        # normalize X for each batch
        # X_normalized = X / (max_val + 1e-6)
        X_normalized = X

        # scale X_normalized to be between -1 and 1

        batch_size, m, z = X_normalized.shape  # (batch,1,16320)
        newshape = (batch_size, z)
        X = torch.reshape(X_normalized, newshape)
        pred_stft = torch.stft(X, n_fft=320, hop_length=160, win_length=320)  # (Bs, F, T, 2)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        pred_stft_real[:,41:81,:] = pred_stft_real[:,1:41,:]
        pred_stft_real[:, 81:121, :] = pred_stft_real[:, 1:41, :]
        pred_stft_real[:, 121:161, :] = pred_stft_real[:, 1:41, :]
        pred_stft_imag[:, 41:81, :] = pred_stft_imag[:, 1:41, :]
        pred_stft_imag[:, 81:121, :] = pred_stft_imag[:, 1:41, :]
        pred_stft_imag[:, 121:161, :] = pred_stft_imag[:, 1:41, :]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-6)
        if torch.isnan(pred_stft).any():
            print("nan error")
            input("press enter pred_stft nan error")
        if torch.isnan(pred_stft_real).any():
            print("nan error")
            input("press enter pred_stft_real nan error")
        if torch.isnan(pred_stft_imag).any():
            print("nan error")
            input("press enter pred_stft_imag nan error")
        if torch.isnan(pred_mag).any():
            print("nan error")
            input("press enter pred_mag nan error")

        # pred_mag = torch.log(pred_mag ** 2 + 1e-6)  # 将计算幅度谱改成计算对数幅度谱
        # pred_mag = (pred_mag - mu) / sigma

        # batch_size, m, z = pred_mag.shape # (16, 161, 103) (Bs, F, T)
        # print(out.shape)
        input = pred_mag.permute(0, 2, 1)  # 指定维度新的位置 (Bs, T, F)
        # h1 = torch.randn(3*2, batch_size, 161).to(device)  # [num_layers(=3) * num_directions(=2), batch_size, n_hidden]
        # c1 = torch.randn(3*2, batch_size, 161).to(device)
        out, (_, _) = self.lstm1(input)
        out = self.fc(out)
        # out = self.relu(out)
        out = self.sigmoid(out)
        out = out.permute(0, 2, 1) # (Bs, F, T)
        mag = torch.mul(out,pred_mag)
        # mag = mag * sigma + mu
        # mag = torch.exp(mag/2)  # 将对数幅度谱转换成幅度谱
        # mag = torch.abs(mag)
        eps = 1e-6
        # assert torch.ge(mag,-eps)
        if torch.any(mag < 0):
            print("error  mag < 0 ")
            input("mag < 0 ")
        if torch.isnan(mag).any():
            print("nan error")
            input("press enter mag nan error")
        pha = torch.atan2(pred_stft_imag, (pred_stft_real + 1e-6))  # ([16, 161, 103])

        if torch.isnan(pha).any():
            print("nan error")
            input("press enter pha nan error")
        mask = torch.isnan(pha) | torch.isinf(pha)  # 找到 pha 中的异常值
        pha = torch.where(mask, torch.zeros_like(pha), pha)  # 将异常值替换为 0
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        model = torch.istft(torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1), n_fft=320, hop_length=160,
                            win_length=320)
        if torch.isnan(model).any():
            print("nan error")
            input("press enter nan error")
        model = torch.unsqueeze(model, 1)
        # model = model * max_val

        return model

class RESCOPYLSTM(nn.Module):
    def __init__(self):
        super(RESCOPYLSTM, self).__init__()
        self.blstm1 = BLSTM()
        self.blstm2 = BLSTM()
        self.blstm3 = BLSTM()

    def forward(self, X):
        # mu = -10.401364  # bone   # air -3.48766 9.9217825
        # sigma = 19.957256
        # calculate maximum and minimum values of X for each batch
        # max_val, _ = torch.max(torch.abs(X), dim=2, keepdim=True)

        # normalize X for each batch
        # X_normalized = X / max_val
        X_normalized = X

        # scale X_normalized to be between -1 and 1

        batch_size, m, z = X_normalized.shape  # (batch,1,16320)
        newshape = (batch_size, z)
        X = torch.reshape(X_normalized, newshape)
        pred_stft = torch.stft(X, n_fft=320, hop_length=160, win_length=320)  # (Bs, F, T, 2)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        pred_stft_real[:, 41:81, :] = pred_stft_real[:, 1:41, :]
        pred_stft_real[:, 81:121, :, :] = pred_stft_real[:, 1:41, :]
        pred_stft_real[:, 121:161, :] = pred_stft_real[:, 1:41, :]
        pred_stft_imag[:, 41:81, :] = pred_stft_imag[:, 1:41, :]
        pred_stft_imag[:, 81:121, :] = pred_stft_imag[:, 1:41, :]
        pred_stft_imag[:, 121:161, :] = pred_stft_imag[:, 1:41, :]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-6)
        # pred_mag = torch.log(pred_mag ** 2 + 1e-12)  # 将计算幅度谱改成计算对数幅度谱
        # pred_mag = (pred_mag - mu) / sigma

        # batch_size, m, z = pred_mag.shape # (16, 161, 103) (Bs, F, T)
        # print(out.shape)
        input = pred_mag.permute(0, 2, 1)  # 指定维度新的位置 (Bs, T, F)
        # h1 = torch.randn(3*2, batch_size, 161).to(device)  # [num_layers(=3) * num_directions(=2), batch_size, n_hidden]
        # c1 = torch.randn(3*2, batch_size, 161).to(device)
        out = self.blstm1(input)
        out = self.blstm2(out)
        out = self.blstm3(out)

        mag = out.permute(0, 2, 1)
        # mag = mag * sigma + mu
        # mag = torch.exp(mag/2)  # 将对数幅度谱转换成幅度谱
        # if (mag < 0).any():
        #     print("error")
        pha = torch.atan2(pred_stft_imag, pred_stft_real)  # ([16, 161, 103])
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        model = torch.istft(torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1), n_fft=320, hop_length=160,
                            win_length=320)
        if torch.isnan(model).any():
            print("nan error")
            input("press enter ")
        model = torch.unsqueeze(model, 1)
        # model = model * max_val

        return model
