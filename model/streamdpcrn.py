from typing import Union, Tuple
import torch
import torch.nn as nn
import numpy as np

class StreamConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 *args, **kargs):
        super(StreamConv, self).__init__(*args, **kargs)
        """
        流式卷积实现。
        默认 kernel_size = [T_size, F_size]
        """
        self.Conv2d = nn.Conv2d(in_channels = in_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding,
                                dilation = dilation,
                                groups = groups,
                                bias = bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        if type(kernel_size) is int:
            self.T_size = kernel_size
            self.F_size = kernel_size
        elif type(kernel_size) in [list, tuple]:
            self.T_size, self.F_size = kernel_size
        else:
            raise ValueError('Invalid kernel size')

    def forward(self, x, cache):
        """
        x: [bs,C,1,F]
        cache: [bs,C,T-1,F]
        """
        inp = torch.cat([cache ,x], dim = 2)
        outp = self.Conv2d(inp)
        # 这里也可以输出x，把更新cache放到外面
        out_cache = inp[: ,: ,1:]
        return outp, out_cache


class StreamConvTranspose(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 *args, **kargs):
        super(StreamConvTranspose, self).__init__(*args, **kargs)
        """
        流式转置卷积实现。
        默认 kernel_size = [T_size, F_size]
        默认 stride = [T_stride, F_stride] 且 T_stride == 1
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        if type(kernel_size) is int:
            self.T_size = kernel_size
            self.F_size = kernel_size
        elif type(kernel_size) in [list, tuple]:
            self.T_size, self.F_size = kernel_size
        else:
            raise ValueError('Invalid kernel size.')

        if type(stride) is int:
            self.T_stride = stride
            self.F_stride = stride
        elif type(stride) in [list, tuple]:
            self.T_stride, self.F_stride = stride
        else:
            raise ValueError('Invalid stride size.')

        assert self.T_stride == 1

        # 我们使用权重时间反向的Conv2d实现转置卷积
        self.Conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=(self.T_stride, 1),  # F维度stride不为1，将在forward中使用额外的上采样算子
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias)

    @staticmethod
    def get_indices(inp, F_stride):
        """
        根据 input 的维度和 F维度上采样维度得到上采样之后的维度
        inp: [bs,C,T,F]
        return:
            indices: [bs,C,T,F]
        由于只对F上采样，因此输出的维度为 [bs,C,T,F_out]
        其中F_out = (F - 1) * (F_stride - 1) + F, 即向原来的每一个元素里面插入F_stride-1个零
        """
        bs, C, T, F = inp.shape
        # indices: [bs,C,T,F]
        F_out = (F - 1) * (F_stride - 1) + F
        indices = np.zeros([bs * 1 * T * F])
        index = 0
        for i in range(bs * 1 * T * F):
            indices[i] = index
            if (i + 1) % F == 0:
                index += 1
            else:
                index += F_stride
        indices = torch.from_numpy(np.repeat(indices.reshape([bs, 1, T, F]).astype('int64'), C, axis=1))
        return indices, F_out

    def forward(self, x, cache):
        """
        x: [bs,C,1,F]
        cache: [bs,C,T-1,F]
        """
        # [bs,C,T,F]
        inp = torch.cat([cache, x], dim=2)
        out_cache = inp[:, :, 1:]
        bs, C, T, F = inp.shape
        # 添加上采样算子
        if self.F_stride >= 1:
            # [bs,C,T,F] -> [bs,C,T,F,1] -> [bs,C,T,F,F_stride] -> [bs,C,T,F_out]
            inp = torch.concat([inp[:, :, :, :, None], torch.zeros([bs, C, T, F, self.F_stride - 1])], dim=-1).reshape(
                [bs, C, T, -1])
            left_pad = self.F_stride - 1
            if self.F_size > 1:
                if left_pad <= self.F_size - 1:
                    inp = torch.nn.functional.pad(inp, pad=[self.F_size - 1, self.F_size - 1 - left_pad, 0, 0])
                else:
                    inp = torch.nn.functional.pad(inp, pad=[self.F_size - 1, 0, 0, 0])[:, :, :,
                          : - (left_pad - self.F_stride + 1)]
            else:
                inp = inp[:, :, :, :-left_pad]

        outp = self.Conv2d(inp)
        # 这里也可以输出x，把更新cache放到外面

        return outp, out_cache

'''
DPRNN
'''
class DPRNN(nn.Module):
    def __init__(self, numUnits, width, channel, **kwargs):
        super(DPRNN, self).__init__(**kwargs)
        self.numUnits = numUnits

        self.intra_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits//2, batch_first = True, bidirectional = True)

        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)

        self.intra_ln = nn.InstanceNorm2d(width,eps=1e-8,affine=True, track_running_stats= True)

        self.inter_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits, batch_first = True, bidirectional = False)

        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)

        self.inter_ln = nn.InstanceNorm2d(channel, eps=1e-8,affine=True, track_running_stats= True)

        self.width = width
        self.channel = channel

    def forward(self,x):
        # x.shape = (Bs, C, T, F)
        self.intra_rnn.flatten_parameters()
        self.inter_rnn.flatten_parameters()
        x = x.permute(0,2,3,1) #(Bs, T, F, C)
        if not x.is_contiguous():
            x = x.contiguous()
        ## Intra RNN
        intra_LSTM_input = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) #(Bs*T, F, C)
        intra_LSTM_out = self.intra_rnn(intra_LSTM_input)[0] #(Bs*T, F, C)
        intra_dense_out = self.intra_fc(intra_LSTM_out)
        intra_ln_input = intra_dense_out.view(x.shape[0], -1, self.width, self.channel) #(Bs, T, F, C)
        intra_ln_input = intra_ln_input.permute(0,2,1,3) #(Bs, F, T, C)
        intra_out = self.intra_ln(intra_ln_input)
        intra_out = intra_out.permute(0,2,1,3) #(Bs, T, F, C)
        intra_out = torch.add(x, intra_out)
        ## Inter RNN
        inter_LSTM_input = intra_out.permute(0,2,1,3) #(Bs, F, T, C)
        inter_LSTM_input = inter_LSTM_input.contiguous()
        inter_LSTM_input = inter_LSTM_input.view(inter_LSTM_input.shape[0] * inter_LSTM_input.shape[1], inter_LSTM_input.shape[2], inter_LSTM_input.shape[3]) #(Bs * F, T, C)
        inter_LSTM_out = self.inter_rnn(inter_LSTM_input)[0]
        inter_dense_out = self.inter_fc(inter_LSTM_out)
        inter_dense_out = inter_dense_out.view(x.shape[0], self.width, -1, self.channel) #(Bs, F, T, C)
        inter_ln_input = inter_dense_out.permute(0,3,2,1) #(Bs, C, T, F)
        inter_out = self.inter_ln(inter_ln_input)
        inter_out = inter_out.permute(0,2,3,1) #(Bs, T, F, C)
        inter_out = torch.add(intra_out, inter_out)
        inter_out = inter_out.permute(0,3,1,2)
        inter_out = inter_out.contiguous()

        return inter_out

'''
DPRNN_RT
'''
class DPRNN_RT(nn.Module):
    def __init__(self, numUnits, width, channel, **kwargs):
        super(DPRNN_RT, self).__init__(**kwargs)
        self.numUnits = numUnits

        self.intra_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits//2, batch_first = True, bidirectional = True)

        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)

        self.intra_ln = nn.InstanceNorm2d(width,eps=1e-8,affine=True, track_running_stats= True)

        self.inter_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits, batch_first = True, bidirectional = False)

        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)

        self.inter_ln = nn.InstanceNorm2d(channel, eps=1e-8,affine=True, track_running_stats= True)

        self.width = width
        self.channel = channel

    def forward(self,x, intra_hidden, inter_hidden):
        # x.shape = (Bs, C, T, F)
        self.intra_rnn.flatten_parameters()
        self.inter_rnn.flatten_parameters()
        x = x.permute(0,2,3,1) #(Bs, T, F, C)
        if not x.is_contiguous():
            x = x.contiguous()
        ## Intra RNN
        intra_LSTM_input = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) #(Bs*T, F, C)
        intra_LSTM_out, intra_next_hidden = self.intra_rnn(intra_LSTM_input,intra_hidden) #(Bs*T, F, C)
        intra_dense_out = self.intra_fc(intra_LSTM_out)
        intra_ln_input = intra_dense_out.view(x.shape[0], -1, self.width, self.channel) #(Bs, T, F, C)
        intra_ln_input = intra_ln_input.permute(0,2,1,3) #(Bs, F, T, C)
        intra_out = self.intra_ln(intra_ln_input)
        intra_out = intra_out.permute(0,2,1,3) #(Bs, T, F, C)
        intra_out = torch.add(x, intra_out)
        ## Inter RNN
        inter_LSTM_input = intra_out.permute(0,2,1,3) #(Bs, F, T, C)
        inter_LSTM_input = inter_LSTM_input.contiguous()
        inter_LSTM_input = inter_LSTM_input.view(inter_LSTM_input.shape[0] * inter_LSTM_input.shape[1], inter_LSTM_input.shape[2], inter_LSTM_input.shape[3]) #(Bs * F, T, C)
        inter_LSTM_out, inter_next_hidden = self.inter_rnn(inter_LSTM_input, inter_hidden)
        inter_dense_out = self.inter_fc(inter_LSTM_out)
        inter_dense_out = inter_dense_out.view(x.shape[0], self.width, -1, self.channel) #(Bs, F, T, C)
        inter_ln_input = inter_dense_out.permute(0,3,2,1) #(Bs, C, T, F)
        inter_out = self.inter_ln(inter_ln_input)
        inter_out = inter_out.permute(0,2,3,1) #(Bs, T, F, C)
        inter_out = torch.add(intra_out, inter_out)
        inter_out = inter_out.permute(0,3,1,2)
        inter_out = inter_out.contiguous()

        return inter_out, intra_next_hidden, inter_next_hidden

'''
Encoder
'''
class Encoder(nn.Module):

    def __init__(self, auto_encoder = True):
        super(Encoder, self).__init__()

        self.auto_encoder = auto_encoder

        #---------------------------whole learnt-----------------------
        # self.flc = nn.Linear(self.F, self.F_c, bias=False)
        # self.flc.weight = nn.Parameter(torch.from_numpy(Sc), requires_grad=self.auto_encoder)
        #--------------------------------------------------------------

        # self.ln = nn.InstanceNorm2d(2, eps=1e-8, affine=True)
        self.conv_1 = nn.Conv2d(2, 32, kernel_size=(2,5),stride=(1,2),padding=(0,1))
        self.bn_1 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_1 = nn.PReLU(32)

        self.conv_2 = nn.Conv2d(32,32,kernel_size=(2,3),stride=(1,2),padding=(0,1))
        self.bn_2 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_2 = nn.PReLU(32)

        self.conv_3 = nn.Conv2d(32,32,kernel_size=(2,3),stride=(1,1),padding=(0,1))
        self.bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_3 = nn.PReLU(32)

        self.conv_4 = nn.Conv2d(32,64,kernel_size=(2,3),stride=(1,1),padding=(0,1))
        self.bn_4 = nn.BatchNorm2d(64, eps=1e-8)
        self.act_4 = nn.PReLU(64)

        self.conv_5 = nn.Conv2d(64,128,kernel_size=(2,3),stride=(1,1),padding=(0,1))
        self.bn_5 = nn.BatchNorm2d(128, eps=1e-8)
        self.act_5 = nn.PReLU(128)

    def forward(self,x):
        #x.shape = (Bs, F, T, 2)
        x = x.permute(0,3,2,1) #(Bs, 2, T, F)
        x = x.to(torch.float32)
        # x = self.flc(x)
        # x = self.ln(x)
        x_1 = self.act_1(self.bn_1(self.conv_1(torch.nn.functional.pad(x, [0, 0, 1, 0]))))
        x_2 = self.act_2(self.bn_2(self.conv_2(torch.nn.functional.pad(x_1, [0, 0, 1, 0]))))
        x_3 = self.act_3(self.bn_3(self.conv_3(torch.nn.functional.pad(x_2, [0, 0, 1, 0]))))
        x_4 = self.act_4(self.bn_4(self.conv_4(torch.nn.functional.pad(x_3, [0, 0, 1, 0]))))
        x_5 = self.act_5(self.bn_5(self.conv_5(torch.nn.functional.pad(x_4, [0, 0, 1, 0]))))
        return [x_1,x_2,x_3,x_4,x_5]

'''
Encoder_RT
'''
class Encoder_RT(nn.Module):

    def __init__(self, auto_encoder = True):
        super(Encoder_RT, self).__init__()

        self.auto_encoder = auto_encoder

        #---------------------------whole learnt-----------------------
        # self.flc = nn.Linear(self.F, self.F_c, bias=False)
        # self.flc.weight = nn.Parameter(torch.from_numpy(Sc), requires_grad=self.auto_encoder)
        #--------------------------------------------------------------

        # self.ln = nn.InstanceNorm2d(2, eps=1e-8, affine=True)
        self.conv_1 = StreamConv(2, 32, [2, 5], [1, 2], [0, 1])
        self.bn_1 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_1 = nn.PReLU(32)

        self.conv_2 = StreamConv(32, 32, [2, 3], [1, 2], [0, 1])
        self.bn_2 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_2 = nn.PReLU(32)

        self.conv_3 = StreamConv(32, 32, [2, 3], [1, 1], [0, 1])
        self.bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_3 = nn.PReLU(32)

        self.conv_4 = StreamConv(32, 64, [2, 3], [1, 1], [0, 1])
        self.bn_4 = nn.BatchNorm2d(64, eps=1e-8)
        self.act_4 = nn.PReLU(64)

        self.conv_5 = StreamConv(64, 128, [2, 3], [1, 1], [0, 1])
        self.bn_5 = nn.BatchNorm2d(128, eps=1e-8)
        self.act_5 = nn.PReLU(128)

    def forward(self,x,cache,cache1,cache2,cache3,cache4):
        #x.shape = (Bs, F, T, 2)
        x = x.permute(0,3,2,1) #(Bs, 2, T, F)
        x = x.to(torch.float32)
        # x = self.flc(x)
        # x = self.ln(x)
        x_1, cache_out = self.conv_1(x,cache)
        x_1 = self.act_1(self.bn_1(x_1))
        x_2, cache1_out = self.conv_2(x_1, cache1)
        x_2 = self.act_2(self.bn_2(x_2))
        x_3, cache2_out = self.conv_3(x_2, cache2)
        x_3 = self.act_3(self.bn_3(x_3))
        x_4, cache3_out = self.conv_4(x_3, cache3)
        x_4 = self.act_4(self.bn_4(x_4))
        x_5, cache4_out = self.conv_5(x_4, cache4)
        x_5 = self.act_5(self.bn_5(x_5))
        return [x_1,x_2,x_3,x_4,x_5], cache_out, cache1_out, cache2_out, cache3_out, cache4_out

'''
Decoder
'''
class Real_Decoder(nn.Module):
    def __init__(self, auto_encoder=True):
        super(Real_Decoder, self).__init__()
        self.auto_encoder = auto_encoder

        self.real_dconv_1 = nn.ConvTranspose2d(256, 64, kernel_size=(2,3), stride=(1,1))
        self.real_bn_1 = nn.BatchNorm2d(64, eps=1e-8)
        self.real_act_1 = nn.PReLU(64)

        self.real_dconv_2 = nn.ConvTranspose2d(128, 32, kernel_size=(2,3), stride=(1,1))
        self.real_bn_2 = nn.BatchNorm2d(32, eps=1e-8)
        self.real_act_2 = nn.PReLU(32)

        self.real_dconv_3 = nn.ConvTranspose2d(64, 32, kernel_size=(2,3), stride=(1,1))
        self.real_bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.real_act_3 = nn.PReLU(32)

        self.real_dconv_4 = nn.ConvTranspose2d(64, 32, kernel_size=(2,3), stride=(1,2))
        self.real_bn_4 = nn.BatchNorm2d(32, eps=1e-8)
        self.real_act_4 = nn.PReLU(32)

        self.real_dconv_5 = nn.ConvTranspose2d(64, 1, kernel_size=(2,5), stride=(1,2))
        self.real_bn_5 = nn.BatchNorm2d(1, eps=1e-8)
        self.real_act_5 = nn.PReLU(1)

        #--------------------------------------------------------------

    def forward(self, dprnn_out, encoder_out):
        skipcon_1 = torch.cat([encoder_out[4],dprnn_out],1)
        x_1 = self.real_act_1(self.real_bn_1(self.real_dconv_1(skipcon_1)[:,:,:-1,:-2]))
        skipcon_2 = torch.cat([encoder_out[3],x_1],1)
        x_2 = self.real_act_2(self.real_bn_2(self.real_dconv_2(skipcon_2)[:,:,:-1,:-2]))
        skipcon_3 = torch.cat([encoder_out[2],x_2],1)
        x_3 = self.real_act_3(self.real_bn_3(self.real_dconv_3(skipcon_3)[:,:,:-1,:-2]))
        skipcon_4 = torch.cat([encoder_out[1],x_3],1)
        x_4 = self.real_act_4(self.real_bn_4(self.real_dconv_4(skipcon_4)[:,:,:-1,:-1]))
        skipcon_5 = torch.cat([encoder_out[0],x_4],1)
        x_5 = self.real_act_5(self.real_bn_5(self.real_dconv_5(skipcon_5)[:,:,:-1,:-2]))


        return x_5


'''
Decoder
'''
class Real_Decoder_RT(nn.Module):
    def __init__(self, auto_encoder=True):
        super(Real_Decoder_RT, self).__init__()
        self.auto_encoder = auto_encoder

        self.real_dconv_1 = StreamConvTranspose(256, 64, [2, 3], [1, 1])
        self.real_bn_1 = nn.BatchNorm2d(64, eps=1e-8)
        self.real_act_1 = nn.PReLU(64)

        self.real_dconv_2 = StreamConvTranspose(128, 32, [2, 3], [1, 1])
        self.real_bn_2 = nn.BatchNorm2d(32, eps=1e-8)
        self.real_act_2 = nn.PReLU(32)

        self.real_dconv_3 = StreamConvTranspose(64, 32, [2, 3], [1, 1])
        self.real_bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.real_act_3 = nn.PReLU(32)

        self.real_dconv_4 = StreamConvTranspose(64, 32, [2, 3], [1, 2])
        self.real_bn_4 = nn.BatchNorm2d(32, eps=1e-8)
        self.real_act_4 = nn.PReLU(32)

        self.real_dconv_5 = StreamConvTranspose(64, 1, [2, 5], [1, 2])
        self.real_bn_5 = nn.BatchNorm2d(1, eps=1e-8)
        self.real_act_5 = nn.PReLU(1)

        #--------------------------------------------------------------

    def forward(self, dprnn_out, encoder_out,cache,cache1,cache2,cache3,cache4):
        skipcon_1 = torch.cat([encoder_out[4],dprnn_out],1)
        x_1, cache_out = self.real_dconv_1(skipcon_1, cache)
        x_1 = self.real_act_1(self.real_bn_1(x_1))[:, :, :, :-2]
        skipcon_2 = torch.cat([encoder_out[3],x_1],1)
        x_2, cache1_out = self.real_dconv_2(skipcon_2, cache1)
        x_2 = self.real_act_2(self.real_bn_2(x_2))[:, :, :, :-2]
        skipcon_3 = torch.cat([encoder_out[2],x_2],1)
        x_3, cache2_out = self.real_dconv_3(skipcon_3, cache2)
        x_3 = self.real_act_3(self.real_bn_3(x_3))[:, :, :, :-2]
        skipcon_4 = torch.cat([encoder_out[1],x_3],1)
        x_4, cache3_out = self.real_dconv_4(skipcon_4, cache3)
        x_4 = self.real_act_4(self.real_bn_4(x_4))[:, :, :, :-1]
        skipcon_5 = torch.cat([encoder_out[0],x_4],1)
        x_5, cache4_out = self.real_dconv_5(skipcon_5, cache4)
        x_5 = self.real_act_5(self.real_bn_5(x_5))[:, :, :, :-2]

        return x_5, cache_out, cache1_out, cache2_out, cache3_out, cache4_out

class Imag_Decoder(nn.Module):
    def __init__(self, auto_encoder=True):
        super(Imag_Decoder, self).__init__()
        self.auto_encoder = auto_encoder

        self.imag_dconv_1 = nn.ConvTranspose2d(256, 64, kernel_size=(2, 3), stride=(1, 1))
        self.imag_bn_1 = nn.BatchNorm2d(64, eps=1e-8)
        self.imag_act_1 = nn.PReLU(64)

        self.imag_dconv_2 = nn.ConvTranspose2d(128, 32, kernel_size=(2, 3), stride=(1, 1))
        self.imag_bn_2 = nn.BatchNorm2d(32, eps=1e-8)
        self.imag_act_2 = nn.PReLU(32)

        self.imag_dconv_3 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 3), stride=(1, 1))
        self.imag_bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.imag_act_3 = nn.PReLU(32)

        self.imag_dconv_4 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 3), stride=(1, 2))
        self.imag_bn_4 = nn.BatchNorm2d(32, eps=1e-8)
        self.imag_act_4 = nn.PReLU(32)

        self.imag_dconv_5 = nn.ConvTranspose2d(64, 1, kernel_size=(2, 5), stride=(1, 2))
        self.imag_bn_5 = nn.BatchNorm2d(1, eps=1e-8)
        self.imag_act_5 = nn.PReLU(1)
        #--------------------------------------------------------------

    def forward(self, dprnn_out, encoder_out):
        skipcon_1 = torch.cat([encoder_out[4],dprnn_out],1)
        x_1 = self.imag_act_1(self.imag_bn_1(self.imag_dconv_1(skipcon_1)[:,:,:-1,:-2]))
        skipcon_2 = torch.cat([encoder_out[3],x_1],1)
        x_2 = self.imag_act_2(self.imag_bn_2(self.imag_dconv_2(skipcon_2)[:,:,:-1,:-2]))
        skipcon_3 = torch.cat([encoder_out[2],x_2],1)
        x_3 = self.imag_act_3(self.imag_bn_3(self.imag_dconv_3(skipcon_3)[:,:,:-1,:-2]))
        skipcon_4 = torch.cat([encoder_out[1],x_3],1)
        x_4 = self.imag_act_4(self.imag_bn_4(self.imag_dconv_4(skipcon_4)[:,:,:-1,:-1]))
        skipcon_5 = torch.cat([encoder_out[0],x_4],1)
        x_5 = self.imag_act_5(self.imag_bn_5(self.imag_dconv_5(skipcon_5)[:,:,:-1,:-2]))

        return x_5

class Imag_Decoder_RT(nn.Module):
    def __init__(self, auto_encoder=True):
        super(Imag_Decoder_RT, self).__init__()
        self.auto_encoder = auto_encoder

        self.imag_dconv_1 = StreamConvTranspose(256, 64, [2, 3], [1, 1])
        self.imag_bn_1 = nn.BatchNorm2d(64, eps=1e-8)
        self.imag_act_1 = nn.PReLU(64)

        self.imag_dconv_2 = StreamConvTranspose(128, 32, [2, 3], [1, 1])
        self.imag_bn_2 = nn.BatchNorm2d(32, eps=1e-8)
        self.imag_act_2 = nn.PReLU(32)

        self.imag_dconv_3 = StreamConvTranspose(64, 32, [2, 3], [1, 1])
        self.imag_bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.imag_act_3 = nn.PReLU(32)

        self.imag_dconv_4 = StreamConvTranspose(64, 32, [2, 3], [1, 2])
        self.imag_bn_4 = nn.BatchNorm2d(32, eps=1e-8)
        self.imag_act_4 = nn.PReLU(32)

        self.imag_dconv_5 = StreamConvTranspose(64, 1, [2, 5], [1, 2])
        self.imag_bn_5 = nn.BatchNorm2d(1, eps=1e-8)
        self.imag_act_5 = nn.PReLU(1)
        #--------------------------------------------------------------

    def forward(self, dprnn_out, encoder_out,cache,cache1,cache2,cache3,cache4):

        skipcon_1 = torch.cat([encoder_out[4], dprnn_out], 1)
        x_1, cache_out = self.imag_dconv_1(skipcon_1, cache)
        x_1 = self.imag_act_1(self.imag_bn_1(x_1))[:, :, :, :-2]
        skipcon_2 = torch.cat([encoder_out[3], x_1], 1)
        x_2, cache1_out = self.imag_dconv_2(skipcon_2, cache1)
        x_2 = self.imag_act_2(self.imag_bn_2(x_2))[:, :, :, :-2]
        skipcon_3 = torch.cat([encoder_out[2], x_2], 1)
        x_3, cache2_out = self.imag_dconv_3(skipcon_3, cache2)
        x_3 = self.imag_act_3(self.imag_bn_3(x_3))[:, :, :, :-2]
        skipcon_4 = torch.cat([encoder_out[1], x_3], 1)
        x_4, cache3_out = self.imag_dconv_4(skipcon_4, cache3)
        x_4 = self.imag_act_4(self.imag_bn_4(x_4))[:, :, :, :-1]
        skipcon_5 = torch.cat([encoder_out[0], x_4], 1)
        x_5, cache4_out = self.imag_dconv_5(skipcon_5, cache4)
        x_5 = self.imag_act_5(self.imag_bn_5(x_5))[:, :, :, :-2]

        return x_5, cache_out, cache1_out, cache2_out, cache3_out, cache4_out

'''
DPCRN
'''

class DPCRN(nn.Module):
    #autoencoder = True
    def __init__(self):
        super(DPCRN,self).__init__()
        self.encoder = Encoder()
        self.dprnn_1 = DPRNN(128, 50, 128)
        self.real_decoder = Real_Decoder()
        self.imag_decoder = Imag_Decoder()


    def forward(self, x):
        # x --> audio batch
        # shape --> [Bs, sequence length]
        x_1 = x
        encoder_out = self.encoder(x_1)
        dprnn_out_1 = self.dprnn_1(encoder_out[4])
        # dprnn_out_2 = self.dprnn_2(dprnn_out_1)
        enh_real = self.real_decoder(dprnn_out_1, encoder_out)
        enh_imag = self.imag_decoder(dprnn_out_1, encoder_out)
        enh_real = enh_real.permute(0,3,2,1)
        enh_imag = enh_imag.permute(0,3,2,1)
        enh_stft = torch.cat([enh_real, enh_imag], -1)
        # enh_stft = self.mk_mask(x, torch.squeeze(enh_real), torch.squeeze(enh_imag))
        return enh_stft

class DPCRN_RT(nn.Module):
    #autoencoder = True
    def __init__(self):
        super(DPCRN_RT,self).__init__()
        self.encoder = Encoder_RT()
        self.dprnn_1 = DPRNN_RT(128, 50, 128)
        self.real_decoder = Real_Decoder_RT()
        self.imag_decoder = Imag_Decoder_RT()


    def forward(self, x,encoder_cache1,encoder_cache2,encoder_cache3,encoder_cache4
        ,encoder_cache5
        ,intra_hidden, inter_hidden
        ,realdec_cache1,realdec_cache2,realdec_cache3,realdec_cache4,realdec_cache5
        ,imagdec_cache1,imagdec_cache2,imagdec_cache3,imagdec_cache4,imagdec_cache5):
        # x --> audio batch
        # shape --> [Bs, sequence length]
        x_1 = x
        encoder_out,encoder_cache1_out,encoder_cache2_out,encoder_cache3_out,encoder_cache4_out,encoder_cache5_out = self.encoder(x_1,encoder_cache1,encoder_cache2,encoder_cache3,encoder_cache4,encoder_cache5)
        dprnn_out_1 , intra_hidden_out, inter_hidden_out = self.dprnn_1(encoder_out[4], intra_hidden, inter_hidden)
        # dprnn_out_2 = self.dprnn_2(dprnn_out_1)
        enh_real ,realdec_cache1_out,realdec_cache2_out,realdec_cache3_out,realdec_cache4_out,realdec_cache5_out = self.real_decoder(dprnn_out_1, encoder_out,realdec_cache1,realdec_cache2,realdec_cache3,realdec_cache4,realdec_cache5)
        enh_imag ,imagdec_cache1_out,imagdec_cache2_out,imagdec_cache3_out,imagdec_cache4_out,imagdec_cache5_out = self.imag_decoder(dprnn_out_1, encoder_out,imagdec_cache1,imagdec_cache2,imagdec_cache3,imagdec_cache4,imagdec_cache5)
        enh_real = enh_real.permute(0,3,2,1)
        enh_imag = enh_imag.permute(0,3,2,1)
        enh_stft = torch.cat([enh_real, enh_imag], -1)
        # enh_stft = self.mk_mask(x, torch.squeeze(enh_real), torch.squeeze(enh_imag))
        return enh_stft,encoder_cache1_out,encoder_cache2_out,encoder_cache3_out,encoder_cache4_out,encoder_cache5_out,intra_hidden_out, inter_hidden_out,realdec_cache1_out,realdec_cache2_out,realdec_cache3_out,realdec_cache4_out,realdec_cache5_out,imagdec_cache1_out,imagdec_cache2_out,imagdec_cache3_out,imagdec_cache4_out,imagdec_cache5_out

class DPCRNmask(nn.Module):
    #autoencoder = True
    def __init__(self):
        super(DPCRNmask,self).__init__()
        self.encoder = Encoder()
        self.dprnn_1 = DPRNN(128, 50, 128)
        self.real_decoder = Real_Decoder()
        self.imag_decoder = Imag_Decoder()

    def mk_mask(self, noisy_stft, mask_real, mask_imag):
        noisy_real = noisy_stft[:,:,:,0]
        noisy_imag = noisy_stft[:,:,:,1]

        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        return torch.stack((enh_real, enh_imag), -1)

    def forward(self, x):
        # x --> audio batch
        # shape --> [Bs, sequence length]
        x_1 = x
        encoder_out = self.encoder(x_1)
        dprnn_out_1 = self.dprnn_1(encoder_out[4])
        # dprnn_out_2 = self.dprnn_2(dprnn_out_1)
        enh_real = self.real_decoder(dprnn_out_1, encoder_out)
        enh_imag = self.imag_decoder(dprnn_out_1, encoder_out)
        enh_real = enh_real.permute(0,3,2,1)
        enh_imag = enh_imag.permute(0,3,2,1)
        # enh_stft = torch.cat([enh_real, enh_imag], -1)
        enh_stft = self.mk_mask(x, torch.squeeze(enh_real), torch.squeeze(enh_imag))
        return enh_stft

class DPCRNmask_RT(nn.Module):
    #autoencoder = True
    def __init__(self):
        super(DPCRNmask_RT,self).__init__()
        self.encoder = Encoder_RT()
        self.dprnn_1 = DPRNN_RT(128, 50, 128)
        self.real_decoder = Real_Decoder_RT()
        self.imag_decoder = Imag_Decoder_RT()

    def mk_mask(self, noisy_stft, mask_real, mask_imag):
        noisy_real = noisy_stft[:,:,:,0]
        noisy_imag = noisy_stft[:,:,:,1]

        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        return torch.stack((enh_real, enh_imag), -1)

    def forward(self, x,encoder_cache1,encoder_cache2,encoder_cache3,encoder_cache4
        ,encoder_cache5
        ,intra_hidden, inter_hidden
        ,realdec_cache1,realdec_cache2,realdec_cache3,realdec_cache4,realdec_cache5
        ,imagdec_cache1,imagdec_cache2,imagdec_cache3,imagdec_cache4,imagdec_cache5):
        # x --> audio batch
        # shape --> [Bs, sequence length]
        x_1 = x
        encoder_out,encoder_cache1_out,encoder_cache2_out,encoder_cache3_out,encoder_cache4_out,encoder_cache5_out = self.encoder(x_1,encoder_cache1,encoder_cache2,encoder_cache3,encoder_cache4,encoder_cache5)
        dprnn_out_1 , intra_hidden_out, inter_hidden_out = self.dprnn_1(encoder_out[4], intra_hidden, inter_hidden)
        # dprnn_out_2 = self.dprnn_2(dprnn_out_1)
        enh_real ,realdec_cache1_out,realdec_cache2_out,realdec_cache3_out,realdec_cache4_out,realdec_cache5_out = self.real_decoder(dprnn_out_1, encoder_out,realdec_cache1,realdec_cache2,realdec_cache3,realdec_cache4,realdec_cache5)
        enh_imag ,imagdec_cache1_out,imagdec_cache2_out,imagdec_cache3_out,imagdec_cache4_out,imagdec_cache5_out = self.imag_decoder(dprnn_out_1, encoder_out,imagdec_cache1,imagdec_cache2,imagdec_cache3,imagdec_cache4,imagdec_cache5)
        enh_real = enh_real.permute(0,3,2,1)
        enh_imag = enh_imag.permute(0,3,2,1)
        # enh_stft = torch.cat([enh_real, enh_imag], -1)
        enh_stft = self.mk_mask(x, torch.squeeze(enh_real), torch.squeeze(enh_imag))
        return enh_stft,encoder_cache1_out,encoder_cache2_out,encoder_cache3_out,encoder_cache4_out,encoder_cache5_out,intra_hidden_out, inter_hidden_out,realdec_cache1_out,realdec_cache2_out,realdec_cache3_out,realdec_cache4_out,realdec_cache5_out,imagdec_cache1_out,imagdec_cache2_out,imagdec_cache3_out,imagdec_cache4_out,imagdec_cache5_out


def init_hidden(D,bs,hs):
    init_h = torch.zeros(D, bs, hs)
    init_c = torch.zeros(D, bs, hs)
    return (init_h, init_c)


if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(1,201,3,2)
    model = DPCRN()
    model1 = DPCRN_RT()

    # model_dict = model.state_dict()
    # for k in model_dict:
    #     if 'weight' in k:
    #         print(k)
    enc = Encoder().eval()
    enc1 = Encoder_RT().eval()
    dec1 = Real_Decoder_RT().eval()
    Stream_dict = enc1.state_dict()
    Conv_dict = enc.state_dict()
    model_dict = model.state_dict()
    model1_dict = model1.state_dict()
    for k in model1_dict:
        print(k)
    Stream_dict["conv_1.Conv2d.weight"] = Conv_dict["conv_1.weight"]
    Stream_dict["conv_1.Conv2d.bias"] = Conv_dict["conv_1.bias"]
    Stream_dict["bn_1.weight"] = Conv_dict["bn_1.weight"]
    Stream_dict["bn_1.bias"] = Conv_dict["bn_1.bias"]
    Stream_dict["bn_1.running_mean"] = Conv_dict["bn_1.running_mean"]
    Stream_dict["bn_1.running_var"] = Conv_dict["bn_1.running_var"]
    Stream_dict["bn_1.num_batches_tracked"] = Conv_dict["bn_1.num_batches_tracked"]
    Stream_dict["act_1.weight"] = Conv_dict["act_1.weight"]
    Stream_dict["conv_2.Conv2d.weight"] = Conv_dict["conv_2.weight"]
    Stream_dict["conv_2.Conv2d.bias"] = Conv_dict["conv_2.bias"]
    Stream_dict["bn_2.weight"] = Conv_dict["bn_2.weight"]
    Stream_dict["bn_2.bias"] = Conv_dict["bn_2.bias"]
    Stream_dict["bn_2.running_mean"] = Conv_dict["bn_2.running_mean"]
    Stream_dict["bn_2.running_var"] = Conv_dict["bn_2.running_var"]
    Stream_dict["bn_2.num_batches_tracked"] = Conv_dict["bn_2.num_batches_tracked"]
    Stream_dict["act_2.weight"] = Conv_dict["act_2.weight"]
    Stream_dict["conv_3.Conv2d.weight"] = Conv_dict["conv_3.weight"]
    Stream_dict["conv_3.Conv2d.bias"] = Conv_dict["conv_3.bias"]
    Stream_dict["bn_3.weight"] = Conv_dict["bn_3.weight"]
    Stream_dict["bn_3.bias"] = Conv_dict["bn_3.bias"]
    Stream_dict["bn_3.running_mean"] = Conv_dict["bn_3.running_mean"]
    Stream_dict["bn_3.running_var"] = Conv_dict["bn_3.running_var"]
    Stream_dict["bn_3.num_batches_tracked"] = Conv_dict["bn_3.num_batches_tracked"]
    Stream_dict["act_3.weight"] = Conv_dict["act_3.weight"]
    Stream_dict["conv_4.Conv2d.weight"] = Conv_dict["conv_4.weight"]
    Stream_dict["conv_4.Conv2d.bias"] = Conv_dict["conv_4.bias"]
    Stream_dict["bn_4.weight"] = Conv_dict["bn_4.weight"]
    Stream_dict["bn_4.bias"] = Conv_dict["bn_4.bias"]
    Stream_dict["bn_4.running_mean"] = Conv_dict["bn_4.running_mean"]
    Stream_dict["bn_4.running_var"] = Conv_dict["bn_4.running_var"]
    Stream_dict["bn_4.num_batches_tracked"] = Conv_dict["bn_4.num_batches_tracked"]
    Stream_dict["act_4.weight"] = Conv_dict["act_4.weight"]
    Stream_dict["conv_5.Conv2d.weight"] = Conv_dict["conv_5.weight"]
    Stream_dict["conv_5.Conv2d.bias"] = Conv_dict["conv_5.bias"]
    Stream_dict["bn_5.weight"] = Conv_dict["bn_5.weight"]
    Stream_dict["bn_5.bias"] = Conv_dict["bn_5.bias"]
    Stream_dict["bn_5.running_mean"] = Conv_dict["bn_5.running_mean"]
    Stream_dict["bn_5.running_var"] = Conv_dict["bn_5.running_var"]
    Stream_dict["bn_5.num_batches_tracked"] = Conv_dict["bn_5.num_batches_tracked"]
    Stream_dict["act_5.weight"] = Conv_dict["act_5.weight"]
    
    enc1.load_state_dict(Stream_dict)
    with torch.no_grad():
        z = enc(x)
        # print(z[1])
        print(z[1].shape)
        cache = torch.zeros([1, 2, 1, 201])
        cache1 = torch.zeros([1, 32, 1, 100])
        cache2 = torch.zeros([1, 32, 1, 50])
        cache3 = torch.zeros([1, 32, 1, 50])
        cache4 = torch.zeros([1, 64, 1, 50])
        cache5 = torch.zeros([1, 256, 1, 50])
        cache6 = torch.zeros([1, 128, 1, 50])
        cache7 = torch.zeros([1, 64, 1, 50])
        cache8 = torch.zeros([1, 64, 1, 50])
        cache9 = torch.zeros([1, 64, 1, 100])
        cache10 = torch.zeros([1, 256, 1, 50])
        cache11 = torch.zeros([1, 128, 1, 50])
        cache12 = torch.zeros([1, 64, 1, 50])
        cache13 = torch.zeros([1, 64, 1, 50])
        cache14 = torch.zeros([1, 64, 1, 100])
        # Streaming
        print("Streaming")
        crn1 = torch.zeros([1, 128, 1, 50])
        intra_init_hidden = init_hidden(2, 1, 64)
        inter_init_hidden = init_hidden(1, 50, 128)
        for i in range(3):
            # z1, cache, cache1, cache2, cache3, cache4 = enc1(x[:, :, i:i + 1], cache, cache1, cache2, cache3, cache4)
            # print(z1[1].shape)
            # print(z1[1])
            # z2,cache5, cache6, cache7, cache8, cache9 = dec1(crn1,z1,cache5,cache6, cache7, cache8, cache9)
            # print(z2.shape)
            output1, cache,cache1,cache2,cache3,cache4,intra_init_hidden,inter_init_hidden,cache5,cache6,cache7,cache8,cache9,cache10,cache11,cache12,cache13,cache14 = model1(x[:, :, i:i + 1],cache,cache1,cache2,cache3,cache4,intra_init_hidden,inter_init_hidden,cache5,cache6,cache7,cache8,cache9,cache10,cache11,cache12,cache13,cache14)
            print(output1.shape)


