# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:32:17 2022

@author: Zhongshu.Hou & Qinwen.hu

Modules
"""
import torch
from torch import nn
import numpy as np
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)

'''
Import initialized SCM matrix
'''
# Sc = np.load('./SpecCompress.npy').astype(np.float32)

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

        self.ln = nn.InstanceNorm2d(2, eps=1e-8, affine=True)
        self.conv_1 = nn.Conv2d(2, 32, kernel_size=(2,5),stride=(1,2),padding=(1,1))
        self.bn_1 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_1 = nn.PReLU(32)

        self.conv_2 = nn.Conv2d(32,32,kernel_size=(2,3),stride=(1,2),padding=(1,1))
        self.bn_2 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_2 = nn.PReLU(32)

        self.conv_3 = nn.Conv2d(32,32,kernel_size=(2,3),stride=(1,1),padding=(1,1))
        self.bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_3 = nn.PReLU(32)

        self.conv_4 = nn.Conv2d(32,64,kernel_size=(2,3),stride=(1,1),padding=(1,1))
        self.bn_4 = nn.BatchNorm2d(64, eps=1e-8)
        self.act_4 = nn.PReLU(64)

        self.conv_5 = nn.Conv2d(64,128,kernel_size=(2,3),stride=(1,1),padding=(1,1))
        self.bn_5 = nn.BatchNorm2d(128, eps=1e-8)
        self.act_5 = nn.PReLU(128)

    def forward(self,x):
        #x.shape = (Bs, F, T, 2)
        x = x.permute(0,3,2,1) #(Bs, 2, T, F)
        x = x.to(torch.float32)
        # x = self.flc(x)
        x = self.ln(x)
        x_1 = self.act_1(self.bn_1(self.conv_1(x)[:,:,:-1,:]))
        x_2 = self.act_2(self.bn_2(self.conv_2(x_1)[:,:,:-1,:]))
        x_3 = self.act_3(self.bn_3(self.conv_3(x_2)[:,:,:-1,:]))
        x_4 = self.act_4(self.bn_4(self.conv_4(x_3)[:,:,:-1,:]))
        x_5 = self.act_5(self.bn_5(self.conv_5(x_4)[:,:,:-1,:]))
        return [x_1,x_2,x_3,x_4,x_5]

'''
DPRNN
'''
class DPRNN(nn.Module):
    def __init__(self, numUnits, width, channel, **kwargs):
        super(DPRNN, self).__init__(**kwargs)
        self.numUnits = numUnits

        self.intra_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits//2, batch_first = True, bidirectional = True)

        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)

        self.intra_ln = nn.InstanceNorm2d(width,eps=1e-8,affine=True)

        self.inter_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits, batch_first = True, bidirectional = False)

        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)

        self.inter_ln = nn.InstanceNorm2d(channel, eps=1e-8,affine=True)

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


class SNR_Estimator(nn.Module):
    def __init__(self, numUnits, insize ,width):
        super(SNR_Estimator, self).__init__()
        self.numUnits = numUnits
        self.insize = insize
        self.width = width
        self.LSTM = nn.LSTM(input_size=self.insize, hidden_size=self.numUnits, batch_first=True,
                                 bidirectional=False)
        self.conv = nn.Conv1d(self.width, 1, kernel_size=5,
                  stride=1, padding=2)
        #.fc = nn.Linear(self.width, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x.shape = (Bs, C, T, F)
        self.LSTM.flatten_parameters()

        x = x.permute(0, 3, 2, 1)  # (Bs, F, T, C)
        if not x.is_contiguous():
            x = x.contiguous()
        LSTM_input = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) #(Bs*F, T, C)
        inter_LSTM_out = self.LSTM(LSTM_input)[0]
        dense_in = inter_LSTM_out.view(x.shape[0], self.width, -1)  # (Bs, F, T)

        #dense_in = dense_in.permute(0,2,1)  # (Bs, T, F)
        dense_out = self.conv(dense_in)
        dense_out = self.act(dense_out)
        dense_out = torch.squeeze(dense_out)
        return dense_out

'''
DPCRN
'''

class DPCRN(nn.Module):
    #autoencoder = True
    def __init__(self):
        super(DPCRN,self).__init__()
        self.encoder = Encoder()
        self.dprnn_1 = DPRNN(128, 50, 128)
        self.dprnn_2 = DPRNN(128, 50, 128)
        self.real_decoder = Real_Decoder()
        self.imag_decoder = Imag_Decoder()
        self.estimator = SNR_Estimator(1,128,50)

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
        # snr_estimated = self.estimator(dprnn_out_2)
        enh_real = self.real_decoder(dprnn_out_1, encoder_out)
        enh_imag = self.imag_decoder(dprnn_out_1, encoder_out)
        enh_real = enh_real.permute(0,3,2,1)
        enh_imag = enh_imag.permute(0,3,2,1)
        enh_stft = torch.cat([enh_real, enh_imag], -1)
        # enh_stft = self.mk_mask(x, torch.squeeze(enh_real), torch.squeeze(enh_imag))
        return enh_stft

