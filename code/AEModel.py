"""
  @ Author       : Ailovejinx
  @ Date         : 2023-04-13 09:44:59
  @ LastEditors  : Ailovejinx
  @ LastEditTime : 2023-04-15 14:42:27
  @ FilePath     : AEModel.py
  @ Description  : 
  @ Copyright (c) 2023 by Ailovejinx, All Rights Reserved. 
"""
import torch
import torch.nn as nn

import PCEncoderDecoder

from comm import is_main_process

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, chkpt_file=None):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bneck_size = encoder.bneck_size
        self.load_chkpt(chkpt_file=chkpt_file)

    def load_chkpt(self, chkpt_file=None):
        if(chkpt_file is not None and chkpt_file is not ""):
            if is_main_process: print("=> loading checkpoint '{}'".format(chkpt_file))
            checkpoint = torch.load(chkpt_file)
            self.load_state_dict(checkpoint['state_dict'])
            if is_main_process(): print("=> loaded checkpoint '{}' (epoch {})"
                  .format(chkpt_file, checkpoint['epoch']))

    def forward(self, X):
        return self.decoder(self.encoder(X))

    def encode(self, X):
        return self.encoder(X)

    def decode(self, X):
        return self.decoder(X)


class PCAutoEncoder(AutoEncoder):
    '''
    An Auto-Encoder for point-clouds.
    '''
    def __init__(self, bneck_size=128, chkpt_file=None):
        self.input_size = 2048
        encoder = PCEncoderDecoder.Encoder(
            bneck_size=bneck_size, input_size=self.input_size)
        decoder = PCEncoderDecoder.Decoder(bneck_size=bneck_size)
        super().__init__(encoder, decoder, chkpt_file=chkpt_file)
