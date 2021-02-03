import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from collections import OrderedDict
import copy


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


def create_model_layers(layers_dict, in_ch, out_ch, power):
    # if power == 1:  # end condition
    layers_dict['d_1__EN_stage_N1'] = RSU7(in_ch, 16, 64)
    layers_dict['d_1__EN_pool_N12'] = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    layers_dict['d_1__EN_stage_N2'] = RSU6(64, 16, 64)
    layers_dict['d_1__EN_pool_N23'] = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    layers_dict['d_1__EN_stage_N3'] = RSU5(64, 16, 64)
    layers_dict['d_1__EN_pool_N34'] = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    layers_dict['d_1__EN_stage_N4'] = RSU4(64, 16, 64)
    layers_dict['d_1__EN_pool_N45'] = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    layers_dict['d_1__EN_stage_N5'] = RSU4F(64, 16, 64)
    layers_dict['d_1__EN_pool_N56'] = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    layers_dict['d_1__bottom_stage_N6'] = RSU4F(64, 16, 64)
    # decoder
    layers_dict['d_1__DE_stage_N5'] = RSU4F(128, 16, 64)
    layers_dict['d_1__DE_stage_N4'] = RSU4(128, 16, 64)
    layers_dict['d_1__DE_stage_N3'] = RSU5(128, 16, 64)
    layers_dict['d_1__DE_stage_N2'] = RSU6(128, 16, 64)
    layers_dict['d_1__DE_stage_N1'] = RSU7(128, 16, 64)
    # return layers_dict

    # else not at the end
    layers_dict['d_2__side_N1'] = nn.Conv2d(64, out_ch, 3, padding=1)
    layers_dict['d_2__side_N2'] = nn.Conv2d(64, out_ch, 3, padding=1)
    layers_dict['d_2__side_N3'] = nn.Conv2d(64, out_ch, 3, padding=1)
    layers_dict['d_2__side_N4'] = nn.Conv2d(64, out_ch, 3, padding=1)
    layers_dict['d_2__side_N5'] = nn.Conv2d(64, out_ch, 3, padding=1)
    layers_dict['d_2__side_N6'] = nn.Conv2d(64, out_ch, 3, padding=1)

    layers_dict['d_2__outconv'] = nn.Conv2d(6, out_ch, 1)


### U^2-Net dynamic small ###
class U2NETPDyn(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, power=2):
        super(U2NETPDyn, self).__init__()
        self.layers_dict = OrderedDict()

        create_model_layers(self.layers_dict, in_ch=3, out_ch=1, power=2)
        for cur_layer_key in self.layers_dict:
            exec(f'self.{cur_layer_key} = self.layers_dict[cur_layer_key]')

    def forward(self, x):
        hx = x
        save_output_dict = dict()
        for cur_layer_key in self.layers_dict:
            # print(cur_layer_key)
            if 'EN_stage' in cur_layer_key:
                cur_output_key = 'hx' + cur_layer_key.split('_N')[1]
                # print('output_' + cur_output_key)
                exec(f'save_output_dict[cur_output_key] = self.{cur_layer_key}(hx)')
            if 'EN_pool' in cur_layer_key:
                cur_input_key = 'hx' + cur_layer_key.split('_N')[1][0]
                # print('input_' + cur_input_key)
                hx = eval(f'self.{cur_layer_key}')(save_output_dict[cur_input_key])

            if 'bottom' in cur_layer_key:
                cur_output_key = 'hx' + cur_layer_key.split('_N')[1] + 'd'
                # print('output_' + cur_output_key)
                exec(f'save_output_dict[cur_output_key] = self.{cur_layer_key}(hx)')

            if 'DE_stage' in cur_layer_key:
                prev_input_layer_key = 'hx' + str(int(cur_layer_key.split('_N')[1]) + 1) + 'd'
                cur_input_layer_key = 'hx' + cur_layer_key.split('_N')[1]
                cur_upsampled_layer_key = 'hx' + str(int(cur_layer_key.split('_N')[1]) + 1) + 'up'
                save_output_dict[cur_upsampled_layer_key] = _upsample_like(save_output_dict[prev_input_layer_key],
                                                                           save_output_dict[cur_input_layer_key])
                cur_output_layer_key = 'hx' + str(cur_layer_key.split('_N')[1]) + 'd'
                save_output_dict[cur_output_layer_key] = eval(f'self.{cur_layer_key}')(
                    torch.cat((save_output_dict[cur_upsampled_layer_key], save_output_dict[cur_input_layer_key]), 1))

                # print(save_output_dict[cur_output_layer_key].shape)

            if 'side' in cur_layer_key:
                cur_input_layer_key = 'hx' + cur_layer_key.split('_N')[1] + 'd'
                cur_output_layer_key = 'd' + cur_layer_key.split('_N')[1]
                prev_output_layer_key = 'd' + str(int(cur_layer_key.split('_N')[1]) - 1)
                save_output_dict[cur_output_layer_key] = eval(f'self.{cur_layer_key}')(
                    save_output_dict[cur_input_layer_key])
                if int(cur_layer_key.split('_N')[1]) > 1:
                    save_output_dict[cur_output_layer_key] = _upsample_like(save_output_dict[cur_output_layer_key],
                                                                            save_output_dict[prev_output_layer_key])

        d1 = save_output_dict['d1']
        d2 = save_output_dict['d2']
        d3 = save_output_dict['d3']
        d4 = save_output_dict['d4']
        d5 = save_output_dict['d5']
        d6 = save_output_dict['d6']

        d0 = self.d_2__outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)


''' Unnet with class '''

''' Unnet with class '''


### U^x-Net small ###
class UxNETP(nn.Module):

    def __init__(self, unet_inner_encoder_layer_list=[], unet_inner_bottom_layer_list=[],
                 unet_inner_decoder_layer_list=[], cur_power=10):
        super(UxNETP, self).__init__()
        self.net_depth = len(unet_inner_encoder_layer_list)
        self.encoder_layers_name_list = []
        self.encoder_pooling_layers_name_list = []
        self.bottom_layers_name_list = []
        self.decoder_layers_name_list = []
        self.cur_power = cur_power

        # print(f'uxnet init - power:{cur_power}')

        for ii, cur_layer in enumerate(unet_inner_encoder_layer_list):
            exec(f'self.stage{ii + 1} = cur_layer')
            exec(f'self.pool{ii + 1}{ii + 2} = nn.MaxPool2d(2, stride=2, ceil_mode=True)')
            self.encoder_layers_name_list.append(f'stage{ii + 1}')
            self.encoder_pooling_layers_name_list.append(f'pool{ii + 1}{ii + 2}')

        exec(f'self.stage{ii + 2} = unet_inner_bottom_layer_list[0]')
        self.bottom_layers_name_list.append(f'stage{ii + 2}')

        for ii, cur_layer in zip(sorted(list(range(self.net_depth)), reverse=True), unet_inner_decoder_layer_list):
            exec(f'self.stage{ii + 1}d = cur_layer')
            self.decoder_layers_name_list.append(f'stage{ii + 1}d')

    def forward(self, x):
        # print(f'uxnet forward - cur_power:{self.cur_power}')
        hx = x
        save_output_dict = dict()
        # print(f'len of U: {len(self.encoder_layers_name_list)}')
        for ii, (cur_enc_layer_name, cur_enc_pool_layer_name) in enumerate(
                zip(self.encoder_layers_name_list, self.encoder_pooling_layers_name_list)):
            cur_output_key = 'hx' + str(ii + 1)
            # print(eval(f'self.{cur_enc_layer_name}'))
            save_output_dict[cur_output_key] = eval(f'self.{cur_enc_layer_name}(hx)')
            # print(f'Enc - Power: {self.cur_power}, {cur_output_key} = {cur_enc_layer_name}(hx) -- {save_output_dict[cur_output_key].shape} <- {hx.shape}')

            hx = eval(f'self.{cur_enc_pool_layer_name}(save_output_dict[cur_output_key])')
            # print(f'Enc - Power: {self.cur_power}, hx = {cur_enc_pool_layer_name}({cur_output_key}) -- {hx.shape} <- {save_output_dict[cur_output_key].shape}')

        for cur_bottom_layer_name in self.bottom_layers_name_list:
            cur_output_key = 'hx' + str(ii + 2) + 'd'
            save_output_dict[cur_output_key] = eval(f'self.{cur_bottom_layer_name}(hx)')
            # print(f'Bot - Power: {self.cur_power}, {cur_output_key} = {cur_bottom_layer_name}(hx) -- {save_output_dict[cur_output_key].shape} <- {hx.shape}')


        for ii, cur_dec_layer_name in zip(sorted(list(range(self.net_depth)), reverse=True),
                                          self.decoder_layers_name_list):
            prev_layer_name = 'hx' + str(ii + 2) + 'd'
            prev_layer_up_sample_name = 'hx' + str(ii + 2) + 'up'
            cur_layer_input_name = 'hx' + str(ii + 1)
            cur_layer_output_name = 'hx' + str(ii + 1) + 'd'

            save_output_dict[prev_layer_up_sample_name] = _upsample_like(save_output_dict[prev_layer_name],
                                                                         save_output_dict[cur_layer_input_name])
            save_output_dict[cur_layer_output_name] = eval(f'self.{cur_dec_layer_name}')(
                torch.cat((save_output_dict[prev_layer_up_sample_name], save_output_dict[cur_layer_input_name]), 1))

            # print(f'Dec - Power: {self.cur_power}, {prev_layer_up_sample_name} = _upsample_like({prev_layer_name}, {cur_layer_input_name}) -- {save_output_dict[prev_layer_up_sample_name].shape}')
            # print(f'Dec - Power: {self.cur_power}, {cur_layer_output_name} = {cur_dec_layer_name}(cat({prev_layer_up_sample_name},{cur_layer_input_name})) -- {save_output_dict[cur_layer_output_name].shape}')


        return save_output_dict[cur_layer_output_name] + save_output_dict[cur_layer_output_name[:-1]]


class UxNETP_TOP(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, unet_inner_encoder_layer_list=[], unet_inner_bottom_layer_list=[],
                 unet_inner_decoder_layer_list=[],cur_power=10):
        super(UxNETP_TOP, self).__init__()
        self.net_depth = len(unet_inner_encoder_layer_list)
        self.encoder_layers_name_list = []
        self.encoder_pooling_layers_name_list = []
        self.bottom_layers_name_list = []
        self.decoder_layers_name_list = []
        self.cur_power = cur_power

        # print(f'uxnet top init - power:{cur_power}')

        for ii, cur_layer in enumerate(unet_inner_encoder_layer_list):
            exec(f'self.stage{ii + 1} = cur_layer')
            exec(f'self.pool{ii + 1}{ii + 2} = nn.MaxPool2d(2, stride=2, ceil_mode=True)')
            self.encoder_layers_name_list.append(f'stage{ii + 1}')
            self.encoder_pooling_layers_name_list.append(f'pool{ii + 1}{ii + 2}')

        exec(f'self.stage{ii + 2} = unet_inner_bottom_layer_list[0]')
        self.bottom_layers_name_list.append(f'stage{ii + 2}')

        for ii, cur_layer in zip(sorted(list(range(self.net_depth)), reverse=True), unet_inner_decoder_layer_list):
            exec(f'self.stage{ii + 1}d = cur_layer')
            self.decoder_layers_name_list.append(f'stage{ii + 1}d')

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        # print(f'uxnet top forward - cur_power:{self.cur_power}')

        hx = x
        save_output_dict = dict()

        for ii, (cur_enc_layer_name, cur_enc_pool_layer_name) in enumerate(
                zip(self.encoder_layers_name_list, self.encoder_pooling_layers_name_list)):
            cur_output_key = 'hx' + str(ii + 1)
            save_output_dict[cur_output_key] = eval(f'self.{cur_enc_layer_name}(hx)')
            hx = eval(f'self.{cur_enc_pool_layer_name}(save_output_dict[cur_output_key])')

        for cur_bottom_layer_name in self.bottom_layers_name_list:
            cur_output_key = 'hx' + str(ii + 2) + 'd'
            save_output_dict[cur_output_key] = eval(f'self.{cur_bottom_layer_name}(hx)')

        for ii, cur_dec_layer_name in zip(sorted(list(range(self.net_depth)), reverse=True),
                                          self.decoder_layers_name_list):
            prev_layer_name = 'hx' + str(ii + 2) + 'd'
            prev_layer_up_sample_name = 'hx' + str(ii + 2) + 'up'
            cur_layer_input_name = 'hx' + str(ii + 1)
            cur_layer_output_name = 'hx' + str(ii + 1) + 'd'

            save_output_dict[prev_layer_up_sample_name] = _upsample_like(save_output_dict[prev_layer_name],
                                                                         save_output_dict[cur_layer_input_name])
            save_output_dict[cur_layer_output_name] = eval(f'self.{cur_dec_layer_name}')(
                torch.cat((save_output_dict[prev_layer_up_sample_name], save_output_dict[cur_layer_input_name]), 1))

        # side output
        d1 = self.side1(save_output_dict['hx1d'])

        d2 = self.side2(save_output_dict['hx2d'])
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(save_output_dict['hx3d'])
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(save_output_dict['hx4d'])
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(save_output_dict['hx5d'])
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(save_output_dict['hx6d'])
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)


def create_single_unnet_layers(unet_inner_encoder_layer_list, unet_inner_bottom_layer_list,
                               unet_inner_decoder_layer_list, cur_power):
    unet_depth = len(unet_inner_encoder_layer_list)

    # create inside unet nodes
    cur_unet_encoder_layer_list = []
    cur_unet_bottom_layer_list = []
    cur_unet_decoder_layer_list = []

    ''' create encoder layers'''
    for ii in range(unet_depth):
        if ii == 0:
            cur_node = UxNETP(unet_inner_encoder_layer_list=copy.deepcopy(unet_inner_encoder_layer_list),
                              unet_inner_bottom_layer_list=copy.deepcopy(unet_inner_bottom_layer_list),
                              unet_inner_decoder_layer_list=copy.deepcopy(unet_inner_decoder_layer_list),
                              cur_power=cur_power)
        else:
            cur_node = UxNETP(unet_inner_encoder_layer_list=copy.deepcopy(unet_inner_encoder_layer_list[ii:]),
                              unet_inner_bottom_layer_list=copy.deepcopy(unet_inner_bottom_layer_list),
                              unet_inner_decoder_layer_list=copy.deepcopy(unet_inner_decoder_layer_list[:-ii]),
                              cur_power=cur_power)
        cur_unet_encoder_layer_list.append(cur_node)

    ''' create bottom layer '''
    cur_node = copy.deepcopy(cur_unet_encoder_layer_list[-1])
    cur_unet_bottom_layer_list.append(cur_node)


    ''' create decoder layers '''
    # unet_inner_encoder_layer_list = copy.deepcopy(unet_inner_decoder_layer_list)
    # unet_inner_encoder_layer_list.reverse()
    for ii in range(unet_depth):
        if ii == 0:
            unet_inner_encoder_layer_list[0] = copy.deepcopy(unet_inner_decoder_layer_list[-1])
            cur_node = UxNETP(unet_inner_encoder_layer_list=unet_inner_encoder_layer_list,
                              unet_inner_bottom_layer_list=unet_inner_bottom_layer_list,
                              unet_inner_decoder_layer_list=unet_inner_decoder_layer_list,
                              cur_power=cur_power)
        else:
            unet_inner_encoder_layer_list[ii] = copy.deepcopy(unet_inner_decoder_layer_list[-ii-1])
            cur_node = UxNETP(unet_inner_encoder_layer_list=unet_inner_encoder_layer_list[ii:],
                              unet_inner_bottom_layer_list=unet_inner_bottom_layer_list,
                              unet_inner_decoder_layer_list=unet_inner_decoder_layer_list[:-ii],
                              cur_power=cur_power)
        cur_unet_decoder_layer_list.append(cur_node)

    cur_unet_decoder_layer_list.reverse()

    return cur_unet_encoder_layer_list, cur_unet_bottom_layer_list, cur_unet_decoder_layer_list


def create_unnet(power=3):
    in_ch = 3
    out_ch = 1
    cur_unet_encoder_layer_list, cur_unet_bottom_layer_list, cur_unet_decoder_layer_list = None, None, None
    for cur_power in range(2, power):
        print(f'cur_power: {cur_power}')
        if cur_power == 2:
            cur_unet_encoder_layer_list = [RSU7(in_ch, 16, 64),
                                           RSU6(64, 16, 64),
                                           RSU5(64, 16, 64),
                                           RSU4(64, 16, 64),
                                           RSU4F(64, 16, 64)]
            cur_unet_bottom_layer_list = [RSU4F(64, 16, 64)]
            cur_unet_decoder_layer_list = [RSU4F(128, 16, 64),
                                           RSU4(128, 16, 64),
                                           RSU5(128, 16, 64),
                                           RSU6(128, 16, 64),
                                           RSU7(128, 16, 64)]
            cur_unet_encoder_layer_list, cur_unet_bottom_layer_list, cur_unet_decoder_layer_list = create_single_unnet_layers(
                cur_unet_encoder_layer_list, cur_unet_bottom_layer_list, cur_unet_decoder_layer_list, cur_power)
        else:
            cur_unet_encoder_layer_list, cur_unet_bottom_layer_list, cur_unet_decoder_layer_list = create_single_unnet_layers(
                copy.deepcopy(cur_unet_encoder_layer_list),
                copy.deepcopy(cur_unet_bottom_layer_list),
                copy.deepcopy(cur_unet_decoder_layer_list), cur_power)

    top_Unet = UxNETP_TOP(in_ch=3, out_ch=1,
                          unet_inner_encoder_layer_list=cur_unet_encoder_layer_list,
                          unet_inner_bottom_layer_list=cur_unet_bottom_layer_list,
                          unet_inner_decoder_layer_list=cur_unet_decoder_layer_list,
                          cur_power=power)

    return top_Unet
