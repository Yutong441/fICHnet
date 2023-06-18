import re
from functools import partial
import torch
from torch.nn import Parameter
from torchvision import models
from CNN.all_models import choose_models


def prevent_update(model, layer_regex):
    ''' prevent parameter updating'''
    with torch.no_grad():
        for m, n in model.named_parameters():
            if layer_regex in m:
                n.requires_grad = False


def load_encoder(model1, model2):
    ''' Load the weights of the encoder block of model1 from model2 '''
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    with torch.no_grad():
        # load the weights in the encoding block
        for m in state_dict1.keys():
            if 'encoder' in m:
                state_dict1[m] = state_dict2[m]
        model1.load_state_dict(state_dict1)
    prevent_update(model1, 'encoder')


def load_rnn_decoder(model1, model2):
    ''' Load the weights of the rnn layer of model1 from model2.
    Model1 decoder must be a fused decoder `CNN.decoder.CNN_decoder` '''
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    with torch.no_grad():
        # load the weights in the encoding block
        for m in state_dict1.keys():
            if 'decoder.decoder1' in m:
                m2 = re.sub('decoder1\\.', '', m)
                state_dict1[m] = state_dict2[m2]
        model1.load_state_dict(state_dict1)
    prevent_update(model1, 'decoder.decoder1')


def load_CNN_weight(model, cf):
    '''
    Load the weights of the image processing stream of a fusion model
    Args:
        `cf`: contain a `old_save_path` entry containing the path to the model
        with only image processing stream
    '''
    # create a new config wih image processing stream only
    cf_old = cf.copy()
    cf_old['output_features'] = None
    old_model = choose_models(cf_old)
    old_model.load_state_dict(torch.load(cf['old_save_path'],
                              map_location=torch.device('cpu')))
    load_encoder(model, old_model)
    load_rnn_decoder(model, old_model)


def rep_weight(x, rep_num, grad=True, img3D=True):
    '''
    Replicate weights along the depth dimension
    Args:
        `x`: a 4D tensor: [B, C, H, W]
        `rep_num`: how many times is `x` repeated
    Return: a 5D tensor: [B, C, D, H, W]
    '''
    if img3D:
        x_rep = torch.stack([x]*rep_num, axis=-1)
        return Parameter(x_rep.permute(0, 1, 4, 2, 3), requires_grad=grad)
    else:
        return Parameter(x, requires_grad=grad)

# --------------------Resnet--------------------
# manually load the weight of a pretrained resnet: for all kinds of resnet
def load_gate(model, pretrained, grad=True):
    with torch.no_grad ():
        # do not need to load the weight of the first layer
        model.encoder.gate[1].weight = Parameter (pretrained.bn1.weight,
                requires_grad=grad)
        model.encoder.gate[1].bias =  Parameter (pretrained.bn1.bias,
                requires_grad=grad)

def load_conv_subblock (model, pretrained, mi, i, j, rw):
    '''mi = layer, i = subblock, j= sub-subblock'''
    layer = 'layer{}'.format(mi+1)
    if hasattr(getattr (pretrained, layer)[i], 'conv'+str(j+1)):
        try:
            N = model.encoder.blocks[mi].blocks[i].blocks[2*j][0].weight.shape[2]
            model.encoder.blocks[mi].blocks[i].blocks[2*j][0].weight = rw(
                    getattr(getattr (pretrained, layer)[i], 
                        'conv'+str(j+1)).weight, N)
            return 'nobreak'
        except: return 'break'
    else: 
        print ('layer {}, subblock {}, sub-subblock {} does not exist'.format(
            mi, i, j))
        return 'break'

def load_conv_layer (model, pretrained, mi, grad=True, img3D=True):
    rw = partial (rep_weight, grad=grad, img3D=img3D)
    with torch.no_grad ():
        layer = 'layer{}'.format(mi+1)
        for i in range(len (getattr (pretrained, layer))):
            for j in range (3):
                breaking = load_conv_subblock (model, pretrained, mi, i, j, rw)
                if breaking == 'break': break

        # load short cut connection weight
        layer0 = getattr (pretrained, layer)[0]
        if hasattr (layer0, 'downsample'):
            downsample_layer = getattr (layer0, 'downsample')
            if downsample_layer is not None:
                N =model.encoder.blocks[mi].blocks[0].shortcut[0].weight.shape[2]
                model.encoder.blocks[mi].blocks[0].shortcut[0].weight = \
                        rw(downsample_layer[0].weight, N)

def load_bn_subblock (model, pretrained, mi, i, j, grad):
    '''mi = layer, i = subblock, j= sub-subblock'''
    layer = 'layer{}'.format(mi+1)
    if hasattr(getattr (pretrained, layer)[i], 'bn'+str(j+1)):
        try:
            rw = partial (Parameter, requires_grad=grad)
            model.encoder.blocks[mi].blocks[i].blocks[2*j][1].weight = rw(
                    getattr(getattr (pretrained, layer)[i], 
                        'bn'+str(j+1)).weight)
            model.encoder.blocks[mi].blocks[i].blocks[2*j][1].bias= rw(
                    getattr(getattr (pretrained, layer)[i], 
                        'bn'+str(j+1)).bias)
            return 'nobreak'
        except: return 'break'
    else: 
        print ('layer {}, subblock {}, sub-subblock {} does not exist'.format(
            mi, i, j))
        return 'break'

def load_bn_layer (model, pretrained, mi, grad=True):
    rw = partial (Parameter, requires_grad=grad)
    with torch.no_grad ():
        layer = 'layer{}'.format(mi+1)
        for i in range(len (getattr (pretrained, layer))):
            for j in range (3):
                breaking = load_bn_subblock (model, pretrained, mi, i, j,
                        grad=grad)
                if breaking == 'break': break

        # load short cut connection weight
        layer0 = getattr (pretrained, layer)[0]
        if hasattr (layer0, 'downsample'):
            downsample_layer = getattr (layer0, 'downsample')
            if downsample_layer is not None:
                model.encoder.blocks[mi].blocks[0].shortcut[1].weight= \
                        rw(downsample_layer[1].weight)
                model.encoder.blocks[mi].blocks[0].shortcut[1].bias= \
                        rw(downsample_layer[1].bias)

def load_resnet (model, resnet_type, grad=True, img3D=True):
    '''
    Load the pretrained weights of resnet into a 2D/3D resnet, except the
    first convolution layer and the last linear layer.
    Args:
        `model`: a 2D/3D resnet18 model
        `grad`: whether to still update the pretrained weights
        `img3D`: for 2D or 3D resnet
    '''
    resnet_torch = getattr (models, resnet_type)
    resnet_model = resnet_torch(pretrained=True)
    load_gate (model, resnet_model, grad=grad)
    for i in range(4):
        load_conv_layer (model, resnet_model, i, grad=grad, img3D=img3D)
        load_bn_layer (model, resnet_model, i, grad=grad)

# --------------------VGG--------------------
def load_vgg_layer (model, vgg_type, grad=True, img3D=True):
    vgg_torch = getattr (models, vgg_type)
    vgg_model = vgg_torch(pretrained=True)
    for index, layer in enumerate (vgg_model.features):
        if hasattr (layer, 'weight'):
            print (layer.weight.shape)
            if len(layer.weight.shape) ==4 and index != 0:
                N = model.features[index].weight.shape[2]
                model.features[index].weight = rep_weight (layer.weight, N,
                        grad=grad, img3D=img3D)
                model.features[index].bias = Parameter(layer.bias,
                        requires_grad=grad)
            elif len (layer.weight.shape) == 1:
                model.features[index].weight = Parameter(layer.weight,
                        requires_grad=grad)
                model.features[index].bias = Parameter(layer.bias,
                        requires_grad=grad)

# ----------Unet----------
def load_unet_conv_block (model, unet, block_name, grad=True):
    block_weight = getattr (unet, block_name)
    if block_name=='encoder1': load_layers = [2]
    else: load_layers = [1,2]
    for i in load_layers:
        layer_name = re.sub ('oder', '', block_name)+'conv'+str(i)
        layer_weight = getattr (block_weight, layer_name)
        N = getattr (getattr (model, block_name), layer_name).weight.shape[2]
        getattr (getattr (model, block_name), layer_name).weight = \
            rep_weight (layer_weight.weight, N, grad=grad)

def load_unet_bn_block (model, unet, block_name, grad=True):
    block_weight = getattr (unet, block_name)
    for i in [1,2]:
        layer_name = re.sub ('oder', '', block_name)+'norm'+str(i)
        layer_weight = getattr (block_weight, layer_name)
        getattr (getattr (model, block_name), layer_name).weight = \
            Parameter(layer_weight.weight, requires_grad=grad)
        getattr (getattr (model, block_name), layer_name).bias = \
            Parameter(layer_weight.bias, requires_grad=grad)

def load_unet (model, grad=True):
    from CNN.unet_2d import UNet
    unet = UNet (in_channels=3)
    unet_pt = torch.load ('CNN/unet.pt', map_location=torch.device('cpu'))
    unet.load_state_dict (unet_pt)

    all_blocks = ['encoder'+str(i) for i in range (1,5)]
    all_blocks.append ('bottleneck')
    all_blocks.extend(['decoder'+str(i) for i in range (1,5)])
    for one_block in all_blocks:
        load_unet_conv_block (model, unet, one_block, grad=grad)
        load_unet_bn_block (model, unet, one_block, grad=grad)

# --------------------Summary--------------------
def load_any_net (model, cf):
    if 'unet' in cf ['model_type']:
        load_unet (model, grad=cf ['train_weight'])
    elif 'resCRNN' in cf ['model_type']:
        restype='resnet'+re.sub ('resCRNN', '', cf['model_type'])
        if restype in ['resnet9', 'resnet14']: restype='resnet18'
        load_resnet (model, resnet_type = restype, grad=cf ['train_weight'],
                img3D=False)
    elif 'resnet2.5d' in cf ['model_type']:
        restype=re.sub ('2\.5d_', '', cf['model_type'])
        load_resnet (model, resnet_type = restype, grad=cf ['train_weight'],
                img3D=False)
    elif 'resnet' in cf ['model_type']:
        load_resnet (model, resnet_type=cf ['model_type'], 
                    grad=cf ['train_weight'])
    elif 'vggCRNN' in cf ['model_type']:
        vgg_type = re.sub ('CRNN', '', cf ['model_type'])
        load_vgg_layer (model, vgg_type=vgg_type, 
                grad = cf ['train_weight'], img3D=False)
    elif 'vgg' in cf ['model_type']:
        load_vgg_layer (model, vgg_type=cf ['model_type'], 
                grad = cf ['train_weight'])
    elif 'simple' in cf ['model_type']:
        print ('no initial weight available')
    elif 'CNN2.5d' in cf ['model_type']:
        print ('no initial weight available')
    elif 'CNN3d' in cf ['model_type']:
        print ('no initial weight available')
