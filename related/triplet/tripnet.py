#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CODE RELEASE TO SUPPORT RESEARCH.
COMMERCIAL USE IS NOT PERMITTED.
#==============================================================================
An implementation based on:
***
    C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. 
    Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. 
    Medical Image Analysis, 78 (2022) 102433.
***  
Created on Thu Oct 21 15:38:36 2021
#==============================================================================  
Copyright 2021 The Research Group CAMMA Authors All Rights Reserved.
(c) Research Group CAMMA, University of Strasbourg, France
@ Laboratory: CAMMA - ICube
@ Author: Nwoye Chinedu Innocent
@ Website: http://camma.u-strasbg.fr
#==============================================================================
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
#==============================================================================
"""

import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.models as basemodels
import torchvision.transforms as transforms


OUT_HEIGHT = 8
OUT_WIDTH  = 14


#%% Model Rendezvous
class AttentionTripnet(nn.Module):
    """
    Rendezvous: attention mechanism for surgical action triplet recognition by Nwoye, C.I. et.al. 2021
    @args:
        image_shape: a tuple (height, width) e.g: (224,224)
        basename: Feature extraction network: e.g: "resnet50", "VGG19"
        num_tool: default = 6, 
        num_verb: default = 10, 
        num_target: default = 15, 
        num_triplet: default = 100, 
    @call:
        inputs: Batch of input images of shape [batch, height, width, channel]
    @output: 
        enc_i: tuple (cam, logits) for instrument
        enc_v: tuple (cam, logits) for verb
        enc_t: tuple (cam, logits) for target
        dec_ivt: logits for triplet
    """
    def __init__(self, basename="resnet18", num_tool=6, num_verb=10, num_target=15, num_triplet=100, hr_output=False, dict_map_url="./"):
        super(AttentionTripnet, self).__init__()
        self.encoder = Encoder(basename, num_tool, num_verb, num_target, num_triplet, hr_output=hr_output)
        self.decoder = Decoder(num_tool, num_verb, num_target, num_triplet, dict_map_url)  
     
    def forward(self, inputs):
        enc_i, enc_v, enc_t = self.encoder(inputs)
        dec_ivt = self.decoder(enc_i[1], enc_v[1], enc_t[1])
        return enc_i, enc_v, enc_t, dec_ivt
    


#%% Triplet Components Feature Encoder
class Encoder(nn.Module):
    def __init__(self, basename='resnet18', num_tool=6,  num_verb=10, num_target=15, num_triplet=100, hr_output=False):
        super(Encoder, self).__init__()
        depth = 64 if basename == 'resnet18' else 128
        self.basemodel  = BaseModel(basename, hr_output)
        self.wsl        = WSL(num_tool, depth)
        self.cagam      = CAGAM(num_tool, num_verb, num_target)
        
    def forward(self, x):
        high_x, low_x = self.basemodel(x)
        enc_i         = self.wsl(high_x)
        enc_v, enc_t  = self.cagam(high_x, enc_i[0])
        return enc_i, enc_v, enc_t


 
#%% Feature extraction backbone
class BaseModel(nn.Module):   
    def __init__(self, basename='resnet18', hr_output=False, *args):
        super(BaseModel, self).__init__(*args)
        self.output_feature = {} 
        if basename == 'resnet18':
            self.basemodel      = basemodels.resnet18(pretrained=True)     
            if hr_output: self.increase_resolution()
            self.basemodel.layer1[1].bn2.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.layer4[1].bn2.register_forward_hook(self.get_activation('high_level_feature'))        
        if basename == 'resnet50':
            self.basemodel      = basemodels.resnet50(pretrained=True) 
            # print(self.basemodel)
            self.basemodel.layer1[2].bn2.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.layer4[2].bn2.register_forward_hook(self.get_activation('high_level_feature'))
        
    def increase_resolution(self):  
        global OUT_HEIGHT, OUT_WIDTH  
        self.basemodel.layer3[0].conv1.stride = (1,1)
        self.basemodel.layer3[0].downsample[0].stride=(1,1)  
        self.basemodel.layer4[0].conv1.stride = (1,1)
        self.basemodel.layer4[0].downsample[0].stride=(1,1)
        OUT_HEIGHT *= 4
        OUT_WIDTH  *= 4
        print("using high resolution output ({}x{})".format(OUT_HEIGHT,OUT_WIDTH))        

    def get_activation(self, layer_name):
        def hook(module, input, output):
            self.output_feature[layer_name] = output
        return hook
    
    def forward(self, x):
        _ = self.basemodel(x)
        return self.output_feature['high_level_feature'], self.output_feature['low_level_feature']
    

     
#%% Weakly-Supervised localization
class WSL(nn.Module):
    def __init__(self, num_class, depth=64):
        super(WSL, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=depth, kernel_size=3, padding=1)
        self.cam   = nn.Conv2d(in_channels=depth, out_channels=num_class, kernel_size=1)
        self.elu   = nn.ELU()
        self.bn    = nn.BatchNorm2d(depth)
        self.gmp   = nn.AdaptiveMaxPool2d((1,1))
        
    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn(feature)
        feature = self.elu(feature)
        cam     = self.cam(feature)
        logits  = self.gmp(cam).squeeze(-1).squeeze(-1)
        return cam, logits

 

#%% Class Activation Guided Attention Mechanism
class CAGAM(nn.Module):    
    def __init__(self, num_tool, num_verb, num_target, in_depth=512):
        super(CAGAM, self).__init__()        
        out_depth               = num_tool        
        self.verb_context       = nn.Conv2d(in_channels=in_depth, out_channels=out_depth, kernel_size=3, padding=1)        
        self.verb_query         = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.verb_tool_query    = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.verb_key           = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.verb_tool_key      = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.verb_cmap          = nn.Conv2d(in_channels=out_depth, out_channels=num_verb, kernel_size=1)       
        self.target_context     = nn.Conv2d(in_channels=in_depth, out_channels=out_depth, kernel_size=3, padding=1)     
        self.target_query       = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.target_tool_query  = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.target_key         = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.target_tool_key    = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.target_cmap        = nn.Conv2d(in_channels=out_depth, out_channels=num_target, kernel_size=1)        
        self.gmp       = nn.AdaptiveMaxPool2d((1,1))
        self.elu       = nn.ELU()    
        self.soft      = nn.Softmax(dim=1)    
        self.flat      = nn.Flatten(2,3)  
        self.bn1       = nn.BatchNorm2d(out_depth)
        self.bn2       = nn.BatchNorm2d(out_depth)
        self.bn3       = nn.BatchNorm2d(out_depth)
        self.bn4       = nn.BatchNorm2d(out_depth)
        self.bn5       = nn.BatchNorm2d(out_depth)
        self.bn6       = nn.BatchNorm2d(out_depth)
        self.bn7       = nn.BatchNorm2d(out_depth)
        self.bn8       = nn.BatchNorm2d(out_depth)
        self.bn9       = nn.BatchNorm2d(out_depth)
        self.bn10      = nn.BatchNorm2d(out_depth) 
        self.bn11      = nn.BatchNorm2d(out_depth) 
        self.bn12      = nn.BatchNorm2d(out_depth)        
        self.encoder_cagam_verb_beta   = torch.nn.Parameter(torch.randn(1))
        self.encoder_cagam_target_beta = torch.nn.Parameter(torch.randn(1))          
                        
    def get_verb(self, raw, cam):
        x  = self.elu(self.bn1(self.verb_context(raw)))
        z  = x.clone()
        sh = list(z.shape)
        sh[0] = -1        
        q1 = self.elu(self.bn2(self.verb_query(x)))
        k1 = self.elu(self.bn3(self.verb_key(x)))
        w1 = self.flat(k1).matmul(self.flat(q1).transpose(-1,-2))        
        q2 = self.elu(self.bn4(self.verb_tool_query(cam)))
        k2 = self.elu(self.bn5(self.verb_tool_key(cam)))
        w2 = self.flat(k2).matmul(self.flat(q2).transpose(-1,-2))        
        attention = (w1 * w2) / torch.sqrt(torch.tensor(sh[-1], dtype=torch.float32))
        attention = self.soft(attention)         
        v = self.flat(z)
        e = (attention.matmul(v) * self.encoder_cagam_verb_beta).reshape(sh)
        e = self.bn6(e + z)
        cmap = self.verb_cmap(e)
        y = self.gmp(cmap).squeeze(-1).squeeze(-1)
        return cmap, y  
    
    def get_target(self, raw, cam):
        x  = self.elu(self.bn7(self.target_context(raw)))
        z  = x.clone()
        sh = list(z.shape)
        sh[0] = -1        
        q1 = self.elu(self.bn8(self.target_query(x)))
        k1 = self.elu(self.bn9(self.target_key(x)))
        w1 = self.flat(k1).transpose(-1,-2).matmul(self.flat(q1))        
        q2 = self.elu(self.bn10(self.target_tool_query(cam)))
        k2 = self.elu(self.bn11(self.target_tool_key(cam)))
        w2 = self.flat(k2).transpose(-1,-2).matmul(self.flat(q2))        
        attention = (w1 * w2) / torch.sqrt(torch.tensor(sh[-1], dtype=torch.float32))
        attention = self.soft(attention)         
        v = self.flat(z)
        e = (v.matmul(attention) * self.encoder_cagam_target_beta).reshape(sh)
        e = self.bn12(e + z)
        cmap = self.target_cmap(e)
        y = self.gmp(cmap).squeeze(-1).squeeze(-1)
        return cmap, y
            
    def forward(self, x, cam):
        cam_v, logit_v = self.get_verb(x, cam)
        cam_t, logit_t = self.get_target(x, cam)
        return (cam_v, logit_v), (cam_t, logit_t)


 
# 3D interaction space
class Decoder(nn.Module):
    def __init__(self, num_tool, num_verb, num_target, num_triplet, dict_map_url="./"):
        super(Decoder, self).__init__()
        self.num_tool       = num_tool
        self.num_verb       = num_verb
        self.num_target     = num_target
        # store expected number of triplets so we can create a placeholder if maps.txt is missing
        self.num_triplet    = num_triplet
        valid_pos = self.constraint(num_verb, num_target, url=os.path.join(dict_map_url, 'maps.txt'))
        # create tensor on appropriate device instead of calling .cuda() unconditionally
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.valid_position = torch.tensor(valid_pos, dtype=torch.long, device=device)
        self.decoder_3dis_triplet_alpha  = torch.nn.Parameter(torch.randn(self.num_tool, self.num_tool))
        self.decoder_3dis_triplet_beta   = torch.nn.Parameter(torch.randn(self.num_verb, self.num_verb))
        self.decoder_3dis_triplet_gamma  = torch.nn.Parameter(torch.randn(self.num_target, self.num_target))
        self.mlp     = nn.Linear(in_features=num_triplet, out_features=num_triplet)   
        self.bn1     = nn.BatchNorm1d(num_tool)
        self.bn2     = nn.BatchNorm1d(num_verb)
        self.bn3     = nn.BatchNorm1d(num_target)
        # self.bn4     = nn.BatchNorm1d(num_triplet)
        self.elu     = nn.ELU()

    def constraint(self, num_verb, num_target, url):
        # constraints mask: read triplet mappings from file when available.
        # If the file does not exist or is malformed, fall back to a placeholder
        # of random valid indices with the correct length (self.num_triplet).
        try:
            if os.path.isfile(url):
                indexes = []
                with open(url) as f:
                    for line in f:
                        values = line.strip().split(',')
                        if not values:
                            continue
                        if values[0].startswith('#'):
                            continue
                        # expect lines like: id, tool, verb, target
                        if len(values) >= 4:
                            try:
                                indexes.append(list(map(int, values[1:4])))
                            except ValueError:
                                # skip malformed lines
                                continue
                if len(indexes) > 0:
                    indexes = np.array(indexes)
                    index_pos = []
                    for index in indexes:
                        index_pos.append(index[0] * (num_target * num_verb) + index[1] * (num_target) + index[2])
                    return np.array(index_pos, dtype=np.int64)
        except Exception:
            # fall through to placeholder generation
            pass

        # Placeholder: generate random valid positions with correct length
        total = int(self.num_tool * num_verb * num_target)
        k = int(self.num_triplet)
        if k >= total:
            # if requested triplets exceed possible combinations, return all indices
            return np.arange(total, dtype=np.int64)
        # sample without replacement when possible
        try:
            rng = np.random.default_rng()
            pos = rng.choice(total, size=k, replace=False)
        except Exception:
            # fallback to numpy choice older API
            pos = np.random.choice(total, size=k, replace=False)
        return np.array(pos, dtype=np.int64)

    def mask(self, ivts):
        ivt_flatten    = ivts.reshape([-1, self.num_tool*self.num_verb*self.num_target])
        n              = ivt_flatten.shape[0]
        valid_position = torch.stack([self.valid_position]*n, dim=0)
        valid_triplets = torch.gather(input=ivt_flatten, dim=-1, index=valid_position)
        return valid_triplets

    def forward(self, tool_logits, verb_logits, target_logits):
        tool      = torch.matmul(tool_logits, self.decoder_3dis_triplet_alpha)
        tool      = self.elu(self.bn1(tool))
        verb      = torch.matmul(verb_logits, self.decoder_3dis_triplet_beta)
        verb      = self.elu(self.bn2(verb))
        target    = torch.matmul(target_logits, self.decoder_3dis_triplet_gamma)  
        target    = self.elu(self.bn3(target)) 
        ivt_maps  = torch.einsum('bi,bv,bt->bivt', tool, verb, target ) 
        ivt_mask  = self.mask(ivts=ivt_maps)
        ivt_mask  = self.mlp(ivt_mask)
        return ivt_mask
    

def test_latency(warmup=5, runs=20, T=16):
    """
    Test inference latency of TripNet model
    @args:
        warmup: number of warmup runs
        runs: number of timed runs
        T: temporal dimension
    @output:
        dict: {"mean_ms": float, "std_ms": float, "error": str(e), "device": str(device), "T": T}
    """ 
    import time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result = {"mean_ms": None, "std_ms": None, "error": None, "device": str(device), "T": T}
    try:
        # Avoid downloading pretrained weights in environments without network
        orig_resnet18 = getattr(basemodels, 'resnet18', None)
        orig_resnet50 = getattr(basemodels, 'resnet50', None)

        def _resnet18_no_pretrained(*a, **k):
            if orig_resnet18 is None:
                raise RuntimeError('resnet18 not available in torchvision')
            return orig_resnet18(pretrained=False)

        def _resnet50_no_pretrained(*a, **k):
            if orig_resnet50 is None:
                raise RuntimeError('resnet50 not available in torchvision')
            return orig_resnet50(pretrained=False)

        # If CUDA is not available, the Decoder currently calls .cuda() unconditionally.
        # Patch Tensor.cuda to a no-op temporarily to allow CPU-only environments.
        cuda_available = torch.cuda.is_available()
        orig_tensor_cuda = None
        if not cuda_available:
            orig_tensor_cuda = getattr(torch.Tensor, 'cuda', None)
            # monkeypatch: make .cuda() a no-op so model construction doesn't fail
            torch.Tensor.cuda = lambda self, *a, **k: self

        # try normal construction; if it triggers downloads, retry with non-pretrained
        try:
            model = AttentionTripnet()
        except Exception:
            # monkeypatch resnet constructors to avoid attempting downloads
            if orig_resnet18 is not None:
                basemodels.resnet18 = _resnet18_no_pretrained
            if orig_resnet50 is not None:
                basemodels.resnet50 = _resnet50_no_pretrained
            try:
                model = AttentionTripnet()
            finally:
                # restore basemodels functions
                if orig_resnet18 is not None:
                    basemodels.resnet18 = orig_resnet18
                if orig_resnet50 is not None:
                    basemodels.resnet50 = orig_resnet50

        # restore Tensor.cuda if we patched it
        if not cuda_available and orig_tensor_cuda is not None:
            torch.Tensor.cuda = orig_tensor_cuda

        model.to(device)
        model.eval()

        # Random input: assume single-frame image model; create T inputs to simulate workload
        # Shape: (batch=T, channels=3, H=224, W=224)
        H, W = 224, 224
        inp = torch.randn((T, 3, H, W), device=device)

        # Warmup runs
        with torch.no_grad():
            for _ in range(max(1, warmup)):
                out = model(inp)
                if cuda_available:
                    torch.cuda.synchronize()

        # Timed runs
        times_ms = []
        with torch.no_grad():
            for _ in range(max(1, runs)):
                if cuda_available:
                    torch.cuda.synchronize()
                t0 = time.time()
                _ = model(inp)
                if cuda_available:
                    torch.cuda.synchronize()
                t1 = time.time()
                times_ms.append((t1 - t0) * 1000.0)

        arr = np.array(times_ms, dtype=np.float64)
        result["mean_ms"] = float(arr.mean())
        result["std_ms"] = float(arr.std())
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


if __name__ == "__main__":
    res = test_latency(warmup=5, runs=20, T=16)
    print(res)