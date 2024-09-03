# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

from layers.back_projection_layers import BackProjectionModule
from layers.NF_layers import SparseSqueezeLayer, SparseInvertibleConv1x1, SparseCouplingLayer
from layers.common_layers import SparseAttModule, SparseEnhModule
from models.compress_model_base import CompressionModel
from models.hyperprior import HyperAnalysisTransform, HyperSynthesisTransform
from compressai.entropy_models import GaussianConditional
from compressai.models.utils import update_registered_buffers
import math
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiFunctional import _wrap_tensor


class CompressBlock(nn.Module):
    def __init__(self,in_channels,out_channels,squeeze_type):
        super(CompressBlock,self).__init__()
        self.operations = nn.ModuleList()
        b = SparseSqueezeLayer(in_channels,2,squeeze_type)
        self.operations.append(b)
        in_channels *= 8
        b = SparseInvertibleConv1x1(in_channels)
        self.operations.append(b)
        b = SparseCouplingLayer(in_channels // 4, 3 * in_channels // 4, 3)
        self.operations.append(b)
        b = SparseCouplingLayer(in_channels // 4, 3 * in_channels // 4, 3)
        self.operations.append(b)
        b = BackProjectionModule(in_channels,out_channels,3)
        self.operations.append(b)


class InvCore(nn.Module):
    def __init__(self, channels=[3,12,24,96], squeeze_type="avg"):
        super(InvCore, self).__init__()
        self.channels = channels
        self.levels = nn.ModuleList()
        self.n_levels = len(channels)-1
        self.squeeze_type = squeeze_type
        # 1st level
        for i in range(self.n_levels):
            b = CompressBlock(self.channels[i], self.channels[i+1] ,self.squeeze_type)
            self.levels.append(b)

    def forward(self, x, reverse=False):
        if not reverse:
            for i, level in enumerate(self.levels):
                for op in level.operations:
                    x = op.forward(x, False)
        else:
            for i, level in reversed(list(enumerate(self.levels))):
                for op in reversed(level.operations):
                    x = op.forward(x, True)
        return x


class RNF_BP(CompressionModel):

    def __init__(self, M=3*8*8, channels=[3,12,24,96], num_scales=64,scale_min=.11,scale_max=256,enh=0, attention=0, squeeze_type="avg", **kwargs):
        super().__init__(entropy_bottleneck_channels=M)

        self.num_scales = num_scales
        self.scale_min = scale_min
        self.scale_max = scale_max

        self.hyper_analysis_transform = HyperAnalysisTransform(conv_filters=[M,M,M,M,M,M],
                                          conv_kernel_size = [3,3,3,3,3],
                                          conv_strides = [1,1,2,1,2])
        self.hyper_synthesis_transform = HyperSynthesisTransform(conv_filters=[M,M,M,2*M,2*M,2*M],
                                           conv_kernel_size = [3,3,3,3,3],
                                           conv_strides = [2,1,2,1,1])
        if not enh==0:
            self.enh = SparseEnhModule(enh)
        else:
            self.enh = None
        
        self.inv = InvCore(channels=channels,squeeze_type=squeeze_type)
        
        if not attention==0:
            self.attention = SparseAttModule(attention)
        else:
            self.attention = None
        
        self.gaussian_conditional = GaussianConditional(None)
        
    def analysis_transform(self, x):
        if self.enh is not None:
            x = self.enh(x)
        
        x = self.inv(x)
        
        if self.attention is not None:
            x = self.attention(x)

        return x

    def synthesis_transform(self, x):
        if self.enh is not None:
            x = self.attention(x, reverse = True)
        
        x = self.inv(x, reverse=True)
        
        if self.attention is not None:
            x = self.enh(x, reverse=True)
        
        return x

    def forward(self, x):
        y = self.analysis_transform(x)
        z = self.hyper_analysis_transform(y)
        
        # Needs to unsqueeze and transpose to use compressAI bottleneck
        z_hat, side_bits = self.entropy_bottleneck(torch.unsqueeze(torch.transpose(z.F,0,1),dim=0))

        # Put the features back in the sparse tensor
        z_hat_sparse = _wrap_tensor(z, torch.transpose(torch.squeeze(z_hat,dim=0),0,1))

        gaussian_params = self.hyper_synthesis_transform(z_hat_sparse)
        ######################################################################################
        scales_hat, means_hat = gaussian_params.F.chunk(2, 1)
        y_hat, bits = self.gaussian_conditional(torch.unsqueeze(torch.transpose(y.F,0,1),dim=0), 
                                                torch.unsqueeze(torch.transpose(scales_hat,0,1),dim=0), 
                                                means=torch.unsqueeze(torch.transpose(means_hat,0,1),dim=0))
        
        # Put the features back in the sparse tensor
        y_hat_sparse = _wrap_tensor(y,torch.transpose(torch.squeeze(y_hat,dim=0),0,1))

        x_hat = self.synthesis_transform(y_hat_sparse)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": bits, "z": side_bits}
        }

    def compress(self, x, debug=False):

        y = self.analysis_transform(x)
        z = self.hyper_analysis_transform(y)

        z_strings = self.entropy_bottleneck.compress(torch.unsqueeze(torch.transpose(z.F,0,1),dim=0))
        z_hat = self.entropy_bottleneck.decompress(z_strings, [z.size()[0]])

        z_hat_sparse = _wrap_tensor(z,torch.transpose(torch.squeeze(z_hat,dim=0),0,1))

        gaussian_params = self.hyper_synthesis_transform(z_hat_sparse)
        scales_hat, means_hat = gaussian_params.F.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)

        y_strings = self.gaussian_conditional.compress(torch.unsqueeze(torch.transpose(y.F,0,1),dim=0),
                                                        torch.unsqueeze(torch.transpose(indexes,0,1),dim=0),
                                                        means=torch.unsqueeze(torch.transpose(means_hat,0,1),dim=0))

        if not debug:
            return {"strings": [y_strings, z_strings], "shape": [z.size()[0]]}

        else:
            return {"strings": [y_strings, z_strings], 
                    "shape": [z.size()[0]],
                    "y" : y,
                    "z" : z,
                    "z_hat_sparse": z_hat_sparse,
                    "gaussian_params": gaussian_params,
                    "indexes": indexes,
                    }


    def decompress(self, strings, latent_coords, hyper_latent_coords):
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], [hyper_latent_coords.size()[0]])

        z_hat_sparse = _wrap_tensor(hyper_latent_coords, torch.transpose(torch.squeeze(z_hat,dim=0),0,1))

        gaussian_params = self.hyper_synthesis_transform(z_hat_sparse)
        scales_hat, means_hat = gaussian_params.F.chunk(2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)

        y_hat = self.gaussian_conditional.decompress(strings[0],
                                                    torch.unsqueeze(torch.transpose(indexes,0,1),dim=0),
                                                    means=torch.unsqueeze(torch.transpose(means_hat,0,1),dim=0))

        y_hat_sparse = _wrap_tensor(latent_coords, torch.transpose(torch.squeeze(y_hat,dim=0),0,1))

        # Remove batch dimension, and crop away any extraneous padding.
        x_hat = self.synthesis_transform(y_hat_sparse)

        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def get_scale_table(self):
        return torch.exp(torch.linspace(math.log(self.scale_min), math.log(self.scale_max), self.num_scales))

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = self.get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated