import torch.nn as nn
import torch
import copy

from model_hyperbolic_poincare import manifolds
from model_hyperbolic_poincare.layers import *


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Model(nn.Module):
    def __init__(self,config, num_classes, img_h, img_w):
        super(Model, self).__init__()
        self.config = config
        self.in_planes = 768
        self.num_classes = num_classes

        self.h_resolution = int((img_h - 16) // 16 + 1)
        self.w_resolution = int((img_w - 16) // 16 + 1)
        self.vision_stride_size = 16
        clip_model, _ = load_clip_to_cpu('ViT-B-16', self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")
        self.image_encoder = clip_model.visual

        self.classifier = Classifier(self.num_classes)
        self.classifier_proj = Classifier2(self.num_classes)

        self.classifier_TP = Classifier(self.num_classes)

        self.sequence_length = config.sequence_length

        self.attn = nn.MultiheadAttention(768, num_heads=12)

        self.ln = nn.LayerNorm(768)
        self.ln_post2 = copy.deepcopy(self.image_encoder.ln_post)
        self.cross_attn = nn.MultiheadAttention(768, num_heads=12, batch_first=True)


        self.manifold = getattr(manifolds, 'PoincareBall')()

        self.hyp_image_encoder = HyperbolicTransformer(layers=1, d_model=768, heads=12, dropout=0.1, manifold=self.manifold, c=1.0)

        self.hyp_cross_attn = HyperbolicCrossAttentionBlock(self.manifold, c=1.0, embedding_dim=768, n_head=12, T=self.sequence_length)

        self.hyp_classifier_TP = HyperbolicPrototypicalClassifier(self.in_planes, self.num_classes, self.manifold)


    def encode_image(self, x: torch.Tensor, cv_emb=None):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        if cv_emb != None:
            x[:, 0] = x[:, 0] + cv_emb
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.ln_pre(x)

        BT, N, C = x.size()
        B = BT // self.sequence_length

        # HHSA: HGI builds hierarchical memory; DGF reads and fuses it below.
        x = x.permute(1, 0, 2)
        for i in range(11):

            x = self.image_encoder.transformer.resblocks[i](x)
            # HGI: interact across frame CLS tokens after every second ViT block.
            if i % 2 == 1:

                x_cls = x[0:1, :, :].permute(1, 0, 2)
                bs = x_cls.size(0) // self.sequence_length
                x_t = x_cls.view(bs, self.sequence_length, -1)
                hyp_x = self.to_hyper(x_t)

                hyp_x = self.hyp_image_encoder.resblocks[0](hyp_x)


                x_cls = self.to_tangent(hyp_x)

                x_cls = x_cls.reshape(1, bs * self.sequence_length, -1)
                x_cls = self.attn(x[0:1, :, :], x_cls, x_cls)[0]
                # Replace CLS and append its projected representation to the memory.
                x = torch.cat((x_cls, x[1:], x_cls), dim=0)


        x11 = x

        x12 = self.image_encoder.transformer.resblocks[11](x11)

        x_cls = x12[0:1, :, :].permute(1, 0, 2)
        bs = x_cls.size(0) // self.sequence_length
        x12_cls = x_cls.view(bs, self.sequence_length, -1)


        # HGI at the final (12th) ViT block.
        hyp_x = self.to_hyper(x12_cls)
        hyp_x = self.hyp_image_encoder.resblocks[0](hyp_x)
        tan_x = self.to_tangent(hyp_x)

        x_cls = tan_x.reshape(1, bs * self.sequence_length, -1)
        x_cls = self.attn(x12[0:1, :, :], x_cls, x_cls)[0]
        x = torch.cat((x_cls, x12[1:], x_cls), dim=0)


        # DGF: each final CLS queries the tracklet-wide, cross-layer memory in both geometries.
        hierarchical_memory = x[163:].permute(1, 0, 2)
        num_hyp_layers = hierarchical_memory.size(1)
        hierarchical_memory = hierarchical_memory.reshape(bs, num_hyp_layers * self.sequence_length, -1)
        euclidean_frame_features = self.cross_attn(x12_cls, hierarchical_memory, hierarchical_memory)[0]
        hyperbolic_frame_features = self.hyp_cross_attn(x12_cls, hierarchical_memory)
        fused_frame_features = euclidean_frame_features + hyperbolic_frame_features
        fused_frame_features = self.image_encoder.ln_post(fused_frame_features)


        x11 = x11.permute(1, 0, 2)
        x12 = x12.permute(1, 0, 2)

        x12 = self.image_encoder.ln_post(x12)
        if self.image_encoder.proj is not None:
            xproj = x12 @ self.image_encoder.proj


        return x11, x12, xproj, fused_frame_features, hyp_x

    def to_hyper(self, x, c=1.0):
        if self.manifold.name == 'PoincareBall':
            x_tan = self.manifold.proj_tan0(x, c)
            x_hyp = self.manifold.expmap0(x_tan, c)
            x_hyp = self.manifold.proj(x_hyp, c)
        elif self.manifold.name == 'Lorentz':
            x_tan = self.manifold.proju0(x, c)
            x_hyp = self.manifold.expmap0(x_tan, c)
        return x_hyp

    def to_tangent(self, x, c=1.0):
        if self.manifold.name == 'PoincareBall':
            x_tan = self.manifold.logmap0(x, c)
            x_tan = self.manifold.proj_tan0(x_tan, c)
        elif self.manifold.name == 'Lorentz':
            x_tan = self.manifold.log_map_zero(x, c)
            x_tan = self.manifold.normalize_tangent_zero(x_tan, c)
        return x_tan

    def forward(self, x1=None, x2=None):
        if x1 is not None and x2 is not None:
            image_features_maps = torch.cat([x1, x2], dim=0)
            image_features_last, image_features, image_features_proj, image_features_tp, hyp_image_features_tp = self.encode_image(image_features_maps)

            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

            img_feature_last = self.tem_pool(img_feature_last, self.sequence_length)
            img_feature = self.tem_pool(img_feature, self.sequence_length)
            img_feature_proj = self.tem_pool(img_feature_proj, self.sequence_length)
            img_feature_tp = image_features_tp.mean(1)
            # GMA uses a tracklet embedding pooled from the final HGI hyperbolic CLS tokens.
            tan_image_features_tp = self.to_tangent(hyp_image_features_tp)

            tan_img_feature_tp = tan_image_features_tp.mean(1)
            hyp_img_feature_tp = self.to_hyper(tan_img_feature_tp)


            cls_scores, _ = self.classifier(img_feature)
            cls_scores_proj, _ = self.classifier_proj(img_feature_proj)
            cls_scores_tp, _ = self.classifier_TP(img_feature_tp)
            # HPA: shared hyperbolic prototypes produce the identity-classification logits.
            hyp_cls_scores_tp, _ = self.hyp_classifier_TP(hyp_img_feature_tp)


            return [img_feature_last, img_feature, img_feature_proj, img_feature_tp, hyp_img_feature_tp], [cls_scores, cls_scores_proj, cls_scores_tp, hyp_cls_scores_tp], \


        else:
            image_features_last1, image_features1, image_features_proj1, image_features_tp, _ = self.encode_image(x1)
            img_feature1 = image_features1[:,0]


            img_feature1 = self.tem_pool(img_feature1, self.sequence_length)

            img_feature_tp = image_features_tp.mean(1)

            return torch.cat([img_feature1, img_feature_tp], dim=1)

    def tem_pool(self,features,t=6):
        features = features.squeeze()
        features = features.view(features.size(0)//t, t, -1).permute(1, 0, 2)
        features = features.mean(0)
        return features


class Classifier(nn.Module):
    def __init__(self, pid_num):
        super(Classifier, self, ).__init__()
        self.pid_num = pid_num

        self.BN = nn.BatchNorm1d(768)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(768, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2_norm = Normalize(2)

    def forward(self, features):
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return cls_score, self.l2_norm(bn_features)

class Classifier2(nn.Module):
    def __init__(self, pid_num):
        super(Classifier2, self, ).__init__()
        self.pid_num = pid_num
        self.BN = nn.BatchNorm1d(512)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(512, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.l2_norm = Normalize(2)

    def forward(self, features):
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return cls_score, self.l2_norm(features)


class HyperbolicPrototypicalClassifier(nn.Module):
    """
    Hyperbolic prototypical classifier in Poincare Ball.
    - Prototypes are parameterized in tangent space at origin for stability.
    - Logits are negative squared hyperbolic distances with learnable temperature.
    """
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        manifold,
        c: float = 1.0,
        temp_init: float = 0.07,
        learn_temp: bool = True,
        use_bias: bool = False,
    ):
        super().__init__()
        self.manifold = manifold
        self.c = float(c)
        self.num_classes = int(num_classes)
        self.in_dim = int(in_dim)


        self.prototypes = nn.Parameter(torch.randn(num_classes, in_dim) * 1e-2)

        logit_scale_init = float(torch.log(torch.tensor(1.0 / temp_init)))
        if learn_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)
        else:
            self.register_buffer("logit_scale", torch.ones([]) * logit_scale_init)

        self.bias = nn.Parameter(torch.zeros(num_classes)) if use_bias else None

    def to_hyper(self, x, c=1.0):
        if self.manifold.name == 'PoincareBall':
            x_tan = self.manifold.proj_tan0(x, c)
            x_hyp = self.manifold.expmap0(x_tan, c)
            x_hyp = self.manifold.proj(x_hyp, c)
        elif self.manifold.name == 'Lorentz':
            x_tan = self.manifold.proju0(x, c)
            x_hyp = self.manifold.expmap0(x_tan, c)
        return x_hyp


    def prototypes_hyp(self):

        return self.to_hyper(self.prototypes, c=self.c)

    def forward(self, x):
        x_hyp = x

        p_hyp = self.to_hyper(self.prototypes)

        if hasattr(self.manifold, "sqdist"):
            with torch.cuda.amp.autocast(enabled=False):
                dist2 = self.manifold.sqdist(x_hyp.unsqueeze(1), p_hyp.unsqueeze(0), c=self.c)
        else:
            dist = self.manifold.dist(x_hyp.unsqueeze(1), p_hyp.unsqueeze(0), c=self.c)
            dist2 = dist * dist

        scale = self.logit_scale.exp().clamp(1e-6, 1e6)
        logits = -dist2 * scale

        if self.bias is not None:
            logits = logits + self.bias

        return logits, x_hyp


class HyperbolicCrossAttentionBlock(nn.Module):
    def __init__(self, manifold, c, embedding_dim: int, n_head: int, attn_mask: torch.Tensor = None, dropout=0., T=0, ):
        super().__init__()
        self.T = T
        self.manifold = manifold
        self.c = c

        self.hyp_cross_attn = HyperbolicMultiheadAttention2(embedding_dim, n_head, manifold, c=self.c, dropout=dropout)

    def to_hyper(self, x, c=1.0):
        if self.manifold.name == 'PoincareBall':
            x_tan = self.manifold.proj_tan0(x, c)
            x_hyp = self.manifold.expmap0(x_tan, c)
            x_hyp = self.manifold.proj(x_hyp, c)
        elif self.manifold.name == 'Lorentz':
            x_tan = self.manifold.proju0(x, c)
            x_hyp = self.manifold.expmap0(x_tan, c)
        return x_hyp

    def to_tangent(self, x, c=1.0):
        if self.manifold.name == 'PoincareBall':
            x_tan = self.manifold.logmap0(x, c)
            x_tan = self.manifold.proj_tan0(x_tan, c)
        elif self.manifold.name == 'Lorentz':
            x_tan = self.manifold.log_map_zero(x, c)
            x_tan = self.manifold.normalize_tangent_zero(x_tan, c)
        return x_tan

    def forward(self, x, y=None):

        BT, p, C = x.size()
        B = BT // self.T
        hyp_global = self.to_hyper(y, c=self.c)
        hyp_part = self.to_hyper(x, c=self.c)

        hyp_part, _ = self.hyp_cross_attn(hyp_part, hyp_global, hyp_global)

        hyp_x = hyp_part
        x = self.to_tangent(hyp_x)

        return x


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    print(model_path)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model,model.state_dict()
