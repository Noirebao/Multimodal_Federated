import torch
import torch.nn as nn
from timm.models.registry import register_model
from beit3_base import BEiT3Wrapper, _get_base_config


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def feature_select(x, split_position):
    img_cls = x[:, 0, :]
    text_cls = x[:, split_position, :]
    return img_cls, text_cls


class Fusion(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.img_norm = norm_layer(input_features)
        self.text_norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features * 2, output_features)
        self.activation = nn.Tanh()

    def forward(self, img_cls, text_cls):
        img_cls = self.img_norm(img_cls)
        text_cls = self.text_norm(text_cls)
        cls = torch.cat((img_cls, text_cls), dim=-1)
        pooled_output = self.dense(cls) / 2.0
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BEiT3ForVisualQuestionAnswering(BEiT3Wrapper):
    def __init__(self, args, num_classes, norm_layer=nn.LayerNorm, mode='vl', **kwargs):
        super(BEiT3ForVisualQuestionAnswering, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.mode = mode
        self.split_position = int((args.img_size // args.patch_size) ** 2 + 1)
        self.fusion = Fusion(
            input_features=embed_dim,
            output_features=embed_dim,
            norm_layer=norm_layer,
        )
        self.fusion.apply(self._init_weights)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            norm_layer(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, num_classes),
        )
        self.head.apply(self._init_weights)

    def forward(self, image, question, padding_mask, prototype=None, **kwargs):
        if self.mode == "vl":
            outputs = self.beit3(
                textual_tokens=question,
                visual_tokens=image,
                text_padding_position=padding_mask,
            )
            x = outputs["encoder_out"]
            img_rep, text_rep = feature_select(x, self.split_position)
            fusion_rep = self.fusion(img_rep, text_rep)
            return self.head(fusion_rep), {"img_rep": img_rep, "text_rep": text_rep, "fusion_rep": fusion_rep}
        elif self.mode == "v":
            if prototype is None:
                raise KeyError('prototype is None while only image')
            else:
                outputs = self.beit3(visual_tokens=image, textual_tokens=None, text_padding_position=None)
                x = outputs["encoder_out"]
                img_rep = x[:, 0, :]
                fusion_rep = self.fusion(img_rep, prototype)
                return self.head(fusion_rep), {"img_rep": img_rep, "text_rep": None, "fusion_rep": fusion_rep}
        elif self.mode == "l":
            if prototype is None:
                raise KeyError('prototype is None while only text')
            else:
                outputs = self.beit3(visual_tokens=None, textual_tokens=question, text_padding_position=padding_mask)
                x = outputs["encoder_out"]
                text_rep = x[:, 0, :]
                fusion_rep = self.fusion(prototype, text_rep)
                return self.head(fusion_rep), {"img_rep": None, "text_rep": text_rep, "fusion_rep": fusion_rep}
        else:
            raise NotImplementedError(f"Undefined mode {self.mode} during model's forword")

    def head_forward(self, fusion_rep):
        return self.head(fusion_rep)

    def set_mode(self, new_mode):
        if new_mode in ['v', 'l', 'vl']:
            self.mode = new_mode
        else:
            raise ValueError('BEIT3 get an undefined mode')


@register_model
def beit3_base_patch16_480_vqav2(pretrained=False, **kwargs):
    args = _get_base_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=310, **kwargs)
    return model


