
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
import torch.nn.init as init
import numpy as np





class SDA(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):

        super(SDA, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out



def build_activation_layer(act_type):

    if act_type is None:
        return nn.Identity()
    elif act_type == 'GELU':
        return nn.GELU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    elif act_type == 'SiLU':
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")


class ElementWiseScale(nn.Module):

    def __init__(self, channels, init_value=1e-5, requires_grad=True):
        super(ElementWiseScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, channels, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale



class FDA(nn.Module):

    def __init__(self,
                 embed_dims,
                 reduction_ratio=16,
                 act_type='GELU',
                 dropout_rate=0.0):
        super(FDA, self).__init__()

        self.embed_dims = embed_dims
        self.intermediate_channels = int(embed_dims // 2)


        self.conv_initial = nn.Sequential(
            nn.Conv2d(in_channels=embed_dims,
                      out_channels=self.intermediate_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(self.intermediate_channels),
            build_activation_layer(act_type)
        )


        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.intermediate_channels, self.intermediate_channels // reduction_ratio, 1, bias=False),
            build_activation_layer(act_type),
            nn.Conv2d(self.intermediate_channels // reduction_ratio, self.intermediate_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.intermediate_channels, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )


        self.decompose_conv = nn.Conv2d(
            in_channels=self.intermediate_channels,
            out_channels=1,
            kernel_size=1
        )
        self.sigma = ElementWiseScale(self.intermediate_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_activation_layer(act_type)


        self.conv_final = nn.Sequential(
            nn.Conv2d(in_channels=self.intermediate_channels,
                      out_channels=embed_dims,
                      kernel_size=1),
            nn.BatchNorm2d(embed_dims),
            build_activation_layer(act_type)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def feat_decompose(self, x):

        x_global = self.decompose_act(self.decompose_conv(x))  # [B, 1, H, W]
        x = x + self.sigma(x - x_global)
        return x

    def forward(self, x):

        x = self.conv_initial(x)

        ca = self.channel_attention(x)
        x = x * ca

        sa = self.spatial_attention(x)
        x = x * sa


        x = self.feat_decompose(x)

        x = self.dropout(x)


        x = self.conv_final(x)
        x = self.dropout(x)

        return x



class DMI(nn.Module):
    def __init__(self, feature_dim, dropout_prob=0.1):
        super(DMI, self).__init__()
        self.text_linear = nn.Linear(feature_dim, feature_dim)
        self.extra_linear = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        self.text_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        self.image_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )

        self.text_scale = nn.Parameter(torch.ones(1, feature_dim), requires_grad=True)
        self.image_scale = nn.Parameter(torch.ones(1, feature_dim), requires_grad=True)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, mode='text'):

        if query.shape[-1] != 768:
            query = self.text_linear(query)
        if key.shape[-1] != 768:
            key = self.extra_linear(key)
            value = self.extra_linear(value)


        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(key.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)


        attended_values = torch.matmul(attention_weights, value)
        attended_values = self.dropout(attended_values)

        if mode == 'text':
            gated_values = torch.sigmoid(self.text_gate(attended_values)) * attended_values * self.text_scale
        elif mode == 'image':
            gated_values = torch.sigmoid(self.image_gate(attended_values)) * attended_values * self.image_scale
        else:
            raise ValueError("Invalid mode. Choose either 'text' or 'image'.")

        return gated_values

    


class ICIN(nn.Module):
    def __init__(self, args):
        super(ICIN, self).__init__()


        self.model = CLIPModel.from_pretrained(" ",
                                               local_files_only=True)

        self.text_linear = nn.Sequential(
            nn.Linear(args.text_size, 768),
            nn.Dropout(args.dropout_rate),
            nn.GELU()
        )
        self.image_linear = nn.Sequential(
            nn.Linear(args.image_size, args.image_size),
            nn.Dropout(args.dropout_rate),
            nn.GELU()
        )

        self.self_attention_layer_text = SDA(d_model=768, d_k=64, d_v=64, h=8, dropout=0.1)

        self.FDA = FDA(embed_dims=args.image_size)

        self.DMI = DMI(feature_dim=768, dropout_prob=0.1)

        self.classifier_fuse = nn.Linear(args.image_size, args.label_number)
        self.loss_fct = nn.CrossEntropyLoss()

        self.initial_lambda = 0.5

    def jensen_shannon_divergence(self, p_logits, q_logits):

        p_probs = F.softmax(p_logits, dim=-1)
        q_probs = F.softmax(q_logits, dim=-1)


        m_probs = 0.5 * (p_probs + q_probs)


        kl_pm = F.kl_div(F.log_softmax(p_logits, dim=-1), m_probs, reduction='batchmean')
        kl_qm = F.kl_div(F.log_softmax(q_logits, dim=-1), m_probs, reduction='batchmean')


        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd

    def forward(self, inputs, labels=None):

        output = self.model(**inputs)
        text_feature = output['text_model_output']['pooler_output']
        image_feature = output['vision_model_output']['pooler_output']


        text_feature = self.text_linear(text_feature)
        image_feature = self.image_linear(image_feature)

        text_feature = self.self_attention_layer_text(text_feature.unsqueeze(1), text_feature.unsqueeze(1),
                                                      text_feature.unsqueeze(1)).squeeze(1)

        image_feature = image_feature.unsqueeze(-1).unsqueeze(-1)  # (B, C) -> (B, C, 1, 1)
        image_feature = self.FDA(image_feature).squeeze(-1).squeeze(-1)


        cross_feature_text = self.DMI(text_feature, image_feature, image_feature, mode='text')
        cross_feature_image = self.DMI(image_feature, text_feature, text_feature, mode='image')
        cross_feature_text = cross_feature_text.squeeze(1)  # [batch_size, 768]
        cross_feature_image = cross_feature_image.squeeze(1)  # [batch_size, 768]


        fuse_feature = cross_feature_text + cross_feature_image
 

        logits_fuse = self.classifier_fuse(fuse_feature)
        fuse_score = F.softmax(logits_fuse, dim=-1)

        outputs = (fuse_score,)
        if labels is not None:

            loss_main = self.loss_fct(logits_fuse, labels)

            lambda_co_reg = self.initial_lambda * (1 / (1 + torch.exp(-loss_main)))
            js_divergence = self.jensen_shannon_divergence(text_feature, image_feature)

            total_loss = loss_main + lambda_co_reg * js_divergence

            outputs = (total_loss,) + outputs

        return outputs
