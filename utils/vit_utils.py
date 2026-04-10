import numpy as np
import torch


def convert_npz_to_torchvision(npz_path, num_layers, embed_dim, num_heads):
    """Convert Google ViT .npz weights to torchvision state_dict.

    Args:
        npz_path:   path to the .npz file
        num_layers: number of transformer blocks (12 for Base, 24 for Large)
        embed_dim:  embedding dimension (768 for Base, 1024 for Large)
        num_heads:  number of attention heads (12 for Base, 16 for Large)
    """
    w = np.load(npz_path)
    state_dict = {}

    # Patch embedding: (H, W, C_in, C_out) -> (C_out, C_in, H, W)
    state_dict['conv_proj.weight'] = torch.from_numpy(
        w['embedding/kernel'].transpose(3, 2, 0, 1).copy()
    )
    state_dict['conv_proj.bias'] = torch.from_numpy(w['embedding/bias'].copy())

    # CLS token
    state_dict['class_token'] = torch.from_numpy(w['cls'].copy())

    # Positional embedding
    state_dict['encoder.pos_embedding'] = torch.from_numpy(
        w['Transformer/posembed_input/pos_embedding'].copy()
    )

    # Final encoder LayerNorm
    state_dict['encoder.ln.weight'] = torch.from_numpy(
        w['Transformer/encoder_norm/scale'].copy()
    )
    state_dict['encoder.ln.bias'] = torch.from_numpy(
        w['Transformer/encoder_norm/bias'].copy()
    )

    # Classification head
    state_dict['heads.head.weight'] = torch.from_numpy(
        w['head/kernel'].transpose().copy()
    )
    state_dict['heads.head.bias'] = torch.from_numpy(w['head/bias'].copy())

    # Encoder blocks
    for i in range(num_layers):
        prefix_jax = f'Transformer/encoderblock_{i}'
        prefix_pt = f'encoder.layers.encoder_layer_{i}'

        # LayerNorm 1 (pre-attention)
        state_dict[f'{prefix_pt}.ln_1.weight'] = torch.from_numpy(
            w[f'{prefix_jax}/LayerNorm_0/scale'].copy()
        )
        state_dict[f'{prefix_pt}.ln_1.bias'] = torch.from_numpy(
            w[f'{prefix_jax}/LayerNorm_0/bias'].copy()
        )

        # LayerNorm 2 (pre-MLP)
        state_dict[f'{prefix_pt}.ln_2.weight'] = torch.from_numpy(
            w[f'{prefix_jax}/LayerNorm_2/scale'].copy()
        )
        state_dict[f'{prefix_pt}.ln_2.bias'] = torch.from_numpy(
            w[f'{prefix_jax}/LayerNorm_2/bias'].copy()
        )

        # Self-attention Q, K, V
        attn_prefix = f'{prefix_jax}/MultiHeadDotProductAttention_1'
        q_w = w[f'{attn_prefix}/query/kernel'].reshape(embed_dim, embed_dim).transpose()
        k_w = w[f'{attn_prefix}/key/kernel'].reshape(embed_dim, embed_dim).transpose()
        v_w = w[f'{attn_prefix}/value/kernel'].reshape(embed_dim, embed_dim).transpose()
        state_dict[f'{prefix_pt}.self_attention.in_proj_weight'] = torch.from_numpy(
            np.concatenate([q_w, k_w, v_w], axis=0).copy()
        )

        q_b = w[f'{attn_prefix}/query/bias'].reshape(embed_dim)
        k_b = w[f'{attn_prefix}/key/bias'].reshape(embed_dim)
        v_b = w[f'{attn_prefix}/value/bias'].reshape(embed_dim)
        state_dict[f'{prefix_pt}.self_attention.in_proj_bias'] = torch.from_numpy(
            np.concatenate([q_b, k_b, v_b], axis=0).copy()
        )

        # Output projection
        state_dict[f'{prefix_pt}.self_attention.out_proj.weight'] = torch.from_numpy(
            w[f'{attn_prefix}/out/kernel'].reshape(embed_dim, embed_dim).transpose().copy()
        )
        state_dict[f'{prefix_pt}.self_attention.out_proj.bias'] = torch.from_numpy(
            w[f'{attn_prefix}/out/bias'].copy()
        )

        # MLP
        mlp_prefix = f'{prefix_jax}/MlpBlock_3'
        state_dict[f'{prefix_pt}.mlp.linear_1.weight'] = torch.from_numpy(
            w[f'{mlp_prefix}/Dense_0/kernel'].transpose().copy()
        )
        state_dict[f'{prefix_pt}.mlp.linear_1.bias'] = torch.from_numpy(
            w[f'{mlp_prefix}/Dense_0/bias'].copy()
        )
        state_dict[f'{prefix_pt}.mlp.linear_2.weight'] = torch.from_numpy(
            w[f'{mlp_prefix}/Dense_1/kernel'].transpose().copy()
        )
        state_dict[f'{prefix_pt}.mlp.linear_2.bias'] = torch.from_numpy(
            w[f'{mlp_prefix}/Dense_1/bias'].copy()
        )

    return state_dict


class ViTStatesExtractor:
    def __init__(self, model, layer_indices=None, average_attn_weights=False, extract_attention=False, double_cls_token=False):
        self.model = model
        self.extract_attention = extract_attention
        self.first_layer_input = None
        self.attention_maps = {}
        self.layers_mixer_output = {}
        self.layers_output = {}
        self.hooks = []
        self.average_attn_weights = average_attn_weights
        self.double_cls_token = double_cls_token

        if layer_indices is None:
            layer_indices = range(len(model.encoder.layers))

        self.layer_indices = layer_indices

        for i in layer_indices:
            layer = model.encoder.layers[i]
            hook = layer.self_attention.register_forward_hook(
                self._get_attention_hook(i)
            )
            self.hooks.append(hook)

    def _get_attention_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if len(output) == 2 and output[1] is not None:
                self.attention_maps[f'layer_{layer_idx}'] = output[1].detach()
        return hook_fn

    def _duplicate_cls_token(self, tensor):
        """Divide the first CLS token by 2 and append a copy at the end of the sequence.

        Converts a single-CLS [B, 1+P, D] representation into the double-CLS
        [B, 1+P+1, D] format expected by the MambaFormer student.
        """
        tensor[:, 0, :] = tensor[:, 0, :] / 2
        return torch.cat([tensor, tensor[:, 0:1, :]], dim=1)

    def get_vit_states(self, x):
        """Extract attention maps for input x."""
        self.attention_maps = {}
        self.layers_mixer_output = {}
        self.layers_output = {}

        with torch.no_grad():
            for layer in self.model.encoder.layers:
                layer.self_attention.training = False
            output = self._forward_with_attention(x)

        if self.double_cls_token:
            for key, value in self.attention_maps.items():
                attn = value
                batch, num_heads, seq_len, _ = attn.shape
                # Add a new row of zeros at the end
                attn = torch.cat([attn, torch.zeros(batch, num_heads, 1, seq_len, device=attn.device, dtype=attn.dtype)], dim=2)
                # Add a new column of zeros at the end
                attn = torch.cat([attn, torch.zeros(batch, num_heads, seq_len+1, 1, device=attn.device, dtype=attn.dtype)], dim=3)
                # Copy the first CLS token (row and column) to the second CLS token
                attn[:, :, -1, :] = attn[:, :, 0, :]
                attn[:, :, :, -1] = attn[:, :, :, 0]
                # Divide both CLS columns by 2
                attn[:, :, :, 0] = attn[:, :, :, 0] / 2
                attn[:, :, :, -1] = attn[:, :, :, -1] / 2
                self.attention_maps[key] = attn

        return output, {
            'first_layer_input': self.first_layer_input,
            'attention_maps': self.attention_maps,
            'layers_mixer_output': self.layers_mixer_output,
            'layers_output': self.layers_output,
        }

    def _forward_with_attention(self, x):
        """Custom forward pass that forces attention weight computation."""
        x = self.model._process_input(x)
        n = x.shape[0]

        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = x + self.model.encoder.pos_embedding
        x = self.model.encoder.dropout(x)

        self.first_layer_input = x.clone().detach()
        if self.double_cls_token:
            self.first_layer_input = self._duplicate_cls_token(self.first_layer_input)

        for i, layer in enumerate(self.model.encoder.layers):
            x_norm = layer.ln_1(x)

            need_weights = i in self.layer_indices
            attn_out, attn_weights = layer.self_attention(
                x_norm, x_norm, x_norm,
                need_weights=need_weights,
                average_attn_weights=self.average_attn_weights
            )

            if self.average_attn_weights and attn_weights is not None:
                attn_weights = attn_weights.unsqueeze(1)

            if attn_weights is not None:
                self.attention_maps[f'layer_{i}'] = attn_weights.detach()

            if attn_out is not None:
                self.layers_mixer_output[f'layer_{i}'] = attn_out.clone().detach()
                if self.double_cls_token:
                    self.layers_mixer_output[f'layer_{i}'] = self._duplicate_cls_token(
                        self.layers_mixer_output[f'layer_{i}']
                    )

            x = layer.dropout(attn_out) + x

            y = layer.ln_2(x)
            y = layer.mlp(y)
            x = x + y

            if attn_out is not None:
                self.layers_output[f'layer_{i}'] = x.clone().detach()
                if self.double_cls_token:
                    self.layers_output[f'layer_{i}'] = self._duplicate_cls_token(
                        self.layers_output[f'layer_{i}']
                    )

        x = self.model.encoder.ln(x)
        x = x[:, 0]
        x = self.model.heads(x)

        return x

    def reconstruct_original_attention(self, attention_weight):
        """Inverse of double_cls_token: merge the duplicated CLS back into one."""
        if not self.double_cls_token:
            return attention_weight

        attn_weight = attention_weight.clone()
        if attn_weight.dim() == 2:
            attn_weight = attn_weight.unsqueeze(0).unsqueeze(1)

        attn_weight[:, :, :, 0] = attn_weight[:, :, :, 0] + attn_weight[:, :, :, -1]
        attn_weight = attn_weight[:, :, :-1, :-1]
        return attn_weight if attention_weight.dim() == 4 else attn_weight.squeeze()

    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
