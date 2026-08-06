"""CPU smoke checks for coherent stochastic RVQ depth regularization."""

from types import SimpleNamespace

import torch
from torch import nn

from miburi.models.rvq_depth_regularization import encode_stochastic_rvq


class _Layer(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.project_in = nn.Identity()
        self.project_out = nn.Identity()
        self.register_buffer("embedding", torch.tensor(embedding).float())

    def decode(self, codes):
        values = self.embedding[codes]
        return values.transpose(1, 2)


class _Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Identity()
        self.n_q = 2
        self.vq = SimpleNamespace(
            layers=nn.ModuleList(
                [
                    _Layer(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
                    _Layer(((0.0, 0.0), (0.5, 0.0), (0.0, 0.5))),
                ]
            )
        )

    def encode(self, latent):
        residual = latent
        codes = []
        for layer in self.vq.layers:
            vectors = residual.transpose(1, 2)
            distances = torch.cdist(vectors, layer.embedding.unsqueeze(0))
            indices = distances.argmin(dim=-1)
            residual = residual - layer.decode(indices)
            codes.append(indices)
        return torch.stack(codes, dim=1)


class _Codec:
    def __init__(self):
        self.quantizer = _Quantizer()

    def _encode_to_unquantized_latent(self, motion):
        return motion


def test_zero_sampling_reproduces_codec_path():
    codec = _Codec()
    latent = torch.tensor(
        [
            [
                [0.9, 0.1, 0.2],
                [0.1, 0.8, 0.2],
            ]
        ]
    )
    deterministic, stochastic = encode_stochastic_rvq(
        codec,
        latent,
        topk=3,
        temperature=0.5,
        sample_probability=0.0,
        distance_chunk_size=2,
    )
    torch.testing.assert_close(stochastic.codes, deterministic)
    torch.testing.assert_close(
        stochastic.topk_probabilities.sum(dim=-1),
        torch.ones_like(stochastic.topk_probabilities[..., 0]),
    )
    assert stochastic.topk_indices.shape == (1, 2, 3, 3)
    assert stochastic.changed_from_deterministic_fraction.item() == 0


def test_sampled_path_has_valid_codes_and_finite_soft_targets():
    torch.manual_seed(7)
    codec = _Codec()
    latent = torch.rand(2, 2, 5)
    deterministic, stochastic = encode_stochastic_rvq(
        codec,
        latent,
        topk=3,
        temperature=1.0,
        sample_probability=1.0,
        distance_chunk_size=4,
    )
    assert stochastic.codes.shape == deterministic.shape == (2, 2, 5)
    assert stochastic.codes.min() >= 0
    assert stochastic.codes.max() < 3
    assert torch.isfinite(stochastic.topk_probabilities).all()
    assert torch.isfinite(stochastic.target_entropy)


if __name__ == "__main__":
    test_zero_sampling_reproduces_codec_path()
    test_sampled_path_has_valid_codes_and_finite_soft_targets()
    print("rvq depth regularization smoke checks passed")
