"""Geometry-aware targets and coherent stochastic paths for RVQ depth.

The stochastic path is *coherent*: after a code is sampled at depth ``k``,
its decoded vector is subtracted before candidates for depth ``k+1`` are
computed.  This differs from independently corrupting already encoded token
IDs, which can create a prefix that no residual quantizer would produce.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StochasticRVQEncoding:
    """Compact top-k soft labels plus one sampled RVQ path."""

    codes: torch.Tensor
    topk_indices: torch.Tensor
    topk_probabilities: torch.Tensor
    target_entropy: torch.Tensor
    sampled_nonnearest_fraction: torch.Tensor
    changed_from_deterministic_fraction: torch.Tensor


def _topk_distance_distribution(
    vectors: torch.Tensor,
    embedding: torch.Tensor,
    *,
    topk: int,
    temperature: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest code IDs and dimensionless distance probabilities.

    Squared distances have codebook-specific units and scales.  For each
    residual vector we therefore divide distance gaps by the mean gap among
    its top-k candidates before applying the configured temperature.  This
    preserves the embedding geometry while making one temperature usable for
    upper, lower, and face codecs.
    """

    if vectors.dim() != 2 or embedding.dim() != 2:
        raise ValueError("vectors and embedding must both be rank-2 tensors")
    if vectors.shape[-1] != embedding.shape[-1]:
        raise ValueError(
            "Residual/codebook dimensions differ: "
            f"{vectors.shape[-1]} and {embedding.shape[-1]}."
        )
    if temperature <= 0:
        raise ValueError("RVQ sampling temperature must be positive.")
    if chunk_size <= 0:
        raise ValueError("RVQ distance chunk size must be positive.")

    candidate_count = min(int(topk), embedding.shape[0])
    if candidate_count < 2:
        raise ValueError("RVQ soft targets require at least two candidates.")

    embedding_float = embedding.detach().float()
    embedding_norm = embedding_float.square().sum(dim=-1).unsqueeze(0)
    all_indices = []
    all_probabilities = []
    for start in range(0, vectors.shape[0], chunk_size):
        vector_chunk = vectors[start:start + chunk_size].float()
        distances = (
            vector_chunk.square().sum(dim=-1, keepdim=True)
            + embedding_norm
            - 2.0 * vector_chunk @ embedding_float.transpose(0, 1)
        ).clamp_min_(0.0)
        nearest_distances, nearest_indices = distances.topk(
            candidate_count,
            dim=-1,
            largest=False,
            sorted=True,
        )
        gaps = nearest_distances - nearest_distances[:, :1]
        gap_scale = gaps[:, 1:].mean(dim=-1, keepdim=True).clamp_min_(
            1e-8
        )
        probabilities = torch.softmax(
            -gaps / (gap_scale * temperature),
            dim=-1,
        )
        all_indices.append(nearest_indices)
        all_probabilities.append(probabilities)

    return torch.cat(all_indices, dim=0), torch.cat(
        all_probabilities,
        dim=0,
    )


@torch.no_grad()
def encode_stochastic_rvq(
    codec,
    motion: torch.Tensor,
    *,
    topk: int,
    temperature: float,
    sample_probability: float,
    distance_chunk_size: int,
) -> tuple[torch.Tensor, StochasticRVQEncoding]:
    """Encode a deterministic target and a coherent stochastic alternative.

    The motion encoder runs only once.  The ordinary codec quantizer produces
    the deterministic tokens used by the frozen temporal transformer.  A
    second residual traversal produces top-k distance-aware targets and the
    stochastic tokens used as kinematic prefixes/targets.
    """

    if not 0 <= sample_probability <= 1:
        raise ValueError("RVQ sample_probability must lie in [0, 1].")

    latent = codec._encode_to_unquantized_latent(motion)
    if hasattr(codec, "_cast_for_module"):
        latent = codec._cast_for_module(latent, codec.quantizer)

    deterministic_codes = codec.quantizer.encode(latent)
    quantizer = codec.quantizer
    residual = quantizer.input_proj(latent)
    batch, _, time = residual.shape

    sampled_layers = []
    topk_index_layers = []
    topk_probability_layers = []
    sampled_nonnearest = []

    for layer in quantizer.vq.layers[:quantizer.n_q]:
        residual_btd = residual.transpose(1, 2)
        projected = layer.project_in(residual_btd)
        flat_projected = projected.reshape(-1, projected.shape[-1])
        topk_indices, topk_probabilities = _topk_distance_distribution(
            flat_projected,
            layer.embedding,
            topk=topk,
            temperature=temperature,
            chunk_size=distance_chunk_size,
        )

        sampled_candidate = torch.multinomial(
            topk_probabilities,
            num_samples=1,
        ).squeeze(-1)
        sampled_codes = topk_indices.gather(
            1,
            sampled_candidate.unsqueeze(-1),
        ).squeeze(-1)
        nearest_codes = topk_indices[:, 0]
        use_sample = torch.rand(
            sampled_codes.shape,
            device=sampled_codes.device,
        ) < sample_probability
        selected_codes = torch.where(
            use_sample,
            sampled_codes,
            nearest_codes,
        )

        selected_codes_bt = selected_codes.reshape(batch, time)
        quantized = layer.decode(selected_codes_bt).to(residual)
        residual = residual - quantized

        sampled_layers.append(selected_codes_bt)
        topk_index_layers.append(
            topk_indices.reshape(batch, time, -1)
        )
        topk_probability_layers.append(
            topk_probabilities.reshape(batch, time, -1)
        )
        sampled_nonnearest.append(selected_codes != nearest_codes)

    sampled_stack = torch.stack(sampled_layers, dim=1)
    topk_index_stack = torch.stack(topk_index_layers, dim=1)
    topk_probability_stack = torch.stack(
        topk_probability_layers,
        dim=1,
    )
    target_entropy = -(
        topk_probability_stack
        * topk_probability_stack.clamp_min(1e-12).log()
    ).sum(dim=-1).mean()
    sampled_nonnearest_fraction = torch.cat(
        sampled_nonnearest,
        dim=0,
    ).float().mean()
    changed_fraction = (
        sampled_stack != deterministic_codes
    ).float().mean()

    result = StochasticRVQEncoding(
        codes=sampled_stack,
        topk_indices=topk_index_stack,
        topk_probabilities=topk_probability_stack,
        target_entropy=target_entropy,
        sampled_nonnearest_fraction=sampled_nonnearest_fraction,
        changed_from_deterministic_fraction=changed_fraction,
    )
    return deterministic_codes, result
