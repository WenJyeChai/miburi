"""Native-rate codec reconstruction diagnostics, independent of legacy metrics.

Inputs are already canonically aligned by the codec runner. This module never
resamples, replaces the reconstructed jaw, integrates codec velocities, or
performs a rigid/Procrustes alignment. Positions use metres; SMPL-X geometry
errors are reported in millimetres. These are not legacy ReconMetrics scores.

Each metric stores ``sum``, ``count`` and ``reduction``. RMSE stores squared
errors and is rooted only AFTER pooling clips, never averaged across clip
RMSEs. Counts are valid scalar coordinates, joints or vertices as appropriate;
for fixed regions this weights clips by valid frames. Velocity counts use
adjacent valid frame pairs within each clip, including across internal chunks.
Only PyTorch is required for the analytic core; SMPL-X is imported on request.
"""

from collections.abc import Mapping, MutableMapping
import math
from pathlib import Path

import torch


def _sequence(value, trailing_dims, name):
    value = torch.as_tensor(value).detach()
    if value.ndim == trailing_dims + 1:
        value = value.unsqueeze(0)
    if value.ndim != trailing_dims + 2:
        raise ValueError(f"{name} has incompatible shape {tuple(value.shape)}")
    return value.to(dtype=torch.float64)


def _pair(target, reconstructed, trailing_dims, name):
    if target is None or reconstructed is None:
        raise ValueError(f"Both target and reconstructed {name} are required")
    target = _sequence(target, trailing_dims, f"target {name}")
    reconstructed = _sequence(reconstructed, trailing_dims, f"reconstructed {name}")
    if target.shape != reconstructed.shape:
        raise ValueError(f"{name} shapes differ: {target.shape} != {reconstructed.shape}")
    return target, reconstructed.to(device=target.device)


def _mask(reference, valid_mask, warmup_frames):
    if not isinstance(warmup_frames, int) or warmup_frames < 0:
        raise ValueError("warmup_frames must be a nonnegative integer")
    shape = reference.shape[:2]
    if valid_mask is None:
        valid = torch.ones(shape, device=reference.device, dtype=torch.bool)
    else:
        valid = torch.as_tensor(valid_mask, device=reference.device, dtype=torch.bool)
        if valid.ndim == 1 and shape[0] == 1:
            valid = valid.unsqueeze(0)
        if tuple(valid.shape) != tuple(shape):
            raise ValueError(f"valid_mask shape {valid.shape} != batch/time shape {shape}")
        valid = valid.clone()
    valid[:, :warmup_frames] = False
    return valid


def _fps(fps):
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    return fps


def _stat(values, valid, reduction="mean"):
    selected = values[valid].reshape(-1)
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("A valid reconstruction metric contains NaN or infinity")
    return {"sum": float(selected.sum().item()), "count": selected.numel(),
            "reduction": reduction}


def _indices(indices, size, name, device):
    values = torch.as_tensor(list(indices), dtype=torch.long, device=device)
    if values.ndim != 1 or not values.numel():
        raise ValueError(f"{name} must contain at least one index")
    if bool((values < 0).any()) or bool((values >= size).any()):
        raise ValueError(f"{name} contains an index outside [0, {size})")
    if values.unique().numel() != values.numel():
        raise ValueError(f"{name} contains duplicate indices")
    return values


def merge_metric_sums(accumulator: MutableMapping, metrics: Mapping):
    """Add sufficient statistics in place and return ``accumulator``."""
    for name, value in metrics.items():
        reduction = value["reduction"]
        if reduction not in ("mean", "rmse"):
            raise ValueError(f"Unknown reduction for {name}: {reduction}")
        if value["count"] < 0 or not math.isfinite(float(value["sum"])):
            raise ValueError(f"Invalid sufficient statistics for {name}")
        if name not in accumulator:
            accumulator[name] = {"sum": float(value["sum"]),
                                 "count": int(value["count"]),
                                 "reduction": reduction}
        else:
            existing = accumulator[name]
            if existing["reduction"] != reduction:
                raise ValueError(f"Cannot mix reductions for {name}")
            existing["sum"] += float(value["sum"])
            existing["count"] += int(value["count"])
    return accumulator


def finalize_metrics(metrics: Mapping):
    """Return pooled scalar metrics; zero valid observations produce ``None``."""
    result = {}
    for name, value in metrics.items():
        if value["count"] == 0:
            result[name] = None
        elif value["reduction"] == "rmse":
            result[name] = math.sqrt(max(0.0, value["sum"] / value["count"]))
        elif value["reduction"] == "mean":
            result[name] = value["sum"] / value["count"]
        else:
            raise ValueError(f"Unknown reduction for {name}: {value['reduction']}")
    return result


def reconstruction_metrics(
    *, target_rotations=None, reconstructed_rotations=None, rotation_regions=None,
    target_expressions=None, reconstructed_expressions=None,
    target_translation=None, reconstructed_translation=None,
    target_translation_velocity=None, reconstructed_translation_velocity=None,
    target_contacts=None, reconstructed_contacts=None,
    valid_mask=None, warmup_frames=0, fps=25,
):
    """Measure part-local canonical rotations, expressions and root translation.

    Rotations: [T,J,3,3] or [B,T,J,3,3], assumed proper rotation matrices.
    Regions map names to part-local joint indices. Vectors: [T,D] or [B,T,D].
    Translation and direct decoder velocity have exactly three coordinates.
    Contacts have four coordinates in their native 0/1 target units; decoded
    values are scored directly, without clipping or a classification threshold.
    Supplied decoder velocities are already m/s and are NOT multiplied by FPS;
    the separate finite-difference translation metric uses FPS explicitly.
    """
    fps = _fps(fps)
    metrics = {}
    reference_shape = None

    def checked_pair(target, reconstructed, trailing_dims, name):
        nonlocal reference_shape
        target, reconstructed = _pair(target, reconstructed, trailing_dims, name)
        if reference_shape is not None and target.shape[:2] != reference_shape:
            raise ValueError("All modalities must share the same batch/time shape")
        reference_shape = target.shape[:2]
        return target, reconstructed, _mask(target, valid_mask, warmup_frames)

    if target_rotations is not None or reconstructed_rotations is not None:
        target, predicted, valid = checked_pair(
            target_rotations, reconstructed_rotations, 3, "rotations")
        if target.shape[-2:] != (3, 3):
            raise ValueError("Rotations must end in [3,3]")
        relative = predicted @ target.transpose(-1, -2)
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
        skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                            relative[..., 0, 2] - relative[..., 2, 0],
                            relative[..., 1, 0] - relative[..., 0, 1]), dim=-1)
        sine = torch.linalg.vector_norm(skew, dim=-1) / 2
        angles = torch.rad2deg(torch.atan2(sine, cosine))
        regions = rotation_regions if rotation_regions is not None else {"all": range(target.shape[2])}
        for region, ids in regions.items():
            ids = _indices(ids, target.shape[2], region, target.device)
            metrics[f"{region}_rotation_geodesic_deg"] = _stat(angles.index_select(2, ids), valid)
    elif rotation_regions is not None:
        raise ValueError("rotation_regions requires rotation tensors")

    for target_value, predicted_value, name, dimension in (
        (target_expressions, reconstructed_expressions, "expression_rmse", None),
        (target_translation, reconstructed_translation, "translation_rmse_m", 3),
        (target_translation_velocity, reconstructed_translation_velocity,
         "translation_velocity_direct_rmse_m_s", 3),
        (target_contacts, reconstructed_contacts, "contact_rmse", 4),
    ):
        if target_value is None and predicted_value is None:
            continue
        target, predicted, valid = checked_pair(target_value, predicted_value, 1, name)
        if dimension is not None and target.shape[-1] != dimension:
            raise ValueError(f"{name} requires {dimension} coordinates")
        metrics[name] = _stat((predicted - target).square(), valid, "rmse")
        if name == "translation_rmse_m":
            pairs = valid[:, 1:] & valid[:, :-1]
            velocity_error = ((predicted[:, 1:] - predicted[:, :-1])
                              - (target[:, 1:] - target[:, :-1])) * fps
            metrics["translation_velocity_rmse_m_s"] = _stat(velocity_error.square(), pairs, "rmse")
    if reference_shape is None:
        raise ValueError("At least one target/reconstruction pair is required")
    return metrics


def geometry_metrics(
    *, target_joints=None, reconstructed_joints=None, joint_regions=None,
    target_face_vertices=None, reconstructed_face_vertices=None,
    face_metric_prefix="face_vertices", valid_mask=None, warmup_frames=0, fps=25,
):
    """Measure supplied metre-valued joints/vertices [T,N,3] or [B,T,N,3].

    MPJPE and mean vertex L2 average Euclidean distances. Coordinate MAE/RMSE
    average individual x/y/z errors. Vertex velocity is a coordinate RMSE in
    mm/s using differences of predicted frames and differences of GT frames.
    No translation, rotation, jaw or other alignment is modified here.
    """
    fps = _fps(fps)
    metrics = {}
    reference_shape = None
    if target_joints is not None or reconstructed_joints is not None:
        target, predicted = _pair(target_joints, reconstructed_joints, 2, "joints")
        if target.shape[-1] != 3:
            raise ValueError("Joints must have three coordinates")
        reference_shape = target.shape[:2]
        valid = _mask(target, valid_mask, warmup_frames)
        distances = torch.linalg.vector_norm((predicted - target) * 1000, dim=-1)
        regions = joint_regions if joint_regions is not None else {"all": range(target.shape[2])}
        for region, ids in regions.items():
            ids = _indices(ids, target.shape[2], region, target.device)
            metrics[f"{region}_mpjpe_mm"] = _stat(distances.index_select(2, ids), valid)
    elif joint_regions is not None:
        raise ValueError("joint_regions requires joint tensors")
    if target_face_vertices is not None or reconstructed_face_vertices is not None:
        target, predicted = _pair(target_face_vertices, reconstructed_face_vertices, 2, "face vertices")
        if target.shape[-1] != 3:
            raise ValueError("Vertices must have three coordinates")
        if reference_shape is not None and reference_shape != target.shape[:2]:
            raise ValueError("Joints and vertices must have matching batch/time dimensions")
        valid = _mask(target, valid_mask, warmup_frames)
        error_mm = (predicted - target) * 1000
        metrics[f"{face_metric_prefix}_coordinate_mae_mm"] = _stat(error_mm.abs(), valid)
        metrics[f"{face_metric_prefix}_coordinate_rmse_mm"] = _stat(error_mm.square(), valid, "rmse")
        metrics[f"{face_metric_prefix}_mean_vertex_l2_mm"] = _stat(torch.linalg.vector_norm(error_mm, dim=-1), valid)
        pairs = valid[:, 1:] & valid[:, :-1]
        velocity_error = ((predicted[:, 1:] - predicted[:, :-1])
                          - (target[:, 1:] - target[:, :-1])) * (1000 * fps)
        metrics[f"{face_metric_prefix}_velocity_coordinate_rmse_mm_s"] = _stat(velocity_error.square(), pairs, "rmse")
    return metrics


class SMPLXGeometryEvaluator:
    """Optional chunked geometry audit on an explicitly selected device.

    Joint MPJPE uses the supplied full poses and canonical world translation.
    Face deformation uses neutral body/hands/global orientation/translation,
    with each stream's OWN jaw, expression and eye rotations. Without explicit
    face_vertex_indices it measures all mesh vertices and says so in every
    metric name; SMPL-X's neutral-body forward is not itself a face ROI.
    """

    def __init__(self, args, device="cpu", chunk_size=32, face_vertex_indices=None):
        import smplx

        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.device = torch.device(device)
        self.chunk_size = chunk_size
        self.num_betas = int(getattr(args, "smplx_num_betas", 300))
        self.num_expressions = int(getattr(args, "smplx_num_expression_coeffs", 100))
        model_path = getattr(args, "smplx_model_path", None)
        if model_path is None:
            model_path = Path(args.deps_path) / "smplx_2020"
        self.model = smplx.create(
            str(model_path), model_type="smplx",
            gender=getattr(args, "smplx_gender", "NEUTRAL_2020"),
            ext="npz", use_face_contour=False, use_pca=False,
            num_betas=self.num_betas, num_expression_coeffs=self.num_expressions,
        ).to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        self.face_vertex_indices = None if face_vertex_indices is None else list(face_vertex_indices)
        self.face_metric_prefix = ("face_deformation_all_vertices" if face_vertex_indices is None
                                   else "face_deformation_roi_vertices")

    @staticmethod
    def _pose(value, name):
        value = torch.as_tensor(value)
        if value.shape[-1] == 165:
            value = value.reshape(*value.shape[:-1], 55, 3)
        value = _sequence(value, 2, name)
        if value.shape[-2:] != (55, 3):
            raise ValueError(f"{name} must contain 55 axis-angle joints")
        return value

    def _forward(self, pose, expressions, translation, betas, face_only):
        n = pose.shape[0]
        zero3 = torch.zeros(n, 3, device=self.device)
        kwargs = dict(
            betas=betas, expression=expressions,
            jaw_pose=pose[:, 22], leye_pose=pose[:, 23], reye_pose=pose[:, 24],
            transl=zero3 if face_only else translation,
            global_orient=zero3 if face_only else pose[:, 0],
            body_pose=torch.zeros(n, 63, device=self.device) if face_only else pose[:, 1:22].reshape(n, 63),
            left_hand_pose=torch.zeros(n, 45, device=self.device) if face_only else pose[:, 25:40].reshape(n, 45),
            right_hand_pose=torch.zeros(n, 45, device=self.device) if face_only else pose[:, 40:55].reshape(n, 45),
        )
        output = self.model(**kwargs, return_verts=face_only)
        if not face_only:
            return output.joints[:, :55].detach().cpu()
        vertices = output.vertices
        if self.face_vertex_indices is not None:
            ids = _indices(self.face_vertex_indices, vertices.shape[1], "face_vertex_indices", vertices.device)
            vertices = vertices.index_select(1, ids)
        return vertices.detach().cpu()

    @torch.no_grad()
    def evaluate(
        self, *, target_pose_aa, reconstructed_pose_aa, betas,
        target_expressions, reconstructed_expressions,
        target_translation, reconstructed_translation, joint_regions,
        valid_mask=None, warmup_frames=0, fps=25,
        include_joints=True, include_face_vertices=True,
    ):
        """Return sufficient statistics without retaining a full-clip mesh.

        Chunks overlap by one frame so velocity metrics do not lose chunk
        boundaries; static observations in the overlap are counted once.
        Padded/invalid frames never enter SMPL-X or a velocity pair.
        Per-part callers should enable joints for upper/lower and face
        vertices for the face codec, avoiding unchanged-GT metric entries.
        """
        fps = _fps(fps)
        target_pose = self._pose(target_pose_aa, "target_pose_aa").cpu()
        predicted_pose = self._pose(reconstructed_pose_aa, "reconstructed_pose_aa").cpu()
        if target_pose.shape != predicted_pose.shape:
            raise ValueError("Target and reconstructed pose shapes differ")
        batch, steps = target_pose.shape[:2]
        expressions = _pair(target_expressions, reconstructed_expressions, 1, "expressions")
        translations = _pair(target_translation, reconstructed_translation, 1, "translation")
        for value in (*expressions, *translations):
            if value.shape[:2] != (batch, steps):
                raise ValueError("All geometry inputs must share batch/time dimensions")
        if expressions[0].shape[-1] != self.num_expressions or translations[0].shape[-1] != 3:
            raise ValueError("Expression/translation dimensions do not match the geometry model")
        expressions = tuple(value.cpu() for value in expressions)
        translations = tuple(value.cpu() for value in translations)
        valid = _mask(target_pose, valid_mask, warmup_frames)
        beta = torch.as_tensor(betas).detach().cpu()
        if beta.ndim == 1:
            beta = beta[None, None].expand(batch, steps, -1)
        elif beta.ndim == 2:
            if beta.shape[0] == batch:
                beta = beta[:, None].expand(-1, steps, -1)
            elif batch == 1 and beta.shape[0] == steps:
                beta = beta[None]
            else:
                raise ValueError("Betas must be per clip or per frame")
        if beta.ndim != 3 or beta.shape[:2] != (batch, steps) or beta.shape[-1] < self.num_betas:
            raise ValueError("Betas have incompatible dimensions")
        beta = beta[..., :self.num_betas]
        totals = {}
        for b in range(batch):
            for start in range(0, steps, self.chunk_size):
                begin, end = max(0, start - 1), min(steps, start + self.chunk_size)
                chunk_valid = valid[b, begin:end]
                if not bool(chunk_valid.any()):
                    continue
                ids = chunk_valid.nonzero(as_tuple=True)[0]
                geometry = []
                for pose, expression, translation in zip(
                    (target_pose, predicted_pose), expressions, translations,
                ):
                    inputs = [value[b, begin:end].index_select(0, ids).to(
                        device=self.device, dtype=torch.float32,
                    ) for value in (pose, expression, translation, beta)]
                    if not all(bool(torch.isfinite(value).all()) for value in inputs):
                        raise ValueError("Valid SMPL-X inputs contain NaN or infinity")
                    pieces = {}
                    for name, enabled, face_only in (
                        ("joints", include_joints, False),
                        ("vertices", include_face_vertices, True),
                    ):
                        if enabled:
                            values = self._forward(*inputs, face_only=face_only)
                            full_values = torch.zeros(end - begin, *values.shape[1:])
                            full_values[ids] = values
                            pieces[name] = full_values
                    geometry.append(pieces)
                kwargs = dict(
                    target_joints=geometry[0].get("joints"), reconstructed_joints=geometry[1].get("joints"),
                    target_face_vertices=geometry[0].get("vertices"), reconstructed_face_vertices=geometry[1].get("vertices"),
                    joint_regions=joint_regions if include_joints else None,
                    face_metric_prefix=self.face_metric_prefix,
                    valid_mask=chunk_valid, fps=fps,
                )
                stats = geometry_metrics(**kwargs)
                if begin < start:
                    overlap_kwargs = dict(kwargs)
                    for name in ("target_joints", "reconstructed_joints",
                                 "target_face_vertices", "reconstructed_face_vertices", "valid_mask"):
                        if kwargs[name] is not None:
                            overlap_kwargs[name] = kwargs[name][:1]
                    overlap = geometry_metrics(**overlap_kwargs)
                    for name, value in overlap.items():
                        stats[name]["sum"] -= value["sum"]
                        stats[name]["count"] -= value["count"]
                merge_metric_sums(totals, stats)
        return totals
