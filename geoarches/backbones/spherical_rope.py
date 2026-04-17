import numpy as np
import torch
from torch import nn

try:
    from scipy.special import sph_harm_y as _sph_harm

    def _complex_sph_harm(degree, m, theta, phi):
        return _sph_harm(degree, m, theta, phi)
except ImportError:
    from scipy.speciadegree import sph_harm as _sph_harm

    def _complex_sph_harm(degree, m, theta, phi):
        return _sph_harm(m, degree, phi, theta)


def real_spherical_harmonics(degree, m, theta, phi):
    if m > 0:
        return np.sqrt(2.0) * ((-1) ** m) * np.real(_complex_sph_harm(degree, m, theta, phi))
    if m < 0:
        return np.sqrt(2.0) * ((-1) ** m) * np.imag(_complex_sph_harm(degree, m, theta, phi))
    return np.real(_complex_sph_harm(degree, 0, theta, phi))


def latlon_to_cartesian(colat, lon):
    sin_colat = torch.sin(colat)
    x = sin_colat * torch.cos(lon)
    y = sin_colat * torch.sin(lon)
    z = torch.cos(colat)
    return torch.stack([x, y, z], dim=-1)


def cartesian_to_latlon(xyz):
    xyz = xyz / xyz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z = xyz.unbind(dim=-1)
    colat = torch.acos(z.clamp(-1.0, 1.0))
    lon = torch.remainder(torch.atan2(y, x), 2 * torch.pi)
    return torch.stack([colat, lon], dim=-1)


def pool_sphere_coords_2x2(coords):
    lat, lon, _ = coords.shape
    if lat % 2 != 0 or lon % 2 != 0:
        raise ValueError(f"Expected even spatial size, got {(lat, lon)}")

    xyz = latlon_to_cartesian(coords[..., 0], coords[..., 1])
    xyz = xyz.reshape(lat // 2, 2, lon // 2, 2, 3).mean(dim=(1, 3))
    return cartesian_to_latlon(xyz)


def build_stage_coords(raw_spatial_size=(32, 64)):
    raw_lat, raw_lon = raw_spatial_size
    lat_deg = np.linspace(90.0, -90.0, raw_lat, dtype=np.float64)
    lon_deg = np.linspace(0.0, 360.0 - 360.0 / raw_lon, raw_lon, dtype=np.float64)

    colat = np.deg2rad(90.0 - lat_deg)
    lon = np.deg2rad(lon_deg)
    colat2d, lon2d = np.meshgrid(colat, lon, indexing="ij")
    raw_coords = torch.stack(
        [torch.from_numpy(colat2d), torch.from_numpy(lon2d)],
        dim=-1,
    ).float()

    stage1_coords = pool_sphere_coords_2x2(raw_coords)
    stage2_coords = pool_sphere_coords_2x2(stage1_coords)
    return stage1_coords, stage2_coords


def get_spherical_features_from_coords(coords_flat, max_l=7):
    coords_np = coords_flat.detach().cpu().numpy()
    colats = coords_np[:, 0]
    lons = coords_np[:, 1]

    features = []
    for degree in range(max_l + 1):
        for m in range(-degree, degree + 1):
            features.append(real_spherical_harmonics(degree, m, colats, lons))

    features = np.stack(features, axis=-1)
    features_norm = np.sqrt((features**2).mean(axis=0, keepdims=True) + 1e-12)
    features = features / features_norm
    return torch.tensor(features, dtype=torch.float32)


def build_stage_mesh_features(raw_spatial_size=(32, 64), max_l=7):
    stage1_coords, stage2_coords = build_stage_coords(raw_spatial_size)
    stage1_flat = stage1_coords.reshape(-1, 2)
    stage2_flat = stage2_coords.reshape(-1, 2)
    stage1_mesh = get_spherical_features_from_coords(stage1_flat, max_l=max_l)
    stage2_mesh = get_spherical_features_from_coords(stage2_flat, max_l=max_l)
    return stage1_mesh, stage2_mesh


class SphericalRopeLayer(nn.Module):
    def __init__(self, head_dim, n_heads, mesh_features, alpha=1.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")

        self.head_dim = head_dim
        self.n_heads = n_heads
        self.alpha = alpha

        mesh_features = mesh_features.float()
        self.register_buffer("mesh_features", mesh_features)
        self.weights = nn.Parameter(torch.randn(n_heads, mesh_features.shape[-1], head_dim // 2))

    def _angles(self):
        angles = torch.einsum("nm,hmd->hnd", self.mesh_features, self.weights)
        angles = angles * self.alpha
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(2)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(2)
        return cos, sin

    def forward(self, tensor):
        if tensor.shape[-1] != self.head_dim:
            raise ValueError(f"Expected last dim {self.head_dim}, got {tensor.shape[-1]}")
        if tensor.shape[1] != self.n_heads:
            raise ValueError(f"Expected head dim {self.n_heads}, got {tensor.shape[1]}")
        if tensor.shape[-2] != self.mesh_features.shape[0]:
            raise ValueError(
                f"Expected token dim {self.mesh_features.shape[0]}, got {tensor.shape[-2]}"
            )

        cos, sin = self._angles()
        cos = cos.to(dtype=tensor.dtype, device=tensor.device)
        sin = sin.to(dtype=tensor.dtype, device=tensor.device)

        x_even = tensor[..., 0::2]
        x_odd = tensor[..., 1::2]
        r_even = x_even * cos - x_odd * sin
        r_odd = x_even * sin + x_odd * cos
        return torch.stack([r_even, r_odd], dim=-1).reshape_as(tensor)
