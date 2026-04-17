#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import zarr
from loguru import logger
from numcodecs import Blosc
from omegaconf import OmegaConf
from tqdm import tqdm

from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.lightning_modules import load_module

# ============================================================
# Settings
# ============================================================
device = "cuda:0"
rollout_iterations = 40  # 10-day rollout (lead_time_hours=6 -> 4 steps/day)

model_path = ""
logger.add("run_det.log", enqueue=True, level="INFO", mode="w")
ensemble_modules = [
    # "archesweather-m-seed0",
    # "archesweather-m-seed1",
    "archesweather-m-seed0-6h-32x64-block16-Rope",
    # "archesweather-m-skip-seed1",
]

out_zarr = "/lustre/fswork/projects/rech/jwo/ukv19eg/geoarches/forecast_result/forecast_output_noskip_det_block16_Rope.zarr"

# ============================================================
# Load data
# ============================================================
logger.info("Loading data...")
ds = Era5Forecast(
    path="data/era5_240/full",
    load_prev=False,
    norm_scheme="pangu",
    domain="test",
    lead_time_hours=6,
)


def _resolve_model_dir(model_ref: str) -> Path:
    model_dir = Path("modelstore") / model_ref
    return model_dir if model_dir.exists() else Path(model_ref)


def _assert_linvert_disabled(model_ref: str) -> None:
    model_dir = _resolve_model_dir(model_ref)
    cfg = OmegaConf.load(model_dir / "config.yaml")
    first_interaction_layer = cfg.module.backbone.get("first_interaction_layer", None)
    logger.info(
        "Model {} first_interaction_layer={}",
        model_ref,
        first_interaction_layer,
    )
    if first_interaction_layer is not None:
        raise ValueError(
            f"Model {model_ref} still enables LinVert "
            f"(first_interaction_layer={first_interaction_layer!r})."
        )


def _init_zarr_store(
    filename: str,
    *,
    n_times: int,
    lead_times: int,
    title: str,
    source: str,
    description: str,
) -> tuple[zarr.Group, Dict[str, np.ndarray], Dict[str, Dict[str, zarr.Array]]]:
    latitudes = np.linspace(-90, 90, 32)
    longitudes = np.linspace(0, 358.5, 64)
    levels = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    surface_level = np.array([1])
    surface_vars = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
    ]
    level_vars = [
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "specific_humidity",
        "vertical_velocity",
    ]
    units_dict = {
        "geopotential": "m²/s²",
        "u_component_of_wind": "m/s",
        "v_component_of_wind": "m/s",
        "temperature": "K",
        "specific_humidity": "kg/kg",
        "vertical_velocity": "Pa/s",
        "10m_u_component_of_wind": "m/s",
        "10m_v_component_of_wind": "m/s",
        "2m_temperature": "K",
        "mean_sea_level_pressure": "Pa",
    }

    root = zarr.group(store=filename, overwrite=True)
    root.attrs.update(
        {
            "title": title,
            "institution": "IRNS",
            "source": source,
            "history": f"Created at {np.datetime64('now')}",
            "Conventions": "CF-1.8",
            "description": description,
        }
    )

    coords = {
        "start_time": np.arange(n_times, dtype=np.int32),
        "lead_time": np.arange(lead_times, dtype=np.int32),
        "level": levels.astype(np.int32),
        "latitude": latitudes.astype(np.float32),
        "longitude": longitudes.astype(np.float32),
        "surface_level": surface_level.astype(np.int32),
    }
    for name, data in coords.items():
        # Use fill_value=None for coordinate arrays to avoid masking valid 0 values
        # (e.g., start_time=0, lead_time=0, longitude=0) as NaN when read by xarray.
        arr = root.create_dataset(name, data=data, overwrite=True, fill_value=None)
        arr.attrs["_ARRAY_DIMENSIONS"] = [name]

    compressor = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
    level_chunks = (
        1,
        lead_times,
        len(levels),
        len(latitudes),
        len(longitudes),
    )
    surface_chunks = (
        1,
        lead_times,
        len(surface_level),
        len(latitudes),
        len(longitudes),
    )

    level_handles: Dict[str, zarr.Array] = {}
    surface_handles: Dict[str, zarr.Array] = {}
    for var_name in level_vars:
        var = root.create_dataset(
            var_name,
            shape=(n_times, lead_times, len(levels), len(latitudes), len(longitudes)),
            chunks=level_chunks,
            dtype="f4",
            compressor=compressor,
            fill_value=np.nan,
        )
        var.attrs["_ARRAY_DIMENSIONS"] = [
            "start_time",
            "lead_time",
            "level",
            "latitude",
            "longitude",
        ]
        var.attrs["long_name"] = var_name
        if var_name in units_dict:
            var.attrs["units"] = units_dict[var_name]
        level_handles[var_name] = var
    for var_name in surface_vars:
        var = root.create_dataset(
            var_name,
            shape=(
                n_times,
                lead_times,
                len(surface_level),
                len(latitudes),
                len(longitudes),
            ),
            chunks=surface_chunks,
            dtype="f4",
            compressor=compressor,
            fill_value=np.nan,
        )
        var.attrs["_ARRAY_DIMENSIONS"] = [
            "start_time",
            "lead_time",
            "surface_level",
            "latitude",
            "longitude",
        ]
        var.attrs["long_name"] = var_name
        if var_name in units_dict:
            var.attrs["units"] = units_dict[var_name]
        surface_handles[var_name] = var

    meta = {
        "levels": levels,
        "latitudes": latitudes,
        "longitudes": longitudes,
        "level_vars": level_vars,
        "surface_vars": surface_vars,
    }
    handles = {"level": level_handles, "surface": surface_handles}
    return root, meta, handles


def _write_step_to_zarr(
    handles: Dict[str, Dict[str, zarr.Array]],
    *,
    start_time: int,
    level_data: np.ndarray,
    surface_data: np.ndarray,
    level_vars: List[str],
    surface_vars: List[str],
) -> None:
    for i, var_name in enumerate(level_vars):
        handles["level"][var_name][start_time, :, :, :, :] = level_data[:, i, :, :, :]
    for i, var_name in enumerate(surface_vars):
        handles["surface"][var_name][start_time, :, :, :, :] = surface_data[:, i, :, :, :]


def main() -> None:
    for model_name in ensemble_modules:
        _assert_linvert_disabled(model_path + model_name)

    n_times = len(ds)
    if os.path.isdir(out_zarr):
        shutil.rmtree(out_zarr)
        logger.info("Removed existing output directory: {}", out_zarr)
    elif os.path.exists(out_zarr):
        os.remove(out_zarr)
        logger.info("Removed existing output file: {}", out_zarr)

    logger.info("Loading deterministic avg model...")
    model, _config = load_module(
        model_path + ensemble_modules[0],
        avg_with_modules=[model_path + m for m in ensemble_modules[1:]],
        ckpt_fname="270000",
    )
    model = model.to(device)
    torch.set_grad_enabled(False)
    logger.info("Deterministic avg model loaded.")

    logger.info("Rolling out deterministic model...")
    store_det, meta_det, handles_det = _init_zarr_store(
        out_zarr,
        n_times=n_times,
        lead_times=rollout_iterations,
        title="archesweather deterministic ensemble forecast (avg model)",
        source="archesweather-m x4 (avg)",
        description=(
            "deterministic avg forecast (single deterministic output), "
            f"n_times={n_times}, lead_times={rollout_iterations}"
        ),
    )
    with torch.inference_mode():
        for itime in tqdm(range(n_times), desc="start_time"):
            if itime == 0:
                logger.info("Rolling out deterministic model across {} start_times", n_times)
            batch = {k: v[None].to(device, non_blocking=True) for k, v in ds[itime].items()}
            pred = model.forward_multistep(batch, iters=rollout_iterations, use_avg=True)
            pred = ds.denormalize(pred)
            level_np = pred["level"].squeeze(0).cpu().numpy()
            surface_np = pred["surface"].squeeze(0).cpu().numpy()
            _write_step_to_zarr(
                handles_det,
                start_time=itime,
                level_data=level_np,
                surface_data=surface_np,
                level_vars=meta_det["level_vars"],
                surface_vars=meta_det["surface_vars"],
            )
    logger.info("All done.")


if __name__ == "__main__":
    main()
