#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple
import traceback
import re

import h5py
import numpy as np

INPUT_DIR = Path(r"D:\data\机场")
OUTPUT_DIR = INPUT_DIR / "converted_envi"
# Export ancillary lat/lon files. Set True only when needed.
EXPORT_LATLON = False
# For AVIRIS L1B/RDN: use geolocation_lookup_table to output orthorectified ENVI grid.
ORTHORECTIFY_L1B_WITH_GLT = True
# Attach richer source metadata into ENVI header as custom fields.
INCLUDE_NC_METADATA_IN_HDR = True
# Also write additional ENVI-standard-like optional fields for better UX in ENVI.
INCLUDE_OPTIONAL_ENVI_STANDARD_FIELDS = True
# Write ENVI bad bands list (bbl) if available or derivable.
INCLUDE_BBL = True
# If no explicit bbl exists in nc, derive from wavelength ranges (nm).
AUTO_BBL_FROM_WAVELENGTH = True
# Default set tuned to reproduce the cor-1 style BBL (249/284 kept).
# Intervals are in nanometers and are applied only when no explicit BBL exists in nc.
DEFAULT_BAD_BAND_RANGES_NM = [
    (1350.0, 1433.0),
    (1804.0, 1968.5),
]

CANDIDATE_CUBE_PATHS = [
    "/radiance/radiance",
    "/reflectance/reflectance",
]
CANDIDATE_WAVELENGTH_PATHS = [
    "/radiance/wavelength",
    "/reflectance/wavelength",
    "/wavelength",
]
CANDIDATE_FWHM_PATHS = [
    "/radiance/fwhm",
    "/reflectance/fwhm",
    "/fwhm",
]
CANDIDATE_LAT_PATHS = ["/lat", "/latitude"]
CANDIDATE_LON_PATHS = ["/lon", "/longitude"]

# Used for axis inference.
CANDIDATE_LINES_PATHS = ["/lines", "/line", "/northing", "/y"]
CANDIDATE_SAMPLES_PATHS = ["/samples", "/sample", "/easting", "/x"]

# Used for projected header metadata.
CANDIDATE_EASTING_PATHS = ["/easting", "/x", "/geolocation_lookup_table/easting"]
CANDIDATE_NORTHING_PATHS = ["/northing", "/y", "/geolocation_lookup_table/northing"]

# Used for L1B orthorectification.
CANDIDATE_GLT_LINE_PATHS = ["/geolocation_lookup_table/line"]
CANDIDATE_GLT_SAMPLE_PATHS = ["/geolocation_lookup_table/sample"]
CANDIDATE_BBL_PATHS = [
    "/bbl",
    "/radiance/bbl",
    "/reflectance/bbl",
    "/good_wavelengths",
    "/radiance/good_wavelengths",
    "/reflectance/good_wavelengths",
    "/quality/bbl",
    "/quality/good_wavelengths",
]

SKIP_ATTR_KEYS_COMMON = {"DIMENSION_LIST", "REFERENCE_LIST", "_Netcdf4Coordinates", "CLASS", "NAME"}
SKIP_ATTR_KEYS_GLOBAL = {"_NCProperties"}


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return [x.decode("utf-8", errors="ignore") for x in value.tolist()]
    return value


def _find_existing_path(h5: h5py.File, candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p in h5:
            return p
    return None


def _find_first_3d_dataset(h5: h5py.File) -> Optional[str]:
    found: List[str] = []

    def visitor(name: str, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            found.append("/" + name)

    h5.visititems(visitor)
    return found[0] if found else None


def _infer_units_from_values(values: np.ndarray) -> str:
    values = np.asarray(values).astype(float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "Nanometers"
    vmax = float(np.nanmax(finite))
    return "Nanometers" if vmax > 100 else "Micrometers"


def _format_envi_list(values: np.ndarray, per_line: int = 8) -> str:
    flat = np.asarray(values).reshape(-1)
    chunks = []
    for i in range(0, flat.size, per_line):
        row = ", ".join(f"{float(x):.6f}" for x in flat[i : i + per_line])
        chunks.append("  " + row)
    return "{\n" + ",\n".join(chunks) + "\n}"


def _format_envi_int_list(values: np.ndarray, per_line: int = 32) -> str:
    flat = np.asarray(values).reshape(-1).astype(np.int32)
    chunks = []
    for i in range(0, flat.size, per_line):
        row = ", ".join(str(int(x)) for x in flat[i : i + per_line])
        chunks.append("  " + row)
    return "{\n" + ",\n".join(chunks) + "\n}"


def _format_envi_str_list(values: List[str], per_line: int = 4) -> str:
    safe = [_clean_header_text(v, max_len=120).replace(",", ";") for v in values]
    chunks = []
    for i in range(0, len(safe), per_line):
        row = ", ".join(safe[i : i + per_line])
        chunks.append("  " + row)
    return "{\n" + ",\n".join(chunks) + "\n}"


def _clean_header_text(text: str, max_len: int = 1200) -> str:
    t = str(text).replace("\n", " ").replace("\r", " ").strip()
    t = t.replace("{", "(").replace("}", ")")
    t = re.sub(r"\s+", " ", t)
    if len(t) > max_len:
        t = t[: max_len - 3] + "..."
    return t


def _sanitize_header_key(name: str) -> str:
    key = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_").lower()
    return key or "attr"


def _to_attr_python(value):
    v = _decode_attr(value)
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return v.item()
        return np.asarray(v).reshape(-1).tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _format_attr_value_for_hdr(value) -> Optional[str]:
    v = _to_attr_python(value)
    if v is None:
        return None

    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return None
        return f"{float(v):.12g}"
    if isinstance(v, str):
        return "{" + _clean_header_text(v) + "}"

    if isinstance(v, (list, tuple)):
        flat = []
        for x in v:
            xv = _to_attr_python(x)
            if isinstance(xv, (list, tuple)):
                flat.extend(xv)
            else:
                flat.append(xv)
        if len(flat) == 0:
            return None
        if len(flat) > 24:
            return "{[" + str(len(flat)) + " values omitted]}"

        if all(isinstance(x, (int, float, np.integer, np.floating, bool)) for x in flat):
            vals = []
            for x in flat:
                if isinstance(x, bool):
                    vals.append("1" if x else "0")
                elif isinstance(x, (int, np.integer)):
                    vals.append(str(int(x)))
                else:
                    xf = float(x)
                    if not np.isfinite(xf):
                        return None
                    vals.append(f"{xf:.12g}")
            return "{ " + ", ".join(vals) + " }"

        vals = [_clean_header_text(str(x), max_len=120) for x in flat]
        return "{ " + ", ".join(vals) + " }"

    return "{" + _clean_header_text(str(v)) + "}"


def _normalize_bbl(values: np.ndarray, bands: int) -> Optional[np.ndarray]:
    arr = np.asarray(values).reshape(-1)
    if arr.size != bands:
        return None

    if arr.dtype.kind == "b":
        return arr.astype(np.int32)

    if arr.dtype.kind in {"i", "u", "f"}:
        out = np.where(np.asarray(arr, dtype=float) > 0, 1, 0).astype(np.int32)
        return out

    # String-like masks: support common tokens.
    try:
        tokens = np.asarray(arr).astype(str)
    except Exception:
        return None
    out = np.ones(bands, dtype=np.int32)
    false_tokens = {"0", "false", "f", "bad", "invalid", "no"}
    true_tokens = {"1", "true", "t", "good", "valid", "yes"}
    for i, t in enumerate(tokens):
        tt = t.strip().lower()
        if tt in false_tokens:
            out[i] = 0
        elif tt in true_tokens:
            out[i] = 1
        else:
            return None
    return out


def _derive_bbl_from_wavelengths(
    wavelengths: Optional[np.ndarray],
    bands: int,
    units: Optional[str],
) -> Optional[np.ndarray]:
    if wavelengths is None:
        return None

    wl = np.asarray(wavelengths).reshape(-1).astype(float)
    if wl.size != bands:
        return None

    # Work in nanometers.
    if units == "Micrometers":
        wl_nm = wl * 1000.0
    else:
        wl_nm = wl

    bbl = np.ones(bands, dtype=np.int32)
    for lo, hi in DEFAULT_BAD_BAND_RANGES_NM:
        bad = (wl_nm >= lo) & (wl_nm <= hi)
        bbl[bad] = 0
    return bbl


def _get_bbl(
    h5: h5py.File,
    group_prefix: str,
    bands: int,
    wavelengths: Optional[np.ndarray],
    units: Optional[str],
) -> Tuple[Optional[np.ndarray], str]:
    cand = [f"{group_prefix}/bbl"] + CANDIDATE_BBL_PATHS
    bbl_path = _find_existing_path(h5, cand)
    if bbl_path:
        norm = _normalize_bbl(h5[bbl_path][...], bands)
        if norm is not None:
            return norm, f"dataset:{bbl_path}"

    # Some producers put mask-like attrs on wavelength variable.
    wl_path = _find_existing_path(h5, [f"{group_prefix}/wavelength"] + CANDIDATE_WAVELENGTH_PATHS)
    if wl_path:
        for key in ("bbl", "good_wavelengths", "good_bands", "band_mask", "mask"):
            if key in h5[wl_path].attrs:
                norm = _normalize_bbl(h5[wl_path].attrs[key], bands)
                if norm is not None:
                    return norm, f"attr:{wl_path}:{key}"

    if AUTO_BBL_FROM_WAVELENGTH:
        derived = _derive_bbl_from_wavelengths(wavelengths, bands, units)
        if derived is not None:
            return derived, "derived:wavelength_ranges"

    return None, "none"


def _append_attrs_to_hdr_lines(
    hdr_lines: List[str],
    attrs,
    prefix: str,
    skip_keys: Optional[set] = None,
) -> None:
    skip = set(skip_keys or set())
    for key in sorted(attrs.keys()):
        if key in skip:
            continue
        value_text = _format_attr_value_for_hdr(attrs.get(key))
        if value_text is None:
            continue
        hdr_key = f"{prefix}_{_sanitize_header_key(key)}"
        hdr_lines.append(f"{hdr_key} = {value_text}")


def _append_attrs_from_path(
    h5: h5py.File,
    hdr_lines: List[str],
    path: Optional[str],
    prefix: str,
    skip_keys: Optional[set] = None,
) -> None:
    if not path:
        return
    if path not in h5:
        return
    _append_attrs_to_hdr_lines(hdr_lines, h5[path].attrs, prefix, skip_keys=skip_keys)


def _get_attr_str(attrs, key: str) -> Optional[str]:
    if key not in attrs:
        return None
    v = _to_attr_python(attrs[key])
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        v = " ".join(str(x) for x in v)
    s = str(v).strip()
    return s if s else None


def _build_band_names_from_wavelengths(wavelengths: np.ndarray, units: Optional[str]) -> List[str]:
    wl = np.asarray(wavelengths).reshape(-1).astype(float)
    if units == "Micrometers":
        return [f"Band {i+1}: {w:.4f} um" for i, w in enumerate(wl)]
    return [f"Band {i+1}: {w:.2f} nm" for i, w in enumerate(wl)]


def _select_default_rgb_bands(wavelengths: np.ndarray, units: Optional[str]) -> Optional[Tuple[int, int, int]]:
    wl = np.asarray(wavelengths).reshape(-1).astype(float)
    if wl.size < 3:
        return None

    if units == "Micrometers":
        targets = [0.65, 0.55, 0.45]  # R, G, B
    else:
        targets = [650.0, 550.0, 450.0]  # R, G, B

    used = set()
    chosen = []
    for t in targets:
        order = np.argsort(np.abs(wl - t))
        pick = None
        for idx in order:
            i = int(idx)
            if i not in used:
                pick = i
                break
        if pick is None:
            return None
        used.add(pick)
        chosen.append(pick + 1)  # ENVI bands are 1-based

    return int(chosen[0]), int(chosen[1]), int(chosen[2])


def _scalar_len(ds: h5py.Dataset) -> Optional[int]:
    arr = np.asarray(ds[...])
    if arr.ndim == 0:
        return None
    return int(arr.size)


def _detect_axes_from_dimension_scales(h5: h5py.File, cube_path: str) -> Optional[Tuple[int, int, int]]:
    if cube_path not in h5:
        return None
    ds = h5[cube_path]
    if ds.ndim != 3:
        return None

    band_axis = None
    line_axis = None
    sample_axis = None

    for ax in range(3):
        try:
            dim = ds.dims[ax]
            if len(dim) < 1:
                continue
            scale = dim[0]
            scale_name = getattr(scale, "name", None)
            if not scale_name:
                continue
            base = str(scale_name).split("/")[-1].lower()
        except Exception:
            continue

        if base in {"wavelength", "wavelengths", "band", "bands"}:
            band_axis = ax
        elif base in {"line", "lines", "northing", "y"}:
            line_axis = ax
        elif base in {"sample", "samples", "easting", "x"}:
            sample_axis = ax

    if None in {band_axis, line_axis, sample_axis}:
        return None
    if len({band_axis, line_axis, sample_axis}) != 3:
        return None

    return int(sample_axis), int(line_axis), int(band_axis)


def _detect_axes(
    cube_shape: Tuple[int, int, int],
    wavelength_count: Optional[int],
    line_len: Optional[int],
    sample_len: Optional[int],
) -> Tuple[int, int, int]:
    shape = tuple(int(x) for x in cube_shape)
    axes = [0, 1, 2]

    band_axis: Optional[int] = None
    if wavelength_count is not None:
        matches = [ax for ax, n in enumerate(shape) if n == wavelength_count]
        if len(matches) == 1:
            band_axis = matches[0]
        elif len(matches) > 1:
            filtered = [
                ax
                for ax in matches
                if (line_len is None or shape[ax] != line_len)
                and (sample_len is None or shape[ax] != sample_len)
            ]
            if len(filtered) == 1:
                band_axis = filtered[0]

    if band_axis is None:
        band_axis = int(np.argmin(shape))

    spatial_axes = [ax for ax in axes if ax != band_axis]

    line_axis: Optional[int] = None
    sample_axis: Optional[int] = None

    if line_len is not None:
        line_matches = [ax for ax in spatial_axes if shape[ax] == line_len]
        if len(line_matches) == 1:
            line_axis = line_matches[0]

    if sample_len is not None:
        sample_matches = [ax for ax in spatial_axes if shape[ax] == sample_len]
        if len(sample_matches) == 1:
            sample_axis = sample_matches[0]

    if line_axis is None and sample_axis is not None:
        line_axis = [ax for ax in spatial_axes if ax != sample_axis][0]
    if sample_axis is None and line_axis is not None:
        sample_axis = [ax for ax in spatial_axes if ax != line_axis][0]

    if line_axis is None or sample_axis is None:
        # Fallback: preserve original order among the two spatial axes.
        line_axis, sample_axis = spatial_axes[0], spatial_axes[1]

    if len({band_axis, line_axis, sample_axis}) != 3:
        raise RuntimeError(
            f"Axis detection failed. cube_shape={shape}, wavelength_count={wavelength_count}, "
            f"line_len={line_len}, sample_len={sample_len}, "
            f"detected=(sample_axis={sample_axis}, line_axis={line_axis}, band_axis={band_axis})"
        )

    return sample_axis, line_axis, band_axis


def _write_binary_bsq(cube: np.ndarray, out_path: Path, axes: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {cube.shape!r}")

    sample_axis, line_axis, band_axis = axes
    samples = int(cube.shape[sample_axis])
    lines = int(cube.shape[line_axis])
    bands = int(cube.shape[band_axis])

    arr = np.asarray(cube, dtype=np.float32)
    arr_bsq = np.transpose(arr, (band_axis, line_axis, sample_axis))  # (bands, lines, samples)
    arr_bsq.tofile(out_path)
    return samples, lines, bands


def _write_binary_bsq_with_glt(
    cube: np.ndarray,
    out_path: Path,
    axes: Tuple[int, int, int],
    glt_line: np.ndarray,
    glt_sample: np.ndarray,
    fill_value: Optional[float],
) -> Tuple[int, int, int]:
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {cube.shape!r}")
    if glt_line.shape != glt_sample.shape or glt_line.ndim != 2:
        raise ValueError(f"GLT shape mismatch: line={glt_line.shape}, sample={glt_sample.shape}")

    sample_axis, line_axis, band_axis = axes
    src = np.transpose(np.asarray(cube, dtype=np.float32), (band_axis, line_axis, sample_axis))
    bands, src_lines, src_samples = src.shape

    # GLT uses 1-based indices. 0 means no data. sample may contain negative values
    # to indicate nearest-neighbor infill in the orthorectified product; abs() keeps mapping.
    line_idx = np.abs(np.rint(np.asarray(glt_line))).astype(np.int64) - 1
    sample_idx = np.abs(np.rint(np.asarray(glt_sample))).astype(np.int64) - 1
    valid = (
        (line_idx >= 0)
        & (sample_idx >= 0)
        & (line_idx < src_lines)
        & (sample_idx < src_samples)
    )

    out_lines, out_samples = glt_line.shape
    nodata = -9999.0 if fill_value is None else float(fill_value)
    out = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(bands, out_lines, out_samples))
    out[:] = nodata

    valid_flat = np.flatnonzero(valid.ravel())
    src_flat_idx = (line_idx * src_samples + sample_idx).ravel()[valid_flat]

    for b in range(bands):
        out_band = out[b].reshape(-1)
        src_band = src[b].reshape(-1)
        out_band[valid_flat] = src_band[src_flat_idx]

    out.flush()
    del out
    return out_samples, out_lines, bands


def _orthorectify_2d_with_glt(
    arr2d: np.ndarray,
    glt_line: np.ndarray,
    glt_sample: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    if arr2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr2d.shape!r}")
    if glt_line.shape != glt_sample.shape or glt_line.ndim != 2:
        raise ValueError(f"GLT shape mismatch: line={glt_line.shape}, sample={glt_sample.shape}")

    src = np.asarray(arr2d, dtype=np.float32)
    src_lines, src_samples = src.shape
    line_idx = np.abs(np.rint(np.asarray(glt_line))).astype(np.int64) - 1
    sample_idx = np.abs(np.rint(np.asarray(glt_sample))).astype(np.int64) - 1
    valid = (
        (line_idx >= 0)
        & (sample_idx >= 0)
        & (line_idx < src_lines)
        & (sample_idx < src_samples)
    )

    out = np.full(glt_line.shape, float(fill_value), dtype=np.float32)
    out_flat = out.ravel()
    src_flat = src.ravel()
    valid_flat = np.flatnonzero(valid.ravel())
    src_flat_idx = (line_idx * src_samples + sample_idx).ravel()[valid_flat]
    out_flat[valid_flat] = src_flat[src_flat_idx]
    return out


def _decode_text_attr(attrs, key: str) -> Optional[str]:
    if key not in attrs:
        return None
    val = _decode_attr(attrs[key])
    if isinstance(val, list):
        val = "".join(str(x) for x in val)
    if isinstance(val, str):
        txt = val.strip()
        return txt if txt else None
    return None


def _parse_utm_info_from_wkt(wkt: Optional[str]) -> Tuple[Optional[int], Optional[str], str]:
    if not wkt:
        return None, None, "WGS-84"

    datum = "WGS-84" if "WGS 84" in wkt.upper() else "Unknown"

    m = re.search(r"UTM zone\s*(\d+)\s*([NS])", wkt, flags=re.IGNORECASE)
    if m:
        zone = int(m.group(1))
        hemi = "North" if m.group(2).upper() == "N" else "South"
        return zone, hemi, datum

    m = re.search(r'EPSG","(326|327)(\d{2})"', wkt)
    if m:
        zone = int(m.group(2))
        hemi = "North" if m.group(1) == "326" else "South"
        return zone, hemi, datum

    return None, None, datum


def _parse_geotransform_attr(value) -> Optional[Tuple[float, float, float, float, float, float]]:
    if value is None:
        return None
    val = _decode_attr(value)
    if isinstance(val, list):
        val = " ".join(str(x) for x in val)

    nums: List[float] = []
    if isinstance(val, str):
        nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", val)]
    else:
        try:
            arr = np.asarray(val).reshape(-1).astype(float)
            nums = [float(x) for x in arr.tolist()]
        except Exception:
            return None

    if len(nums) < 6:
        return None
    return nums[0], nums[1], nums[2], nums[3], nums[4], nums[5]


def _build_mapinfo_and_crs_lines(
    h5: h5py.File,
    lines: int,
    samples: int,
    prefer_glt: bool,
) -> List[str]:
    if prefer_glt:
        easting_candidates = ["/geolocation_lookup_table/easting"] + CANDIDATE_EASTING_PATHS
        northing_candidates = ["/geolocation_lookup_table/northing"] + CANDIDATE_NORTHING_PATHS
    else:
        easting_candidates = CANDIDATE_EASTING_PATHS + ["/geolocation_lookup_table/easting"]
        northing_candidates = CANDIDATE_NORTHING_PATHS + ["/geolocation_lookup_table/northing"]

    e_path = _find_existing_path(h5, easting_candidates)
    n_path = _find_existing_path(h5, northing_candidates)
    if not e_path or not n_path:
        return []

    easting = np.asarray(h5[e_path][...]).reshape(-1).astype(float)
    northing = np.asarray(h5[n_path][...]).reshape(-1).astype(float)
    if easting.size != samples or northing.size != lines:
        return []
    if easting.size < 2 or northing.size < 2:
        return []

    wkt = None
    gt = None
    if "/transverse_mercator" in h5:
        tm = h5["/transverse_mercator"]
        wkt = _decode_text_attr(tm.attrs, "crs_wkt") or _decode_text_attr(
            h5["/transverse_mercator"].attrs, "spatial_ref"
        )
        gt = _parse_geotransform_attr(tm.attrs.get("GeoTransform"))

    if gt is not None:
        gt0, gt1, gt2, gt3, gt4, gt5 = gt
        # GeoTransform uses pixel corner; ENVI map info expects center of reference pixel (1,1).
        x_ref = float(gt0 + 0.5 * (gt1 + gt2))
        y_ref = float(gt3 + 0.5 * (gt4 + gt5))
        x_res = float((gt1 * gt1 + gt4 * gt4) ** 0.5)
        y_res = float((gt2 * gt2 + gt5 * gt5) ** 0.5)
        rotation = float(np.degrees(np.arctan2(gt4, gt1)))
    else:
        x_ref = float(easting[0])
        y_ref = float(northing[0])
        x_res = float(abs(np.nanmedian(np.diff(easting))))
        y_res = float(abs(np.nanmedian(np.diff(northing))))
        rotation = 0.0

    zone, hemi, datum = _parse_utm_info_from_wkt(wkt)
    out: List[str] = []
    if zone is not None and hemi is not None:
        out.append(
            f"map info = {{UTM, 1, 1, {x_ref:.6f}, {y_ref:.6f}, {x_res:.6f}, {y_res:.6f}, {zone}, {hemi}, {datum}, units=Meters, rotation={rotation:.6f}}}"
        )
    else:
        out.append(
            f"map info = {{Arbitrary, 1, 1, {x_ref:.6f}, {y_ref:.6f}, {x_res:.6f}, {y_res:.6f}, units=Meters, rotation={rotation:.6f}}}"
        )

    if wkt:
        wkt_clean = wkt.replace("\n", " ").replace("\r", " ").strip()
        out.append(f"coordinate system string = {{{wkt_clean}}}")

    return out


def _write_2d_envi(
    arr2d: np.ndarray,
    out_base: Path,
    description: str,
    extra_hdr_lines: Optional[List[str]] = None,
) -> None:
    arr2d = np.asarray(arr2d, dtype=np.float32)
    if arr2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr2d.shape!r}")

    # AVIRIS lat/lon are (lines, samples)
    lines, samples = arr2d.shape
    arr2d.astype(np.float32).tofile(out_base)

    out_hdr = Path(str(out_base) + ".hdr")
    hdr_lines = [
        "ENVI",
        f"description = {{{_clean_header_text(description)}}}",
        f"samples = {samples}",
        f"lines   = {lines}",
        "bands   = 1",
        "header offset = 0",
        "file type = ENVI Standard",
        "data type = 4",
        "interleave = bsq",
        "byte order = 0",
    ]
    if extra_hdr_lines:
        hdr_lines.extend(extra_hdr_lines)
    out_hdr.write_text("\n".join(hdr_lines) + "\n", encoding="utf-8")


def convert_one_nc(nc_path: Path, out_dir: Path, export_latlon: bool = False) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = f"{nc_path.stem}_data"
    out_data = out_dir / out_stem
    out_hdr = out_dir / f"{out_stem}.hdr"

    with h5py.File(nc_path, "r") as h5:
        cube_path = _find_existing_path(h5, CANDIDATE_CUBE_PATHS) or _find_first_3d_dataset(h5)
        if not cube_path:
            raise RuntimeError("Could not find a 3D data cube in the .nc file.")

        cube_ds = h5[cube_path]
        if cube_ds.ndim != 3:
            raise RuntimeError(f"Selected dataset is not 3D: {cube_path} shape={cube_ds.shape!r}")

        group_prefix = str(Path(cube_path).parent)
        wavelength_path = _find_existing_path(h5, [f"{group_prefix}/wavelength"] + CANDIDATE_WAVELENGTH_PATHS)
        fwhm_path = _find_existing_path(h5, [f"{group_prefix}/fwhm"] + CANDIDATE_FWHM_PATHS)
        lines_path = _find_existing_path(h5, CANDIDATE_LINES_PATHS)
        samples_path = _find_existing_path(h5, CANDIDATE_SAMPLES_PATHS)
        easting_path = _find_existing_path(h5, ["/easting", "/x"])
        northing_path = _find_existing_path(h5, ["/northing", "/y"])
        lat_path = _find_existing_path(h5, CANDIDATE_LAT_PATHS)
        lon_path = _find_existing_path(h5, CANDIDATE_LON_PATHS)
        glt_line_path = _find_existing_path(h5, CANDIDATE_GLT_LINE_PATHS)
        glt_sample_path = _find_existing_path(h5, CANDIDATE_GLT_SAMPLE_PATHS)

        wavelengths = None
        if wavelength_path:
            wavelengths = np.asarray(h5[wavelength_path][...]).reshape(-1)

        fwhm = None
        if fwhm_path:
            fwhm = np.asarray(h5[fwhm_path][...]).reshape(-1)

        line_len = _scalar_len(h5[lines_path]) if lines_path else None
        sample_len = _scalar_len(h5[samples_path]) if samples_path else None
        wavelength_count = int(wavelengths.size) if wavelengths is not None else None

        axes = _detect_axes_from_dimension_scales(h5, cube_path)
        if axes is None:
            axes = _detect_axes(tuple(int(x) for x in cube_ds.shape), wavelength_count, line_len, sample_len)
        sample_axis, line_axis, band_axis = axes
        bands = int(cube_ds.shape[band_axis])

        if wavelengths is not None and wavelengths.size != bands:
            raise RuntimeError(
                f"Wavelength count ({wavelengths.size}) does not match detected bands ({bands}). "
                f"cube_shape={cube_ds.shape}, axes(sample,line,band)={axes}"
            )
        if fwhm is not None and fwhm.size != bands:
            raise RuntimeError(
                f"FWHM count ({fwhm.size}) does not match detected bands ({bands}). "
                f"cube_shape={cube_ds.shape}, axes(sample,line,band)={axes}"
            )

        units = None
        if wavelength_path:
            units_attr = _decode_attr(h5[wavelength_path].attrs.get("units"))
            if isinstance(units_attr, str) and units_attr.strip():
                low = units_attr.lower().strip()
                if "nano" in low or low == "nm":
                    units = "Nanometers"
                elif "micro" in low or "micron" in low or low in {"um", "μm"}:
                    units = "Micrometers"
        if units is None and wavelengths is not None:
            units = _infer_units_from_values(wavelengths)

        bbl = None
        bbl_source = "none"
        if INCLUDE_BBL:
            bbl, bbl_source = _get_bbl(
                h5=h5,
                group_prefix=group_prefix,
                bands=bands,
                wavelengths=wavelengths,
                units=units,
            )

        fill_value = _decode_attr(cube_ds.attrs.get("_FillValue"))
        if isinstance(fill_value, np.ndarray):
            fill_value = float(np.asarray(fill_value).reshape(-1)[0])
        elif fill_value is not None:
            fill_value = float(fill_value)

        cube = np.asarray(cube_ds[...])

        use_glt = (
            ORTHORECTIFY_L1B_WITH_GLT
            and cube_path.startswith("/radiance/")
            and (glt_line_path is not None)
            and (glt_sample_path is not None)
        )

        if use_glt:
            glt_line = np.asarray(h5[glt_line_path][...])
            glt_sample = np.asarray(h5[glt_sample_path][...])
            samples, lines, bands = _write_binary_bsq_with_glt(cube, out_data, axes, glt_line, glt_sample, fill_value)
        else:
            samples, lines, bands = _write_binary_bsq(cube, out_data, axes)

        title_attr = _get_attr_str(h5.attrs, "title")
        summary_attr = _get_attr_str(h5.attrs, "summary")
        desc_extra_parts = []
        if title_attr:
            desc_extra_parts.append(f"title={_clean_header_text(title_attr, max_len=220)}")
        if summary_attr:
            desc_extra_parts.append(f"summary={_clean_header_text(summary_attr, max_len=260)}")
        desc_extra = "; ".join(desc_extra_parts)

        base_desc = (
            f"Converted from {nc_path.name}, source dataset {cube_path}, "
            f"cube_shape={cube_ds.shape}, axes(sample,line,band)={axes}, "
            f"orthorectified={'True' if use_glt else 'False'}"
        )
        if desc_extra:
            base_desc = f"{base_desc}; {desc_extra}"

        hdr_lines = [
            "ENVI",
            f"description = {{{_clean_header_text(base_desc)}}}",
            f"samples = {samples}",
            f"lines   = {lines}",
            f"bands   = {bands}",
            "header offset = 0",
            "file type = ENVI Standard",
            "data type = 4",
            "interleave = bsq",
            "byte order = 0",
        ]

        if fill_value is not None:
            hdr_lines.append(f"data ignore value = {fill_value:.6e}")

        if wavelengths is not None and units is not None:
            hdr_lines.append(f"wavelength units = {units}")
            hdr_lines.append(f"wavelength = {_format_envi_list(wavelengths)}")
        elif wavelengths is not None:
            hdr_lines.append(f"wavelength = {_format_envi_list(wavelengths)}")

        if fwhm is not None:
            hdr_lines.append(f"fwhm = {_format_envi_list(fwhm)}")
        if bbl is not None:
            hdr_lines.append(f"bbl = {_format_envi_int_list(bbl)}")
            hdr_lines.append(f"nc_bbl_source = {{{bbl_source}}}")

        if INCLUDE_OPTIONAL_ENVI_STANDARD_FIELDS:
            sensor_type = _get_attr_str(h5.attrs, "sensor")
            if sensor_type:
                hdr_lines.append(f"sensor type = {{{_clean_header_text(sensor_type, max_len=120)}}}")

            acq_start = _get_attr_str(h5.attrs, "time_coverage_start")
            acq_end = _get_attr_str(h5.attrs, "time_coverage_end")
            if acq_start and acq_end:
                hdr_lines.append(
                    f"acquisition time = {{{_clean_header_text(acq_start, max_len=80)} / {_clean_header_text(acq_end, max_len=80)}}}"
                )
            elif acq_start:
                hdr_lines.append(f"acquisition time = {{{_clean_header_text(acq_start, max_len=80)}}}")

            cube_units = _get_attr_str(cube_ds.attrs, "units")
            if not cube_units and cube_path.startswith("/reflectance/"):
                cube_units = "unitless"
            if cube_units:
                hdr_lines.append(f"data units = {{{_clean_header_text(cube_units, max_len=120)}}}")

            if wavelengths is not None and wavelengths.size == bands:
                band_names = _build_band_names_from_wavelengths(wavelengths, units)
                if band_names:
                    hdr_lines.append(f"band names = {_format_envi_str_list(band_names)}")

                rgb = _select_default_rgb_bands(wavelengths, units)
                if rgb is not None:
                    r, g, b = rgb
                    hdr_lines.append(f"default bands = {{{r}, {g}, {b}}}")

        mapinfo_lines = _build_mapinfo_and_crs_lines(h5, lines=lines, samples=samples, prefer_glt=use_glt)
        hdr_lines.extend(mapinfo_lines)

        if INCLUDE_NC_METADATA_IN_HDR:
            _append_attrs_to_hdr_lines(hdr_lines, h5.attrs, "nc_global", skip_keys=SKIP_ATTR_KEYS_GLOBAL)
            _append_attrs_to_hdr_lines(hdr_lines, cube_ds.attrs, "nc_cube", skip_keys=SKIP_ATTR_KEYS_COMMON)
            if wavelength_path:
                _append_attrs_to_hdr_lines(
                    hdr_lines,
                    h5[wavelength_path].attrs,
                    "nc_wavelength",
                    skip_keys=SKIP_ATTR_KEYS_COMMON,
                )
            if fwhm_path:
                _append_attrs_to_hdr_lines(
                    hdr_lines,
                    h5[fwhm_path].attrs,
                    "nc_fwhm",
                    skip_keys=SKIP_ATTR_KEYS_COMMON,
                )
            if "/transverse_mercator" in h5:
                _append_attrs_to_hdr_lines(
                    hdr_lines,
                    h5["/transverse_mercator"].attrs,
                    "nc_crs",
                    skip_keys=set(),
                )
            _append_attrs_from_path(
                h5, hdr_lines, "/geolocation_lookup_table", "nc_glt", skip_keys=SKIP_ATTR_KEYS_COMMON
            )
            _append_attrs_from_path(
                h5, hdr_lines, "/geolocation_lookup_table/easting", "nc_glt_x", skip_keys=SKIP_ATTR_KEYS_COMMON
            )
            _append_attrs_from_path(
                h5, hdr_lines, "/geolocation_lookup_table/northing", "nc_glt_y", skip_keys=SKIP_ATTR_KEYS_COMMON
            )
            _append_attrs_from_path(
                h5, hdr_lines, glt_line_path, "nc_glt_line", skip_keys=SKIP_ATTR_KEYS_COMMON
            )
            _append_attrs_from_path(
                h5, hdr_lines, glt_sample_path, "nc_glt_sample", skip_keys=SKIP_ATTR_KEYS_COMMON
            )
            _append_attrs_from_path(h5, hdr_lines, easting_path, "nc_x", skip_keys=SKIP_ATTR_KEYS_COMMON)
            _append_attrs_from_path(h5, hdr_lines, northing_path, "nc_y", skip_keys=SKIP_ATTR_KEYS_COMMON)
            _append_attrs_from_path(h5, hdr_lines, lines_path, "nc_lines", skip_keys=SKIP_ATTR_KEYS_COMMON)
            _append_attrs_from_path(
                h5, hdr_lines, samples_path, "nc_samples", skip_keys=SKIP_ATTR_KEYS_COMMON
            )
            _append_attrs_from_path(h5, hdr_lines, lat_path, "nc_lat", skip_keys=SKIP_ATTR_KEYS_COMMON)
            _append_attrs_from_path(h5, hdr_lines, lon_path, "nc_lon", skip_keys=SKIP_ATTR_KEYS_COMMON)

        hdr_lines.append(f"dataset names = {cube_path}")
        out_hdr.write_text("\n".join(hdr_lines) + "\n", encoding="utf-8")

        if export_latlon:
            if lat_path and lon_path:
                lat = np.asarray(h5[lat_path][...], dtype=np.float32)
                lon = np.asarray(h5[lon_path][...], dtype=np.float32)

                if use_glt:
                    lat_fill = _to_attr_python(h5[lat_path].attrs.get("_FillValue"))
                    lon_fill = _to_attr_python(h5[lon_path].attrs.get("_FillValue"))
                    lat_fill = -9999.0 if lat_fill is None else float(np.asarray(lat_fill).reshape(-1)[0])
                    lon_fill = -9999.0 if lon_fill is None else float(np.asarray(lon_fill).reshape(-1)[0])
                    lat = _orthorectify_2d_with_glt(lat, glt_line, glt_sample, lat_fill)
                    lon = _orthorectify_2d_with_glt(lon, glt_line, glt_sample, lon_fill)

                _write_2d_envi(
                    lat,
                    out_dir / f"{out_stem}_lat",
                    f"Latitude from {nc_path.name}",
                    extra_hdr_lines=mapinfo_lines,
                )
                _write_2d_envi(
                    lon,
                    out_dir / f"{out_stem}_lon",
                    f"Longitude from {nc_path.name}",
                    extra_hdr_lines=mapinfo_lines,
                )

    return out_data, out_hdr


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nc_files = sorted(INPUT_DIR.glob("*.nc"))
    if not nc_files:
        print(f"No .nc files found in: {INPUT_DIR}")
        return

    total = len(nc_files)
    ok = 0
    failed = 0

    print(f"Input dir : {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Found .nc : {total}")
    print("=" * 80)

    for i, nc_path in enumerate(nc_files, start=1):
        print(f"[{i}/{total}] converting: {nc_path.name}")
        try:
            out_data, out_hdr = convert_one_nc(nc_path, OUTPUT_DIR, export_latlon=EXPORT_LATLON)
            print(f"  ok: {out_data.name}")
            print(f"  ok: {out_hdr.name}")
            ok += 1
        except Exception as e:
            failed += 1
            print(traceback.format_exc())
            print(f"  failed: {nc_path.name}")
            print(f"  reason: {e}")
        print("-" * 80)

    print("Conversion finished")
    print(f"Success: {ok}")
    print(f"Failed : {failed}")
    print(f"Output : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
