# aviris-nc-to-envi-legacy

Convert AVIRIS NetCDF (`.nc`) products to ENVI binary (`BSQ + HDR`) with automatic axis detection and better legacy ENVI compatibility.

This project is designed to make AVIRIS data that may require ENVI 6.2 open reliably in ENVI 5.6 or older versions.

## Features

- Supports AVIRIS products:
  - `L1B RDN` (radiance)
  - `L2A RFL_ORT` (reflectance, orthorectified)
- Automatic 3D cube axis detection (sample/line/band)
- Correct wavelength/FWHM handling and ENVI header writing
- Optional GLT-based orthorectification for `L1B RDN`
- Writes georeferencing metadata into HDR:
  - `map info`
  - `coordinate system string`
- Optional export of `lat/lon` companion rasters
- Optional attachment of rich NetCDF metadata into HDR (`nc_global_*`, `nc_cube_*`, `nc_crs_*`, etc.)
- Output format tuned for ENVI 5.6+ compatibility

## Why This Repo

Some AVIRIS NetCDF files are not straightforward to open in older ENVI versions due to axis/layout and metadata interpretation differences.

This script normalizes the cube layout and header fields so the converted data can be opened more consistently across ENVI versions.

## Requirements

- Python 3.8+
- `numpy`
- `h5py`

Install:

```bash
pip install numpy h5py
```

## Usage

1. Edit configuration at the top of `aviris_nc_to_envi_fixed_v2.py`:

```python
INPUT_DIR = Path(r"D:\data")
OUTPUT_DIR = INPUT_DIR / "converted_envi"
EXPORT_LATLON = False
ORTHORECTIFY_L1B_WITH_GLT = True
INCLUDE_NC_METADATA_IN_HDR = True
```

2. Run:

```bash
python aviris_nc_to_envi_fixed_v2.py
```

## Output

For each input file:

- Main data binary: `<input_stem>_data`
- Main ENVI header: `<input_stem>_data.hdr`
- Optional lat/lon rasters (if `EXPORT_LATLON=True`):
  - `<input_stem>_data_lat` + `.hdr`
  - `<input_stem>_data_lon` + `.hdr`

## Notes on Georeferencing

- `L2A RFL_ORT`: already orthorectified; script keeps projected grid and writes CRS/header info.
- `L1B RDN`: sensor geometry; if GLT is available and `ORTHORECTIFY_L1B_WITH_GLT=True`, script uses GLT to output an orthorectified ENVI grid.

## Header Metadata Enrichment

When `INCLUDE_NC_METADATA_IN_HDR=True`, extra source metadata is written as custom ENVI header fields, for example:

- `nc_global_sensor`
- `nc_global_processing_level`
- `nc_cube_units`
- `nc_crs_grid_mapping_name`

This improves traceability when sharing exported ENVI files.

## Compatibility Goal

Primary goal: convert AVIRIS NetCDF into ENVI products that open and behave correctly in ENVI 5.6+ (without requiring ENVI 6.2).

## Limitations

- Very large scenes can require significant disk I/O and time.
- GLT behavior depends on source product quality and available lookup tables.
- Custom metadata fields are designed for traceability, not strict ENVI standard keys.

## GitHub Topics

`aviris`, `netcdf`, `envi`, `hyperspectral`, `remote-sensing`, `orthorectification`, `glt`, `utm`, `geospatial`

## License

Add a license file before publishing (recommended: MIT).
