# Iris Metadata Guide for IMPROVER Compatibility

This document explains the exact metadata structure IMPROVER expects
on Iris cubes. Getting this right is the #1 technical challenge when
applying IMPROVER to non-Met Office models.

## Golden rule

**Clone the `improver_test_data` repo and use it as your reference.**

```bash
git clone https://github.com/metoppv/improver_test_data.git
export IMPROVER_ACC_TEST_DIR=$PWD/improver_test_data
```

Load test files with Iris and inspect every attribute, coordinate,
and dimension ordering. Your converted data must match this structure.

## Required coordinate structure

### Dimension coordinates (ordered)

For a typical 2D field (single time, single member):

```
latitude     (latitude)   float64  — must be monotonically increasing
longitude    (longitude)  float64  — must be monotonically increasing
```

For ensemble data:

```
realization  (realization) int32   — ensemble member index (0, 1, 2, ...)
latitude     (latitude)    float64
longitude    (longitude)   float64
```

### Auxiliary (scalar) coordinates

```
forecast_reference_time  — init time, units: "hours since 1970-01-01 00:00:00"
forecast_period          — lead time in hours, units: "hours"
time                     — valid time = ref_time + period, same units as ref_time
height                   — for screen-level vars: 2.0 (m) for T2m, 10.0 for wind
```

### Coordinate system

All spatial coordinates MUST have a `coord_system`:

```python
from iris.coord_systems import GeogCS
WGS84 = GeogCS(semi_major_axis=6378137.0, semi_minor_axis=6356752.314245)

lat_coord.coord_system = WGS84
lon_coord.coord_system = WGS84
```

### Bounds

Latitude and longitude coordinates MUST have bounds:

```python
lat_coord.guess_bounds()
lon_coord.guess_bounds()
```

## Required attributes

```python
cube.attributes["model_id"] = "gefs"           # for blending
cube.attributes["model_configuration"] = "gefs"  # some CLIs check this
```

## Common GEFS → IMPROVER conversion issues

| GEFS (GRIB2)           | IMPROVER expected      | Fix                          |
|------------------------|------------------------|------------------------------|
| `TMP:2 m above ground` | `air_temperature`      | Set `standard_name`          |
| Latitude 90→-90        | Latitude -90→90        | Reverse with `[::-1]`        |
| Longitude 0→360        | Longitude 0→360 (OK)   | Or convert to -180→180       |
| Float64 data           | Float32 preferred      | `.astype(np.float32)`        |
| No coord_system        | WGS84 GeogCS           | Assign to all spatial coords |
| Missing bounds         | Bounds required         | `coord.guess_bounds()`       |
| GRIB parameter table   | CF standard_name       | Map manually                 |

## Variable name mapping

| GEFS GRIB2              | CF standard_name              | Units |
|-------------------------|-------------------------------|-------|
| TMP:2 m above ground    | air_temperature               | K     |
| UGRD:10 m above ground  | x_wind                        | m s-1 |
| VGRD:10 m above ground  | y_wind                        | m s-1 |
| PRMSL                   | air_pressure_at_sea_level     | Pa    |
| APCP                    | precipitation_amount          | kg m-2|

## Testing your conversion

```python
from src.data.iris_convert import validate_improver_compatibility

cube = your_conversion_function(data)
issues = validate_improver_compatibility(cube)
assert len(issues) == 0, f"IMPROVER compatibility issues: {issues}"
```

## IMPROVER acceptance test workflow

Once your conversion is working, run IMPROVER's own acceptance tests
with your data to catch any remaining issues:

```bash
# Run a single CLI test
pytest -v -s -m acc -k test_threshold
```
