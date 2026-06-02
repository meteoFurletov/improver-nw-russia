"""Tests for Iris cube conversion."""

import numpy as np
import pytest
import xarray as xr
from datetime import datetime


def _make_sample_dataarray(nlat=10, nlon=15):
    """Create a sample 2D DataArray for testing."""
    lat = np.linspace(55.0, 70.0, nlat)
    lon = np.linspace(28.0, 55.0, nlon)
    data = np.random.uniform(250, 290, (nlat, nlon)).astype(np.float32)
    return xr.DataArray(
        data,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
        attrs={"units": "K"},
    )


class TestXarrayToIrisCube:
    """Test xarray → Iris cube conversion."""

    def test_basic_conversion(self):
        from src.data.iris_convert import xarray_to_iris_cube

        da = _make_sample_dataarray()
        cube = xarray_to_iris_cube(
            da=da,
            standard_name="air_temperature",
            units="K",
            model_id="test",
            forecast_ref_time=datetime(2022, 1, 1, 0),
            forecast_period_hours=24,
            realization=0,
        )

        assert cube.standard_name == "air_temperature"
        assert str(cube.units) == "K"
        assert cube.shape == (10, 15)

    def test_has_required_coordinates(self):
        from src.data.iris_convert import xarray_to_iris_cube

        da = _make_sample_dataarray()
        cube = xarray_to_iris_cube(
            da=da,
            standard_name="air_temperature",
            units="K",
            model_id="gefs",
            forecast_ref_time=datetime(2022, 1, 1, 0),
            forecast_period_hours=48,
            realization=5,
        )

        # Check all required coordinates exist
        coord_names = [c.standard_name for c in cube.coords()]
        assert "latitude" in coord_names
        assert "longitude" in coord_names
        assert "forecast_reference_time" in coord_names
        assert "forecast_period" in coord_names
        assert "time" in coord_names
        assert "realization" in coord_names
        assert "height" in coord_names

    def test_coordinate_system_assigned(self):
        from src.data.iris_convert import xarray_to_iris_cube

        da = _make_sample_dataarray()
        cube = xarray_to_iris_cube(
            da=da,
            standard_name="air_temperature",
            units="K",
            model_id="test",
            forecast_ref_time=datetime(2022, 1, 1, 0),
            forecast_period_hours=0,
        )

        lat = cube.coord("latitude")
        lon = cube.coord("longitude")
        assert lat.coord_system is not None
        assert lon.coord_system is not None

    def test_bounds_exist(self):
        from src.data.iris_convert import xarray_to_iris_cube

        da = _make_sample_dataarray()
        cube = xarray_to_iris_cube(
            da=da,
            standard_name="air_temperature",
            units="K",
            model_id="test",
            forecast_ref_time=datetime(2022, 1, 1, 0),
            forecast_period_hours=0,
        )

        assert cube.coord("latitude").has_bounds()
        assert cube.coord("longitude").has_bounds()

    def test_model_id_attribute(self):
        from src.data.iris_convert import xarray_to_iris_cube

        da = _make_sample_dataarray()
        cube = xarray_to_iris_cube(
            da=da,
            standard_name="air_temperature",
            units="K",
            model_id="gefs",
            forecast_ref_time=datetime(2022, 1, 1, 0),
            forecast_period_hours=0,
        )

        assert cube.attributes["model_id"] == "gefs"

    def test_improver_compatibility_check(self):
        from src.data.iris_convert import xarray_to_iris_cube, validate_improver_compatibility

        da = _make_sample_dataarray()
        cube = xarray_to_iris_cube(
            da=da,
            standard_name="air_temperature",
            units="K",
            model_id="gefs",
            forecast_ref_time=datetime(2022, 1, 1, 0),
            forecast_period_hours=24,
            realization=0,
        )

        issues = validate_improver_compatibility(cube)
        assert len(issues) == 0, f"Compatibility issues: {issues}"
