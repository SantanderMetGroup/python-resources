import sys
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import xarray as xr
import xesmf as xe


class Interpolator:
    """
    A class for interpolating data using the specified method and resolution.

    Parameters
    ----------
    interpolation_attrs (dict): A dictionary containing the
        interpolation method, lons/lats or resolution. Only "conservative_normed" method has been tested.
    data (xarray): The data to be interpolated.
    """

    def __init__(
        self,
        interpolation_attrs: dict
    ):
        self.var_name = interpolation_attrs['var_name']
        self.interpolation_method = interpolation_attrs['interpolation_method']
        if ('lons' in interpolation_attrs.keys()) and ('lats' in interpolation_attrs.keys()):
            self.lons = interpolation_attrs['lons']
            self.lats = interpolation_attrs['lats']
            self.resolution = None
        else:
            self.lons = None
            self.lats = None
            self.resolution = interpolation_attrs['resolution']

    def __call__(self, data):
        """
        Interpolate the data and store the interpolated data in the output directory.

        Returns
        -------
        The interpolated data.
        """
        df_inter = interpolation(data, 
                             self.interpolation_method, 
                             self.var_name,
                             self.resolution, 
                             self.lons,
                             self.lats)
        return df_inter

def estimate_boundaries(
    lon_values: np.ndarray, lat_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the cell boundaries from the cell location.

    Original Function from Andreas Prein (NCAR). Javier Diez-Sierra and Jesús Fernández
    included some changes to deal with curvilinear grids and Polar regions.

    Parameters
    ----------
    lon_values : np.ndarray
        The longitudes of the grid cells.
    lat_values : np.ndarray
        The latitudes of the grid cells.

    Returns
    -------
    lon_b : np.ndarray
        The longitude boundaries of the grid cells.
    lat_b : np.ndarray
        The latitude boundaries of the grid cells.
    """
    # Dimensions lats/lons
    # nlon = lon_values.shape[1]
    # nlat = lat_values.shape[0]

    # Rearrange lat/lons
    # lons_row = lon_values.flatten()
    # lats_row = lat_values.flatten()

    # Allocate lat/lon corners
    # lons_cor = np.zeros((lons_row.size * 4))
    # lats_cor = np.zeros((lats_row.size * 4))

    lons_crnr = np.empty((lon_values.shape[0] + 1, lon_values.shape[1] + 1))
    lons_crnr[:] = np.nan
    lats_crnr = np.empty((lat_values.shape[0] + 1, lat_values.shape[1] + 1))
    lats_crnr[:] = np.nan

    # -------- Calculating corners --------- #

    # Loop through all grid points except at the boundaries
    for lat in range(1, lon_values.shape[0]):
        for lon in range(1, lon_values.shape[1]):

            buffer = lon_values[(lat - 1) : (lat + 1), (lon - 1) : (lon + 1)]
            ave = np.mean(buffer)
            if abs(lon_values[lat, lon] - ave) > 20:
                buffer = np.where(buffer < 0, buffer + 360, buffer)
                ave = np.mean(buffer)
            lons_crnr[lat, lon] = ave if ave <= 180 else ave - 360.0

            lats_crnr[lat, lon] = np.mean(
                lat_values[(lat - 1) : (lat + 1), (lon - 1) : (lon + 1)]
            )

    # Grid points at boundaries
    lons_crnr[0, :] = lons_crnr[1, :] - (lons_crnr[2, :] - lons_crnr[1, :])
    lons_crnr[-1, :] = lons_crnr[-2, :] + (lons_crnr[-2, :] - lons_crnr[-3, :])
    lons_crnr[:, 0] = lons_crnr[:, 1] + (lons_crnr[:, 1] - lons_crnr[:, 2])
    lons_crnr[:, -1] = lons_crnr[:, -2] + (lons_crnr[:, -2] - lons_crnr[:, -3])

    lats_crnr[0, :] = lats_crnr[1, :] - (lats_crnr[2, :] - lats_crnr[1, :])
    lats_crnr[-1, :] = lats_crnr[-2, :] + (lats_crnr[-2, :] - lats_crnr[-3, :])
    lats_crnr[:, 0] = lats_crnr[:, 1] - (lats_crnr[:, 1] - lats_crnr[:, 2])
    lats_crnr[:, -1] = lats_crnr[:, -2] + (lats_crnr[:, -2] - lats_crnr[:, -3])

    return lons_crnr, lats_crnr


def generate_reference_grid(ds: xr.Dataset, var_name: str) -> xr.Dataset:
    """
    Set the format of the reference dataset to be interpolated with xESMF.

    Parameters
    ----------
    ds : xr.Dataset
        The reference dataset to be interpolated.
    var_name : str
        The name of the variable.

    Returns
    -------
    ds_input : xr.Dataset
        The reference dataset, formatted to be compatible with xESMF.
    """
    # Check longitud or latitude variable name
    if [var for var in ds.variables if "lon" == var]:
        lon_name = "lon"
        lat_name = "lat"
    elif [var for var in ds.variables if "longitude" == var]:
        lon_name = "longitude"
        lat_name = "latitude"
    else:
        print("longitud or lon not available in variable names")
        print([var for var in ds.variables])
        sys.exit(0)
    # Identify vector (regular) or matrix (irregular) coordinates
    if len(np.shape(ds[lon_name].values)) == 1:  # if is regular
        lons, lats = np.meshgrid(ds[lon_name].values, ds[lat_name].values)
    else:
        lons = ds[lon_name].values
        lats = ds[lat_name].values
    # Convert 360 to 180
    if (lons > 180).any():
        lons[lons > 180] = lons[lons > 180] - 360
    # Calculate grid vertices
    lon_bnds, lat_bnds = estimate_boundaries(lons, lats)
    # Rename dims
    new_dims = []
    for dim in ds[var_name].dims:
        if lon_name == dim:
            dim = "x"
        elif lat_name == dim:
            dim = "y"
        elif 'rlon' == dim:
            dim = "x"
        elif 'rlat' == dim:
            dim = "y"
        new_dims.append(dim)
    # Create new xarray
    ds_input = xr.Dataset(
        {
            var_name: (new_dims, ds[var_name].data),
            "lat": (["y", "x"], lats),
            "lon": (["y", "x"], lons),
            "lat_b": (["y_b", "x_b"], lat_bnds),
            "lon_b": (["y_b", "x_b"], lon_bnds),
        }
    )
    # Add 'bounds' attributes to lat/lon
    ds_input.lon.attrs["bounds"] = "lon_b"
    ds_input.lat.attrs["bounds"] = "lat_b"

    return ds_input


def generate_destination_grid(
    res: float = None, x: np.ndarray = None, y: np.ndarray = None
) -> xr.Dataset:
    """
    Generate a destination file with a specific grid resolution.

    Parameters
    ----------
    res : float, optional
        The longitude/latitude resolution.
    x : np.array, optional
        A vector containing the longitude values of the grid centers.
    y : np.array, optional
        A vector containing the latitude values of the grid centers.

    Returns
    -------
    ds_grid : xr.Dataset
        The destination grid, compatible with xESMF.
    """
    if res:
        ds_grid = xe.util.grid_2d(-180.0, 180.0, res, -90.0, 90.0, res)
    else:
        xx, yy = np.meshgrid(x, y)
        xx_bnds, yy_bnds = estimate_boundaries(xx, yy)

        ds_grid = xr.Dataset(
            {
                "lon": (["y", "x"], xx),
                "lat": (["y", "x"], yy),
                "lat_b": (["y_b", "x_b"], yy_bnds),
                "lon_b": (["y_b", "x_b"], xx_bnds),
            }
        )
    return ds_grid


def reorder_boundaries_to_2d(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorder the longitude/latitude bounds in a 2D matrix to be CF-compliant.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the longitude/latitude bounds in 1D format.

    Returns
    -------
    lon_boundaries : np.array
        A 2D matrix containing the longitude bounds.
    lat_boundaries : np.array
        A 2D matrix containing the latitude bounds.
    """
    # format lon_boundaries
    lon_boundaries = np.ones((len(ds["lon"].data[0, :]), 2)) * np.nan
    for n_l in range(len(ds["lon_b"].data[0, :]) - 1):
        lon_boundaries[n_l, 0] = ds["lon_b"].data[0, :][n_l]
        lon_boundaries[n_l, 1] = ds["lon_b"].data[0, :][n_l + 1]
    # format lat_boundaries
    lat_boundaries = np.ones((len(ds["lat"].data[:, 0]), 2)) * np.nan
    for n_l in range(len(ds["lat_b"].data[:, 0]) - 1):
        lat_boundaries[n_l, 0] = ds["lat_b"].data[:, 0][n_l]
        lat_boundaries[n_l, 1] = ds["lat_b"].data[:, 0][n_l + 1]

    return lon_boundaries, lat_boundaries


def make_cf_compliant(
    ds: xr.Dataset, ds_dest: xr.Dataset, ds_inter: xr.Dataset, var_name: str
) -> xr.Dataset:
    """
    Convert the interpolated file into a CF-compliant file.

    Parameters
    ----------
    ds : xr.Dataset
        The original dataset to be interpolated.
    ds_dest : xr.Dataset
        The destination dataset.
    ds_inter : xr.Dataset
        The interpolated dataset.
    var_name : str
        The name of the variable.

    Returns
    -------
    grid : xr.Dataset
        The CF-compliant interpolated dataset.
    """    
    # Check longitud or latitude variable name
    if [var for var in ds.variables if "lon" == var]:
        lon_name = "lon"
        lat_name = "lat"
    elif [var for var in ds.variables if "longitude" == var]:
        lon_name = "longitude"
        lat_name = "latitude"
    else:
        print("longitud or lon not available in variable names")
        print([var for var in ds.variables])
    # Rename dims
    new_dims = []
    for dim in ds[var_name].dims:
        if lon_name == dim:
            dim = "lon"
        elif lat_name == dim:
            dim = "lat"
        elif 'rlon' == dim:
            dim = "lon"
        elif 'rlat' == dim:
            dim = "lat"
        new_dims.append(dim)

    # generate lon and lat bonds in the correct format
    lon_bnds, lat_bnds = reorder_boundaries_to_2d(ds_dest)
    x_grid = ds_dest["lon"].data[0, :]
    y_grid = ds_dest["lat"].data[:, 0]

    xr_dict = {
        "dims": {
            "bnds": 2,
            "lon": len(x_grid),
            "lat": len(y_grid),
            "time": len(ds.time),
        },
        "coords": {
            "lon": {
                "dims": ("lon",),
                "attrs": {
                    "units": "degrees_east",
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "axis": "X",
                    "bounds": "lon_bnds",
                },
                "data": x_grid,
            },
            "lat": {
                "dims": ("lat",),
                "attrs": {
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "axis": "Y",
                    "bounds": "lat_bnds",
                },
                "data": y_grid,
            },
            "time": {"dims": ("time",), "attrs": ds.time.attrs, "data": ds.time.data},
        },
        "data_vars": {
            var_name: {
                "dims": new_dims,
                "attrs": ds[var_name].attrs,
                "data": ds_inter[var_name].data,
            },
            "lon_bnds": {"dims": ("lon", "bnds"), "data": lon_bnds},
            "lat_bnds": {"dims": ("lat", "bnds"), "data": lat_bnds},
            "crs": {
                "dims": {},
                "dtype": np.dtype("int"),
                "attrs": {
                    "grid_mapping_name": "latitude_longitude",
                    "longitude_of_prime_meridian": 0.0,
                    "semi_major_axis": 6378137.0,
                    "inverse_flattening": 298.257223563,
                },
                "data": 0,
            },
        },
    }

    # Include time_bnds if the original only if it is available in the original datset
    if "time_bnds" in ds.variables:
        xr_dict["data_vars"]["time_bnds"] = {
            "dims": ("time", "bnds"),
            "attrs": {},
            "dtype": np.dtype("float64"),
            "data": ds.time_bnds.data,
        }
    # Generate dataset
    grid = xr.Dataset.from_dict(xr_dict)
    # heredate fill/missing value
    for var in grid.variables:
        if var in ds.variables:
            for snf in ['missing_value', 'fill_value']:
                if snf in ds[var].encoding:
                    grid[var].encoding[snf] = ds[var].encoding[snf]
    # Add plev/ensemble/height
    for vrr in ['plev', 'ensemble', 'height']:
        if vrr in ds.variables:
            grid = grid.assign(plev=ds[vrr])
    # Add some extra attributes
    grid[var_name].attrs["grid_mapping"] = "crs"
    for attr in ds.attrs:
        grid.attrs[attr] = ds.attrs[attr]

    return grid


def interpolation(
    ds: xr.Dataset,
    interpolation_method: str,
    var_name: str,
    resolution: float = None,
    lon_values: np.array = None,
    lat_values: np.array = None,
) -> xr.Dataset:
    """
    Apply an interpolation method to the data using the xESMF package.

    Parameters
    ----------
    ds : xr.Dataset
        The data stored by dimensions.
    interpolation_method : str
        The interpolation method. Only "conservative_normed" has been tested.
    var_name : str
        The name of the main variable.
    resolution : float, optional
        The output resolution of the new dataset. If not provided, the `lon_values` and
        `lat_values` arguments must be provided instead.
    lon_values : np.array, optional
        A vector containing the longitude values of the reference grid
        (i.e. the grid that the data will be interpolated onto).
        This argument is only used when the `resolution` argument is not provided.
    lat_values : np.array, optional
        A vector containing the latitude values of the reference grid
        (i.e. the grid that the data will be interpolated onto).
        This argument is only used when the `resolution` argument is not provided.
    output_path : pathlib.Path
        Path where the interpolated data will be stored.
    clobber : bool
        Boolean indicating whether

    Returns
    -------
    ds_inter (xarray.Dataset): The interpolated dataset.
    """
    # Format original dataset
    ds_ref = generate_reference_grid(ds, var_name)

    # Create reference dataset
    if resolution:
        ds_dest = generate_destination_grid(res=resolution)
    else:
        ds_dest = generate_destination_grid(x=lon_values, y=lat_values)

    # Add mask to the referece and destination datasets
    dims = [dim for dim in ds_ref[var_name].dims if not dim in ['x', 'y']]
    ds_ref["mask"] = xr.where(~np.isnan(ds_ref[var_name].mean(dims)), 1, 0) # set nan equal to 0 and 1 for the rest
    ds_dest["mask"] = xr.where(~np.isnan(ds_dest['lon']), 1, 1) # set all values equal to 1

    # Interpolation
    regridder = xe.Regridder(
        ds_ref, ds_dest, interpolation_method, periodic=True, unmapped_to_nan=True,
        ignore_degenerate=True
    )
    ## ignore_degenerate (bool) – Ignore degenerate cells when checking the input Grids or   Meshes for errors. If this is set to True, then the regridding proceeds, but degenerate cells will be skipped. If set to False, a degenerate cell produces an error. This currently only applies to CONSERVE, other regrid methods currently always skip degenerate cells. If None, defaults to False.
    ds_inter = regridder(ds_ref)
    ds_output = make_cf_compliant(ds, ds_dest, ds_inter, var_name)

    return ds_output
