import pandas as pd
import datetime as dt
import xarray as xr

def latest_version(df: pd.DataFrame) -> str:
    """
    Select latest version from a column dataframe with a list of versions  
    with a specific (e.g. v20201025)"
    """
    last_v = max([dt.datetime.strptime(v[1:], '%Y%m%d') for v in df["version"]])
    last_v_str = "v" + str(last_v.year ) + str(last_v.month).zfill(2) + str(last_v.day).zfill(2)
    return last_v_str

def fix_360_longitudes(
    dataset: xr.Dataset, project: str, lonname: str = "lon"
) -> xr.Dataset:
    """
    Fix longitude values.

    Function to transform datasets where longitudes are in (0, 360) to (-180, 180).

    Parameters
    ----------
    dataset (xarray.Dataset): data stored by dimensions
    project (str): project of the process e.g CMIP6, CORDEX...
    lonname (str): name of the longitude dimension

    Returns
    -------
    dataset (xarray.Dataset): data with the new longitudes
    """
    lonname = lonname if "CORDEX" not in project else "lon"
    lon = dataset[lonname]
    if lon.max() > 180:
        dataset[lonname] = dataset[lonname].where(lon <= 180, other=lon - 360)
    if "CORDEX" not in project:
        dataset = dataset.reindex(**{lonname: sorted(dataset[lonname])})
    return dataset
