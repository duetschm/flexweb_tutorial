import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import cartopy
import cartopy.crs as ccrs
from pyproj import Geod
from cartopy.mpl.contour import GeoContourSet
from matplotlib.animation import FuncAnimation

plt.rcParams['animation.embed_limit'] = 50

def colormap_vintage():
    """
    Create the FLEXWEB colormap with darkened RdYlBu_r colors.
    
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Custom colormap with white values at the beginning and darkened RdYlBu_r values.
    """
    cmap = "RdYlBu_r"
    cmap_values = plt.get_cmap(cmap)(np.linspace(0.05, 1.0, 256))
    # darken cmap_values
    cmap_values = np.minimum(
        cmap_values
        * (
            1.0
            - np.sin(np.arange(len(cmap_values)) * np.pi / (len(cmap_values) - 1))
            * 0.05
        )[:, np.newaxis],
        1.0,
    )
    white_weight = np.linspace(1.0, 0.0, 50)[:, np.newaxis]
    gamma = 1 / 0.8
    white_values = np.array([1.0, 1.0, 1.0, 0.0]) * white_weight**gamma + cmap_values[
        0, :
    ] * (1 - white_weight**gamma)
    new_cmap_values = np.vstack([white_values, cmap_values])
    newcmap = mcolors.LinearSegmentedColormap.from_list(
        cmap + "_white100", new_cmap_values
    )
    return newcmap

def draw_map(ax, lon, lat, release_lon, release_lat):
    """
    Draw a map with coastlines, borders, lakes, and grid on a Cartopy axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Cartopy axes object to draw on.
    lon : array_like
        Longitude coordinates of the domain.
    lat : array_like
        Latitude coordinates of the domain.
    release_lon : float
        Longitude of the release point.
    release_lat : float
        Latitude of the release point.
    
    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object with map features.
    """

    ax.add_feature(cartopy.feature.OCEAN, facecolor=[1.0] * 3)
    ax.add_feature(cartopy.feature.LAND, facecolor=[0.9] * 3)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=0.2, edgecolor=[0.4] * 3)
    lakes = cartopy.feature.NaturalEarthFeature(
        "physical", "lakes", "50m",
        edgecolor="black",
        facecolor="white",
    )
    ax.add_feature(lakes, zorder=2)

    # plot release point
    plt.scatter(
        release_lon,
        release_lat,
        color="black",
        linewidth=1.5,
        marker="x",
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    # plot outgrid
    lons, lats = np.meshgrid(lon, lat)
    plt.plot(
        np.hstack([lons[:, 0], lons[-1, :], lons[::-1, -1], lons[0, ::-1]]),
        np.hstack([lats[:, 0], lats[-1, :], lats[::-1, -1], lats[0, ::-1]]),
        color="black",
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
    )

    # plot gridlines with labels
    gl = ax.gridlines(
        draw_labels=True,
        color="lightgrey",
        linestyle="dashed",
        linewidth=0.5,
        xlocs=np.arange(-180, 180, 20),
        ylocs=np.arange(-80, 90, 10),
    )

    ax.coastlines()

    return ax

def open_flexpart(filename):
    """
    Open and process a FLEXPART output file.
    
    Parameters
    ----------
    filename : str
        Path to the FLEXPART NetCDF file.
    
    Returns
    -------
    ds : xarray.Dataset
        Loaded FLEXPART dataset with reduced dimensions.
    heights_exp : numpy.ndarray
        Heights array with 0 prepended.
    """
    # load dataset
    ds = xr.open_dataset(filename, decode_timedelta=True)
    
    # remove unnecessary dimensions
    ds = ds.isel(numpoint=0, numspec=0, nageclass=0, pointspec=0)
    
    # assign new height values
    heights_exp = np.concatenate([[0.], ds["height"]])

    return ds, heights_exp

def open_topography(filename):
    """
    Open and process a topography dataset.
    
    Parameters
    ----------
    filename : str
        Path to the topography NetCDF file.
    
    Returns
    -------
    xarray.Dataset
        Topography dataset with masked negative values and updated coordinates.
    """
    etopo = xr.open_dataset(filename)
    etopo["topo"].values = np.ma.MaskedArray(etopo["topo"], mask=etopo["topo"]<0.0).filled(0.0)
    etopo = etopo.assign_coords(lon=("lon", etopo["topo_lon"].values), lat=("lat", etopo["topo_lat"].values))

    return etopo

def interpolate_values(ds, etopo, lon1, lat1, lon2, lat2):
    """
    Interpolate FLEXPART and topography data along a line.
    
    Parameters
    ----------
    ds : xarray.Dataset
        FLEXPART dataset to interpolate.
    etopo : xarray.Dataset
        Topography dataset to interpolate.
    lon1 : float
        Starting longitude of the line.
    lat1 : float
        Starting latitude of the line.
    lon2 : float
        Ending longitude of the line.
    lat2 : float
        Ending latitude of the line.
    
    Returns
    -------
    ds_int : xarray.Dataset
        Interpolated FLEXPART data along the line.
    distance : numpy.ndarray
        Cumulative distance along the line in km.
    etopo_int_avg : xarray.Dataset
        Topography interpolated at midpoints between line segments.
    etopo_int : xarray.Dataset
        Topography interpolated at all line points.
    """
    
    # build points along the line
    npts = 201
    lons = np.linspace(lon1, lon2, npts)
    lats = np.linspace(lat1, lat2, npts)
    lons_avg = (lons[1:] + lons[:-1]) / 2.
    lats_avg = (lats[1:] + lats[:-1]) / 2.
    
    # cumulative distance (km)
    g = Geod(ellps="WGS84")
    d = [0.0]
    for i in range(1, npts):
        _,_,dist = g.inv(lons[i-1], lats[i-1], lons[i], lats[i])
        d.append(d[-1] + dist/1000.0)
    distance = np.array(d)
    distance_avg = (distance[1:] + distance[:-1])/2.

    # flexpart
    ds_int = ds.interp(longitude=("points", lons_avg), latitude=("points", lats_avg))
    ds_int = ds_int.assign_coords(distance=("points", distance_avg))
    ds_int["distance"].attrs.update(units="km", long_name="cumulative distance")

    # topography
    etopo_int_avg = etopo.interp(lon=("points", lons_avg), lat=("points", lats_avg))
    etopo_int = etopo.interp(lon=("points", lons), lat=("points", lats))

    return ds_int, distance, etopo_int_avg, etopo_int

def setup_fig(ds):
    """
    Set up a figure with orthographic projection and map.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing longitude, latitude, and release point information.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        Cartopy axes with orthographic projection and map features.
    gs : matplotlib.gridspec.GridSpec
        GridSpec object for layout management.
    """
    gs = gridspec.GridSpec(1, 3, width_ratios=[40,1,2])
    
    central_lon = float(np.mean(ds["longitude"]))
    central_lat = float(np.mean(ds["latitude"]))
    
    proj = ccrs.Orthographic(
        central_longitude=central_lon,
        central_latitude=central_lat,
    )
    
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(gs[0,0], projection=proj)
    ax = draw_map(ax, ds["longitude"], ds["latitude"], ds["RELLNG1"], ds["RELLAT1"])

    return fig, ax, gs

def plot_map(ds, cmap, levels, norm, lon1=None, lat1=None, lon2=None, lat2=None):
    """
    Plot a map with FLEXPART concentration data.
    
    Parameters
    ----------
    ds : xarray.Dataset
        FLEXPART dataset at one time step.
    cmap : str or matplotlib.colors.Colormap
        Colormap to use for plotting.
    levels : array_like
        Contour levels for the plot.
    norm : matplotlib.colors.Normalize
        Normalization for colormap scaling.
    lon1 : float, optional
        Starting longitude for cross-section line.
    lat1 : float, optional
        Starting latitude for cross-section line.
    lon2 : float, optional
        Ending longitude for cross-section line.
    lat2 : float, optional
        Ending latitude for cross-section line.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the map plot.
    """
    # plot map
    fig, ax, gs = setup_fig(ds)
    
    # mask data
    data = ds["spec001_mr"].isel(time=0).weighted(ds["height"]).mean(dim="height")
    data.values = np.ma.MaskedArray(data, mask=data == 0.0).filled(1e-10)
    
    # add title
    ax.set_title(ds["spec001_mr"].long_name, loc="left")
    ax.set_title(np.datetime_as_string(data["time"], unit="h"), loc="right")
    
    # plot data
    h1 = ax.contourf(
        data["longitude"],
        data["latitude"],
        data,
        norm=norm,
        cmap=cmap,
        levels=levels,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    
    if all(l is not None for l in [lon1, lat1, lon2, lat2]):
        plt.plot([lon1, lon2], [lat1, lat2], transform=ccrs.PlateCarree(), color='black')
        plt.scatter([lon1], [lat1], marker='^', transform=ccrs.PlateCarree(), color='black', zorder=3)
        plt.scatter([lon2], [lat2], marker='v', transform=ccrs.PlateCarree(), color='black', zorder=3)
    
    cax = fig.add_subplot(gs[0,2])
    cbar = plt.colorbar(h1, ax=ax, cax=cax, label=ds["spec001_mr"].units)

    nearest_int = np.rint(np.log10(levels))
    is_whole = np.isclose(np.log10(levels), nearest_int, atol=1e-12)
    cbar.set_ticks(levels[is_whole])

    return fig

def plot_map_anim(ds, cmap, levels, norm):
    """
    Create an animated map showing FLEXPART concentration evolution over time.
    
    Parameters
    ----------
    ds : xarray.Dataset
        FLEXPART dataset with time dimension.
    cmap : str or matplotlib.colors.Colormap
        Colormap to use for plotting.
    levels : array_like
        Contour levels for the plot.
    norm : matplotlib.colors.Normalize
        Normalization for colormap scaling.
    
    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object showing concentration data over time.
    """
    # plot map
    fig, ax, gs = setup_fig(ds)

    # mask data
    data = ds["spec001_mr"].weighted(ds["height"]).mean(dim="height").sortby("time")
    data.values = np.ma.MaskedArray(data, mask=data == 0.0).filled(1e-10)

    # add title
    ax.set_title(ds["spec001_mr"].long_name, loc="left")

    # plot data
    def update(n, data, cmap, levels, norm):
        nall = data["time"].size-1
        print(f"\rPlotting frame {n}/{nall}", end="", flush=True)
        
        h1_list = ax.findobj(lambda x: isinstance(x, GeoContourSet))
        if len(h1_list) > 0:
            for h1 in h1_list:
                h1.remove()
        h1 = ax.contourf(
            data["longitude"],
            data["latitude"],
            data.isel(time=n),
            norm=norm,
            cmap=cmap,
            levels=levels,
            extend="both",
            transform=ccrs.PlateCarree(),
            zorder=2,
        )

        ax.set_title(np.datetime_as_string(data["time"].isel(time=n), unit="h"), loc="right")

        return h1

    # initialize h1 for colorbar
    h1 = update(0, data, cmap, levels, norm)
    cax = fig.add_subplot(gs[0,2])
    cbar = plt.colorbar(h1, ax=ax, cax=cax, label=ds["spec001_mr"].units)

    nearest_int = np.rint(np.log10(levels))
    is_whole = np.isclose(np.log10(levels), nearest_int, atol=1e-12)
    cbar.set_ticks(levels[is_whole])

    # plot animation
    ani = FuncAnimation(fig, update, fargs=(data, cmap, levels, norm), frames=data["time"].size, blit=False, interval=50)
    plt.close(fig)

    return ani

def plot_cross_section(ds_int, distance, heights_exp, cmap, norm, etopo_int, etopo_int_avg):
    """
    Plot a cross-section of FLEXPART concentration with topography.
    
    Parameters
    ----------
    ds_int : xarray.Dataset
        Interpolated FLEXPART data along the cross-section line.
    distance : numpy.ndarray
        Cumulative distance along the cross-section line in km.
    heights_exp : numpy.ndarray
        Height levels for the cross-section.
    cmap : str or matplotlib.colors.Colormap
        Colormap to use for plotting.
    norm : matplotlib.colors.Normalize
        Normalization for colormap scaling.
    etopo_int : xarray.Dataset
        Topography interpolated at all cross-section points.
    etopo_int_avg : xarray.Dataset
        Topography interpolated at midpoints between cross-section points.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the cross-section plot.
    """
    # plot cross section
    
    distance2d = np.repeat(distance[None,:], heights_exp.size, axis=0)
    heightstopo = etopo_int["topo"].values[None,:]+heights_exp[:,None]
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    h1 = ax.pcolor(
        distance2d, 
        heightstopo, 
        ds_int["spec001_mr"].isel(time=0), 
        cmap=cmap, 
        shading="flat", 
        norm=norm,
    )
    ax.plot(ds_int["distance"], etopo_int_avg["topo"], color="black")
    ax.fill_between(ds_int["distance"], etopo_int_avg["topo"], color=[0.9]*3)
    ax.plot(distance2d.T, heightstopo.T, color="black", linewidth=0.5, alpha=0.1)
    ax.scatter([0], [-0.07], marker="^", linestyle="None", transform=ax.transAxes, clip_on=False, color="black")
    ax.scatter([1], [-0.07], marker="v", linestyle="None", transform=ax.transAxes, clip_on=False, color="black")
    
    plt.ylim((0., 7000))
    plt.xlabel("distance (km)")
    plt.ylabel("height (m)")
    plt.colorbar(h1, label=ds_int["spec001_mr"].units)

    return fig