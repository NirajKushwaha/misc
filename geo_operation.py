from .utils import *
from shapely.geometry import Point
import geopandas as gpd

def polygons_intersection_mapping(gdf1, gdf2):
    """
    Get the mapping of polygons in gdf2 to polygons in gdf1
    based on intersection area.

    Parameters
    ----------
    gdf1 : GeoDataFrame
        The other GeoDataFrame to which the mapping is to be done.
    gdf2 : GeoDataFrame
        The GeoDataFrame for which the mapping is to be done.

    Returns
    -------
    dict
        A dictionary where keys are the indices of polygons 
        in gdf2 and values are the indices of polygons in gdf1.
    """

    intersections = gpd.sjoin(gdf2, gdf1, predicate="intersects")["index_right"]

    adm_mapping = {}
    for polygon_2_ix in gdf2.index:
        intersection_areas = {}
        polygons_1s = intersections.loc[polygon_2_ix]

        if(isinstance(polygons_1s, np.int64)):
            adm_mapping[polygon_2_ix] = polygons_1s
        else:
            for polygon_1_ix in polygons_1s:
                intersection_areas[polygon_1_ix] = gdf2.geometry.loc[polygon_2_ix].intersection(gdf1.geometry.loc[polygon_1_ix]).area

            adm_mapping[polygon_2_ix] = max(intersection_areas, key=intersection_areas.get)
            
    return adm_mapping

def country_UTM(ISO3):
    """
    Get the UTM zone of a country based on its centroid.

    Parameters
    ----------
    ISO3 : str
        The ISO3 code of the country.

    Returns
    -------
    str
        The EPSG code of the UTM zone of the country
    """

    try:
        import utm
    except ImportError:
        raise ImportError("'utm' package not found. Please install it using either pip or conda.")

    ISO3 = ISO3.upper()

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    country = world[world['iso_a3'] == f"{ISO3}"] 

    country_centroid = country.centroid
    lon, lat = float(country_centroid.x), float(country_centroid.y)

    _,__,utm_zone_number,hemisphere = utm.from_latlon(lat, lon)

    if hemisphere == 'N':
        epsg_code = f"EPSG:326{utm_zone_number}"
    else:
        epsg_code = f"EPSG:327{utm_zone_number}"
        
    return epsg_code

def create_buffer_circle(lon, lat, radius_m):
    """
    Creates a buffer circle around a given point with a specified radius in meters.

    Parameters
    ----------
    lon : float
        Longitude of the center of the circle.
    lat : float
        Latitude of the center of the circle.
    radius_m : float
        Radius of the circle in meters.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the circle as a single geometry
    """

    try:
        from pyproj import Transformer
    except ImportError:
        raise ImportError("'pyproj' package not found. Please install it using either pip or conda.")
    try:
        from shapely.ops import transform
    except ImportError:
        raise ImportError("'shapely' package not found. Please install it using either pip or conda.")

    point = Point(lon, lat)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    point_metric = transform(transformer.transform, point)

    circle_metric = point_metric.buffer(radius_m)

    transformer_reverse = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    circle = transform(transformer_reverse.transform, circle_metric)

    return gpd.GeoDataFrame({'geometry': [circle]}, crs="EPSG:4326")

def latlong_to_shapely(dataframe, lat_column="latitude", lon_column="longitude"):
    """
    Convert dataframe containing latitude and longitude columns to a GeoDataFrame containing shapely points.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing latitude and longitude columns.
    lat_column : str, "latitude"
        Name of the latitude column.
    lon_column : str, "longitude"
        Name of the longitude column.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the shapely points.
    """

    dataframe.columns = dataframe.columns.str.lower()

    coordinates = [Point(lon, lat) for lat, lon in zip(dataframe[f"{lat_column}"], dataframe[f"{lon_column}"])]
    dataframe = gpd.GeoDataFrame(dataframe, geometry=coordinates)

    return dataframe
