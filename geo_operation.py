from utils import *
import utm

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