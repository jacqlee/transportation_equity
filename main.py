import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
import skmob
import math
from shapely import geometry
from skmob.tessellation import tilers
from datetime import datetime
from equity_graphs import process_graphs
from numpy.core.function_base import linspace

crs = 'epsg:4326'
mode_names = ['walk', 'bike', 'bus', 'car', 'subway','train', 'airplane', 'boat', 'run', 'motorcycle', 'taxi']
mode_ids = {s: i + 1 for i, s in enumerate(mode_names)}
p_distance = 0

def poly_reverse_xy(poly):
    new_polys = []
    if isinstance(poly, geometry.MultiPolygon):
        polys = list(poly)
    else:
        polys = list([poly])
    for polygon in polys:
        points = []
        for (x,y) in polygon.exterior.coords:
            points.append(geometry.Point(y, x))
        new_polys.append(geometry.Polygon(points))
    if isinstance(poly, geometry.MultiPolygon):
        return geometry.MultiPolygon(new_polys)
    else:
        return new_polys[0]


def equity_gap(grid_gdf, trips_gdf, start_time, end_time, demo_relevant_cols=tuple(["F"]), urban_vit_cols=tuple(["y-HOUSINGP"]), e=0.5):
    """
    Calculates the equity gap in the data frame for income
    :param grid_gdf: GeoDataFrame object for geographic grid with demographic information
    :param trips_filename: GeoDataFrame object for trips
    :param demo_relevant_cols: columns in grid_gdf for comparison ratio with POP column
    :param urban_vit_cols: columns in grid_gdf for comparison with median (urban vitality data)
    :param start_time: starting time for period examined
    :param end_time: end time for period examined
    :return: equity_gap
    """
    window_trips_gdf = trips_gdf.loc[trips_gdf["time"] >= start_time]
    window_trips_gdf = window_trips_gdf.loc[window_trips_gdf["time"] <= end_time]
    df = gpd.sjoin(window_trips_gdf, grid_gdf)

    block_year_month_counts = df.groupby(["block", "year", "month"]).size()
    avg_trips = pd.DataFrame(block_year_month_counts.groupby("block").mean(), columns=["avg_trips"])
    # type(avg_trips)
    # .groupby("block").mean()
    cols = ["block", "POP"]
    cols.extend(demo_relevant_cols)
    cols.extend(urban_vit_cols)
    simple_df = df[cols]
    new_df = simple_df.join(avg_trips, on="block").fillna(0)

    counts = df.groupby("block").size()
    N = len(df.groupby("block").groups)
    # N = df.groupby("block").size()
    avg = counts.mean()

    equity_measures = {}

    if e < 1:
        atkinson = 1 - ((counts.apply(lambda y: y**(1-e)).sum()/N)**(1/(1-e))/avg)
    else:
        atkinson = 1 - ((counts.sum())**(1/N))/avg
    equity_measures["atkinson"] = atkinson

    for metric in demo_relevant_cols:
        ratio = new_df[metric]/new_df["POP"]
        disadv_num = new_df["avg_trips"].rmul(ratio).sum()
        disadv_den = new_df["POP"].rmul(ratio).sum()

        adv_num = new_df["avg_trips"].rmul(1 - ratio).sum()
        adv_den = new_df["POP"].rmul(1-ratio).sum()

        equity_measures[metric + "_gap"] = (adv_num/adv_den) - (disadv_num/disadv_den)

    for metric in urban_vit_cols:
        quart = new_df[metric].quantile(0.25)
        adv_group = new_df.loc[new_df[metric] >= quart]
        disadv_group = new_df.loc[new_df[metric] < quart]
        equity_measures[metric + "_gap"] = adv_group["avg_trips"].sum()/adv_group["POP"].sum() - disadv_group["avg_trips"].sum()/disadv_group["POP"].sum()

    return equity_measures


def process_beijing_trips(trip_type=0):
    df = pd.read_pickle("geolife.pkl")
    if trip_type != 0:
        #   Specified trip type
        df = df.loc[df['label'] == trip_type]

    df = df.loc[df.index == 0]
    trip_geometries = [geometry.Point(xy) for xy in zip(df.lat, df.lon)]
    points = gpd.GeoDataFrame(df, crs=crs, geometry=trip_geometries)

    bounding_box = [(39.77750000, 116.17944444), (39.77750000, 116.58888889), (40.04722222, 116.58888889),
                 (40.04722222, 116.17944444)]

    poly = geometry.Polygon(bounding_box)
    spoly = gpd.GeoSeries([poly], crs=crs)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    beijing_trips = points[points.within(spoly.geometry.iloc[0])]
    beijing_trips["geometry"] = beijing_trips.geometry.apply(lambda pt: geometry.Point(pt.y, pt.x))
    beijing_trips.plot(ax=ax1)
    if p_distance > 0:
        beijing_trips = spatial_cloaking(points, bounding_box, privacy_distance=p_distance)
        beijing_trips = beijing_trips.drop("index_right", axis=1)
        beijing_trips["geometry"] = beijing_trips.geometry.apply(lambda pt: geometry.Point(pt.y, pt.x))
        beijing_trips.plot(ax=ax2)
    print('Number of points within Beijing: ', beijing_trips.shape[0])
    plt.show()

    # REVERSE BACK
    beijing_trips["geometry"] = beijing_trips.geometry.apply(lambda pt: geometry.Point(pt.y, pt.x))

    df = pd.DataFrame(beijing_trips)
    dates = pd.to_datetime(df.time)

    df["month"] = dates.apply(lambda x: x.month)
    df["year"] = dates.apply(lambda x: x.year)
    return df


def coord_distance(lat1, lon1, lat2, lon2):
    """
    Returns distance between two coordinates
    :param lat1:
    :param lon1:
    :param lat2:
    :param lon2:
    :return:
    """
    R = 6373.0
    dlon = math.radians(lon2) - math.radians(lon1)
    dlat = math.radians(lat2) - math.radians(lat1)

    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance


def spatial_cloaking(gdf, bounding_box, privacy_distance):
    """
    Maps points in GeoDataFrame to centroid coordinates of grid of (privacy_distance) length boxes
    :param gdf: GeoDataFrame with geometry column of points
    :param bounding_box: list of coordinates for bounding box, starting at bottom left and going clockwise
    :param privacy_distance: number of km for spatial cloaking grid
    :return: new_df: DataFrame where coordinates are now cloaked based on privacy distance
    """
    bottom_left_corner = bounding_box[0]
    bottom_right_corner= bounding_box[1]
    top_right_corner = bounding_box[2]
    top_left_corner = bounding_box[3]

    lat_distance = coord_distance(bottom_left_corner[0], bottom_left_corner[1],
                                  top_left_corner[0], top_left_corner[1])
    lon_distance = coord_distance(bottom_left_corner[0], bottom_left_corner[1],
                                  bottom_right_corner[0], bottom_right_corner[1])

    lat_grid_coords = linspace(bottom_left_corner[0], top_left_corner[0], math.ceil(lat_distance/privacy_distance))
    lon_grid_coords = linspace(bottom_left_corner[1], bottom_right_corner[1], math.floor(lon_distance/privacy_distance))

    square_grid_polygons = []
    square_grid_centroids = []

    for i in range(len(lat_grid_coords) - 1):
        for j in range(len(lon_grid_coords) - 1):
            corners = [geometry.Point(lat_grid_coords[i], lon_grid_coords[j]), geometry.Point(lat_grid_coords[i+1], lon_grid_coords[j]),
                       geometry.Point(lat_grid_coords[i+1], lon_grid_coords[j+1]), geometry.Point(lat_grid_coords[i], lon_grid_coords[j+1])]
            poly = geometry.Polygon(corners)
            square_grid_polygons.append(poly)
            square_grid_centroids.append(poly.centroid)
    # d = {"polygons": square_grid_polygons, "centroids": square_grid_centroids}
    grid_df = gpd.GeoDataFrame({"polygons": square_grid_polygons, "centroids": square_grid_centroids},
                               geometry="polygons", crs=crs)
    merged_df = gpd.sjoin(gdf, grid_df)
    merged_df = merged_df.drop("geometry", axis=1)
    merged_df = merged_df.rename(columns={"centroids": "geometry"})
    return merged_df


def grid_data():
    beijing_data_df = pd.read_pickle("beijing_data.pkl")
    beijing_data_df["geometry"] = beijing_data_df["geometry_x"].apply(poly_reverse_xy)
    beijing_data_gdf = gpd.GeoDataFrame(beijing_data_df, crs=crs,
                                        geometry=beijing_data_df.geometry)

    urban_vit_df = gpd.read_file("demographic_data/DT36_UrbanVitality/GridsDP24512.shp")
    urban_vit_gdf = gpd.GeoDataFrame(urban_vit_df, crs=crs, geometry=urban_vit_df.geometry)
    beijing_urban_vit_gdf = urban_vit_gdf.loc[urban_vit_gdf["E_NAME"] == "Beijing"]

    demographic_data = gpd.sjoin(beijing_urban_vit_gdf, beijing_data_gdf).rename(columns={"index_right": "block"})
    return gpd.GeoDataFrame(demographic_data, crs=crs, geometry=demographic_data.geometry_x)

    # df["geometry"] = df.geometry.apply(lambda x: geometry.Point(x.y, x.x))
    # gdf = gpd.GeoDataFrame(df, crs=crs, geometry=df.geometry)
    #
    # return gpd.sjoin(gdf, demographic_data_gdf)


if __name__ == '__main__':
    trip_types = ["walk", "bike", "car", "taxi", "bus", "subway"]
    results = {}
    grid_gdf = grid_data()
    start = np.datetime64("2008-01", "D")
    end = np.datetime64("2009-01", "D")
    overall_gdf = gpd.GeoDataFrame()
    for x in trip_types:
        df = process_beijing_trips(mode_ids[x])
        gdf = gpd.GeoDataFrame(df, crs=crs, geometry=df.geometry)
        overall_gdf = pd.concat([overall_gdf, gdf])
        #   df = process_beijing_trips()
        measures = equity_gap(grid_gdf, gdf, start, end,
                              demo_relevant_cols=["F", "AGE65"], urban_vit_cols=["Y_HOUSINGP", "AMENITIES"])
        results[x] = measures
    trip_types.append("overall")
    # df = process_beijing_trips()
    # gdf = gpd.GeoDataFrame(df, crs=crs, geometry=df.geometry)
    measures = equity_gap(grid_gdf, overall_gdf, start, end,
                          demo_relevant_cols=["F", "AGE65"], urban_vit_cols=["Y_HOUSINGP", "AMENITIES"])
    results["overall"] = measures

    atkinson = []
    gender_gap = []
    age_gap = []
    house_price_gap = []
    amenities_gap = []

    for value in results.values():
        atkinson.append(value["atkinson"])
        gender_gap.append(value["F_gap"])
        age_gap.append(value["AGE65_gap"])
        house_price_gap.append(value["Y_HOUSINGP_gap"])
        amenities_gap.append(value["AMENITIES_gap"])

    equity_gaps = [atkinson, gender_gap, age_gap, house_price_gap, amenities_gap]
    equity_titles = ["Atkinson Equity Measure", "Gender Equity Gap", "Age Equity Gap", "Housing Price Gap",
                     "Amenities Gap"]
    for i in range(len(equity_gaps)):
        if p_distance > 0:
            process_graphs(equity_gaps[i], trip_types, equity_titles[i]+"(p="+str(p_distance)+")")
        else:
            process_graphs(equity_gaps[i], trip_types, equity_titles[i])

    # def equity_gap(grid_gdf, trips_gdf, demo_relevant_cols=tuple(["F"]), urban_vit_cols=tuple(["y-HOUSINGP"]),
    #                start_time, end_time, e=0.5):


