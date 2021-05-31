import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from math import cos, sin
from shapely import geometry
# from skmob.tessellation import tilers
from datetime import datetime
from equity_graphs import process_graphs
from equity_gap import equity_gap
from numpy.core.function_base import linspace
from differential_privacy import addLaplaceNoise

crs = 'epsg:4326'
mode_names = ['walk', 'bike', 'bus', 'car', 'subway','train', 'airplane', 'boat', 'run', 'motorcycle', 'taxi']
mode_ids = {s: i + 1 for i, s in enumerate(mode_names)}
# spatial cloaking distance in km
p_distance = 0
# differential privacy epsilon value (greater e -> less privacy)
differential_epsilon = 0.0001


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


def polar_coordinates(point, center_point):
    r = coord_distance(center_point.x, center_point.y, point.x, point.y)
    a1 = np.arctan2(center_point.x, center_point.y)
    a2 = np.arctan2(point.x, point.y)
    theta = np.rad2deg((a1-a2) % (2*np.pi))
    return [r, theta]


def polar_to_coordinates(r, theta, center_point):
    center_lat = np.radians(center_point.x)
    center_lon = np.radians(center_point.y)

    r2 = np.radians(r)
    theta2 = np.radians(theta)

    b = math.asin(sin(r2)*sin(theta2))
    c = 2*math.atan((math.tan(0.5*(r2-b))*(sin(0.5*(math.pi/2 + theta2))/sin(0.5*(math.pi/2 - theta2)))))
    # c = 2 * math.atan((math.tan(np.radians(0.5 * (r - b)) * (sin(0.5 * (math.pi / 2 + theta)) / sin(0.5 * (math.pi / 2 - theta)))))

    latitude = center_lat + b
    longitude = center_lon + c
    return geometry.Point(np.degrees(latitude), np.degrees(longitude))


def process_beijing_trips(trip_type=0, save_od_df=False, privacy_scheme=0):
    """

    :param trip_type:
    :param save_od_df:
    :param privacy_scheme: 0 if none, 1 if spatial cloaking, 2 if geo-indistinguishability/differential privacy
    :return:
    """
    df = pd.read_pickle("geolife.pkl")
    if trip_type != 0:
        #   Specified trip type
        df = df.loc[df['label'] == trip_type]

    start_points = df.loc[df.index == 0].reset_index(drop=True)

    ind = df.index.append(pd.Int64Index([0])).get_loc(0)
    start_ind = list(filter(lambda i: ind[i], range(len(ind))))
    end_ind = list(j - 1 for j in start_ind)
    end_ind = end_ind[1:]

    end_points = df.iloc[end_ind].reset_index()
    end_points.rename(columns={"index": "endpt_stamp"}, inplace=True)

    df = start_points.merge(end_points, left_index=True, right_index=True)

    if trip_type == 0 and save_od_df:
        pd.to_pickle("geolife_od.pkl", protocol=2)

    trip_ogeometries = [geometry.Point(xy) for xy in zip(df.lat_x, df.lon_x)]
    trip_dgeometries = [geometry.Point(xy) for xy in zip(df.lat_y, df.lon_y)]
    trip_geometries = [geometry.MultiPoint(od) for od in zip(trip_ogeometries, trip_dgeometries)]

    df["origin"] = trip_ogeometries
    df["destination"] = trip_dgeometries

    points = gpd.GeoDataFrame(df, crs=crs, geometry=trip_geometries)

    bounding_box = [(39.77750000, 116.17944444), (39.77750000, 116.58888889), (40.04722222, 116.58888889),
                 (40.04722222, 116.17944444)]

    poly = geometry.Polygon(bounding_box)
    spoly = gpd.GeoSeries([poly], crs=crs)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    beijing_trips = pd.DataFrame(points[points.within(spoly.geometry.iloc[0])])
    beijing_trips["geometry"] = beijing_trips.geometry.apply(lambda multipt:
                                                             geometry.MultiPoint([geometry.Point(pt.y, pt.x)
                                                                                  for pt in multipt]))
    beijing_trips = gpd.GeoDataFrame(beijing_trips)
    # beijing_trips.plot()
    beijing_trips.plot(ax=ax1)
    if privacy_scheme > 0:
        if privacy_scheme == 1:
            beijing_trips = spatial_cloaking(points, bounding_box, privacy_distance=p_distance)
            beijing_trips = beijing_trips.drop("index_right", axis=1)
        else:
            # if isinstance(beijing_trips["geometry"][0], geometry.Point):
            #     beijing_trips["geometry2"] = beijing_trips.geometry.apply(lambda pt: geometry.Point(pt.y, pt.x))
            # else:
            #     beijing_trips["geometry2"] = beijing_trips.geometry.apply(lambda multipt:
            #                                                               geometry.MultiPoint(
            #                                                                   [geometry.Point(pt.y, pt.x)
            #                                                                    for pt in multipt]))
            beijing_trips["geometry2"] = beijing_trips["geometry"].apply(lambda multipt:
                                                geometry.MultiPoint([geometry.Point(
                                                    addLaplaceNoise(
                                                        differential_epsilon, [pt.y, pt.x]))
                                                    for pt in multipt.geoms]))
            # beijing_trips["geometry2"] = beijing_trips["geometry2"].apply(lambda multipt:
            #                                                               geometry.MultiPoint([geometry.Point(
            #                                                                   addLaplaceNoise(
            #                                                                       differential_epsilon,
            #                                                                       polar_coordinates(pt,
            #                                                                                         center_point=poly.centroid)))
            #                                                                   for pt in multipt.geoms]))
            # beijing_trips["geometry2"] = \
            #     beijing_trips["geometry2"].apply(lambda multipt:
            #                                      geometry.MultiPoint([geometry.Point(
            #                                          polar_to_coordinates(pt.x, pt.y, center_point=poly.centroid))
            #                                          for pt in multipt.geoms]))

            if isinstance(beijing_trips["geometry"][0], geometry.Point):
                beijing_trips["geometry2"] = beijing_trips.geometry2.apply(lambda pt: geometry.Point(pt.y, pt.x))
            else:
                beijing_trips["geometry2"] = beijing_trips.geometry2.apply(lambda multipt:
                                                                         geometry.MultiPoint([geometry.Point(pt.y, pt.x)
                                                                                              for pt in multipt]))
            beijing_trips = beijing_trips.set_geometry("geometry2")
            beijing_trips = beijing_trips.drop("geometry", axis=1)
            beijing_trips.rename_geometry("geometry", inplace=True)

        beijing_trips.plot(ax=ax2, color="red")

    print('Number of points within Beijing: ', beijing_trips.shape[0])
    plt.show()

    # REVERSE BACK
    if isinstance(beijing_trips["geometry"][0], geometry.Point):
        beijing_trips["geometry"] = beijing_trips.geometry.apply(lambda pt: geometry.Point(pt.y, pt.x))
    else:
        beijing_trips["geometry"] = beijing_trips.geometry.apply(lambda multipt:
                                                                 geometry.MultiPoint([geometry.Point(pt.y, pt.x)
                                                                                      for pt in multipt]))
    # beijing_trips["geometry"] = beijing_trips.geometry.apply(lambda pt: geometry.Point(pt.y, pt.x))
    # beijing_trips["geometry"] = beijing_trips.geometry.apply(lambda multipt:
    #                                                          geometry.MultiPoint([geometry.Point(pt.y, pt.x)
    #                                                                               for pt in multipt]))
    df = pd.DataFrame(beijing_trips)
    dates = pd.to_datetime(df.time_x)

    df["month"] = dates.apply(lambda dt: dt.month)
    df["year"] = dates.apply(lambda dt: dt.year)
    return df


def coord_distance(lat1, lon1, lat2, lon2):
    """
    Returns distance between two coordinates in km
    :param lat1:
    :param lon1:
    :param lat2:
    :param lon2:
    :return:
    """
    R = 6371.0
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

    beijing_data_gdf = beijing_data_gdf.drop("geometry_y", axis=1)
    # demographic_data = gpd.sjoin(beijing_urban_vit_gdf, beijing_data_gdf).rename(columns={"index_right": "block"})
    # demographic_data = gpd.sjoin(beijing_data_gdf, beijing_urban_vit_gdf).rename(columns={"index_right": "block"})
    demographic_data = gpd.sjoin(beijing_data_gdf, beijing_urban_vit_gdf).drop("index", axis=1).reset_index()

    demographic_data = demographic_data.drop("index_right", axis=1)
    demographic_data.rename(columns={"index": "block"}, inplace=True)
    return gpd.GeoDataFrame(demographic_data, crs=crs, geometry=demographic_data.geometry_x)

    # df["geometry"] = df.geometry.apply(lambda x: geometry.Point(x.y, x.x))
    # gdf = gpd.GeoDataFrame(df, crs=crs, geometry=df.geometry)
    #
    # return gpd.sjoin(gdf, demographic_data_gdf)


if __name__ == '__main__':
    trip_types = ["walk", "bike", "car", "taxi", "bus", "subway"]
    privacy_scheme=2
    results = {}
    grid_gdf = grid_data()
    start = np.datetime64("2008-01", "D")
    end = np.datetime64("2009-01", "D")
    overall_gdf = gpd.GeoDataFrame()
    for x in trip_types:
        df = process_beijing_trips(mode_ids[x], privacy_scheme=privacy_scheme)
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
        # if p_distance > 0:
        if privacy_scheme > 0:
            if privacy_scheme == 1:
                fig_title = equity_titles[i] + "(p=" + str(p_distance) + ")"
            else:
                fig_title = equity_titles[i] + "(e=" + str(differential_epsilon) + ")"
            process_graphs(equity_gaps[i], trip_types, fig_title)
        else:
            process_graphs(equity_gaps[i], trip_types, equity_titles[i])

    # def equity_gap(grid_gdf, trips_gdf, demo_relevant_cols=tuple(["F"]), urban_vit_cols=tuple(["y-HOUSINGP"]),
    #                start_time, end_time, e=0.5):


