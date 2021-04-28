from shapely import geometry
import geopandas as gpd
import pandas as pd
import osmnx as ox
import numpy as np
import skmob
from skmob.tessellation import tilers
from skmob.models.markov_diary_generator import MarkovDiaryGenerator


def process_trajectories():
    df = pd.read_pickle('old pickle files/clean_beijing_bikes.pkl').to_crs(epsg=4326)
    tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon',datetime='time', user_id='user', trajectory_id='trip_num')

    # Generate tesselation square tile grid over Beijing
    tessellation = tilers.tiler.get("squared", base_shape="Beijing, China", meters=1000)
    # Map trajectory data frame to grid
    mtdf = tdf.mapping(tessellation)
    return mtdf


def generate_synthetic_trajectories(tdf, hours, start_date):
    # ctdf = compression.compress(tdf)
    # stdf = detection.stops(ctdf)
    # cstdf = clustering.cluster(stdf)

    num_people = len(tdf.uid.unique())
    mdg = MarkovDiaryGenerator()
    mdg.fit(tdf, n_individuals=num_people, lid='tile_ID')

    mdg._get_location2frequency(tdf, location_column='tile_ID')

    return mdg.generate(diary_length=hours, start_date=start_date)

