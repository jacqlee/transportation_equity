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
from toolz import interleave
from collections import defaultdict


def fairness_gap(trip_df, metric):
    """

    :param trip_df:
    :param metric: column label for equity group
    :return: fairness gap for metric (sum over all blocks)
    """
    gaps = []
    dest_label = metric + "_dest"
    orig_label = metric + "_orig"
    trip_counts_dict = defaultdict(int, trip_df.set_index(["block_orig", "block_dest"]).trip_counts.T.to_dict())
    starts = list(trip_df.groupby("block_orig").indices.keys())
    for start in starts:
        arrive_at_start = trip_df.loc[(trip_df.block_dest == start) & (trip_df.block_orig != start)]
        if len(arrive_at_start) != 0:
            other_blocks_dis = arrive_at_start.apply(lambda x: (((x[dest_label] / x.POP_dest) * x.trip_counts) +
                                                                ((x[orig_label] / x.POP_orig) *
                                                                 trip_counts_dict[(x.block_dest, x.block_orig)])) /
                                                               (x[orig_label] * x[dest_label]), axis=1).sum()
            other_blocks_adv = arrive_at_start.apply(lambda x: ((((x.POP_dest - x[dest_label]) / x.POP_dest) * x.trip_counts) +
                                                                (((x.POP_orig - x[orig_label]) / x.POP_orig) *
                                                                 trip_counts_dict[(x.block_dest, x.block_orig)])) /
                                                               ((x.POP_orig - x[orig_label]) *
                                                                (x.POP_dest - x[dest_label])), axis=1).sum()
        else:
            other_blocks_dis = 0
            other_blocks_adv = 0

        start_same_block = trip_df.loc[(trip_df.block_dest == start) & (trip_df.block_orig == start)]
        if len(start_same_block) != 0:
            start_block_dis = start_same_block.apply(lambda x: (x.trip_counts *
                                                                (x[orig_label] / x.POP_orig)) / x[orig_label] ** 2, axis=1).sum()
            start_block_adv = start_same_block.apply(lambda x: (x.trip_counts *
                                                                ((x.POP_orig - x[orig_label]) / x.POP_orig)) /
                                                               (x.POP_orig - x[orig_label]) ** 2, axis=1).sum()
        else:
            start_block_dis = 0
            start_block_adv = 0

        first_term = other_blocks_adv + start_block_adv
        second_term = other_blocks_dis + start_block_dis
        if not np.isnan(first_term - second_term):
            gaps.append(first_term - second_term)
        else:
            gaps.append(0)

    return np.array(gaps).sum()


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
    window_trips_gdf = trips_gdf.loc[trips_gdf["time_x"] >= start_time]
    window_trips_gdf = window_trips_gdf.loc[window_trips_gdf["time_x"] <= end_time]

    o_window_trips_gdf = window_trips_gdf.set_geometry("origin")
    d_window_trips_gdf = window_trips_gdf.set_geometry("destination")

    o_df = gpd.sjoin(o_window_trips_gdf, grid_gdf).reset_index()
    d_df = gpd.sjoin(d_window_trips_gdf, grid_gdf).reset_index()

    # df.rename({"block": "orig_block", "POP": "orig_POP"}, inplace=True)
    # df["dest_block"] = d_df.block
    # df["dest_POP"] = d_df.POP
    df = o_df.join(d_df, lsuffix="_orig", rsuffix="_dest")

    # df = gpd.sjoin(window_trips_gdf, grid_gdf)

    # block_year_month_counts = df.groupby(["block", "year", "month"]).size()
    block_counts = df.groupby(["block_orig", "block_dest"]).size()
    block_year_month_counts = df.groupby(["block_orig", "block_dest", "year_orig", "month_orig"]).size()
    avg_trips = pd.DataFrame(block_year_month_counts.groupby(["block_orig", "block_dest"]).mean(),
                             columns=["avg_trips"]).reset_index()

    cols = ["block_orig", "block_dest", "POP_orig", "POP_dest"]
    for col in demo_relevant_cols:
        cols.append(col + "_orig")
        cols.append(col + "_dest")
    # cols.extend(demo_relevant_cols)
    for col in urban_vit_cols:
        cols.append(col + "_orig")
        cols.append(col + "_dest")
    # cols.extend(urban_vit_cols)
    simple_df = df[cols]

    trip_df = pd.DataFrame(block_counts, columns=["trip_counts"]).reset_index()
    # trip_df = pd.DataFrame(block_year_month_counts, columns=["trip_counts"]).reset_index()
    demographics = trip_df.apply(lambda row: simple_df.loc[(simple_df.block_orig == row.block_orig) &
                                                           (simple_df.block_dest == row.block_dest)].iloc[0], axis=1)

    # trip_df = pd.concat([trip_df, demographics], axis=1)
    trip_df = pd.concat([demographics, trip_df.trip_counts], axis=1)
    # trip_df = pd.concat([trip_df, demographics.drop(["block_orig", "block_dest"], axis=1)])

    o_trips = pd.DataFrame({"o_tripcount": trip_df.groupby("block_orig").sum().trip_counts}).reset_index()
    d_trips = pd.DataFrame({"d_tripcount": trip_df.groupby("block_dest").sum().trip_counts}).reset_index()

    block_od_counts = pd.merge(o_trips, d_trips, left_on="block_orig", right_on="block_dest")
    # o_new_df = trip_df.merge(avg_trips, on="block_orig").fillna(0)
    # d_new_df = trip_df.merge(avg_trips, on="block_dest").fillna(0)
    new_df = pd.concat([trip_df, avg_trips.avg_trips], axis=1)
        # trip_df.join(avg_trips, on=["block_orig", "block_dest"])

    counts = df.groupby(["block_orig", "block_dest"]).size()
    N = len(df.groupby(["block_orig", "block_dest"]).groups)
    # N = df.groupby("block").size()
    avg = counts.mean()

    equity_measures = {}

    if e < 1:
        atkinson = 1 - ((counts.apply(lambda y: y**(1-e)).sum()/N)**(1/(1-e))/avg)
    else:
        atkinson = 1 - ((counts.sum())**(1/N))/avg
    equity_measures["atkinson"] = atkinson

    for metric in demo_relevant_cols:
        gap = fairness_gap(trip_df, metric)
        # ratio = (o_new_df[metric+"_orig"] + d_new_df[metric+"_dest"])/(o_new_df["POP_orig"]+d_new_df["POP_dest"])
        # # disadv_num = (o_new_df["avg_trips"]+d_new_df["avg_trips"]).rmul(ratio).sum()
        # disadv_num = (trip_df["trip_counts"]+d_new_df["trip_counts"]).rmul(ratio).sum()
        # disadv_den = (o_new_df["POP_orig"]*d_new_df["POP_dest"]).rmul(ratio).sum()
        #
        # adv_num = (o_new_df["avg_trips"]+d_new_df["avg_trips"]).rmul(1-ratio).sum()
        # adv_den = (o_new_df["POP_orig"]*d_new_df["POP_dest"]).rmul(1-ratio).sum()

        # equity_measures[metric + "_gap"] = (adv_num/adv_den) - (disadv_num/disadv_den)
        equity_measures[metric + "_gap"] = gap

    for metric in urban_vit_cols:
        o_quart = (new_df[metric+"_orig"]).quantile(0.25)
        o_adv_group = new_df.loc[new_df[metric+"_orig"] >= o_quart]
        o_disadv_group = new_df.loc[new_df[metric+"_orig"] < o_quart]

        d_quart = (new_df[metric+"_dest"]).quantile(0.25)
        d_adv_group = new_df.loc[new_df[metric+"_dest"] >= d_quart]
        d_disadv_group = new_df.loc[new_df[metric+"_dest"] < d_quart]
        #
        # equity_measures[metric + "_gap"] = (o_adv_group["avg_trips"].sum() + d_adv_group)/(o_adv_group["POP"].sum()) - \
        #                                    o_disadv_group["avg_trips"].sum()/o_disadv_group["POP"].sum()

        equity_measures[metric + "_gap"] = (o_adv_group["avg_trips"].sum() + d_adv_group["avg_trips"].sum()) / \
                                           (o_adv_group["POP_orig"].sum()*d_adv_group["POP_dest"].sum()) - \
                                           (o_disadv_group["avg_trips"].sum() + d_disadv_group["avg_trips"].sum()) / \
                                           (o_disadv_group["POP_orig"].sum()*d_disadv_group["POP_dest"].sum())
    return equity_measures
