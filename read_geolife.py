import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os, sys
import osmnx

def read_plt(plt_file):
    """
    Read GeoLife plot file
    :param plt_file:
    :return: DataFrame containing trajectory data
    """

    points = pd.read_csv(plt_file, skiprows=6, header=None,
                         parse_dates=[[5, 6]], infer_datetime_format=True)

    # for clarity rename columns
    points.rename(inplace=True, columns={'5_6': 'time', 0: 'lat', 1: 'lon', 3: 'alt'})

    # remove unused columns
    points.drop(inplace=True, columns=[2, 4])

    return points

mode_names = ['walk', 'bike', 'bus', 'car', 'subway','train', 'airplane', 'boat', 'run', 'motorcycle', 'taxi']
mode_ids = {s: i + 1 for i, s in enumerate(mode_names)}

def read_labels(labels_file):
    """
    Reads label txt file for labels dictionary to DataFrame
    :param labels_file:
    :return replaces labels in txt file for label id
    """
    labels = pd.read_csv(labels_file, skiprows=1, header=None,
                         parse_dates=[[0, 1], [2, 3]],
                         infer_datetime_format=True, delim_whitespace=True)

    # for clarity rename columns
    labels.columns = ['start_time', 'end_time', 'label']

    # replace 'label' column with integer encoding
    labels['label'] = [mode_ids[i] for i in labels['label']]

    return labels

def apply_labels(points, labels):
    """

    :param points: DataFrame of all trips for single user
    :param labels: DataFrame of labels (ID) for trips
    :return: points with label ID column
    """
    indices = labels['start_time'].searchsorted(points['time'], side='right') - 1
    no_label = (indices < 0) | (points['time'].values >= labels['end_time'].iloc[indices].values)
    points['label'] = labels['label'].iloc[indices].values
    points['label'][no_label] = 0

def read_user(user_folder):
    """

    :param user_folder: folder for GeoLife data for specific user (XXX where X is a number)
    :return: returns DataFrame item for all user's trip data
    """
    labels = None

    plt_files = glob.glob(os.path.join(user_folder, 'Trajectory', '*.plt'))
    df = pd.concat([read_plt(f) for f in plt_files])

    labels_file = os.path.join(user_folder, 'labels.txt')
    if os.path.exists(labels_file):
        labels = read_labels(labels_file)
        apply_labels(df, labels)
    else:
        df['label'] = 0

    return df

def read_all_users(folder):
    """

    :param folder: path to folder of all user folders
    :return: DataFrame of DataFrames for each user and all trip data
    """
    subfolders = list(filter(lambda x: not x.startswith("."), os.listdir(folder)))

    dfs = []
    for i, sf in enumerate(subfolders):
        print('[%d/%d] processing user %s' % (i + 1, len(subfolders), sf))
        df = read_user(os.path.join(folder, sf))
        df['user'] = int(sf)
        dfs.append(df)
    return pd.concat(dfs)