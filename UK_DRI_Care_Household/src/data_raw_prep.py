import typer
from loguru import logger
import sqlite3
import pandas as pd

from sklearn.model_selection import train_test_split
import pickle
from functools import reduce

app = typer.Typer()


# basic db preprocessing
def preproc_db(db_file_path, proc_folder_path):
    logger.info("Importing from db...")

    # loading raw data
    conn = sqlite3.connect(db_file_path)

    # Loading both of the tables
    homes_y = pd.read_sql_query("""   
    SELECT * 
    FROM homes 
    WHERE id IN (SELECT home_id FROM motion);
    """, conn)
    motion = pd.read_sql_query("SELECT * FROM motion", conn)
    conn.close()

    logger.info("Preprocessing into one datatable...")

    all_data_table = pd.merge(homes_y, motion, left_on='id', right_on='home_id', suffixes=('_home', '_motion')).drop(
        columns=['home_id'])

    all_data_table.to_pickle(proc_folder_path / 'full_data.pkl')


def proc_1(table):
    # Table_1 features:
    # motion_count = count of motions overall, between houses
    # hour_mean = mean of the hours of movements
    # hour_std = the std of the mean of the hours of the movements
    table['hour'] = table['datetime'].dt.hour
    features_1 = table.groupby('id_home').agg({
        'id_motion': 'count',  # Number of motion events
        'hour': ['mean', 'std'],  # Mean and standard deviation of the hour of events
    }).reset_index()
    features_1.columns = ['id_home', 'motion_count', 'hour_mean', 'hour_std']
    table_ready = features_1.copy()

    return table_ready


def proc_2(table):
    # Table_2 features:
    # Change count between room types at night

    table['date'] = table['datetime'].dt.date
    # filtering out only the night hours
    mask = (table['hour'] >= 20) | (table['hour'] < 10)
    filtered_df = table[mask]
    filtered_df_sorted = (filtered_df.sort_values(['id_home', 'datetime'])).reset_index()
    # counting the amount of transitions between rooms
    filtered_df_sorted['match'] = filtered_df_sorted.location != filtered_df_sorted.location.shift()
    filtered_df_sorted = (filtered_df_sorted.sort_values(['id_home', 'datetime'])).groupby(
        ['id_home', 'multiple_occupancy']).agg({'match': 'count'}).reset_index()
    table_ready = filtered_df_sorted.copy()

    return table_ready


def preproc_for_model(proc_folder_path, output_file_path):
    proc_folder_path = proc_folder_path / 'full_data.pkl'

    all_data_table = pd.read_pickle(proc_folder_path)
    all_data_table['datetime'] = pd.to_datetime(all_data_table['datetime'])

    ### Step 1: Process and prepare features for train-test split:

    functions_to_proc = [proc_1, proc_2]
    tables_to_merge = []
    for fun_to_proc in functions_to_proc:
        tables_to_merge.append(fun_to_proc(all_data_table))

    full_data_pre_model = reduce(lambda df1, df2: pd.merge(df1, df2, on='id_home'), tables_to_merge)

    ### Step 2: Choose features and Train Test Split

    X = full_data_pre_model[['motion_count', 'match', 'hour_mean']]
    y = full_data_pre_model['multiple_occupancy']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with open(output_file_path, 'wb') as f:
        pickle.dump([X_train, X_test, y_train, y_test], f)
