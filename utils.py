from os import listdir
import os.path as osp
from random import shuffle
import random
import shlex
import subprocess
import sqlite3
import datetime

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import psutil
import pandas as pd
from pandas.io.sql import DatabaseError
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
from psycopg2.sql import SQL, Identifier
import torch
import torch.nn.functional as F


def print_with_time(x):
    now = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print("%s: %s" % (now, x))


class RemainingTasksTaken(Exception):
    pass


class PopulationFinished(Exception):
    pass


class ExploitationNeeded(Exception):
    pass


class ExploitationOcurring(Exception):
    pass


class LossIsNaN(Exception):
    pass


def register_numpy_types():
    # Credit: https://github.com/musically-ut/psycopg2_numpy_ext
    """Register the AsIs adapter for following types from numpy:
      - numpy.int8
      - numpy.int16
      - numpy.int32
      - numpy.int64
      - numpy.float16
      - numpy.float32
      - numpy.float64
      - numpy.float128
    """
    for typ in ['int8', 'int16', 'int32', 'int64',
                'float16', 'float32', 'float64', 'float128',
                'bool_']:
        register_adapter(np.__getattribute__(typ), AsIs)


def get_task_ids_and_scores(connect_str_or_path, use_sqlite, population_id):
    if use_sqlite:
        sqlite_path = connect_str_or_path
        conn = sqlite3.connect(sqlite_path)
        command = """
                  SELECT task_id, score
                  FROM populations
                  WHERE population_id = ?
                  ORDER BY score DESC
                  """
    else:
        db_connect_str = connect_str_or_path
        conn = psycopg2.connect(db_connect_str)
        command = """
                  SELECT task_id, score
                  FROM populations
                  WHERE population_id = %s
                  ORDER BY score DESC
                  """
    cur = conn.cursor()
    cur.execute(command, [population_id])
    results = cur.fetchall()
    cur.close()
    conn.close()
    task_ids = [result[0] for result in results]
    scores = [result[1] for result in results]
    return task_ids, scores


def get_col_from_populations(connect_str_or_path, use_sqlite,
                             population_id, column_name):
    if use_sqlite:
        sqlite_path = connect_str_or_path
        conn = sqlite3.connect(sqlite_path)
        command = "SELECT {} FROM populations WHERE population_id = ?"
        command = command.format(column_name)  # Warning: SQL injection
    else:
        db_connect_str = connect_str_or_path
        conn = psycopg2.connect(db_connect_str)
        command = "SELECT {} FROM populations WHERE population_id = %s"
        command = SQL(command).format(Identifier(column_name))
    cur = conn.cursor()
    cur.execute(command, [population_id])
    column = cur.fetchall()
    cur.close()
    conn.close()
    column = [value[0] for value in column]
    return column


def update_table(connect_str_or_path, use_sqlite, table_name, key_value_pairs,
                 where_string=None, where_variables=None):
    values = [v.__name__ if callable(v) or isinstance(v, type) else v
              for v in key_value_pairs.values()]
    if use_sqlite:
        sqlite_path = connect_str_or_path
        conn = sqlite3.connect(sqlite_path)
        fields = list(key_value_pairs.keys())
        placeholders = get_placeholders(len(key_value_pairs), "{} = ?")
        if where_string is None:
            where_string = "WHERE id = ?"
            row_id = key_value_pairs['id']
            where_variables = [row_id]
        command = " ".join(["UPDATE {}",
                            "SET {}".format(placeholders),
                            where_string])
        command = command.format(table_name, *fields)
    else:
        register_numpy_types()
        db_connect_str = connect_str_or_path
        conn = psycopg2.connect(db_connect_str)
        table_name = Identifier(table_name)
        fields = [Identifier(field) for field in key_value_pairs.keys()]
        placeholders = get_placeholders(len(key_value_pairs), "{} = %s")
        if where_string is None:
            where_string = "WHERE id = %s"
            row_id = key_value_pairs['id']
            where_variables = [row_id]
        command = " ".join(["UPDATE {}",
                            "SET {}".format(placeholders),
                            where_string])
        command = SQL(command).format(table_name, *fields)
    parameters = values + where_variables
    cur = conn.cursor()
    cur.execute(command, parameters)
    conn.commit()
    cur.close()
    conn.close()


def update_task(connect_str_or_path, use_sqlite,
                population_id, task_id, key_value_pairs):
    if use_sqlite:
        where_string = "WHERE population_id = ? AND task_id = ?"
    else:
        where_string = "WHERE population_id = %s AND task_id = %s"
    where_variables = [population_id, task_id]
    update_table(connect_str_or_path, use_sqlite,
                 "populations", key_value_pairs,
                 where_string=where_string, where_variables=where_variables)


def get_a_task(connect_str_or_path, use_sqlite, population_id, interval_limit):
    if use_sqlite:
        sqlite_path = connect_str_or_path
        conn = sqlite3.connect(sqlite_path)
        command_get_task = """
                           SELECT task_id
                           FROM populations
                           WHERE population_id = ?
                           AND ready_for_exploitation = 0
                           AND active = 0
                           LIMIT 1
                           """
        command_lock_task = """
                            UPDATE populations
                            SET active = 1
                            WHERE population_id = ?
                            AND task_id = ?
                            """
        command_get_task_info = """
                                SELECT intervals_trained, seed_for_shuffling
                                FROM populations
                                WHERE population_id = ?
                                AND task_id = ?
                                """
    else:
        db_connect_str = connect_str_or_path
        conn = psycopg2.connect(db_connect_str)
        command_get_task = """
                           SELECT task_id
                           FROM populations
                           WHERE population_id = %s
                           AND ready_for_exploitation = False
                           AND active = False
                           LIMIT 1
                           FOR SHARE
                           """
        command_lock_task = """
                            UPDATE populations
                            SET active = True
                            WHERE population_id = %s
                            AND task_id = %s
                            """
        command_get_task_info = """
                                SELECT intervals_trained, seed_for_shuffling
                                FROM populations
                                WHERE population_id = %s
                                AND task_id = %s
                                """
    cur = conn.cursor()
    cur.execute(command_get_task, [population_id])
    try:
        task_id = cur.fetchone()[0]
        cur.execute(command_lock_task, [population_id, task_id])
        conn.commit()
        cur.execute(command_get_task_info, [population_id, task_id])
        intervals_trained, seed_for_shuffling = cur.fetchone()
        cur.close()
        conn.close()
        return task_id, intervals_trained, seed_for_shuffling
    except TypeError:
        cur.close()
        conn.close()
        activities = get_col_from_populations(
            connect_str_or_path, use_sqlite, population_id, "active")
        any_are_active = [a for a in activities if a]
        if any_are_active:
            raise RemainingTasksTaken
        intervals_trained_col = get_col_from_populations(
            connect_str_or_path, use_sqlite, population_id,
            "intervals_trained")
        unfinished = [i for i in intervals_trained_col
                      if i < interval_limit]
        if not unfinished:
            raise PopulationFinished
        readys = get_col_from_populations(
            connect_str_or_path, use_sqlite, population_id,
            "ready_for_exploitation")
        not_ready = [r for r in readys if not r]
        if not not_ready:
            raise ExploitationNeeded
        else:
            raise ExploitationOcurring


def get_max_of_db_column(connect_str_or_path, use_sqlite, table_name,
                         column_name):
    if use_sqlite:
        sqlite_path = connect_str_or_path
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        parameters = [column_name, table_name]
        cur.execute("SELECT MAX({}) FROM {}".format(*parameters))
    else:
        db_connect_str = connect_str_or_path
        conn = psycopg2.connect(db_connect_str)
        cur = conn.cursor()
        parameters = [Identifier(column_name), Identifier(table_name)]
        cur.execute(SQL("SELECT MAX({}) FROM {}").format(*parameters))
    max_value = cur.fetchone()[0]
    cur.close()
    conn.close()
    return max_value


def insert_into_table(connect_str_or_path, use_sqlite, table_name,
                      key_value_pairs):
    if use_sqlite:
        sqlite_path = connect_str_or_path
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        fields = key_value_pairs.keys()
        values = list(key_value_pairs.values())
        field_placeholders = get_placeholders(len(key_value_pairs), "{}")
        field_placeholders = "({})".format(field_placeholders)
        values_placeholders = get_placeholders(len(key_value_pairs), "?")
        values_placeholders = "({})".format(values_placeholders)
        # Warning: This command is vulnerable to SQL injection via
        # the fields variable.
        command = " ".join(["INSERT INTO populations", field_placeholders,
                            "VALUES", values_placeholders]).format(*fields)
    else:
        # TODO: Clean (see above block)
        db_connect_str = connect_str_or_path
        conn = psycopg2.connect(db_connect_str)
        register_numpy_types()
        table_name = Identifier(table_name)
        fields = [Identifier(field) for field in key_value_pairs.keys()]
        values = [v.__name__ if callable(v) or isinstance(v, type) else v
                  for v in key_value_pairs.values()]
        conn = psycopg2.connect(db_connect_str)
        cur = conn.cursor()
        insert_part = "INSERT INTO {}"
        field_positions = get_placeholders(len(key_value_pairs), "{}")
        fields_part = "({})".format(field_positions)
        value_positions = get_placeholders(len(key_value_pairs), "%s")
        values_part = "VALUES ({})".format(value_positions)
        command = insert_part + " " + fields_part + " " + values_part
        command = SQL(command).format(table_name, *fields)
    cur.execute(command, values)
    conn.commit()
    cur.close()
    conn.close()


def create_table(connect_str_or_path, use_sqlite, command):
    if use_sqlite:
        sqlite_path = connect_str_or_path
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        cur.execute(command)
        conn.commit()
        cur.close()
    else:
        conn = None
        try:
            db_connect_str = connect_str_or_path
            conn = psycopg2.connect(db_connect_str)
            cur = conn.cursor()
            cur.execute(command)
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            if "already exists" not in str(error):
                print(error)
        finally:
            if conn is not None:
                conn.close()


def get_placeholders(num, form):
    """
    Example:
        >>> get_placeholders(num=3, form="%s")
        '%s, %s, %s'
    """
    return ' '.join([form + "," for _ in range(num)])[:-1]


def create_new_population(connect_str_or_path, use_sqlite, population_size):
    if use_sqlite:
        command = """
                  CREATE TABLE populations (
                        population_id INTEGER,
                        task_id INTEGER,
                        intervals_trained INTEGER,
                        ready_for_exploitation INTEGER,
                        active INTEGER,
                        score REAL,
                        seed_for_shuffling INTEGER
                  )
                  """
        ready_for_exploitation = 0
        active = 0
    else:
        command = """
                  CREATE TABLE populations (
                        population_id INTEGER,
                        task_id INTEGER,
                        intervals_trained INTEGER,
                        ready_for_exploitation BOOLEAN,
                        active BOOLEAN,
                        score REAL,
                        seed_for_shuffling INTEGER
                  )
                  """
        ready_for_exploitation = False
        active = False
    table_name = "populations"
    try:
        latest_population_id = get_max_of_db_column(connect_str_or_path,
                                                    use_sqlite,
                                                    table_name,
                                                    "population_id")
        population_id = latest_population_id + 1
    except (sqlite3.OperationalError, psycopg2.ProgrammingError):
        create_table(connect_str_or_path, use_sqlite, command)
        population_id = 0
    for task_id in range(population_size):
        key_value_pairs = dict(population_id=population_id,
                               task_id=task_id,
                               intervals_trained=0,
                               ready_for_exploitation=ready_for_exploitation,
                               active=active,
                               score=None,
                               seed_for_shuffling=123)
        insert_into_table(connect_str_or_path, use_sqlite, table_name,
                          key_value_pairs)
    return population_id


def choose(x):
    return np.random.choice(x)


def print_separator():
    print("-"*80)


def get_database_path(here):
    return osp.join(osp.join(here, "logs"), "database.sqlite")


def load_sqlite_table(database_path, table_name):
    """Returns (table, connection). table is a pandas DataFrame."""
    conn = sqlite3.connect(database_path)
    try:
        df = pd.read_sql("SELECT * FROM %s" % table_name, conn)
        #  print("\nLoading %s table from SQLite3 database." % table_name)
    except DatabaseError as e:
        if 'no such table' in e.args[0]:
            print("\nNo such table: %s" % table_name)
            print("Create the table before loading it. " +
                  "Consider using the create_sqlite_table function")
            raise DatabaseError
        else:
            print(e)
            raise Exception("Failed to create %s table. Unknown error." %
                            table_name)
    return df, conn


def create_sqlite_table(database_path, table_name, table_header):
    """Returns (table, connection). table is a pandas DataFrame."""
    conn = sqlite3.connect(database_path)
    print("\nCreating %s table in SQLite3 database." % table_name)
    df = pd.DataFrame(columns=table_header)
    df.to_sql(table_name, conn, index=False)
    return df, conn


def create_log(filepath, headers):
    if not osp.exists(filepath):
        with open(filepath, 'w') as f:
            f.write(','.join(headers) + '\n')


def get_RAM():
    return psutil.virtual_memory().used


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    return hash


def transform_portrait(img):
    img = np.array(img, dtype=np.uint8)
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    img -= mean_bgr
    img = img.transpose(2, 0, 1)  # HxWxC --> CxHxW
    return img


def split_trn_val(num_train, valid_size=0.2, shuffle=False):
    indices = list(range(num_train))
    if shuffle:
        np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    trn_indices, val_indices = indices[split:], indices[:split]
    return trn_indices, val_indices


def cross_entropy2d(score, target, weight=None, size_average=True):
    log_p = F.log_softmax(score)

    # Flatten the score tensor
    n, c, h, w = score.size()
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    # Remove guesses corresponding to "unknown" labels
    # (labels that are less than zero)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    # Remove "unknown" labels (labels that are less than zero)
    # Also, flatten the target tensor
    # TODO: Replace this entire function with nn.functional.cross_entropy
    #   with ignore_index set to -1.
    mask = target >= 0
    target = target[mask]

    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)

    if size_average:
        loss /= mask.data.sum()
    return loss


def scoretensor2mask(scoretensor):
    """
    - scoretensor (3D torch tensor) (CxHxW): Each channel contains the scores
        for the corresponding category in the image.
    Returns a numpy array.
    """
    _, labels = scoretensor.max(0)  # Get labels w/ highest scores
    labels_np = labels.numpy().astype(np.uint8)
    mask = labels_np * 255
    return mask


def detransform_portrait(img, mean="voc"):
    """
    - img (torch tensor)
    Returns a numpy array.
    """
    if mean == "voc":
        mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    else:
        raise ValueError("unknown mean")
    #  img = img.numpy().astype(np.float64)
    img = img.transpose((1, 2, 0))  # CxHxW --> HxWxC
    #  img *= 255
    img += mean_bgr
    img = img[:, :, ::-1]  # BGR -> RGB
    img = img.astype(np.uint8)
    return img


def detransform_mask(mask):
    #  mask = mask.numpy()
    mask = mask.astype(np.uint8)
    mask *= 255
    return mask


def mask_image(img, mask, opacity=1.00, bg=False):
    """
        - img (PIL)
        - mask (PIL)
        - opacity (float) (default: 1.00)
    Returns a PIL image.
    """
    blank = Image.new('RGB', img.size, color=0)
    if bg:
        masked_image = Image.composite(blank, img, mask)
    else:
        masked_image = Image.composite(img, blank, mask)
    if opacity < 1:
        masked_image = Image.blend(img, masked_image, opacity)
    return masked_image


def show_portrait_pred_mask(portrait, preds, mask, start_iteration,
                            evaluation_interval,
                            opacity=None, bg=False, fig=None):
    """
    Args:
        - portrait (torch tensor)
        - preds (list of np.ndarray): list of mask predictions
        - mask (torch tensor)
    A visualization function.
    Returns nothing.
    """
    # Gather images
    images = []
    titles = []
    cmaps = []

    #  ### Prepare portrait
    portrait_pil = Image.fromarray(portrait)
    images.append(portrait)
    titles.append("input")
    cmaps.append(None)

    #  ### Prepare predictions
    for i, pred in enumerate(preds):
        pred_pil = Image.fromarray(pred)
        if opacity:
            pred_pil = mask_image(portrait_pil, pred_pil, opacity, bg)
        images.append(pred_pil)
        titles.append("iter. %d" % (start_iteration + i * evaluation_interval))
        cmaps.append("gray")

    #  ### Prepare target mask
    if opacity:
        mask_pil = Image.fromarray(mask)
        mask = mask_image(portrait_pil, mask_pil, opacity, bg)
    images.append(mask)
    titles.append("target")
    cmaps.append("gray")

    # Show images
    cols = 5
    rows = int(np.ceil(len(images) / cols))
    w = 12
    h = rows * (w / cols + 1)
    figsize = (w, h)  # width x height
    plots(images, titles=titles, cmap=cmaps, rows=rows, cols=cols,
          figsize=figsize, fig=fig)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_fnames(d, random=False):
    fnames = [d + f for f in listdir(d) if osp.isfile(osp.join(d, f))]
    print("Number of files found in %s: %s" % (d, len(fnames)))
    if random:
        shuffle(fnames)
    return fnames


def rm_dir_and_ext(filepath):
    return filepath.split('/')[-1].split('.')[-2]


def get_flickr_id(portrait_fname):
    """
    Input (string): '../data/portraits/flickr/cropped/portraits/00074.jpg'
    Output (int): 74
    """
    return int(rm_dir_and_ext(portrait_fname))


def get_lines(fname):
    '''Read lines, strip, and split.'''
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip().split() for x in content]
    return content


def hist(data, figsize=(6, 3)):
    plt.figure(figsize=figsize)
    plt.hist(data)
    plt.show()


def plot_portraits_and_masks(portraits, masks):
    assert len(portraits) == len(masks)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.tight_layout()
    for i, ax in enumerate(axes.flat):
        if i < 4:
            ax.imshow(portraits[i], interpolation="spline16")
        else:
            mask = gray2rgb(masks[i-4])
            ax.imshow(mask)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.uint8)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb


def plots(imgs, figsize=(12, 12), rows=None, cols=None,
          interp=None, titles=None, cmap='gray',
          fig=None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [np.array(img) for img in imgs]
    if not isinstance(cmap, list):
        if imgs[0].ndim == 2:
            cmap = 'gray'
        cmap = [cmap] * len(imgs)
    if not isinstance(interp, list):
        interp = [interp] * len(imgs)
    n = len(imgs)
    if not rows and not cols:
        cols = n
        rows = 1
    elif not rows:
        rows = cols
    elif not cols:
        cols = rows
    if not fig:
        rows = int(np.ceil(len(imgs) / cols))
        w = 12
        h = rows * (w / cols + 1)
        figsize = (w, h)
        fig = plt.figure(figsize=figsize)
    fontsize = 13 if cols == 5 else 16
    fig.set_figheight(figsize[1], forward=True)
    fig.clear()
    for i in range(len(imgs)):
        sp = fig.add_subplot(rows, cols, i+1)
        if titles:
            sp.set_title(titles[i], fontsize=fontsize)
        plt.imshow(imgs[i], interpolation=interp[i], cmap=cmap[i])
        plt.axis('off')
        plt.subplots_adjust(0, 0, 1, 1, .1, 0)
        #  plt.tight_layout()
    if fig:
        fig.canvas.draw()
