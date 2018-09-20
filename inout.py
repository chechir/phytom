import os
# import cPickle
import _pickle as cPickle
import json
import optparse
import os
import sys


def append_csv(data, path):
    from ddf import DDF
    assert path.endswith('.csv')
    to_log = data if isinstance(data, DDF) else DDF(data)
    if os.path.isfile(path):
        current_log = DDF.from_csv(path)
        current_log = current_log.append(to_log, axis=0)
    else:
        current_log = to_log
    current_log.to_csv(path)


def ensure_dir_exists(path):
    """ Takes path like:
           /path/to/test.pkl  OR
           /path/to/
        Ensures directory exists so file can be writen
    """
    if '.' in path.split('/')[-1]:
        directory = '/'.join(path.split('/')[:-1])
    else:
        directory = path
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_pickle(path):
    with open(path, 'rb') as f:
        obj = cPickle.load(f)
    return obj


def write_pickle(obj, path):
    with open(path, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)


def write_json(obj, path, encoder=None):
    with open(path, 'wb') as f:
        if encoder is None:
            json.dump(obj, f)
        else:
            json.dump(obj, f, cls=encoder)


def read_json(path):
    with open(path, 'rb') as f:
        obj = json.load(f)
    return obj


def read_json_per_line(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_json_per_line(list_of_data, path):
    """ Each element in list_of_data will be written to path as a json on its own line. """
    ensure_dir_exists(path)
    assert isinstance(list_of_data, list), 'data must be list'
    with open(path,'a') as f:
        for row in list_of_data:
            json_formatted_data = json.dumps(row, cls=NumpyEncoder)
            f.write(json_formatted_data + '\n')
