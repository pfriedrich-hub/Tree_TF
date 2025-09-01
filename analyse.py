import pickle
from pathlib import Path

filename = Path.cwd() / 'data' / '313_4.6_20N' / '313_4.6_20N.pkl'

with open(filename, 'rb') as handle:
    id_dictionary = pickle.load(handle)