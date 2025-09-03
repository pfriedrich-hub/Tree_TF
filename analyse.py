import pickle
from pathlib import Path

filename = Path.cwd() / 'data' /  '332_5_55NO' /  '332_5_55NO.pkl'

with open(filename, 'rb') as handle:
    id_dictionary = pickle.load(handle)