import re

def numeric(fname):
    return int(re.search(r'_s(\d+)\.h5', fname).group(1))