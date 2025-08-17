import time

def time_funtion(func):
    def wrapper(*args):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        return result, end-start
    return wrapper