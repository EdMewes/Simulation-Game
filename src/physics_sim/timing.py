import time

def time_funtion(func):
    def wrapper(*args):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper