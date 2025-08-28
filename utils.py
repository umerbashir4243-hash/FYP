"""
Utility functions for the intrusion detection project.
"""
import time
import functools
import logging

def timing_decorator(func):
    """
    Decorator to measure and log the execution time of functions.
    
    Args:
        func: The function to be decorated
        
    Returns:
        The decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Starting {func.__name__}...")
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logging.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logging.error(f"Error in {func.__name__} after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper 