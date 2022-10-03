import random
from pip._vendor.colorama import Fore, init # colorama always in pip._vendor.colorama
init(autoreset=True) # to avoid reseting color everytime
import time
import logging
import sys
import json
import pprint
import numpy as np
from datetime import datetime
import multiprocessing
from codetiming import Timer as TimerPlus # Does what did with Timer but more sophisticated

################
# Benchmarking #
################

# Note that codetiming Timer more powerful than both those benchmarking decorator and context manager below.

# Code taken from [1] https://stackoverflow.com/questions/1593019/is-there-any-simple-way-to-benchmark-python-script
# more explanation on how to use [2] https://www.blog.pythonlibrary.org/2016/05/24/python-101-an-intro-to-benchmarking-your-code/
def time_func(func):
    """
    st decorator to calculate the total time of a func
    """
    def st_func(*args, **kwargs):
        # t1 = time.time()
        t1 = time.perf_counter()
        r = func(*args, **kwargs)
        t2 = time.perf_counter()
        # t2 = time.time()
        # print("Function=%s, Time=%s" % (func.__name__, t2 - t1))
        # print(f"{Fore.LIGHTMAGENTA_EX}Function={func.__name__}, Time={t2 - t1}s") # file=sys.stdout
        logging.d5bg(f"{Fore.LIGHTMAGENTA_EX}Function={func.__name__}, Time={t2 - t1}s") # logging config same as in other file ?
        return r

    return st_func
# then need to annotate with @time_func the def of function want to time

# taken from [2] https://www.blog.pythonlibrary.org/2016/05/24/python-101-an-intro-to-benchmarking-your-code/
# similar code more polished [3] https://realpython.com/python-timer/
# They made codetiming which could be imported for convenience [4] https://pypi.org/project/codetiming/
# Context Manager
class OldTimer:
    def __init__(self, name):
        self.name = name
        # self.start = time.time() # could use time.perf_counter() ?
        self._start_time = None

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise Exception(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise Exception(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        return elapsed_time

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # end = time.time()
        # runtime = end - self.start
        runtime = self.stop()
        # msg = 'The function took {time} seconds to complete'
        # print(msg.format(time=runtime))
        # print(f'{Fore.LIGHTMAGENTA_EX}Timer {self.name} took {runtime}s to complete')
        logging.d5bg(f'{Fore.LIGHTMAGENTA_EX}Timer {self.name} took {runtime}s to complete')


##########
## SEED ##
##########

# Context manager for seed update that need to revert after
class SeedUpdater:
    def __init__(self, name, param_dict):
        self.name = name
        self.default_seed = param_dict['SEED']
        if self.default_seed is not None:
            self.seed_inc = param_dict['seed_inc']+1 # Note: could overflow if do +1 to many time so could take modulo
            param_dict['seed_inc'] += 1 # use param dict for side effects (modify value inside dict)
        self.check_np_state = False
        if self.check_np_state:
            self.np_state = np.random.get_state()

    def __enter__(self):
        """Start a new timer as a context manager"""
        if self.default_seed is not None:
            logging.debug(f'{Fore.RED}Seed updater {self.name} update seed to {self.default_seed + self.seed_inc}')
            random.seed(self.default_seed + self.seed_inc)
            np.random.seed(self.default_seed + self.seed_inc)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If seed is None need to update it after ? not impactful if does it ?
        np.random.seed(self.default_seed)
        # np.random.set_state(self.np_state) # another way to restore previous state ?
        random.seed(self.default_seed)
        logging.debug(f'{Fore.RED}Seed updater {self.name} reset seed to {self.default_seed}')

        # Note this could crash if state format change
        if self.check_np_state:
            # The state were not equal according to this condition
            cs = np.random.get_state()
            ps = self.np_state

            if not (np.array_equal(ps[1], cs[1]) and ps[0] == cs[0] and ps[2] == cs[2] and ps[3] == cs[3] and ps[4] == cs[4]):
                 logging.info(f'{Fore.RED}Seed updater {self.name} reverted seed to {np.random.get_state()} instead of {self.np_state}')


###################
# Multiprocessing #
###################

# Noted that if there is code outside main function the multiprocessing of preimage attack crashed due to concurrent process creation
# inspiration [1] https://gist.github.com/MInner/9716950ac85b49821b56298117756451
# linked from [2] https://stackoverflow.com/questions/1380860/add-variables-to-tuple
# to solve cannot pickle [3] https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
# Decorator wrapper Note: [does not work as a decorator]
def multiprocess_run_release_gpu(func):

    # problem cannot pickle this function when do multiprocessing as it is a local function
    def worker(output_dict, *args, **kwargs):
        return_value = func(*args, **kwargs)
        if return_value is not None:
            output_dict['return_value'] = return_value

    def outer_wrapper(*args, **kwargs):
        with multiprocessing.Manager() as manager:
            output_dict = manager.dict()
            worker_args = (func, output_dict, ) + args
            process = multiprocessing.Process(target=worker, args=worker_args, kwargs=kwargs)
            process.start()
            process.join() # don't need timeout
            return_value = output_dict.get('return_value', None)

        return return_value
    return outer_wrapper

# def multiprocess_run_release_gpu(func, *args, **kwargs):
#     with multiprocessing.Manager() as manager:
#         output_dict = manager.dict()
#         worker_args = (func, output_dict, ) + args
#         process = multiprocessing.Process(target=worker, args=worker_args, kwargs=kwargs)
#         process.start()
#         process.join() # don't need timeout
#         return_value = output_dict.get('return_value', None)
#
#     return return_value
#
# # In top level of module function can be pickled
# # Need to set FUNC_TO_CALL beforehand
# def worker(func, output_dict, *args, **kwargs):
#         # return_value = func(*args, **kwargs)
#         # return_value = FUNC_TO_CALL(*args, **kwargs)
#         return_value = func(*args, **kwargs)
#         if return_value is not None:
#             output_dict['return_value'] = return_value

FUNC_TO_CALL = None



###########
# Logging #
###########

## New logging levels
# https://docs.python.org/3/library/logging.html#logging-levels
# Default logging level NOTSET 0 < DEBUG 10 < INFO 20 < WARNING 30 < ERROR 40 < CRITICAL 50

# from [2] https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility
# Check answer from `pfa` and `mad physicist`
def create_new_logging_level(level_name: str, LEVEL_VALUE: int, method_name=None):
    """
       Comprehensively adds a new logging level to the `logging` module and the
       currently configured logging class.

       `level_name` becomes an attribute of the `logging` module with the value
       `LEVEL_VALUE`. `method_name` becomes a convenience method for both `logging`
       itself and the class returned by `logging.getLoggerClass()` (usually just
       `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
       used.

       To avoid accidental clobberings of existing attributes, this method will
       raise an `AttributeError` if the level name is already an attribute of the
       `logging` module or if the method name is already present

       Example
       -------
       >> addLoggingLevel('TRACE', logging.DEBUG - 5)
       >> logging.getLogger(__name__).setLevel("TRACE")
       >> logging.getLogger(__name__).trace('that worked')
       >> logging.trace('so did this')
       >> logging.TRACE
       5

       """

    if not method_name:
        method_name = level_name.lower()

    if hasattr(logging, level_name):
        raise AttributeError(f'{level_name} already defined in logging module')
    if hasattr(logging, method_name):
        raise AttributeError(f'{method_name} already defined in logging module')
    if hasattr(logging.getLoggerClass(), method_name):
        raise AttributeError(f'{method_name} already defined in logger class')

    def log_for_level(self, message, *args, **kwargs):
        if self.isEnabledFor(LEVEL_VALUE):
            # here logger takes its '*args' as 'args'.
            self._log(LEVEL_VALUE, message, args, **kwargs)

    def log_to_root(message, *args, **kwargs):
        logging.log(LEVEL_VALUE, message, *args, **kwargs)

    logging.addLevelName(LEVEL_VALUE, level_name.upper())

    # Using the value of a string as a variable name (globals(), locals(), vars() etc):
    # [1] https://docs.python.org/3/library/logging.html#logging-levels
    setattr(logging, level_name.upper(), LEVEL_VALUE) # # logging.LEVEL_NAME = level_value
    setattr(logging.getLoggerClass(), method_name, log_for_level)
    setattr(logging, method_name, log_to_root)

def init_loggers(log_to_stdout=False, log_to_file=True, log_folder=None, filename='empty', fh_lvl=logging.INFO, sh_lvl=logging.INFO):

    # Simple check to see if one level was created as create the same (and other method throws error if attribute already exists)
    if not hasattr(logging, 'D1BG'):
        create_logging_levels()

    f_handler, s_handler = None, None # otherwise could have variable reference before assignment

    # if want encoding prior to python 3.9 need logging.FileHandler and addHandler ?
    root_logger = logging.getLogger()
    # The level set in the logger determines which severity of messages it will pass to its handlers.
    # need this level lower because handlers will filter only what they can see from this prefiltering
    root_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(f'[%(asctime)s:%(name)s:%(levelname)s]: %(message)s')
    # [1] https://docs.python.org/3/howto/logging.html#handler-basic
    ## File handler
    if log_to_file:
        timestamp = datetime.now().strftime('%d-%m-%Y_%Hh%Mm%Ss')  # timestamp for log file name
        # Note: udpate file name to make a custom depending on what variable change
        # default filemode is 'a' for append
        f_handler = logging.FileHandler(f'{"./logs" if log_folder is None else log_folder}/{filename}_[{timestamp}].log', 'w', 'utf-8')
        f_handler.setFormatter(formatter)
        f_handler.setLevel(fh_lvl)  # only set logging for that handler (eg DEBUG, D5BG, logging.INFO)
        root_logger.addHandler(f_handler)
        # logging.basicConfig(filename='int_prog.log', filemode='w', level=logging.INFO) # INFO encoding='utf-8' not in py3.6 ?
        # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) # to have logging to file and console ?

    ## Console logger
    if log_to_stdout:
        s_handler = logging.StreamHandler(sys.stdout)
        s_handler.setLevel(sh_lvl)
        s_handler.setFormatter(formatter)
        # s_handler.addFilter(starts_with_Maximize_filter)
        root_logger.addHandler(s_handler)  # to have logging to file and console ?

    return f_handler, s_handler # root_logger can always be obtained with logging.getLogger()

def create_logging_levels():
    # DEBUG value is 10 and INFO 20 can use value in between for debug lvl of verbose
    create_new_logging_level('D1BG', 11)
    create_new_logging_level('D2BG', 12)
    create_new_logging_level('D3BG', 13)
    create_new_logging_level('D4BG', 14)
    create_new_logging_level('D5BG', 15)


def remove_handlers(handler_list):
    # By default only have root_logger otherwise need to specify logger or how to retrieve it
    root_logger = logging.getLogger()

    for handler in handler_list:
        if handler is not None: # can be None due to init_loggers
            root_logger.removeHandler(handler)
            handler.close()


## Filters
# https://stackoverflow.com/questions/879732/logging-with-filters
class NoDeprecatedFilter(logging.Filter): # From python 3.2+ doc can use a func as a filter
    def filter(self, record) -> bool:
        return 'deprecated' not in record.getMessage().lower() # record does not contain deprecated
        # return not record.getMessage().startswith('parsing') # starts with parsing

def starts_with_Maximize_filter(record) -> bool:
    return not record.getMessage().startswith('Maximize')


## Pretty printer

def pretty_format(obj, ppindent=4, ppwidth=80, ppcompact=False, use_json=False):

    if use_json:
        try:
            # default=str make it use str repr. when cannot serialize e.g. for set etc.
            formatted_obj = json.dumps(obj, sort_keys=True, indent=4, default=str) # cannot serialize set etc
        except TypeError:
            print(f'TypeError occured obj might not be JSON serializable')
            raise
    else:
        # Parameter sort_dicts and underscore_numbers added in 3.8 and 3.10 so omitted
        pp = pprint.PrettyPrinter(indent=ppindent, depth=None, width=ppwidth, compact=ppcompact, stream=sys.stdout)
        formatted_obj = pp.pformat(obj) # pprint print but pformat returns the string

    return formatted_obj

###################
## Miscellaneous ##
###################

# haming distance
# inspired: https://www.tutorialspoint.com/hamming-distance-in-python
def binary_hamming_distance(x, y, prefix_bitlength=64):
    if not 0 <= prefix_bitlength <= 64:
        raise ValueError('Prefix length should be in [0,64]')
    output = 0
    # for i in range(prefix_bitlength-1, -1, -1): # goes from prefix_length-1 to 0 (LSB)
    included_upperbound = 63 # zero-based indexing
    exluded_lowerbound = included_upperbound - prefix_bitlength # if prefix=64 goes to -1 if prefix 0 no computations
    for i in range(included_upperbound, exluded_lowerbound, -1): # goes from 64 (63 0-based index) to 64-prefix_length (LSB)
        bx_i = x >> i & 1
        by_i = y >> i & 1
        output += not(bx_i == by_i) # boolean true is 1 false is 0
        # print(f'{bx_i, by_i}')
    return output


def reverse_target_simhash_bits(target_simhash, hash_bitlength):
    target_bits = bin(target_simhash)[2:2 + hash_bitlength]
    # Reverse target simhash for integer programming constraint order
    r_target_bits = target_bits[::-1]
    # If the bit_length of target simhash is not equal to hash_bitlength
    # need to manually add leading 0 (trailing 0 on the reverse) ?
    if len(r_target_bits) < hash_bitlength:
        logging.debug('target simhash bit length smaller than output hash bitcount so add leading zeroes')
        while len(r_target_bits) < hash_bitlength:
            r_target_bits += '0'
        logging.debug(f'extended len: {len(r_target_bits)} matches output hash len: {hash_bitlength}')
    return r_target_bits


if __name__ == '__main__':
    # Test new logger level
    test_logger_new_lvl = False
    if test_logger_new_lvl:
        dbg_nlvl = create_new_logging_level('D1BG', 11)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        test_logger = logging.getLogger(__name__)
        test_logger.setLevel(logging.D1BG)
        s_handler = logging.StreamHandler(sys.stdout)
        s_handler.setFormatter(logging.Formatter(f'[%(asctime)s:%(name)s:%(levelname)s]: %(message)s'))
        s_handler.setLevel(logging.D1BG)
        root_logger.addHandler(s_handler)

        logging.getLogger(__name__).d1bg('that worked')
        logging.d1bg(f'so did this')
        print(logging.D1BG)
        logging.info(f'Info message')
        logging.debug(f'debug message')

    test_pretty_fmt=False
    if test_pretty_fmt:
        # This dict cannot be created
        target_dict = {
            #'1': {5, {'a', 'b'}, 65, [1,2,3], {'ta':3, 'agasg':True}}, # if contains set typeError unhashable
            '2': {6, {'a':1, 'b':2}, 65, [1, 2, 3], {'ta': 3, 'agasg': True}},
            '3': {6, {'a':1, 'b':2}, 66, [1, 2, 3], {'ta': 3, 'agasg': True}}
        }
        print(pretty_format(target_dict, use_json=True))

    regex_test = False
    if regex_test:
        import re
        expr = f'15666 in common (GAN-target)'
        number = int(re.search(r"\d+", expr).group(0))
        print(number)
        extract_starting_number = re.compile(r'\d+')
        number = int(extract_starting_number.search(expr).group(0))
        print(number)

    test_hamming = True
    if test_hamming:
        x = 0b1111111100000000111111110000000011111111000000001111111100000000
        y = 0b1111111000000000111111110000000011111111000000001111111100000001
        out = binary_hamming_distance(x, y, prefix_bitlength=8)
        # print(f'{out=}') # equality sign from python 3.8
        print(f'hamming distance: {out}')
