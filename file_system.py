from .utils import *
import dill as pickle
import gzip

def find_files(start_dir, file_extension, prefix=None, full_path=True):
    """
    Find files in a directory that start with a given prefix and have a given extension.
    The function traverses across all subdirectories of the start_dir. 

    Parameters
    ----------
    start_dir : str
        The directory in which to start the search.
    file_extension : str
        The extension that the files must have.
    prefix : str, None
        The prefix that the files must start with.
        If None, the files can start with any prefix.
    full_path : bool
        If True, the function returns the full path to the files.
        If False, the function returns only the filenames.

    Returns
    -------
    list
        A list of paths to the files that match the given criteria.
    """

    matching_files = []

    for dirpath, _, filenames in os.walk(start_dir):
        for filename in filenames:
            if(prefix):
                if filename.startswith(prefix) and fnmatch.fnmatch(filename, f"*.{file_extension}"):
                    if(full_path):
                        full_path = os.path.join(dirpath, filename)
                        matching_files.append(full_path)
                    else:
                        matching_files.append(filename)
            else:
                if fnmatch.fnmatch(filename, f"*.{file_extension}"):
                    if(full_path):
                        full_path = os.path.join(dirpath, filename)
                        matching_files.append(full_path)
                    else:
                        matching_files.append(filename)
    
    return matching_files

def save_pickle(data, file_name, compress=False):
    """
    Save data to a pickle file.

    Parameters
    ----------
    data
        The data to be saved.
    file_name : str
        The name of the file along with the path where the data will be saved.
    compress : bool, False
        If True, the file will be compressed using gzip.
        Filename should end in .gz if compress is True and .p otherwise.
        Compressed files take less space but require more time to read/write.
    """

    if compress:
        with gzip.open(file_name, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)

def load_pickle(file_name, compressed=False):
    """
    Load a pickle file.
    
    Parameters
    ----------
    file_name : str
        The name of the file along with the path from which the data will be loaded.
    compressed : bool, False
        If True, the file is expected to be compressed using gzip.
    
    Returns
    -------
    data : object
        The data loaded from the pickle file.
    """

    if compressed:
        with gzip.open(file_name, 'rb') as file:
            data = pickle.load(file)
    else:
        with open(file_name, 'rb') as file:
            data = pickle.load(file)

    return data