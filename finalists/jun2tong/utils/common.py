
"""
General useful functions for machine learning prototyping based on numpy.
"""
import os
import psutil
import shutil


def check_ext_mem(ext_mem_dir):
    """
    Compute recursively the memory occupation on disk of ::ext_mem_dir::
    directory.

        Args:
            ext_mem_dir (str): path to the directory.
        Returns:
            ext_mem (float): Occupation size in Megabytes
    """

    ext_mem = sum(os.path.getsize(os.path.join(dirpath, filename))
                  for dirpath, dirnames, filenames in os.walk(ext_mem_dir) for filename in filenames) / (1024 * 1024)

    return ext_mem


def check_ram_usage():
    """
    Compute the RAM usage of the current process.

        Returns:
            mem (float): Memory occupation in Megabytes
    """

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)

    return mem


def create_code_snapshot(code_dir, dst_dir):
    """
    Copy the code that generated the exps as a backup.

        Args:
            code_dir (str): root dir of the project
            dst_dir (str): where to put the code files
    """

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for dirpath, dirnames, filenames in os.walk(code_dir):
        for filename in filenames:
            if ".py" in filename and ".pyc" not in filename:
                try:
                    shutil.copy(os.path.join(dirpath, filename), dst_dir)
                except shutil.SameFileError:
                    pass


