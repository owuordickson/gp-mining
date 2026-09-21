# SPDX-License-Identifier: MIT
# See the LICENSE file at the root of this
# repository for complete details.

"""
@author: Dickson Owuor
@credits: Thomas Runkler, Edmond Menya, and Anne Laurent
@license: MIT
@email: owuordickson@gmail.com
@created: 21 July 2021

A collection of miscellaneous classes and methods.
"""

import multiprocessing as mp
import os


def get_num_cores() -> int:
    """
    Finds the count of CPU cores in a computer or a SLURM supercomputer.
    :return: Number of cpu cores (int)
    """
    num_cores = get_slurm_cores()
    if not num_cores:
        num_cores = mp.cpu_count()
    return num_cores


def get_slurm_cores() -> int | bool:
    """
    Test the computer to see if it is a SLURM environment, then gets the number of CPU cores.
    :return: Count of CPUs (int) or False
    """
    try:
        cores = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        return cores
    except ValueError:
        try:
            str_cores = str(os.environ["SLURM_JOB_CPUS_PER_NODE"])
            temp = str_cores.split("(", 1)
            cpus = int(temp[0])
            str_nodes = temp[1]
            temp = str_nodes.split("x", 1)
            str_temp = str(temp[1]).split(")", 1)
            nodes = int(str_temp[0])
            cores = cpus * nodes
            return cores
        except ValueError:
            return False
    except KeyError:
        return False


def write_file(data, path, wr=True) -> None:
    """
    Writes data into a file

    :param data: information to be written
    :param path: name of the file and storage path
    :param wr: writes data into the file if True

    :return:
    """
    if wr:
        with open(path, "w") as f:
            f.write(data)
            f.close()
    else:
        pass
