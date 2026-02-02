from rsciio import edax
import os
import numpy as np 
from typing import Any
import numpy.typing as npt
import argparse
from dataclasses import dataclass

datadir = os.environ.get("EXTRACTACOMPDATADIR")


def load_edax_spd(basename: str) -> dict[str, Any]:
    subdir = os.path.join(datadir, basename, "Proprietary EDAX Files")    
    fispd = os.path.join(subdir, f"{basename}.spd")
    fiipr = os.path.join(subdir, f"{basename}.ipr")
    
    ds = edax.file_reader(fispd, ipr_fname=fiipr)[0]
    return ds  



def sum_over_energy_index_range(filename_base: str, 
                                index_start: int, 
                                index_end: int,):
    
    # do NOT pass the memmap handle in. data still 
    # accumulates! 
    # https://stackoverflow.com/a/61472122/9357244
    ds = load_edax_spd(filename_base)
    subsample = np.array(ds['data'][:,:,index_start: index_end])
    return subsample.sum(axis=0).sum(axis=0)
    

def get_n_chunks(filename_base: str, chunksize: int):
    ds = load_edax_spd(filename_base)  
    n_chunks = int(np.ceil(ds['axes'][2]['size'] / chunksize))    
    maxindex = ds['axes'][2]['size']
    return maxindex, n_chunks

def integrate_spatial_dims(filename_base: str, chunksize: int) -> npt.NDArray[np.floating]:
    
    maxindex, n_chunks = get_n_chunks(filename_base, chunksize)    
    channel_sums = np.zeros((maxindex,))

    print(f"integrating {n_chunks} chunks")

    for ichunk in range(n_chunks):
        channel_start = ichunk * chunksize
        channel_end = channel_start + chunksize 
        if channel_end > maxindex:
            channel_end = maxindex 
        sumvalues = sum_over_energy_index_range(filename_base, channel_start, channel_end)
        channel_sums[channel_start:channel_end] = sumvalues

    return channel_sums 


def run_test(filename_base: str, chunksize: int):        
    channel_sums = integrate_spatial_dims(filename_base, chunksize)
    assert np.all(np.isreal(channel_sums))
    


@dataclass
class cmdArgs: 
    filename: str 
    chunksize: int 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='C-12')
    parser.add_argument('--chunksize', type=int, default=50)
    args = parser.parse_args()
    valid_args = cmdArgs(args.filename, args.chunksize)
    run_test(valid_args.filename, valid_args.chunksize)

    

