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

    ds = edax.file_reader(fispd, ipr_fname=fiipr, mode='r')[0]
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


def get_ds_info(filename_base: str, 
                axis: int, 
                chunksize: int, 
                return_ds: bool = False):
    ds = load_edax_spd(filename_base)
    n_chunks = int(np.ceil(ds['axes'][axis]['size'] / chunksize))
    maxindex = ds['axes'][axis]['size']
    if return_ds: 
        ds_out = ds 
    else:
        ds_out = None 
    return n_chunks, maxindex, ds_out


def process_chunk(filename_base: str, 
                  axis: int, 
                  channel_start: int, 
                  channel_end: int,
                  ds = None):
    
    if ds is None:
        ds = load_edax_spd(filename_base)

    match axis:
        case 0:
            subsample = np.array(ds['data'][channel_start:channel_end,:,:])
        case 1:
            subsample = np.array(ds['data'][:,channel_start:channel_end,:])
        case 2:
            # worst case!
            subsample = np.array(ds['data'][:,:,channel_start:channel_end])

    reduced_value = np.sum(subsample)
    return reduced_value, subsample.shape


def run_test(filename_base: str, 
             axis: int, 
             reload_memmap: bool = False,
             chunksize: int = 32):


    if reload_memmap is False: 
        return_ds = True
    else:
        return_ds = False 
    
    n_chunks, maxindex, ds = get_ds_info(filename_base, axis, chunksize, return_ds=return_ds)
        
    for ichunk in range(n_chunks):

        channel_start = ichunk * chunksize
        channel_end = channel_start + chunksize
        if channel_end > maxindex:
            channel_end = maxindex

        _, subshape = process_chunk(filename_base, 
                                    axis, 
                                    channel_start, 
                                    channel_end, 
                                    ds=ds)        
        assert len(subshape) == 3
        




@dataclass
class cmdArgs:
    filename: str
    axis: int
    reload_memmap: bool
    chunksize: int


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='C-12')
    parser.add_argument('--axis', type=int, default=0)
    parser.add_argument('--reload', type=int, default=0)
    parser.add_argument('--chunksize', type=int, default=32)
    args = parser.parse_args()
    valid_args = cmdArgs(args.filename, args.axis, bool(args.reload), args.chunksize)
    run_test(valid_args.filename, 
             valid_args.axis, 
             reload_memmap=valid_args.reload_memmap,
             chunksize=valid_args.chunksize)
    



