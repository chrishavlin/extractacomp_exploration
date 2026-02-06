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

def get_dims(filename_base: str):
    ds = load_edax_spd(filename_base)
    Nx, Ny, N_channels = ds['data'].shape
    del ds
    return int(Nx), int(Ny), int(N_channels)

def get_pixel_energy(filename_base: str, ix:int, iy:int):
    ds = load_edax_spd(filename_base)
    value = np.array(ds['data'][ix,iy,:]).copy()
    del ds
    return value

def run_pixel_itertest(filename_base: str, max_pixels: int = 50000):

    Nx, Ny, N_channels = get_dims(filename_base)

    channel_energy = np.zeros((Ny, N_channels))
    i_pixel= 0

    for ix in range(Nx):
        for iy in range(Ny):
            if (i_pixel % 10 == 0):
                print(f"{i_pixel} pixels processed, {Nx*Ny - i_pixel} pixels left of ({Nx,Ny,N_channels})...")
            value = get_pixel_energy(filename_base, ix, iy)
            channel_energy += value
            i_pixel += 1
            if i_pixel > max_pixels:
                print("hit max pixels, exiting")
                return

@dataclass
class cmdArgs:
    filename: str
    chunksize: int
    testcase: str


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='C-12')
    parser.add_argument('--chunksize', type=int, default=50)
    parser.add_argument('--testcase', type=str, default='spectrum')
    args = parser.parse_args()
    valid_args = cmdArgs(filename=args.filename, chunksize=args.chunksize, testcase=args.testcase)

    if valid_args.testcase == 'spectrum':
        run_test(valid_args.filename, valid_args.chunksize)
    elif valid_args.testcase == 'load':
        ds = load_edax_spd(valid_args.filename)
        assert len(ds['data'].shape) == 3
        print("loaded file, closing now")
    elif valid_args.testcase == 'singleaccess':
        ds = load_edax_spd(valid_args.filename)
        value = ds['data'][100,100,100]
        print(f"single value access at (100,100,100) = {value}")
    elif valid_args.testcase == 'singlemap':
        ds = load_edax_spd(valid_args.filename)
        value = np.array(ds['data'][:,:,100])
        print(f"single map access at (:,:,100) of size {value.shape}")
    elif valid_args.testcase == 'singlepixel':
        ds = load_edax_spd(valid_args.filename)
        value = np.array(ds['data'][500,500,:])
        sumval = np.sum(value)
        print(f"single map access at (500,500,:) of size {value.shape}")
    elif valid_args.testcase == 'pixeliter':
        run_pixel_itertest(valid_args.filename)




