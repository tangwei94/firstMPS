import numpy as np 
import h5py 

def savearr(hf, datasetname, arr):
    hf[datasetname] = arr.flatten()
    hf[datasetname].attrs['shape'] = arr.shape

def readarr(hf, datasetname):
    arr_dat = hf[datasetname].value
    arr_shape = hf[datasetname].attrs['shape']
    arr = np.reshape(arr_dat, arr_shape)
    return arr

if __name__ == '__main__':
    arr = np.reshape(np.arange(20), (2,5,2))
    hf = h5py.File('hallo.h5', 'w')
    savearr(hf, 'data/data', arr)
    hf.close() 

    hf = h5py.File('hallo.h5', 'r')
    print(readarr(hf, 'data/data'))
    hf.close()
    