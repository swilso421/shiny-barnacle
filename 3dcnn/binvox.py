import numpy as np

#=====
#
# binvox IO
#
#=====

class _Voxels(object):

    def __init__(self, data, dims, translate, scale, axis_order):

        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        assert(axis_order in ('xzy', 'xyz'))
        self.axis_order = axis_order

    def clone(self):
        return _Voxels(self.data.copy(), self.dims[:], self.translate[:], self.scale, self.axis_order[:])

    def write(self, fp):
        binvox_write(self, fp)

def _read_binvox_header(flo):
    '''Internal function for parsing header of binvox files'''
    line = flo.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims, translate, scale

def binvox_read(filepath, fix_coords = True):

    with open(filepath, 'rb') as f:

        dims, translate, scale = _read_binvox_header(f)
        raw = np.frombuffer(fp.read(), dtype=np.uint8)

        values, counts = raw[::2], raw[1::2]

        data = np.repeat(values, counts).astype(np.bool)
        data = data.reshape(dims)

        if fix_coords:
            data = np.transpose(data, (0, 2, 1))
            axis_order = 'xyz'
        else:
            axis_order = 'xzy'

        return _Voxels(data, dims, translate, scale, axis_order)

def binvox_write():
    print('NYI')

def loadBinvoxAsNumPy(filepath):
    return binvox_read(filepath).data
