import math
import torch

def build_filter(pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

def get_dct_filter(tile_size_x):
        dct_filter = torch.zeros(tile_size_x, tile_size_x)

        for u_x in range(tile_size_x):
            for t_x in range(tile_size_x):
                dct_filter[u_x, t_x] = build_filter(t_x, u_x, tile_size_x)
                        
        return dct_filter

def get_idct_filter(tile_size_x):
        idct_filter = torch.zeros(tile_size_x, tile_size_x)

        for u_x in range(tile_size_x):
            for t_x in range(tile_size_x):
                idct_filter[t_x, u_x] = build_filter(t_x, u_x, tile_size_x)
                        
        return idct_filter


if __name__ == '__main__':
    arr = torch.randn(1, 10)
    dct_filter = get_dct_filter(10)
    idct_filter = get_idct_filter(10)
    print(arr)
    print((arr @ dct_filter) @ idct_filter)