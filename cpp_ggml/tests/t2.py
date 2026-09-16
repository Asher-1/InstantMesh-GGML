import numpy as np
hd, seq, nh, B = 64, 197, 12, 1
kf = np.fromfile('/tmp/fa_k.bin', dtype=np.float32)
k = kf.reshape((hd, seq, nh, B))
print("kf.shape", kf.shape, "k.shape", k.shape)
print("k.strides", k.strides)
print("kf.strides", kf.strides)
print("shares_memory", np.shares_memory(k, kf))
print("k.base is kf ?", k.base is kf)
print("k.flags", k.flags)
print("kf[0:8] =", kf[0:8])
print("k[:,0,0,0][0:8] =", k[:,0,0,0][0:8])
print("k flat index of [1,0,0,0]:", 1 + 0*64 + 0*64*197 + 0*64*197*12)
print("k[1,0,0,0] =", k[1,0,0,0], " kf[1] =", kf[1])