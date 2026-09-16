import numpy as np
hd, seq, nh, B = 64, 197, 12, 1
kf = np.fromfile('/tmp/fa_k.bin', dtype=np.float32)
k = kf.reshape((hd, seq, nh, B))
print("k[0,0,0,0] =", k[0,0,0,0], " kf[0] =", kf[0])
print("k[1,0,0,0] =", k[1,0,0,0], " kf[1] =", kf[1])
print("k[:,0,0,0][0:4] =", k[:,0,0,0][0:4])
print("kf[0:4] =", kf[0:4])
print("equal =", np.array_equal(k.ravel(), kf))