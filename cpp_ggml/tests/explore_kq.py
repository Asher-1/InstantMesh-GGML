"""Empirically infer ggml_mul_mat semantics from dumped kq.
ggml kq ne=[197,197,12,1] (ne0 fastest). q,k,v dumped as [hd,seq,nh,B].
Find which numpy expression reproduces ggml kq exactly.
"""
import numpy as np

hd, nh, seq, B = 64, 12, 197, 1

def load(fn, shape):
    a = np.fromfile(fn, dtype=np.float32)
    assert a.size == np.prod(shape), f"{fn}: {a.size} vs {np.prod(shape)}"
    return a.reshape(shape)

q = load('/tmp/fa_q.bin', (hd, seq, nh, B))
k = load('/tmp/fa_k.bin', (hd, seq, nh, B))
kq_ggml = load('/tmp/fa_kq.bin', (seq, seq, nh, B))  # [q(n0), kv?(n1), h, B]

scale = 1.0 / np.sqrt(hd)

# k:[hd,seq,nh]=[a,b,c], q:[hd,seq,nh]=[a,d,c]
# H0: dst[n0=q,n1=kv,h] = sum_d k[d,kv,h]*q[d,q,h]  -> [q,kv,nh]  idces[d?]
H0 = np.einsum('abc,adc->dbc', k[:, :, :, 0], q[:, :, :, 0]) * scale  # [q,kv,h]
# H1: dst[n0=kv,n1=q,h] = sum_d k[d,kv,h]*q[d,q,h]  -> [kv,q,nh]
H1 = np.einsum('abc,adc->bdc', k[:, :, :, 0], q[:, :, :, 0]) * scale  # [kv,q,h]
# H2: reduction over seq (ne1): dst[n0,n1,h] = sum_s k[n0,s,h]*q[n1,s,h]
H2 = np.einsum('nsh,msh->nmh', k[:, :, :, 0], q[:, :, :, 0]) * scale  # [n0,n1,h]
# H3: same as H2 swapped
H3 = np.einsum('msh,nsh->nmh', k[:, :, :, 0], q[:, :, :, 0]) * scale  # [n0,n1,h]

cands = {
    'H0[q,kv]': H0, 'H0T[kv,q]': H0.transpose(1, 0, 2),
    'H1[kv,q]': H1, 'H1T[q,kv]': H1.transpose(1, 0, 2),
}
for name, ref in cands.items():
    e = np.abs(kq_ggml[:, :, :, 0] - ref)
    print(f"{name}: max_abs={e.max():.4e}")

print("ggml kq[0,0:4,0] =", kq_ggml[0, 0:4, 0])
print("H1[kv,q] [0,0:4] =", H1[0, 0:4, 0])
print("H1T[q,kv][0,0:4]= ", H1.transpose(1, 0, 2)[0, 0:4, 0])
print("q[0,0:4,0,0] =", q[0, 0:4, 0, 0])
print("k[0,0:4,0,0] =", k[0, 0:4, 0, 0])
print("q range", q.min(), q.max())
print("k range", k.min(), k.max())
# direct H1 element check: kq[0,0,0] vs sum_d k[d,0,0]*q[d,0,0]
print("sum_d k[d,0,0]*q[d,0,0] =", np.sum(k[:,0,0,0]*q[:,0,0,0]))
print("sum_d k[d,0,0]*q[d,0,0]*scale =", np.sum(k[:,0,0,0]*q[:,0,0,0])*scale)
kf = np.fromfile('/tmp/fa_k.bin', dtype=np.float32)
qf = np.fromfile('/tmp/fa_q.bin', dtype=np.float32)
print("flat[0:64] dot flat[0:64] =", np.sum(kf[0:64]*qf[0:64]))
print("kf[0:4] =", kf[0:4])
print("k[:,0,0,0][0:4] =", k[:,0,0,0][0:4])
print("array_equal k[:,0,0,0] vs kf[0:64] =", np.array_equal(k[:,0,0,0], kf[0:64]))
print("WHOLE k.ravel() == kf ?", np.array_equal(k.ravel(), kf))
print("kf.size, k.size =", kf.size, k.size)
# what layout makes k[:,0,0,0] stale? try ne0=seq
K2 = load('/tmp/fa_k.bin', (seq, hd, nh, B))  # [s, d, h, b]
print("K2[s0][0:4] =", K2[:, 0, 0, 0][0:4])
print("kf[0:64] == K2[:,0,0,0] ?", np.array_equal(kf[0:64], K2[:, 0, 0, 0]))