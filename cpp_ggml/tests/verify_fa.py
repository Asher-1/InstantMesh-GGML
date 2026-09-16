"""Verify test_flashattn manual-softmax attention against a numpy reference.
Reads /tmp/fa_{q,k,v,kq,att,out}.bin dumped by ./build/test_flashattn.

ggml dump layout: flat memory is ne0-fastest. Tensors are [hd, seq, nh, B]
(ne0=hd), so the flat file is [B, nh, seq, hd] with hd innermost. We therefore
load as shape (nh, seq, hd) and index k[h, s, d].
"""
import numpy as np

hd, nh, seq, B = 64, 12, 197, 1
scale = 1.0 / np.sqrt(hd)

def load4(fn):
    a = np.fromfile(fn, dtype=np.float32)
    assert a.size == hd*seq*nh*B, f"{fn}: {a.size}"
    return a.reshape((B, nh, seq, hd))[0]  # [h, s, d]

q = load4('/tmp/fa_q.bin')
k = load4('/tmp/fa_k.bin')
v = load4('/tmp/fa_v.bin')
# kq: tensor [kv, q, nh, B] (ne0=kv). flat = [B, nh, q, kv] -> load [nh, q, kv]
kq_f = np.fromfile('/tmp/fa_kq.bin', dtype=np.float32).reshape((B, nh, seq, seq))[0]  # [h,q,kv]
att_f = np.fromfile('/tmp/fa_att.bin', dtype=np.float32).reshape((B, nh, seq, seq))[0]  # [h,q,kv]
# out: tensor after permute(o,2,0,1,3) = [nh, q, hd, B] (ne0=nh). flat=[B,hd,q,nh]->load [nh,q,hd]
out_f = np.fromfile('/tmp/fa_out.bin', dtype=np.float32).reshape((B, nh, seq, hd))[0]  # [nh,q,hd]
# o (mul_mat result): tensor [q, hd, nh, B] (ne0=q). flat = [B, nh, hd, q] -> load [nh, hd, q]
o_f = np.fromfile('/tmp/fa_o.bin', dtype=np.float32).reshape((B, nh, hd, seq))[0]  # [h, hd, q]
# vp: tensor [kv, hd, nh, B] (ne0=kv). flat = [B, nh, hd, kv] -> load [nh, hd, kv]
vp_f = np.fromfile('/tmp/fa_vp.bin', dtype=np.float32).reshape((B, nh, hd, seq))[0]  # [h, hd, kv]

# numpy reference: att[q, kv, h], out[q, hd, h]
att_ref = np.zeros((seq, seq, nh), np.float32)
out_ref = np.zeros((seq, hd, nh), np.float32)
kq_ref = np.zeros((seq, seq, nh), np.float32)
for h in range(nh):
    Q = q[h]        # [seq, hd]
    K = k[h]        # [seq, hd]
    V = v[h]        # [seq, hd]
    kq = (Q @ K.T) * scale   # [q, kv]
    a = np.exp(kq - kq.max(axis=1, keepdims=True))
    a = a / a.sum(axis=1, keepdims=True)
    kq_ref[:, :, h] = kq
    att_ref[:, :, h] = a
    out_ref[:, :, h] = a @ V   # [q, hd]

# compare (transpose numpy refs to ggml flat layout)
e_kq = np.abs(kq_f - kq_ref.transpose(2, 0, 1))      # [h, q, kv] vs ggml[h,q,kv]
e_att = np.abs(att_f - att_ref.transpose(2, 0, 1))
e_out = np.abs(out_f - out_ref.transpose(2, 0, 1))   # ref[q,hd,nh]->[nh,q,hd]
# verify o (mul_mat(att,vp)) directly: o[q, hd, h] = sum_kv att[q,kv,h]*vp[h,kv,hd]
o_ref = np.einsum('qkh,hkd->qdh', att_ref, vp_f.transpose(0, 2, 1))   # [q,hd,h]
e_o = np.abs(o_f - o_ref.transpose(2, 1, 0))  # ggml o_f[h,hd,q] vs ref[q,hd,h]->[h,hd,q]
print(f"[o  ] max_abs={e_o.max():.3e} mean={e_o.mean():.3e}")
print(f"[kq ] max_abs={e_kq.max():.3e} mean={e_kq.mean():.3e}")
print(f"[att] max_abs={e_att.max():.3e} mean={e_att.mean():.3e}")
print(f"[out] max_abs={e_out.max():.3e} mean={e_out.mean():.3e}")

# softmax-over-q hypothesis (normalize over q instead of kv)
att_q = np.zeros((seq, seq, nh), np.float32)
for h in range(nh):
    kq = kq_ref[:, :, h]  # [q, kv]
    a = np.exp(kq - kq.max(axis=0, keepdims=True))
    a = a / a.sum(axis=0, keepdims=True)
    att_q[:, :, h] = a
e_att_q = np.abs(att_f - att_q.transpose(2, 0, 1))
print(f"[att softmax-over-q] max_abs={e_att_q.max():.3e}")

# check row sums of att_f (should be 1 if normalized over kv)
print("att_f rowsum[0:3] (over kv) =", att_f[0, 0, :].sum(), att_f[0, 1, :].sum(), att_f[0, 2, :].sum())
print("att_f colsum[0:3] (over q) =", att_f[0, :, 0].sum(), att_f[0, :, 1].sum(), att_f[0, :, 2].sum())
print("PASS" if e_out.max() < 1e-4 else "FAIL")