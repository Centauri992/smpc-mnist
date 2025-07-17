"""Demo BNN MNIST classifier, vectorized, adapted for PyTorch/NumPy weights & biases."""

import os
import argparse
import numpy as np
import time
from mpyc.runtime import mpc
import mpyc.gmpy as gmpy2
from torchvision import datasets, transforms

secint = None

def load_custom_W_b(export_dir, layer_idx):
    W = np.load(os.path.join(export_dir, f'layer{layer_idx}_weight.npy'))
    b = np.load(os.path.join(export_dir, f'layer{layer_idx}_bias.npy'))
    W = W.astype(np.int64).T
    b = b.astype(np.int64)
    return secint.array(W), secint.array(b)

def load_pytorch_mnist(batch_size=1, offset=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])
    mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    images = []
    labels = []
    for idx in range(offset, offset+batch_size):
        img, label = mnist[idx]
        images.append(img.numpy().reshape(-1))
        labels.append(label)
    arr = np.stack(images)
    return arr, labels

# --- Protocols (unchanged, from np_bnnmnist.py) ---

@mpc.coroutine
async def bsgn_0(a):
    stype = type(a)
    await mpc.returnType((stype, a.shape))
    Zp = stype.sectype.field
    legendre = gmpy2.legendre
    p = gmpy2.mpz(Zp.modulus)
    legendre_p = np.vectorize(lambda a: legendre(a, p), otypes='O')
    n = a.size
    s = mpc.np_random_bits(Zp, n, signed=True)
    r2 = mpc._np_randoms(Zp, n)
    if mpc.options.no_prss:
        r2 = await r2
    r2 **= 2
    r2 = mpc._reshare(r2)
    s, r2 = await mpc.gather(s, r2)
    y = s * r2
    y = await mpc._reshare(y)
    a = await mpc.gather(a)
    y = y * (2*a + 1).reshape(n)
    y = await mpc.output(y, threshold=2*mpc.threshold)
    return (s * legendre_p(y.value)).reshape(a.shape)

@mpc.coroutine
async def bsgn_1(a):
    stype = type(a)
    await mpc.returnType((stype, a.shape))
    Zp = stype.sectype.field
    legendre = gmpy2.legendre
    p = gmpy2.mpz(Zp.modulus)
    legendre_p = np.vectorize(lambda a: legendre(a, p), otypes='O')
    n = a.size
    s = mpc.np_random_bits(Zp, 3*n, signed=True)
    r2 = mpc._np_randoms(Zp, 3*n)
    if mpc.options.no_prss:
        r2 = await r2
    r2 **= 2
    r2 = mpc._reshare(r2)
    s, r2 = await mpc.gather(s, r2)
    s = s.reshape(3, n)
    r2 = r2.reshape(3, n)
    s = np.append(s, [s[0]], axis=0)
    r2 = np.append(r2, [s[1]], axis=0)
    z = s * r2
    z = await mpc._reshare(z)
    a = await mpc.gather(a)
    y = 2*a + 1
    y = y.reshape(1, n)
    y = y + np.array([[-2], [0], [2]])
    y = np.append(y, [s[2]], axis=0)
    y = z * y
    y = await mpc.output(y, threshold=2*mpc.threshold)
    y = y.value
    h = legendre_p(y[:3])
    t = (s[:3] * h).value
    z = h[0] * h[1] * h[2] * y[3]
    q = int((p+1) >> 1)
    return Zp.array((t[0] + t[1] + t[2] - z) * q).reshape(a.shape)

@mpc.coroutine
async def bsgn_2(a):
    stype = type(a)
    await mpc.returnType((stype, a.shape))
    Zp = stype.sectype.field
    legendre = gmpy2.legendre
    p = gmpy2.mpz(Zp.modulus)
    legendre_p = np.vectorize(lambda a: legendre(a, p), otypes='O')
    n = a.size
    s = mpc.np_random_bits(Zp, 6*n, signed=True)
    r2 = mpc._np_randoms(Zp, 6*n)
    if mpc.options.no_prss:
        r2 = await r2
    r2 **= 2
    r2 = mpc._reshare(r2)
    s, r2 = await mpc.gather(s, r2)
    s = s.reshape(6, n)
    r2 = r2.reshape(6, n)
    z = s * r2
    z = await mpc._reshare(z)
    a = await mpc.gather(a)
    y = 2*a + 1
    y = y.reshape(1, n)
    y = y + np.array([[-4], [-2], [0], [2], [4]])
    y = y * z[:5]
    y = await mpc._reshare(y)
    y = np.append(y, [z[5]], axis=0)
    y = await mpc.output(y, threshold=2*mpc.threshold)
    t = np.sum(s[:5] * legendre_p(y[:5].value), axis=0)
    t = await mpc.output(t * y[5])
    return (s[5] * legendre_p(t.value)).reshape(a.shape)

async def main():
    global secint

    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=str, default='binarized_params')
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('-o', '--offset', type=int, default=0)
    parser.add_argument('-d', '--d-k-star', type=int, default=1)
    parser.add_argument('--no-legendre', action='store_true')
    args = parser.parse_args()

    batch_size = args.batch_size
    offset = args.offset
    export_dir = args.export_dir

    if args.no_legendre:
        secint = mpc.SecInt(14)
        bsgn = lambda L: (L >= 0)*2 - 1
    else:
        if args.d_k_star == 0:
            secint = mpc.SecInt(14, p=3546374752298322551)
            bsgn = bsgn_0
        elif args.d_k_star == 1:
            secint = mpc.SecInt(14, p=9409569905028393239)
            bsgn = bsgn_1
        else:
            secint = mpc.SecInt(14, p=15569949805843283171)
            bsgn = bsgn_2

    await mpc.start()

    # --- LOAD DATA ---
    images, labels = load_pytorch_mnist(batch_size, offset)

    # --- Secret-share image ---
    L = secint.array(images.astype(np.int64))
    total_start = time.time()
    
    # --- LAYER 1 ---
    W, b = load_custom_W_b(export_dir, 0)
    L = L @ W + b
    L = (L >= 0)*2 - 1
    await mpc.barrier('after-layer-1')

    # --- LAYER 2 ---
    t2_start = time.time()
    W, b = load_custom_W_b(export_dir, 1)
    L = L @ W + b
    if args.no_legendre:
        secint.bit_length = 10
        L = (L >= 0)*2 - 1
        secint.bit_length = 14
    else:
        L = bsgn(L)
    await mpc.barrier('after-layer-2')
    t2_end = time.time()

    # --- LAYER 3 ---
    t3_start = time.time()
    W, b = load_custom_W_b(export_dir, 2)
    L = L @ W + b
    if args.no_legendre:
        secint.bit_length = 10
        L = (L >= 0)*2 - 1
        secint.bit_length = 14
    else:
        L = bsgn(L)
    await mpc.barrier('after-layer-3')
    t3_end = time.time()

    # --- LAYER 4 ---
    W, b = load_custom_W_b(export_dir, 3)
    L = L @ W + b

    # --- OUTPUT ---
    out_start = time.time()
    n_errors = 0
    for i in range(batch_size):
        logits = await mpc.output(L[i])
        pred = int(np.argmax(logits))
        true_label = labels[i]
        if pred != true_label:
            n_errors += 1
    out_end = time.time()

    total_end = time.time()
    print(f"\nTotal images: {batch_size}")
    print(f"Misclassifications: {n_errors}")
    print(f"Error rate: {n_errors / batch_size:.4f}")
    print(f"Layer 2 time: {t2_end - t2_start:.2f} seconds")
    print(f"Layer 3 time: {t3_end - t3_start:.2f} seconds")
    print(f"Output processing time: {out_end - out_start:.2f} seconds")
    print(f"Total secure inference time: {total_end - total_start:.2f} seconds")

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())