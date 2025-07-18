import os
import logging
import random
import argparse
import numpy as np
import time
from mpyc.runtime import mpc
import mpyc.gmpy as gmpy2

# ---- Torchvision MNIST loader ----
from torchvision import datasets, transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def load_pytorch_mnist(batch_size=1, offset=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).view(-1)),  # Explicitly get [0,255]
    ])
    mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    images, labels = [], []
    for idx in range(offset, offset + batch_size):
        img, label = mnist[idx]
        images.append(img.numpy())
        labels.append(int(label))
    arr = np.stack(images)
    return arr, labels

secint = None

def load_W_b(name):
    """Load signed binary weights W and signed integer bias values b for fully connected layer 'name' from binarized_params directory."""
    param_dir = 'binarized_params'
    W = np.load(os.path.join(param_dir, f'W_{name}.npy'))
    W = np.unpackbits(W, axis=0).astype(np.int8)
    W = W*2 - 1  # map 0->-1, 1->1
    b = np.load(os.path.join(param_dir, f'b_{name}.npy')).astype(object)
    return secint.array(W), secint.array(b)

# ---- (Legendre-based sign functions unchanged) ----

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

# ---- Main logic ----

async def main():
    global secint

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, metavar='B',
                        help='number of images to classify')
    parser.add_argument('-o', '--offset', type=int, metavar='O',
                        help='offset for batch (otherwise random in [0,10000-B])')
    parser.add_argument('-d', '--d-k-star', type=int, metavar='D',
                        help='k=D=0,1,2 for Legendre-based comparison using d_k^*')
    parser.add_argument('--no-legendre', action='store_true',
                        help='disable Legendre-based comparison')
    parser.set_defaults(batch_size=1, offset=-1, d_k_star=1)
    args = parser.parse_args()

    batch_size = args.batch_size
    offset = args.offset
    if args.no_legendre:
        secint = mpc.SecInt(14)  # using vectorized MPyC integer comparison
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

    if offset < 0:
        offset = random.randrange(10000 - batch_size + 1) if mpc.pid == 0 else None
        offset = await mpc.transfer(offset, senders=0)

    logging.info('--------------- INPUT   -------------')
    print(f'Type = {secint.__name__}, range = ({offset}, {offset + batch_size})')

    # ---- Load images/labels from torchvision MNIST ----
    L, labels = load_pytorch_mnist(batch_size, offset)
    print('Labels:', labels)

    if batch_size == 1:
        print("MPyC INPUT min/max:", L[0].min(), L[0].max())
        print("MPyC INPUT first 10:", L[0][:10])

    L = np.round(L).astype(np.uint8)
    if batch_size == 1:
        print("MPyC INPUT (rounded, uint8) first 10:", L[0][:10])

    L = secint.array(L)

    # --- Inference timing starts here ---
    total_start = time.time()

    logging.info('--------------- LAYER 1 -------------')
    W, b = load_W_b('fc1')
    logging.info('- - - - - - - - fc  784 x 4096  - - -')
    L = L @ W + b
    if batch_size == 1:
        fc1_bn1_out = await mpc.output(L[0][:10])
        print("MPyC after FC1+BN1 (first 10):", fc1_bn1_out)
    logging.info('- - - - - - - - bsgn    - - - - - - -')
    L = (L >= 0)*2 - 1
    if batch_size == 1:
        fc1_bin_out = await mpc.output(L[0][:10])
        print("MPyC after FC1+BN1+binarize (first 10):", fc1_bin_out)
    await mpc.barrier('after-layer-1')

    logging.info('--------------- LAYER 2 -------------')
    W, b = load_W_b('fc2')
    logging.info('- - - - - - - - fc 4096 x 4096  - - -')

    layer2_start = time.time()
    L = L @ W + b
    if batch_size == 1:
        fc2_bn2_out = await mpc.output(L[0][:10])
        print("MPyC after FC2+BN2 (first 10):", fc2_bn2_out)
    logging.info('- - - - - - - - bsgn    - - - - - - -')
    if args.no_legendre:
        secint.bit_length = 10
        L = (L >= 0)*2 - 1
        if batch_size == 1:
            fc2_bin_out = await mpc.output(L[0][:10])
            print("MPyC after FC2+BN2+binarize (first 10):", fc2_bin_out)
    else:
        L = bsgn(L)
    await mpc.barrier('after-layer-2')
    layer2_end = time.time()

    logging.info('--------------- LAYER 3 -------------')
    W, b = load_W_b('fc3')
    logging.info('- - - - - - - - fc 4096 x 4096  - - -')

    layer3_start = time.time()
    L = L @ W + b
    if batch_size == 1:
        fc3_bn3_out = await mpc.output(L[0][:10])
        print("MPyC after FC3+BN3 (first 10):", fc3_bn3_out)
    logging.info('- - - - - - - - bsgn    - - - - - - -')
    if args.no_legendre:
        secint.bit_length = 10
        L = (L >= 0)*2 - 1
        if batch_size == 1:
            fc3_bin_out = await mpc.output(L[0][:10])
            print("MPyC after FC3+BN3+binarize (first 10):", fc3_bin_out)
    else:
        L = bsgn(L)
    await mpc.barrier('after-layer-3')
    layer3_end = time.time()

    logging.info('--------------- LAYER 4 -------------')
    W, b = load_W_b('fc4')
    logging.info('- - - - - - - - fc 4096 x 10  - - - -')
    L = L @ W + b

    logging.info('--------------- OUTPUT  -------------')
    if args.no_legendre:
        secint.bit_length = 14

    n_errors = 0
    out_start = time.time()
    for i in range(batch_size):
        logits = await mpc.output(L[i])
        prediction = np.argmax(logits)
        error = '******* ERROR *******' if prediction != labels[i] else ''
        print(f'Image #{offset+i} with label {labels[i]}: {prediction} predicted. {error}')
        print("MPyC output logits (first 10):", logits[:10])
        print("MPyC predicted class:", prediction)
        if prediction != labels[i]:
            n_errors += 1
    out_end = time.time()

    total_end = time.time()
    print(f"\nTotal images: {batch_size}")
    print(f"Misclassifications: {n_errors}")
    print(f"Misclassification rate: {100.0 * n_errors / batch_size:.2f}%")
    print(f"Layer 2 time: {layer2_end - layer2_start:.2f} seconds")
    print(f"Layer 3 time: {layer3_end - layer3_start:.2f} seconds")
    print(f"Output processing time: {out_end - out_start:.2f} seconds")
    print(f"Total secure inference time: {total_end - total_start:.2f} seconds")

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())