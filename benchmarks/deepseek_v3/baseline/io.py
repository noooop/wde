# ruff: noqa: F841, E402

import time

GB = 1 << 30


def test_kvikio(path):
    import cupy
    import kvikio

    a = cupy.random.rand(GB)

    start = time.perf_counter()
    f = kvikio.CuFile(path, "w")

    f.write(a)
    f.close()
    end = time.perf_counter()

    elapsed_time = end - start

    print(f"kvikio: ssd to gpu {8. / elapsed_time} GB/s")

    c = cupy.empty_like(a)
    start = time.perf_counter()
    with kvikio.CuFile(path, "r") as f:
        f.read(c)

    end = time.perf_counter()

    elapsed_time = end - start

    print(f"kvikio: gpu to ssd {8. / elapsed_time} GB/s")


def test_numpy(path):
    import numpy as np

    a = np.random.rand(GB)

    start = time.perf_counter()
    np.save(path, a)
    end = time.perf_counter()

    elapsed_time = end - start

    print(f"numpy: cpu to ssd {8. / elapsed_time} GB/s")

    start = time.perf_counter()
    b = np.load(path + ".npy")

    end = time.perf_counter()
    elapsed_time = end - start
    print(f"numpy: ssd to cpu {8. / elapsed_time} GB/s")

    start = time.perf_counter()
    b[:] = a[:]
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"numpy: cpu to cpu {8. / elapsed_time} GB/s")


def test_torch_cpu(path):
    import torch

    a = torch.rand(GB, dtype=torch.float64)

    start = time.perf_counter()
    torch.save(a, path + ".pt")
    end = time.perf_counter()

    elapsed_time = end - start

    print(f"torch: cpu to ssd {8. / elapsed_time} GB/s")

    start = time.perf_counter()
    b = torch.load(path + ".pt")

    end = time.perf_counter()
    elapsed_time = end - start
    print(f"torch: ssd to cpu {8. / elapsed_time} GB/s")

    start = time.perf_counter()
    b[:] = a[:]
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"torch: cpu to cpu {8. / elapsed_time} GB/s")


def test_torch_gpu(path):
    import torch

    a = torch.rand(GB, dtype=torch.float64, device="cuda")

    start = time.perf_counter()
    torch.save(a, path + ".pt")
    end = time.perf_counter()

    elapsed_time = end - start

    print(f"torch: gpu to ssd {8. / elapsed_time} GB/s")

    start = time.perf_counter()
    b = torch.load(path + ".pt", weights_only=False)

    end = time.perf_counter()
    elapsed_time = end - start
    print(b.device)
    print(f"torch: ssd to gpu {8. / elapsed_time} GB/s")

    start = time.perf_counter()
    b[:] = a[:]
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"torch: gpu to gpu {8. / elapsed_time} GB/s")

    c = torch.randn_like(a, pin_memory=True, device="cpu")

    start = time.perf_counter()
    c[:] = a
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"torch: gpu to cpu {8. / elapsed_time} GB/s")

    del a, b

    start = time.perf_counter()
    e = c.to("cuda")
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"torch: cpu to gpu {8. / elapsed_time} GB/s")


if __name__ == "__main__":
    test_kvikio(path="/share/test_kvikio")
    test_numpy(path="/share/test_numpy")
    test_torch_cpu(path="/share/test_torch_cpu")
    test_torch_gpu(path="/share/test_torch_gpu")
"""
kvikio: ssd to gpu 3.0453657519158623 GB/s
kvikio: gpu to ssd 5.641049281000345 GB/s

numpy: cpu to ssd 3.4756728820261786 GB/s
numpy: ssd to cpu 6.5520971491038305 GB/s
numpy: cpu to cpu 19.668772559522562 GB/s

torch: cpu to ssd 2.120535333725255 GB/s
torch: ssd to cpu 3.6678152118238767 GB/s
torch: cpu to cpu 18.31373723387411 GB/s

torch: gpu to ssd 1.3003456617848537 GB/s
torch: ssd to gpu 2.7394149293058145 GB/s
torch: gpu to gpu 13833.578593932974 GB/s
torch: gpu to cpu 19.467226540853233 GB/s
torch: cpu to gpu 19.653485611881948 GB/s
"""
