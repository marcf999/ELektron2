# HotAisle Fresh Install Guide

## 1. SSH Key Setup for GitHub

```bash
ssh-keygen -t ed25519 -C "hotaisle"
# Enter for defaults, no passphrase if preferred

cat ~/.ssh/id_ed25519.pub
# Copy output → GitHub Settings → SSH and GPG keys → New SSH key

ssh -T git@github.com  # verify
```

## 2. Clone the Repo

```bash
cd ~
git clone git@github.com:marcf/ELektron2.git
cd ELektron2
```

## 3. Install Dependencies

```bash
sudo apt install libboost-all-dev
```

## 4. Build (ROCm / MI300X)

CMake must be configured before building — `cmake --build` alone won't work on a fresh clone.

```bash
cd cpp
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build . --target elektron2_rocm_fp64 -j$(nproc)
```

## 5. Run Spin -Y (4-GPU, 2M electrons)

```bash
cd ~/ELektron2
screen -S spiny
chmod +x cpp/scripts/run_spin_minus_y_4gpu.sh
./cpp/scripts/run_spin_minus_y_4gpu.sh
```

Detach from screen: `Ctrl+A, D`. Reattach: `screen -r spiny`.

Output goes to `results/`.
