# LSIR
This repo is for source code of DASFAA 2024 paper "Learning Social Graph for Inactive User Recommendation". 

# Environment Settings
```

```
GPU: Tesla-V100, Memory 32G \
CPU: Intel(R) Xeon(R) Platinum 8163 CPU @2.50GHz

# Usage
First, download datasets from <https://drive.google.com/drive/folders/12L7Je7p_xQdoPkVxoxqc3mVob24qGrgw?usp=drive_link>. Then, unzip the datasets into the folder /dataset.

Running the following command:
```
CUDA_DEVICES_VISIBLES=0 python main.py yelp

CUDA_DEVICES_VISIBLES=0 python main.py flickr
```

