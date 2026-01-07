import pandas as pd
from datasets import load_dataset

def read_parquet():
    path = "/wl_intelligent/shenzhiwei/model_data/nvidia_distributed/Nemotron-Pretraining-SFT-v1/Nemotron-SFT-General/part_000063.parquet"

    df = pd.read_parquet(path)
    print(len(df))

def read_hf():
    path="/wl_intelligent/shenzhiwei/model_data/nvidia_distributed/Nemotron-Pretraining-SFT-v1"
    # ds=load_dataset(path)
    ds = load_dataset(
        path,
        split="train[:1]"   # 只加载第一条
    )
    print(len(ds))
    print(ds)


if __name__ ==  "__main__":
    read_parquet()