import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU

from related.phase.baesyan import test_latency as BaesyanTestLatency
from related.phase.swag import test_latency as SWAGTestLatency
from related.phase.skit import test_latency as SKITTestLatency
from related.phase.IIA_net import test_latency as IIA_netTestLatency
from related.phase.supr_gan import test_latency as SuprGANTestLatency
from related.phase.must import test_latency as MUSTTestLatency


if __name__ == "__main__":
    results = []  # list of {"mean_ms": None, "std_ms": None, "error": str(e), "device": str(device), "T": T}
    for test_func in [
        BaesyanTestLatency,
        SWAGTestLatency,
        SKITTestLatency,
        IIA_netTestLatency,
        SuprGANTestLatency,
        MUSTTestLatency,
    ]:
        res = test_func(
            warmup=5, runs=20, T=16
        )
        res["model"] = test_func.__module__.split(".")[-1]
        results.append(res)

    # create final table
    import pandas as pd
    df = pd.DataFrame(results)
    # print in psql format
    print(df.to_markdown(tablefmt="psql"))
    
