import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU

from related.triplet.difftriplet import test_latency as DiffTripletTestLatency
from related.triplet.mt4mtl import test_latency as MT4MTLTestLatency
from related.triplet.rendezvous import test_latency as RendezvousTestLatency
from related.triplet.rit import test_latency as RITTestLatency
from related.triplet.tdn import test_latency as TDNTestLatency
from related.triplet.tripnet import test_latency as TripNetTestLatency


if __name__ == "__main__":
    results = []  # list of {"mean_ms": None, "std_ms": None, "error": str(e), "device": str(device), "T": T}
    for test_func in [
        DiffTripletTestLatency,
        MT4MTLTestLatency,
        RendezvousTestLatency,
        RITTestLatency,
        TDNTestLatency,
        TripNetTestLatency,
    ]:
        try:
            res = test_func(warmup=5, runs=20, T=16)
        except Exception as e:
            # If the individual test helper raises, capture error info
            res = {"mean_ms": None, "std_ms": None, "error": str(e), "device": None, "T": 16}
        res["model"] = test_func.__module__.split(".")[-1]
        results.append(res)

    # create final table
    try:
        import pandas as pd

        df = pd.DataFrame(results)
        # save CSV next to this script for easy access
        out_csv = os.path.join(os.path.dirname(__file__), "inference_latency_table.csv")
        df.to_csv(out_csv, index=False)
        # print in psql format if available
        try:
            print(df.to_markdown(tablefmt="psql"))
        except Exception:
            print(df)
        print(f"Wrote CSV to: {out_csv}")
    except Exception:
        # pandas not available; just pretty-print the results
        import json

        print(json.dumps(results, indent=2))
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU

from related.triplet.mt4mtl import test_latency as MT4MTLTestLatency
from related.triplet.rendezvous import test_latency as RendezvousTestLatency
from related.triplet.rit import test_latency as RITTestLatency
from related.triplet.tdn import test_latency as TDNTestLatency
from related.triplet.tripnet import test_latency as TripNetTestLatency
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU

from related.triplet.difftriplet import test_latency as DiffTripletTestLatency
from related.triplet.mt4mtl import test_latency as MT4MTLTestLatency
from related.triplet.rendezvous import test_latency as RendezvousTestLatency
from related.triplet.rit import test_latency as RITTestLatency
from related.triplet.tdn import test_latency as TDNTestLatency
from related.triplet.tripnet import test_latency as TripNetTestLatency


if __name__ == "__main__":
    results = []  # list of {"mean_ms": None, "std_ms": None, "error": str(e), "device": str(device), "T": T}
    for test_func in [
        DiffTripletTestLatency,
        MT4MTLTestLatency,
        RendezvousTestLatency,
        RITTestLatency,
        TDNTestLatency,
        TripNetTestLatency,
    ]:
        try:
            res = test_func(warmup=5, runs=20, T=16)
        except Exception as e:
            # If the individual test helper raises, capture error info
            res = {"mean_ms": None, "std_ms": None, "error": str(e), "device": None, "T": 16}
        res["model"] = test_func.__module__.split(".")[-1]
        results.append(res)

    # create final table
    try:
        import pandas as pd

        df = pd.DataFrame(results)
        # save CSV next to this script for easy access
        out_csv = os.path.join(os.path.dirname(__file__), "inference_latency_table.csv")
        df.to_csv(out_csv, index=False)
        # print in psql format if available
        try:
            print(df.to_markdown(tablefmt="psql"))
        except Exception:
            print(df)
        print(f"Wrote CSV to: {out_csv}")
    except Exception:
        # pandas not available; just pretty-print the results
        import json

        print(json.dumps(results, indent=2))

