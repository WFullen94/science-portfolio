"""NetFlow-v2 schema constants + a schema-accurate synthetic generator.

The NetFlow-v2 standardized feature set (43 numeric features + 2 label columns)
is shared across NF-UNSW-NB15-v2 / NF-ToN-IoT-v2 / NF-CSE-CIC-IDS2018. Defining
it once here lets the pipeline validate real downloads and, when no public
mirror is reachable, fall back to a realistic synthetic sample so every stage
downstream stays runnable and demoable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Identifier columns (strings) — present in the raw data, dropped before modeling.
ID_COLUMNS = ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT"]

# The 39 NetFlow-v2 numeric flow features, in canonical file order.
NUMERIC_FEATURES = [
    "PROTOCOL", "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
]

LABEL_BINARY = "Label"        # 0 = benign, 1 = attack
LABEL_MULTICLASS = "Attack"   # subclass name; 'Benign' for normal traffic

# NetFlow-v2 (ToN-IoT) attack subclasses present in the mirror we download.
ATTACK_CLASSES = [
    "scanning", "xss", "ddos", "password", "dos",
    "injection", "backdoor", "mitm", "ransomware",
]

ALL_COLUMNS = ID_COLUMNS + NUMERIC_FEATURES + [LABEL_BINARY, LABEL_MULTICLASS]


def _rand_ip(rng: np.random.Generator, n: int) -> np.ndarray:
    octets = rng.integers(1, 255, size=(n, 4))
    return np.array([".".join(map(str, row)) for row in octets])


def make_synthetic(n_rows: int = 60_000, attack_frac: float = 0.22,
                   seed: int = 42) -> pd.DataFrame:
    """Generate a schema-accurate synthetic NetFlow-v2 sample.

    Attack flows carry a learnable but noisy signal (heavier packet counts,
    more TCP flags set, shorter durations, higher throughput) so a real model
    trains to a meaningful — not perfect — ROC-AUC, and class imbalance mirrors
    operational NIDS data.
    """
    rng = np.random.default_rng(seed)
    n_atk = int(n_rows * attack_frac)
    n_ben = n_rows - n_atk
    labels = np.concatenate([np.zeros(n_ben, int), np.ones(n_atk, int)])
    rng.shuffle(labels)
    is_atk = labels == 1

    df = pd.DataFrame(index=range(n_rows))
    df[LABEL_BINARY] = labels

    # Base (benign) distributions; attacks shift a subset of features.
    df["PROTOCOL"] = rng.choice([1, 6, 17], size=n_rows, p=[0.05, 0.7, 0.25])
    df["L7_PROTO"] = rng.choice([0, 7, 11, 53, 80, 443], size=n_rows)
    df["IN_BYTES"] = rng.lognormal(6, 1.5, n_rows).astype(int)
    df["OUT_BYTES"] = rng.lognormal(6, 1.5, n_rows).astype(int)
    df["IN_PKTS"] = rng.poisson(8, n_rows) + 1
    df["OUT_PKTS"] = rng.poisson(8, n_rows) + 1
    df["TCP_FLAGS"] = rng.integers(0, 32, n_rows)
    df["CLIENT_TCP_FLAGS"] = rng.integers(0, 32, n_rows)
    df["SERVER_TCP_FLAGS"] = rng.integers(0, 32, n_rows)
    df["FLOW_DURATION_MILLISECONDS"] = rng.lognormal(7, 1.2, n_rows).astype(int)
    df["DURATION_IN"] = (df["FLOW_DURATION_MILLISECONDS"] * rng.uniform(0.3, 0.7, n_rows)).astype(int)
    df["DURATION_OUT"] = df["FLOW_DURATION_MILLISECONDS"] - df["DURATION_IN"]
    df["MIN_TTL"] = rng.integers(30, 64, n_rows)
    df["MAX_TTL"] = rng.integers(64, 128, n_rows)
    df["LONGEST_FLOW_PKT"] = rng.integers(40, 1514, n_rows)
    df["SHORTEST_FLOW_PKT"] = rng.integers(40, 200, n_rows)
    df["MIN_IP_PKT_LEN"] = df["SHORTEST_FLOW_PKT"]
    df["MAX_IP_PKT_LEN"] = df["LONGEST_FLOW_PKT"]
    df["SRC_TO_DST_SECOND_BYTES"] = rng.lognormal(5, 1.5, n_rows)
    df["DST_TO_SRC_SECOND_BYTES"] = rng.lognormal(5, 1.5, n_rows)
    for c in ["RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
              "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS"]:
        df[c] = rng.poisson(0.5, n_rows)
    df["SRC_TO_DST_AVG_THROUGHPUT"] = rng.lognormal(9, 2, n_rows).astype(int)
    df["DST_TO_SRC_AVG_THROUGHPUT"] = rng.lognormal(9, 2, n_rows).astype(int)
    for c in ["NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
              "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
              "NUM_PKTS_1024_TO_1514_BYTES"]:
        df[c] = rng.poisson(3, n_rows)
    df["TCP_WIN_MAX_IN"] = rng.integers(0, 65535, n_rows)
    df["TCP_WIN_MAX_OUT"] = rng.integers(0, 65535, n_rows)
    df["ICMP_TYPE"] = rng.choice([0, 8, 11], size=n_rows)
    df["ICMP_IPV4_TYPE"] = df["ICMP_TYPE"]
    df["DNS_QUERY_ID"] = rng.integers(0, 65535, n_rows)
    df["DNS_QUERY_TYPE"] = rng.choice([0, 1, 28], size=n_rows)
    df["DNS_TTL_ANSWER"] = rng.integers(0, 3600, n_rows)
    df["FTP_COMMAND_RET_CODE"] = rng.choice([0, 200, 331, 530], size=n_rows)

    # Inject the attack signal (noisy, partially overlapping with benign).
    m = is_atk.sum()
    df.loc[is_atk, "IN_PKTS"] += rng.poisson(40, m)
    df.loc[is_atk, "OUT_PKTS"] += rng.poisson(5, m)
    df.loc[is_atk, "TCP_FLAGS"] = rng.integers(16, 64, m)
    df.loc[is_atk, "FLOW_DURATION_MILLISECONDS"] = rng.lognormal(4, 1.5, m).astype(int)
    df.loc[is_atk, "SRC_TO_DST_AVG_THROUGHPUT"] = rng.lognormal(12, 2, m).astype(int)
    df.loc[is_atk, "RETRANSMITTED_IN_PKTS"] += rng.poisson(4, m)
    df.loc[is_atk, "MIN_TTL"] = rng.integers(1, 32, m)

    # Identifier columns.
    df["IPV4_SRC_ADDR"] = _rand_ip(rng, n_rows)
    df["IPV4_DST_ADDR"] = _rand_ip(rng, n_rows)
    df["L4_SRC_PORT"] = rng.integers(1024, 65535, n_rows)
    df["L4_DST_PORT"] = rng.choice([22, 53, 80, 443, 445, 3389, 8080], size=n_rows)

    # Multiclass label: 'Benign' for normal, a sampled subclass for attacks.
    attack_names = np.array(["Benign"] * n_rows, dtype=object)
    attack_names[is_atk] = rng.choice(ATTACK_CLASSES, size=m)
    df[LABEL_MULTICLASS] = attack_names

    return df[ALL_COLUMNS]
