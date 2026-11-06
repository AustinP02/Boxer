"""Read the ASCII/binary PCD formats emitted by Viam. PCD XYZ is in meters."""

import io

import numpy as np


def decode_pcd(blob: bytes):
    stream = io.BytesIO(blob)
    header = {}
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("Truncated PCD header")
        text = line.decode("ascii").strip()
        if not text or text.startswith("#"):
            continue
        key, *values = text.split()
        header[key.upper()] = values
        if key.upper() == "DATA":
            break
    fields = header["FIELDS"]
    sizes = list(map(int, header["SIZE"]))
    types = header["TYPE"]
    counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
    if not len(fields) == len(sizes) == len(types) == len(counts) or any(c != 1 for c in counts):
        raise ValueError("Unsupported PCD field layout")
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError("PCD lacks XYZ")
    if not ({"rgb", "rgba"} & set(fields)) and not {"r", "g", "b"}.issubset(fields):
        raise ValueError("Colored point cloud required; configure both color and depth on cam")
    dtype = np.dtype(
        [(n, f"<{ {'F': 'f', 'U': 'u', 'I': 'i'}[t] }{s}") for n, t, s in zip(fields, types, sizes)]
    )
    n = int(header.get("POINTS", [str(int(header["WIDTH"][0]) * int(header["HEIGHT"][0]))])[0])
    if n < 0 or n > 5_000_000:
        raise ValueError("Invalid PCD point count")
    payload = stream.read()
    if header["DATA"] == ["binary"]:
        if len(payload) < n * dtype.itemsize:
            raise ValueError("Truncated PCD data")
        data = np.frombuffer(payload, dtype=dtype, count=n)
    elif header["DATA"] == ["ascii"]:
        if n == 0:
            data = np.empty(0, dtype=dtype)
        else:
            data = np.atleast_1d(np.loadtxt(io.BytesIO(payload), dtype=dtype))
            if len(data) != n:
                raise ValueError("PCD point count does not match header")
    else:
        raise ValueError("Unsupported PCD compression; request pointcloud/pcd")
    xyz = np.column_stack([data[k] for k in ("x", "y", "z")]).astype(np.float64) * 1000
    packed_name = "rgb" if "rgb" in fields else "rgba"
    if packed_name in fields:
        raw = data[packed_name]
        packed = raw.astype("<f4").view("<u4") if raw.dtype.kind == "f" else raw.astype("<u4")
        rgb = np.column_stack([(packed >> shift) & 255 for shift in (16, 8, 0)]).astype(np.uint8)
    else:
        rgb = np.column_stack([data[k] for k in ("r", "g", "b")]).astype(np.uint8)
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
    return xyz[valid], rgb[valid]
