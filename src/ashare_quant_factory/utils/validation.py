from __future__ import annotations

import re

_BS_CODE = re.compile(r"^(sh|sz)\.[0-9]{6}$", re.IGNORECASE)


def is_valid_baostock_code(code: str) -> bool:
    return bool(_BS_CODE.match(code.strip()))


def normalize_baostock_code(code: str) -> str:
    code = code.strip().lower()
    if not is_valid_baostock_code(code):
        raise ValueError(f"Invalid Baostock code: {code} (expected like sh.600519 / sz.000001)")
    return code
