from ashare_quant_factory.utils.validation import is_valid_baostock_code, normalize_baostock_code


def test_baostock_code_validation():
    assert is_valid_baostock_code("sh.600519")
    assert is_valid_baostock_code("sz.000001")
    assert not is_valid_baostock_code("600519")
    assert normalize_baostock_code("SH.600519") == "sh.600519"
