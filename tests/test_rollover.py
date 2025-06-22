import datetime as dt
import datetime as dt
import torch

from ifera.data_processing import calculate_rollover
from ifera import ConfigManager


def _make_tensor(
    start_ord: int, volumes: list[int], prices: list[float], cut_off: int
) -> torch.Tensor:
    n = len(volumes)
    t = torch.zeros((n, 2, 9), dtype=torch.float32)
    for i in range(n):
        ord_day = start_ord + i
        t[i, :, 2] = ord_day
        t[i, 0, 3] = cut_off - 60
        t[i, 0, 8] = volumes[i]
        t[i, 1, 3] = cut_off
        t[i, 1, 4] = prices[i]
    return t


def test_calculate_rollover_traded_months_and_window(config_manager: ConfigManager):
    base = config_manager.get_base_instrument_config("CL", "30m")
    base.traded_months = "HM"
    base.rollover_vol_alpha = 1.0
    base.rollover_max_days = 1

    cut_off = base.rollover_offset
    start_date = dt.date(2020, 1, 1)
    start_ord = start_date.toordinal()

    j = config_manager.create_derived_base_config(base, contract_code="F20")
    h = config_manager.create_derived_base_config(base, contract_code="H20")
    m = config_manager.create_derived_base_config(base, contract_code="M20")

    h.expiration_date = start_date + dt.timedelta(days=4)
    m.expiration_date = start_date + dt.timedelta(days=35)

    tens_j = _make_tensor(start_ord, [300, 300, 300, 300], [10, 10, 10, 10], cut_off)
    tens_h = _make_tensor(start_ord, [100, 80, 60, 40], [10, 10, 10, 10], cut_off)
    tens_m = _make_tensor(start_ord, [10, 90, 100, 110], [10, 10, 10, 10], cut_off)

    ratios, idx, days = calculate_rollover([j, h, m], [tens_j, tens_h, tens_m])

    assert idx.tolist() == [0, 0, 1, 1]
    # Ratio should be 1 on the rollover day (day index 2)
    assert torch.isnan(ratios[0])
    assert torch.isnan(ratios[1])
    assert ratios[2] == 1
    assert torch.isnan(ratios[3])
