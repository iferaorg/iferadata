import torch
import ifera
import datetime as dt
from tqdm import tqdm

cm = ifera.ConfigManager()
iconfig = cm.get_config("IBKR", "CL:1m")


# @torch.compile()
def run_test(iconfig):
    dm = ifera.DataManager()
    idata = dm.get_instrument_data(iconfig, device=torch.device("cpu"))

    openPolicy = ifera.AlwaysOpenPolicy(direction=1)
    initStopPolicy = ifera.InitialArtrStopLossPolicy(
        instrument_data=idata, atr_multiple=3.0
    )
    maintenancePolicy = ifera.ScaledArtrMaintenancePolicy(
        instrument_data=idata,
        stages=["1m", "5m", "15m", "1h", "4h", "1d"],
        atr_multiple=3.0,
        wait_for_breakeven=False,
        minimum_improvement=0.0,
    )

    tradingPolicy = ifera.TradingPolicy(
        instrument_data=idata,
        open_position_policy=openPolicy,
        initial_stop_loss_policy=initStopPolicy,
        position_maintenance_policy=maintenancePolicy,
    )

    # tradingPolicy = torch.compile(tradingPolicy)

    ms = ifera.MarketSimulatorIntraday(instrument_data=idata)

    t = (
        iconfig.liquid_start - iconfig.trading_start
    ).total_seconds() // iconfig.time_step.total_seconds() - 1

    max_date_idx = idata.data.shape[0] - 1
    max_time_idx = idata.data.shape[1] - 1
    # 3166, 99
    date_idx = torch.arange(0, 100, dtype=torch.int32, device=idata.device)
    time_idx = torch.full_like(date_idx, t, dtype=torch.int32, device=idata.device)
    position = torch.zeros_like(date_idx, dtype=torch.int32, device=idata.device)
    prev_stop_loss = torch.full_like(
        date_idx, torch.nan, dtype=torch.float32, device=idata.device
    )
    execution_price = torch.full_like(
        date_idx, torch.nan, dtype=torch.float32, device=idata.device
    )

    total_profit = torch.zeros_like(date_idx, dtype=torch.float32, device=idata.device)
    open_price = idata.data[date_idx, time_idx, 0]

    # Open position
    action, stop_loss = tradingPolicy(
        date_idx, time_idx, position, prev_stop_loss, execution_price
    )
    time_idx += 1
    profit, position, execution_price, _ = ms.calculate_step(
        date_idx, time_idx, position, action, stop_loss
    )
    execution_price = execution_price.clone()
    position = position.clone()
    total_profit += profit
    done = torch.zeros_like(date_idx, dtype=torch.bool, device=idata.device)
    action = torch.zeros_like(date_idx, dtype=torch.int32, device=idata.device)

    timer = dt.datetime.now()
    steps = idata.data.shape[0] * idata.data.shape[1]

    for _ in tqdm(range(steps)):
        if not position.any():
            break
        prev_stop_loss = stop_loss
        _, stop_loss = tradingPolicy(
            date_idx, time_idx, position, prev_stop_loss, execution_price
        )
        done = done | ((date_idx == max_date_idx) & (time_idx >= max_time_idx))
        position = torch.where(done, 0, position)
        time_idx = torch.where(done, max_time_idx - 1, time_idx)
        date_idx, time_idx = idata.get_next_indices(date_idx, time_idx)
        profit, position, _, _ = ms.calculate_step(
            date_idx, time_idx, position, action, stop_loss
        )
        total_profit += profit

    price_ratio = total_profit / (open_price * iconfig.contract_multiplier)
    min_done_date_idx = (
        torch.cat(
            (
                date_idx[done],
                torch.tensor(
                    [idata.data.shape[0]], dtype=torch.int32, device=idata.device
                ),
            ),
            dim=0,
        )
        .min()
        .item()
    )
    max_idx = price_ratio[date_idx < min_done_date_idx].argmax()
    print(f"Min done date index: {min_done_date_idx}")
    print(
        f"Max index: {max_idx}, Total profit: {total_profit[date_idx < min_done_date_idx][max_idx].item():.4f}, Price ratio: {price_ratio[date_idx < min_done_date_idx][max_idx].item():.6f}"
    )
    print(f"Time elapsed: {dt.datetime.now() - timer}")


torch.set_float32_matmul_precision("high")
run_test(iconfig)
