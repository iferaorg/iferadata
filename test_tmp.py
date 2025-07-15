import pandas as pd
import ifera
import torch
import numpy as np
from torch._higher_order_ops.foreach_map import foreach_map
import yaml
import os
import datetime as dt
import time
from tqdm import tqdm
from rich.live import Live
from rich.table import Table


# dm = ifera.DataManager()
# cm = ifera.ConfigManager()
# instrument = cm.get_base_instrument_config("TN", "5m")
# device=torch.device("cuda:0")

# idata = dm.get_instrument_data(instrument_config=instrument, dtype=torch.float32, device=device, backadjust=True)
# idata = dm.get_instrument_data(instrument_config=instrument, dtype=torch.float32, device=device, backadjust=True)

# idata.data.shape


# cf = ifera.ConfigManager()
# instrument = cf.get_base_instrument_config("ES", "30m")
# contract_instrument = cf.create_derived_base_config(instrument, contract_code="M21")

# ifera.calculate_expiration(contract_instrument.contract_code, ifera.ExpirationRule(contract_instrument.last_trading_day_rule))

# -----------------------------------------------------------
# fm = ifera.FileManager()
# cm = ifera.ConfigManager()

# broker = cm.get_broker_config("IBKR")
# symbols = broker.instruments.keys()


# for symbol in symbols:
#     iconfig = cm.get_base_instrument_config(symbol, "30m")

#     if iconfig.type != "futures":
#         continue

#     print(f"Refreshing backadjusted tensors for {symbol}...")
#     fm.refresh_file(f"s3:tensor_backadjusted/futures/1m/{symbol}.pt.gz")
# ifera.delete_s3_file(f"meta/futures/rollover/{symbol}.yml")
# fm.refresh_file(f"file:meta/futures/rollover/{symbol}.yml")

# ifera.delete_s3_file(f"meta/futures/rollover/PL.yml")
# fm.refresh_file(f"s3:tensor_backadjusted/futures/1m/PL.pt.gz")


# -----------------------------------------------------------

# import ifera

# for s3_key in ifera.list_s3_objects(""):
#     if s3_key.endswith(".pt"):
#         print(f"Deleting S3 file: {s3_key}")
#         ifera.delete_s3_file(s3_key)

# -----------------------------------------------------------

# import ifera

# s3_prefix = "raw/futures_individual/1m/"

# for s3_key in ifera.list_s3_objects(s3_prefix):
#     if s3_key[-8] == "-":
#         # Skip files that already have a dash before the last 3 characters
#         continue
#     # Add a dash to the filename before the last 3 characters (not including the extension)
#     new_s3_key = s3_key[:-7] + "-" + s3_key[-7:]    # Last 3 characters + 4 for the extension
#     print(f"Renaming {s3_key} to {new_s3_key}")
#     ifera.rename_s3_file(old_key=s3_key,new_key=new_s3_key)

# -----------------------------------------------------------

# mismatches = {}
# last_data_date = dt.date(2025, 4, 16)

# for symbol in symbols:
#     ibkr_instrument = cm.get_config(broker_name="IBKR", symbol=symbol, interval="30m")
#     if ibkr_instrument.type != "futures":
#         continue

#     mismatches[symbol] = {}
#     instrument = cm.get_config(broker_name="barchart", symbol=symbol, interval="30m")

#     if instrument.first_notice_day_rule is None:
#         continue

#     if instrument.last_trading_day_rule is None:
#         raise ValueError(
#             f"Instrument {symbol} does not have a last trading day rule defined."
#         )

#     print(f"{symbol}: {instrument.broker_symbol}")
#     url_raw = f"file:data/meta/futures/dates_raw/{symbol}.yml"
#     url = f"file:data/meta/futures/dates/{symbol}.yml"
#     # fm.refresh_file(f"file:raw/futures/30m/{symbol}.zip")
#     fm.refresh_file(url_raw)
#     fm.refresh_file(url)
#     with open(f"data/meta/futures/dates_raw/{symbol}.yml", "r") as f:
#         dates_raw = yaml.safe_load(f)

#     # fm.refresh_file(f"file:data/tensor/futures_rollover/30m/{symbol}.pt")
#     # fm.refresh_file(f"file:data/meta/futures/dates/{symbol}.yml")
#     # fm.refresh_file(f"file:data/meta/futures/dates/{symbol}.yml")
#     # fm.refresh_file(f"file:data/meta/futures/dates/{symbol}.yml")
#     fm.build_subgraph(url, ifera.RuleType.REFRESH)
#     for dep in fm.refresh_graph.successors(url):
#         if not dep.startswith("file:data/tensor/futures_individual/30m/"):
#             continue

#         params = fm.get_node_params(ifera.RuleType.REFRESH, dep)
#         contract_instrument = cm.create_derived_base_config(
#             ibkr_instrument,
#             contract_code=params["contract_code"],
#         )

#         calculated_exp_date = ifera.calculate_expiration(
#             contract_instrument.contract_code,  # type: ignore
#             ifera.ExpirationRule(contract_instrument.last_trading_day_rule),
#             contract_instrument.asset_class
#         )

#         calculated_first_notice_date = ifera.calculate_expiration(
#             contract_instrument.contract_code,  # type: ignore
#             ifera.ExpirationRule(contract_instrument.first_notice_day_rule),
#             contract_instrument.asset_class
#         )

#         exp_date_str = dates_raw[contract_instrument.contract_code].get("expiration_date")
#         first_notice_date_str = dates_raw[contract_instrument.contract_code].get("first_notice_date")
#         if exp_date_str is not None:
#             expiration_date = dt.date.fromisoformat(exp_date_str)
#         else:
#             expiration_date = calculated_exp_date

#         if first_notice_date_str is not None:
#             first_notice_date = dt.date.fromisoformat(first_notice_date_str)
#         else:
#             first_notice_date = calculated_first_notice_date

#         # if expiration_date > last_data_date:
#         #     continue

#         # fm.refresh_file(dep)

#         # Load the data tensor for the contract instrument
#         # contract_tensor = ifera.load_data_tensor(
#         #     instrument=contract_instrument,
#         #     dtype=torch.float32,
#         #     device=torch.device("cpu"),
#         #     strip_date_time=False,
#         # )
#         # if contract_tensor.shape[0] != 0:
#         #     last_trade_date = dt.date.fromordinal(int(contract_tensor[-1, 0, 2].item()))
#         # else:
#         #     last_trade_date = dt.date(
#         #         1900, 1, 1
#         #     )  # Default to a very old date if no data is available


#         # if expiration_date != calculated_exp_date or last_trade_date > calculated_exp_date or (last_trade_date - calculated_exp_date).days < -7:
#         # if expiration_date != calculated_exp_date:
#         #     print(
#         #         f"Warning: Expiration date mismatch for {symbol}-{contract_instrument.contract_code}: "
#         #         f"Expiration date: {expiration_date}, "
#         #         f"Calculated expiration date: {calculated_exp_date}"
#         #     )
#         #     mismatches[symbol][contract_instrument.contract_code] = {
#         #         "expiration_date": expiration_date,
#         #         "calculated_exp_date": calculated_exp_date,
#         #     }

#         if first_notice_date != calculated_first_notice_date:
#             print(
#                 f"Warning: First notice date mismatch for {symbol}-{contract_instrument.contract_code}: "
#                 f"First notice date: {first_notice_date}, "
#                 f"Calculated first notice date: {calculated_first_notice_date}"
#             )
#             mismatches[symbol][contract_instrument.contract_code] = {
#                 "first_notice_date": first_notice_date,
#                 "calculated_first_notice_date": calculated_first_notice_date,
#             }

#         # os.remove(
#         #     f"data/tensor/futures_individual/30m/{symbol}-{contract_instrument.contract_code}.pt"
#         # )

# # mismatches to yaml
# if mismatches:
#     mismatches_file = "data/meta/futures/expiration_mismatches.yml"
#     with open(mismatches_file, "w") as f:
#         yaml.safe_dump(
#             mismatches, f, sort_keys=True, default_flow_style=False, allow_unicode=True
#         )
#     print(f"Expiration mismatches saved to {mismatches_file}")

# -----------------------------------------------------------
# symbol = "AD"

# fm = ifera.FileManager()
# url = f"file:data/meta/futures/rollover/{symbol}.yml"
# fm.build_subgraph(url, ifera.RuleType.REFRESH)
# cm = ifera.ConfigManager()

# base_instrument = cm.get_base_instrument_config(symbol,"30m")
# contract_instruments = []
# contract_tensors = []

# get_params_time = 0
# get_tensor_time = 0

# # Load data/meta/futures/dates/GC.yml
# fm.refresh_file(f"file:data/meta/futures/dates/{symbol}.yml")
# with open(f"data/meta/futures/dates/{symbol}.yml", "r") as f:
#     dates = yaml.safe_load(f)

# for dep in fm.refresh_graph.successors(url):
#     if not dep.startswith("file:data/tensor/futures_individual/30m/"):
#         continue

#     fm.refresh_file(dep)
#     params = fm.get_node_params(ifera.RuleType.REFRESH, dep)

#     contract_instrument = cm.create_derived_base_config(
#         base_instrument,
#         contract_code=params["contract_code"],
#     )
#     if dates[contract_instrument.contract_code]["first_notice_date"] is not None:
#         contract_instrument.first_notice_date = dt.date.fromisoformat(dates[contract_instrument.contract_code]["first_notice_date"])

#     if dates[contract_instrument.contract_code]["expiration_date"] is not None:
#         contract_instrument.expiration_date = dt.date.fromisoformat(dates[contract_instrument.contract_code]["expiration_date"])

#     contract_tensor = ifera.load_data_tensor(
#         instrument=contract_instrument,
#         dtype=torch.float64,
#         device=torch.device("cuda:0"),
#         strip_date_time=False,
#     )
#     contract_tensors.append(contract_tensor)
#     contract_instruments.append(contract_instrument)

# # min_dates = foreach_map(lambda x: x[:, 2].min().to(torch.int64).item(), contract_tensors)
# # max_dates = foreach_map(lambda x: x[:, 2].max().to(torch.int64).item(), contract_tensors)

# # Sort both contract_instruments and contract_tensors by expiration date
# sorted_indices = sorted(range(len(contract_instruments)), key=lambda k: contract_instruments[k].expiration_date)
# contract_instruments_sorted = [contract_instruments[i] for i in sorted_indices]
# contract_tensors_sorted = [contract_tensors[i] for i in sorted_indices]

# ratios, active_idx, all_dates_ord = ifera.calculate_rollover(contract_instruments_sorted, contract_tensors_sorted)
# # ratios = ratios.nan_to_num(1.0)

# # # Calculate the cumulative product of the ratios backwards
# # ratios = torch.flip(ratios, dims=[0])
# # ratios = torch.cumprod(ratios, dim=0)
# # ratios = torch.flip(ratios, dims=[0])
# # print(ratios.shape, ratios.dtype)

# # print(np.array2string(ratios.cpu().numpy(),
# #     formatter={'float_kind': lambda x: f"{x:.4f}"}))
# # print(np.array2string(active_idx.cpu().numpy(),
# #     formatter={'int_kind': lambda x: f"{x}"}))


def make_table(
    data_tensor, date_idx_t, time_idx_t, maintenance_policy, stop_loss_t, total_profit
):
    idx = total_profit.argmax()
    date_idx = date_idx_t[idx].item()
    time_idx = time_idx_t[idx].item()
    stop_loss = stop_loss_t[idx].item()
    ord_date = data_tensor[date_idx, time_idx, 2].to(torch.int64).item()
    time_seconds = data_tensor[date_idx, time_idx, 3].to(torch.int64).item()
    stage_str = (
        maintenance_policy.derived_configs[maintenance_policy.stage].interval
        if maintenance_policy.stage.shape[0] > 0
        else "N/A"
    )
    current_high = data_tensor[date_idx, time_idx, 5].item()
    current_low = data_tensor[date_idx, time_idx, 6].item()
    stop_loss = stop_loss.item()
    total_profit = total_profit.item()
    date_str = dt.date.fromordinal(ord_date).strftime("%Y-%m-%d")
    time_str = f"{time_seconds // 3600:02}:{(time_seconds % 3600) // 60:02}:{time_seconds % 60:02}"
    ord_date = f"{date_str} {time_str}"
    table = Table(title=f"Simulation Status on {ord_date}", show_lines=False)

    table.add_column("Stage", justify="right", style="cyan")
    table.add_column("Current High", justify="right", style="green")
    table.add_column("Current Low", justify="right", style="yellow")
    table.add_column("Stop Loss", justify="right", style="yellow")
    table.add_column("Profit", justify="right", style="magenta")

    table.add_row(
        stage_str,
        f"{current_high:.2f}",
        f"{current_low:.2f}",
        f"{stop_loss:.2f}",
        f"{total_profit:.2f}",
    )

    return table


# symbols = broker.instruments.keys()
# for symbol in ["CL"]:

# if iconfig.type != "futures":
#     continue


import torch
import ifera
from einops import rearrange

fm = ifera.FileManager()
cm = ifera.ConfigManager()
dm = ifera.DataManager()

broker = cm.get_broker_config("IBKR")

symbol = "CL"
iconfig = cm.get_base_instrument_config(symbol, "30m")

base_config = cm.get_base_instrument_config(symbol, "1m")
env = ifera.SingleMarketEnv(
    instrument_config=base_config,
    broker_name="IBKR",
    backadjust=True,
    device=torch.device("cuda:0"),
    dtype=torch.float32,
)
batch_size = 10  # env.instrument_data.data.shape[0] - 250

openPolicy = ifera.OpenOncePolicy(direction=1, batch_size=batch_size, device=env.device)
initStopPolicy = ifera.InitialArtrStopLossPolicy(
    instrument_data=env.instrument_data, atr_multiple=3.0, batch_size=batch_size
)
maintenancePolicy = ifera.ScaledArtrMaintenancePolicy(
    instrument_data=env.instrument_data,
    stages=["1m", "5m", "15m", "1h", "4h", "1d"],
    atr_multiple=3.0,
    wait_for_breakeven=True,
    minimum_improvement=0.0,
    batch_size=batch_size,
)

done_policy = ifera.SingleTradeDonePolicy(batch_size=batch_size, device=env.device)

tradingPolicy = ifera.TradingPolicy(
    instrument_data=env.instrument_data,
    open_position_policy=openPolicy,
    initial_stop_loss_policy=initStopPolicy,
    position_maintenance_policy=maintenancePolicy,
    trading_done_policy=done_policy,
    batch_size=batch_size,
)

date_idx = torch.arange(0, batch_size, device=env.device, dtype=torch.int32)
time_idx = torch.zeros_like(date_idx, dtype=torch.int32, device=env.device)

print(f"Starting simulation for {symbol}")
t = time.time()

total_profit = env.rollout(
    trading_policy=tradingPolicy, start_date_idx=date_idx, start_time_idx=time_idx
)

print(f"Simulation completed in {time.time() - t:.2f} seconds.")

max_idx = total_profit.argmax()
print(
    f"Max index: {max_idx}, Total profit: {total_profit[max_idx].item():.4f}, date_idx: {date_idx[max_idx].item()}, time_idx: {time_idx[max_idx].item()}"
)
