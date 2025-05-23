import pandas as pd
import ifera
import torch
import numpy as np
from torch._higher_order_ops.foreach_map import foreach_map
import yaml
import os


# dm = ifera.DataManager()
# cm = ifera.ConfigManager()
# parent_instrument = cm.get_base_instrument_config("TN:1m")
# instrument = cm.create_derived_base_config(parent_instrument, "5m")
# device=torch.device("cuda:0")

# idata = dm.get_instrument_data(instrument_config=instrument, dtype=torch.float32, device=device)

# idata.data.shape

fm = ifera.FileManager()
cm = ifera.ConfigManager()
s3_objects = ifera.list_s3_objects("raw/futures/30m")
# Extract just the symbol (basename without .zip) and ignore any non-zip entries
symbols = [
    os.path.splitext(os.path.basename(obj))[0]
    for obj in s3_objects
    if obj.endswith(".zip")
]

for symbol in symbols:
    # fm.refresh_file(f"file:raw/futures/30m/{symbol}.zip")
    # ifera.delete_s3_file(f"meta/futures/dates/{symbol}.yml")
    # fm.refresh_file(f"file:data/meta/futures/dates/{symbol}.yml")
    # fm.refresh_file(f"file:data/tensor/futures_rollover/30m/{symbol}.pt")
    # fm.refresh_file(f"file:data/meta/futures/dates/{symbol}.yml")
    # fm.refresh_file(f"file:data/meta/futures/dates/{symbol}.yml")
    # fm.refresh_file(f"file:data/meta/futures/dates/{symbol}.yml")
    instrument = cm.get_config(broker_name="barchart", symbol=symbol, interval="30m")
    print(f"{symbol}: {instrument.broker_symbol}")
    pass



# fm = ifera.FileManager()
# url = "file:data/tensor/futures_rollover/30m/GC.pt"
# fm.build_subgraph(url, ifera.RuleType.REFRESH)
# cm = ifera.ConfigManager()
# base_instrument = cm.get_base_instrument_config("GC","30m")
# contract_instruments = []
# contract_tensors = []

# get_params_time = 0
# get_tensor_time = 0

# # Load data/meta/futures/dates/GC.yml
# fm.refresh_file("file:data/meta/futures/dates/GC.yml")
# with open("data/meta/futures/dates/GC.yml", "r") as f:
#     dates = yaml.safe_load(f)


# for dep in fm.refresh_graph.successors(url):
#     # fm.refresh_file(dep)
#     params = fm.get_node_params(ifera.RuleType.REFRESH, dep)

#     contract_instrument = cm.create_derived_base_config(
#         base_instrument,
#         contract_code=params["contract_code"],
#     )
#     if dates[contract_instrument.contract_code]["first_notice_date"] is not None:
#         contract_instrument.first_notice_date = pd.to_datetime(dates[contract_instrument.contract_code]["first_notice_date"])

#     if dates[contract_instrument.contract_code]["expiration_date"] is not None:
#         contract_instrument.expiration_date = pd.to_datetime(dates[contract_instrument.contract_code]["expiration_date"])

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

# ratios, active_idx = ifera.calculate_rollover(contract_instruments_sorted, contract_tensors_sorted)
# ratios = ratios.nan_to_num(1.0)

# # Calculate the cumulative product of the ratios backwards
# ratios = torch.flip(ratios, dims=[0])
# ratios = torch.cumprod(ratios, dim=0)
# ratios = torch.flip(ratios, dims=[0])
# print(ratios.shape, ratios.dtype)

# print(np.array2string(ratios.cpu().numpy(),
#     formatter={'float_kind': lambda x: f"{x:.4f}"}))
# print(np.array2string(active_idx.cpu().numpy(),
#     formatter={'int_kind': lambda x: f"{x}"}))
