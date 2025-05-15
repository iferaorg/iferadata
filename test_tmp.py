import ifera
import torch
from torch._higher_order_ops.foreach_map import foreach_map

# dm = ifera.DataManager()
# cm = ifera.ConfigManager()
# parent_instrument = cm.get_base_instrument_config("TN:1m")
# instrument = cm.create_derived_base_config(parent_instrument, "5m")
# device=torch.device("cuda:0")

# idata = dm.get_instrument_data(instrument_config=instrument, dtype=torch.float32, device=device)

# idata.data.shape

# fm = ifera.FileManager()
# fm.refresh_file("file:data/tensor/futures_rollover/30m/GC.pt")

fm = ifera.FileManager()
url = "file:data/tensor/futures_rollover/30m/GC.pt"
fm.build_subgraph(url, ifera.RuleType.REFRESH)
cm = ifera.ConfigManager()
base_instrument = cm.get_base_instrument_config("GC","30m")
base_tensor = None
contract_instruments = []
contract_tensors = []

get_params_time = 0
get_tensor_time = 0

for dep in fm.refresh_graph.successors(url):
    fm.refresh_file(dep)
    params = fm.get_node_params(ifera.RuleType.REFRESH, dep)

    if params["type"] == "futures":
        base_tensor = ifera.load_data_tensor(
            instrument=base_instrument,
            reset=False,
            dtype=torch.float32,
            device=torch.device("cuda:0"),
            strip_date_time=False,
        )
    else:
        contract_instrument = cm.create_derived_base_config(
            base_instrument,
            contract_code=params["contract_code"],
        )
        contract_tensor = ifera.load_data_tensor(
            instrument=contract_instrument,
            reset=False,
            dtype=torch.float32,
            device=torch.device("cuda:0"),
            strip_date_time=False,
        )
        contract_tensors.append(contract_tensor)
        contract_instruments.append(contract_instrument)

min_dates = foreach_map(lambda x: x[:, 2].min(), contract_tensors)
max_dates = foreach_map(lambda x: x[:, 2].max(), contract_tensors)

print("min_dates", min_dates)
print("max_dates", max_dates)

print("base min date", base_tensor[:, 2].min())
print("base max date", base_tensor[:, 2].max())

