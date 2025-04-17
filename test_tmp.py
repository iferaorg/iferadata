import ifera.file_refresh
import ifera
import torch

# dm = ifera.DataManager()
# cm = ifera.ConfigManager()
# parent_instrument = cm.get_base_instrument_config("TN:1m")
# instrument = cm.create_derived_base_config(parent_instrument, "5m")
# device=torch.device("cuda:0")

# idata = dm.get_instrument_data(instrument_config=instrument, dtype=torch.float32, device=device)

# idata.data.shape

fm = ifera.FileManager()
fm.refresh_file("file:data/tensor/futures_rollower/30m/TN.pt")

