import ifera

file_url = "file:data/raw/futures/1m/CL.zip"

fm = ifera.FileManager()
fm.refresh_file(file_url)
