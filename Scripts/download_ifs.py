import os, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import netCDF4

BASE    = 'http://rain.physics.ox.ac.uk/ICPAC/training/IFS'
DEST    = 'E:/CGAN/IFS_training'
YEARS   = ['2018', '2019', '2020', '2021']
VARS    = ['tp', 't2m', 'tcwv', 'sp']
WORKERS = 4  # parallel downloads

def is_valid_nc(path):
    try:
        with netCDF4.Dataset(path, 'r') as ds:
            _ = ds.variables  # force read of metadata
        return True
    except Exception:
        return False

def fetch(dest, url):
    for attempt in range(5):
        r = subprocess.run([
            'curl.exe', '-L', '-C', '-', '-o', dest,
            '--retry', '10', '--retry-delay', '5', '--retry-all-errors',
            '--insecure', '--silent', '--show-error', url
        ])
        if r.returncode == 0:
            return
        print(f'  curl exited {r.returncode}, retrying ({attempt+1}/5)...', flush=True)
    raise RuntimeError(f'Failed to download {url} after 5 attempts')

def download(year, var):
    os.makedirs(f'{DEST}/{year}', exist_ok=True)
    dest = f'{DEST}/{year}/{var}.nc'
    url  = f'{BASE}/{year}/{var}.nc'
    size = os.path.getsize(dest) if os.path.exists(dest) else 0

    if size > 6e9 and is_valid_nc(dest):
        return f'SKIP   {year}/{var}.nc  ({size/1e9:.1f} GB, verified OK)'

    if size > 6e9 and not is_valid_nc(dest):
        print(f'CORRUPT {year}/{var}.nc — deleting and re-downloading...', flush=True)
        os.remove(dest)

    print(f'START  {year}/{var}.nc ...', flush=True)
    fetch(dest, url)

    if not is_valid_nc(dest):
        return f'ERROR  {year}/{var}.nc  — file is corrupt after download!'

    return f'DONE   {year}/{var}.nc  ({os.path.getsize(dest)/1e9:.2f} GB, verified OK)'

tasks = [(year, var) for year in YEARS for var in VARS]

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(download, y, v): (y, v) for y, v in tasks}
    for f in as_completed(futures):
        print(f.result(), flush=True)

print('All downloads complete!')
