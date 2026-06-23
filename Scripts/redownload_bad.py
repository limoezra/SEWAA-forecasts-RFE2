import os, subprocess

BASE = 'http://rain.physics.ox.ac.uk/ICPAC/training/IFS'
DEST = 'E:/CGAN/IFS_training'

BAD = [
    ('2018', 'tp'),
    ('2018', 't2m'),
    ('2018', 'tcwv'),
    ('2019', 'tp'),
    ('2019', 't2m'),
    ('2021', 'tcwv'),
]

print(f'Re-downloading {len(BAD)} files...\n')
for year, var in BAD:
    dest = f'{DEST}/{year}/{var}.nc'
    url  = f'{BASE}/{year}/{var}.nc'
    if os.path.exists(dest):
        os.remove(dest)
        print(f'Deleted old {year}/{var}.nc')
    print(f'Downloading {year}/{var}.nc ...')
    subprocess.run([
        'curl.exe', '-L', '-C', '-', '-o', dest,
        '--retry', '10', '--retry-delay', '5',
        '--progress-bar', url
    ], check=True)
    size = os.path.getsize(dest)/1e9
    print(f'Done: {year}/{var}.nc  ({size:.2f} GB)\n')

print('All done!')
