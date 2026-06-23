$years = @('2018','2019','2020','2021')
$base  = 'http://rain.physics.ox.ac.uk/ICPAC/training/IFS'
foreach ($year in $years) {
    $dest = "E:\CGAN\IFS_training\$year\sp.nc"
    if ((Test-Path $dest) -and (Get-Item $dest).Length -gt 6GB) { Write-Host "DONE $year/sp.nc"; continue }
    Write-Host "$(Get-Date -f 'HH:mm:ss') $year/sp.nc..."
    curl.exe -L -C - -o $dest --retry 10 --retry-delay 5 "$base/$year/sp.nc"
}
Write-Host "sp DONE"
