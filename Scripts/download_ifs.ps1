$fields = @('tp','t2m','tcwv','sp')
$years  = @('2018','2019','2020','2021')
$base   = 'http://rain.physics.ox.ac.uk/ICPAC/training/IFS'

foreach ($year in $years) {
    New-Item -ItemType Directory -Force "E:\CGAN\IFS_training\$year" | Out-Null
}

foreach ($year in $years) {
    foreach ($field in $fields) {
        $dest = "E:\CGAN\IFS_training\$year\$field.nc"
        $size = if (Test-Path $dest) { (Get-Item $dest).Length } else { 0 }
        if ($size -gt 6GB) {
            Write-Host "DONE   $year/$field.nc"
            continue
        }
        Write-Host "$(Get-Date -f 'HH:mm:ss') Downloading $year/$field.nc ..."
        curl.exe -L -C - -o $dest --retry 10 --retry-delay 5 --connect-timeout 30 "$base/$year/$field.nc"
        Write-Host "$(Get-Date -f 'HH:mm:ss') Finished $year/$field.nc ($([math]::Round((Get-Item $dest).Length/1GB,2)) GB)"
    }
}
Write-Host "ALL DONE"
