$pipPath = "$env:APPDATA\Python\Python313\Scripts"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$pipPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$pipPath", "User")
    Write-Host "Added pip to PATH"
} else {
    Write-Host "pip already in PATH"
} 