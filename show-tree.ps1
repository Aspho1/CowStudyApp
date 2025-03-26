$excludedDirs = @('.git', '.venv', '__pycache__', '.pytest_cache', 'build', 'dist', 'RB_22', '.mypy_cache')
$excludedFiles = @('*.pyc', '*.pyo', '*.pyd', '*.so')

function Show-ProjectTree {
    param (
        [string]$indent = "",
        [string]$path = "."
    )

    # Get directories first
    Get-ChildItem -Path $path -Directory | 
        Where-Object { $excludedDirs -notcontains $_.Name } | 
        ForEach-Object {
            Write-Host "$indent├── $($_.Name)/"
            Show-ProjectTree -indent "$indent│   " -path $_.FullName
        }

    # Then get files
    Get-ChildItem -Path $path -File | 
        Where-Object { 
            $excluded = $false
            foreach ($pattern in $excludedFiles) {
                if ($_.Name -like $pattern) {
                    $excluded = $true
                    break
                }
            }
            -not $excluded
        } | 
        ForEach-Object {
            Write-Host "$indent├── $($_.Name)"
        }
}

Show-ProjectTree