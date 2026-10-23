param(
    [string]$Apk = (Join-Path (Split-Path $PSScriptRoot -Parent) 'AM_CS_NEW.apk'),
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath $Apk -PathType Leaf)) { throw "APK not found: $Apk" }
    $outputPath = Join-Path $PSScriptRoot ('runs\apk_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
    & $Python -m driftlab inspect-apk --apk $Apk --output $outputPath
    if ($LASTEXITCODE -ne 0) { throw 'APK inspection failed.' }
    Start-Process -FilePath (Join-Path $outputPath 'apk_report.html')
} finally { Pop-Location }
