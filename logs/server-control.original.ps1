param(
    [ValidateSet('Start', 'Check', 'Stop')]
    [string]$Action = 'Start',
    [ValidateRange(1, 65535)]
    [int]$Port = 8002
)

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$serverDir = [System.IO.Path]::GetFullPath((Join-Path $projectRoot 'code\server'))
$logDir = Join-Path $projectRoot 'logs'
$statePath = Join-Path $logDir ("server-{0}.json" -f $Port)
$baseUrl = "http://127.0.0.1:$Port"

function Get-ListenerIds {
    $connections = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    return @($connections | Select-Object -ExpandProperty OwningProcess -Unique)
}

function Get-ProcessRecord([int]$processId) {
    return Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
}

function Test-CommandLine([string]$commandLine, [bool]$requireAbsolutePath) {
    if ([string]::IsNullOrWhiteSpace($commandLine)) { return $false }
    if ($commandLine -notmatch '(?i)-m\s+uvicorn\s+app:app(?:\s|$)') { return $false }
    if ($commandLine -notmatch "(?i)(?:^|\s)--port\s+$Port(?:\s|$)") { return $false }
    if ($commandLine -notmatch '(?i)(?:^|\s)--host\s+127\.0\.0\.1(?:\s|$)') { return $false }
    if ($commandLine.IndexOf($serverDir, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
        return $true
    }
    if ($requireAbsolutePath) { return $false }
    return $commandLine -match '(?i)--app-dir\s+["'']?code[\\/]+server(?:["'']|\s|$)'
}

function Test-Health {
    try {
        $response = Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 3 -ErrorAction Stop
        return $response.status -eq 'ok' -and $response.service -eq 'aigc-local-api'
    } catch {
        return $false
    }
}

function Show-ListenerStatus {
    $listenerIds = @(Get-ListenerIds)
    if ($listenerIds.Count -eq 0) {
        Write-Host "No service is listening on port $Port."
        return $false
    }
    $healthy = Test-Health
    foreach ($listenerId in $listenerIds) {
        $record = Get-ProcessRecord ([int]$listenerId)
        $compatible = $null -ne $record -and (Test-CommandLine $record.CommandLine $false)
        if ($healthy -and $compatible) {
            Write-Host "A compatible service is already running (PID $listenerId): $baseUrl/"
            Write-Host 'It was left untouched.'
            return $true
        }
    }
    Write-Host "Port $Port is occupied by PID(s) $($listenerIds -join ', '), but this is not a verified healthy project service."
    Write-Host 'No process was stopped or restarted.'
    return $false
}

function Start-ProjectServer {
    $listenerIds = @(Get-ListenerIds)
    if ($listenerIds.Count -gt 0) {
        if (Show-ListenerStatus) { return 0 }
        return 1
    }

    if (-not (Test-Path -LiteralPath (Join-Path $serverDir 'app.py') -PathType Leaf)) {
        Write-Host "Server source not found: $serverDir\app.py"
        return 1
    }
    $localPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $pythonCommand = if (Test-Path -LiteralPath $localPython) { Get-Command $localPython } else { Get-Command python -ErrorAction SilentlyContinue }
    if ($null -eq $pythonCommand) {
        Write-Host 'Python is not on PATH. Install Python 3.12 and the project requirements first.'
        return 1
    }
    $pythonExe = $pythonCommand.Source
    $previousErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $sklearnVersion = & $pythonExe -c 'import fastapi, uvicorn, sklearn; print(sklearn.__version__)' 2>$null
        $dependencyExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorPreference
    }
    if ($dependencyExitCode -ne 0 -or $sklearnVersion -ne '1.9.1') {
        Write-Host 'Python dependencies are missing or scikit-learn is not 1.9.1.'
        Write-Host 'From this project folder run: py -3.12 setup_environment.py --with-llm'
        return 1
    }

    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $outLog = Join-Path $logDir ("server-{0}-{1}.out.log" -f $Port, $stamp)
    $errLog = Join-Path $logDir ("server-{0}-{1}.err.log" -f $Port, $stamp)
    $arguments = @('-m', 'uvicorn', 'app:app', '--app-dir', ('"' + $serverDir + '"'), '--host', '127.0.0.1', '--port', [string]$Port)
    $started = Start-Process -FilePath $pythonExe -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $outLog -RedirectStandardError $errLog
    $serviceId = $started.Id
    Write-Host "Starting project service (PID $serviceId)."
    $ready = $false
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        Start-Sleep -Milliseconds 500
        if (Test-Health) {
            $currentOwners = @(Get-ListenerIds)
            if ($currentOwners -contains $serviceId) {
                $ready = $true
                break
            }
        }
        if ($started.HasExited) { break }
    }
    if (-not $ready) {
        Write-Host "Startup did not become ready. Check $errLog"
        Write-Host 'No other process will be stopped automatically.'
        return 1
    }

    $record = Get-ProcessRecord $serviceId
    if ($null -eq $record -or -not (Test-CommandLine $record.CommandLine $true)) {
        Write-Host 'Service responded, but process identity could not be verified; no stop record was written.'
        return 1
    }
    $state = [ordered]@{
        project_root = $projectRoot
        app_dir = $serverDir
        port = $Port
        process_id = $serviceId
        process_created_utc = $record.CreationDate.ToUniversalTime().ToString('o')
        executable = $record.ExecutablePath
        stdout_log = $outLog
        stderr_log = $errLog
    }
    [System.IO.File]::WriteAllText($statePath, ($state | ConvertTo-Json -Depth 4), [System.Text.UTF8Encoding]::new($false))
    Write-Host "Ready: $baseUrl/"
    Write-Host "Logs: $outLog and $errLog"
    return 0
}

function Stop-ProjectServer {
    if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        Write-Host 'No managed PID record exists. The running service, if any, was not started by this script.'
        Write-Host 'No process was stopped.'
        return 1
    }
    try {
        $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Write-Host 'The PID record is unreadable. No process was stopped.'
        return 1
    }
    if ($state.project_root -ne $projectRoot -or $state.app_dir -ne $serverDir -or [int]$state.port -ne $Port) {
        Write-Host 'The PID record does not match this project and port. No process was stopped.'
        return 1
    }
    $serviceId = [int]$state.process_id
    $record = Get-ProcessRecord $serviceId
    if ($null -eq $record) {
        Write-Host 'The recorded process has already exited. No process was stopped.'
        return 1
    }
    $createdUtc = $record.CreationDate.ToUniversalTime().ToString('o')
    $sameExecutable = [string]::Equals($record.ExecutablePath, [string]$state.executable, [System.StringComparison]::OrdinalIgnoreCase)
    $ownerIds = @(Get-ListenerIds)
    if ($createdUtc -ne $state.process_created_utc -or -not $sameExecutable -or -not (Test-CommandLine $record.CommandLine $true) -or $ownerIds -notcontains $serviceId) {
        Write-Host 'PID, creation time, executable, command line or port ownership changed.'
        Write-Host 'No process was stopped.'
        return 1
    }
    Stop-Process -Id $serviceId -ErrorAction Stop
    Remove-Item -LiteralPath $statePath -Force
    Write-Host "Stopped this project's managed service (PID $serviceId)."
    return 0
}

try {
    switch ($Action) {
        'Check' {
            if (Show-ListenerStatus) { exit 0 }
            exit 1
        }
        'Start' { exit (Start-ProjectServer) }
        'Stop' { exit (Stop-ProjectServer) }
    }
} catch {
    Write-Host "Server control failed: $($_.Exception.Message)"
    exit 1
}
