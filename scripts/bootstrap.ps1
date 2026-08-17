$ErrorActionPreference = "Stop"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.11+ is required."
}

python -m pip install --user uv
python -m uv sync --extra dev

$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    winget install --id Ollama.Ollama --exact --accept-package-agreements --accept-source-agreements --silent
    $ollamaPath = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
} else {
    $ollamaPath = $ollama.Source
}

if (-not (Test-Path -LiteralPath $ollamaPath)) {
    throw "Ollama was installed but its executable was not found. Restart PowerShell and retry."
}

& $ollamaPath pull qwen3:4b-instruct
& $ollamaPath pull qwen3-embedding:0.6b

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
}

python -m uv run research-agent doctor
python -m uv run pytest -q

