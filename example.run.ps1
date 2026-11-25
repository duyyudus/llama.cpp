# Copy to run.ps1 and set $models to your local GGUF directory.
# Single-GPU users: remove `--mmproj-device CUDA1` (or set it to CUDA0).

$cuda = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5" }
$env:PATH = "$cuda\bin\x64;$cuda\bin;$env:PATH"

$models = "C:\path\to\models\Inkling-2-Mini-GGUF"

& "$PSScriptRoot\build\bin\llama-server.exe" `
    -m "$models\Inkling-2-Mini-UD-IQ4_XS.gguf" `
    --mmproj "$models\mmproj-F16.gguf" --mmproj-device CUDA1 `
    --alias Inkling-2-Mini-GGUF `
    -c 200000 -ub 512 -np 1 -ctk q8_0 -ctv q8_0 `
    -ngl 99 -fa on -fit off -lm none -bs `
    --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 `
    --host 127.0.0.1 --port 8001 @args
