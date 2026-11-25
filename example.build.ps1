# Copy to build.ps1 and adjust paths/CUDA architecture for your machine.
# Usage:
#   .\build.ps1              # incremental build
#   .\build.ps1 -reconfigure # re-run CMake configure (required after changing GPU arch)

$vs = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
if (-not (Test-Path "$vs\VC\Auxiliary\Build\vcvars64.bat")) {
    $vs = "C:\Program Files\Microsoft Visual Studio\2022\Community"
}
$cuda = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5" }

# Use "native" to auto-detect installed GPUs, or set explicit arch (e.g. 86=Ampere, 89=Ada, 120=Blackwell)
$cudaArch = "native"

cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") { [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) }
}
$env:CUDA_PATH = $cuda
$env:PATH = "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\CMake\bin;$cuda\bin;$cuda\bin\x64;$env:PATH"

if (-not (Test-Path "$PSScriptRoot\build\build.ninja") -or $args -contains "-reconfigure") {
    cmake -S $PSScriptRoot -B "$PSScriptRoot\build" -G Ninja -DCMAKE_BUILD_TYPE=Release `
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$cudaArch `
        -DGGML_CUDA_FA_QUANTS="f16-f16;q8_0-q8_0;q4_0-q4_0;bf16-bf16;f8_e4m3-f8_e4m3" `
        -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

cmake --build "$PSScriptRoot\build" --target llama-bench llama-completion llama-cli llama-server llama-fit-params -j 16
