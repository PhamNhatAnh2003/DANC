# Khởi chạy toàn bộ chatbot: Action Server, Rasa Server, giao diện Streamlit (mỗi cái 1 cửa sổ).
# Chạy: powershell -ExecutionPolicy Bypass -File .\run_all.ps1
$root = $PSScriptRoot

function Start-Window([string]$title, [string]$command) {
    $cmd = "`$host.UI.RawUI.WindowTitle='$title'; `$env:PYTHONUTF8='1'; $command"
    Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-NoProfile", "-Command", $cmd
}

Start-Window "Action Server (5055)" ".\.venv\Scripts\python.exe -m rasa run actions"
Start-Window "Rasa Server (5005)" ".\.venv\Scripts\python.exe -m rasa run --enable-api --cors '*' --model models\soict-admission.tar.gz"
Start-Window "Streamlit (8501)" ".\.venv-ui\Scripts\python.exe -m streamlit run streamlit_app.py --server.headless true"

Write-Host "Đang khởi động 3 cửa sổ. Đợi ~1 phút đến khi:"
Write-Host "  - cửa sổ Action Server hiện 'Starting action endpoint server'"
Write-Host "  - cửa sổ Rasa Server hiện 'Rasa server is up and running'"
Write-Host "rồi mở http://localhost:8501"


# powershell -ExecutionPolicy Bypass -File .\run_all.ps1