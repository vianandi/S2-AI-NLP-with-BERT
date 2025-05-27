if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python tidak ditemukan. Pastikan sudah terinstall dan ditambahkan ke PATH." -ForegroundColor Red
    exit
}

if (-not (Test-Path ".\venv")) {
    Write-Host "Membuat virtual environment 'venv'..."
    python -m venv venv
} else {
    Write-Host "Virtual environment 'venv' sudah ada."
}

Write-Host "Mengaktifkan virtual environment..."
& .\venv\Scripts\Activate.ps1

if (Test-Path ".\requirements.txt") {
    Write-Host "Menginstall dependensi dari requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
    Write-Host "Semua library berhasil diinstall." -ForegroundColor Green
} else {
    Write-Host "File requirements.txt tidak ditemukan!" -ForegroundColor Yellow
}