param(
    [string]$Python = "py -3.11"
)

Write-Host "Creating venv with Python 3.11..."
Invoke-Expression "$Python -m venv .venv"

Write-Host "Activating venv..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip and installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements-dev.txt

Write-Host "Done. Copy .env.example to .env and fill credentials."
