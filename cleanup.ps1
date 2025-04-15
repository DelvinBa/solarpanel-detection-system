# Script to clean up unused files and reduce project size
# Run this script with administrative privileges

# Clean up log files
Write-Host "Cleaning up log files..." -ForegroundColor Green
if (Test-Path -Path "logs") {
    Remove-Item -Path "logs\*" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Logs cleaned." -ForegroundColor Green
} else {
    Write-Host "Logs directory not found." -ForegroundColor Yellow
}

# Clean up MLflow artifacts and runs (these can be recreated)
Write-Host "Cleaning up MLflow artifacts..." -ForegroundColor Green
if (Test-Path -Path "mlartifacts") {
    Remove-Item -Path "mlartifacts\*" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "MLflow artifacts cleaned." -ForegroundColor Green
} else {
    Write-Host "MLflow artifacts directory not found." -ForegroundColor Yellow
}

Write-Host "Cleaning up MLflow runs..." -ForegroundColor Green
if (Test-Path -Path "mlruns") {
    Remove-Item -Path "mlruns\*" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "MLflow runs cleaned." -ForegroundColor Green
} else {
    Write-Host "MLflow runs directory not found." -ForegroundColor Yellow
}

# Clean up MinIO data (test data storage)
Write-Host "Cleaning up MinIO data..." -ForegroundColor Green
if (Test-Path -Path "minio_data") {
    Remove-Item -Path "minio_data\*" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "MinIO data cleaned." -ForegroundColor Green
} else {
    Write-Host "MinIO data directory not found." -ForegroundColor Yellow
}

# Clean up cache directories
Write-Host "Cleaning up cache directories..." -ForegroundColor Green
$cacheDirs = @(".pytest_cache", "__pycache__", ".quarto")
foreach ($dir in $cacheDirs) {
    if (Test-Path -Path $dir) {
        Remove-Item -Path $dir -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "$dir cleaned." -ForegroundColor Green
    } else {
        Write-Host "$dir not found." -ForegroundColor Yellow
    }
}

# Find and clean all __pycache__ directories recursively
Get-ChildItem -Path . -Filter "__pycache__" -Directory -Recurse | 
    ForEach-Object {
        Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Removed $($_.FullName)" -ForegroundColor Green
    }

# Remove YOLOv8 pre-downloaded weights if present
if (Test-Path -Path "yolov8n.pt") {
    Remove-Item -Path "yolov8n.pt" -Force -ErrorAction SilentlyContinue
    Write-Host "Removed YOLOv8 pre-downloaded weights." -ForegroundColor Green
}

Write-Host "Cleanup completed!" -ForegroundColor Green 