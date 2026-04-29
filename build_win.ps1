# Script build cho Windows (PowerShell)
Write-Host "🚀 Đang bắt đầu build Claw Code cho Windows (Release)..." -ForegroundColor Cyan
Set-Location -Path "rust"
cargo build --release --workspace
Write-Host "✅ Build hoàn tất! File thực thi nằm tại: rust\target\release\claw.exe" -ForegroundColor Green
