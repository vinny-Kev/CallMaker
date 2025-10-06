# Quick Deploy Script - PowerShell Version
# Deploys the latest trained model to the API service

Write-Host ""
Write-Host "🚀 Deploying Latest Model to API Service..." -ForegroundColor Cyan
Write-Host ""

# Run the Python deployment script
& "D:/CODE ALL HERE PLEASE/money-printer/.venv/Scripts/python.exe" "D:\CODE ALL HERE PLEASE\money-printer\deploy_model_to_api.py"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Deployment complete! You can now restart your API service." -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Deployment failed. Check the error messages above." -ForegroundColor Red
    Write-Host ""
}
