"""
Deploy Latest Model to API Service
Automatically copies the most recently trained model to the bitcoin-api-service directory
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

def get_latest_model_dir(models_path='data/models'):
    """Find the most recently created model directory"""
    model_dirs = [d for d in Path(models_path).iterdir() if d.is_dir()]
    
    if not model_dirs:
        raise FileNotFoundError(f"No models found in {models_path}")
    
    # Sort by creation time (directory name contains timestamp)
    latest_model = max(model_dirs, key=lambda d: d.name)
    return latest_model

def deploy_model_to_api(api_service_path=r'd:\CODE ALL HERE PLEASE\bitcoin-api-service'):
    """Deploy the latest model to the API service directory"""
    
    print("="*80)
    print(" MODEL DEPLOYMENT TO API SERVICE")
    print("="*80)
    
    # Get latest model directory
    latest_model_dir = get_latest_model_dir()
    print(f"\n📦 Latest Model Found: {latest_model_dir.name}")
    
    # Load and display metadata
    metadata_path = latest_model_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📊 Model Information:")
        print(f"  Symbol: {metadata.get('symbol', 'N/A')}")
        print(f"  Interval: {metadata.get('interval', 'N/A')}")
        print(f"  Features: {metadata.get('n_features', 'N/A')}")
        print(f"  Loss Type: {metadata.get('loss_type', 'N/A')}")
        print(f"  Days Trained: {metadata.get('days_trained', 'N/A')}")
        print(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")
        
        # Display performance metrics if available
        if 'performance' in metadata:
            perf = metadata['performance']
            print(f"\n📈 Performance Metrics:")
            print(f"  Direction Accuracy: {perf.get('direction_accuracy', 'N/A'):.2f}%")
            print(f"  Top Win Rate: {perf.get('top_win_rate', 'N/A'):.2f}%")
            print(f"  Top Trades: {perf.get('top_trades', 'N/A')}")
            print(f"  Avg Return: {perf.get('top_avg_return', 'N/A'):.3f}%")
        
        # Display best strategy if available
        if 'best_strategy' in metadata:
            best = metadata['best_strategy']
            print(f"\n🎯 Best Trading Strategy:")
            print(f"  Strategy: {best.get('pred_type', 'N/A')}_{best.get('threshold', 'N/A')}")
            print(f"  Win Rate: {best.get('win_rate', 'N/A'):.2f}%")
            print(f"  Trades: {best.get('n_trades', 'N/A')}")
            print(f"  Avg Return: {best.get('avg_return', 'N/A'):.3f}%")
    
    # Setup API service path
    api_models_path = Path(api_service_path) / 'data' / 'models'
    
    if not api_models_path.exists():
        print(f"\n⚠️  API models directory not found: {api_models_path}")
        print(f"Creating directory...")
        api_models_path.mkdir(parents=True, exist_ok=True)
    
    # Remove old models from API directory
    print(f"\n🗑️  Removing old models from API directory...")
    old_models_removed = 0
    for old_model in api_models_path.iterdir():
        if old_model.is_dir():
            shutil.rmtree(old_model)
            print(f"  ✓ Removed: {old_model.name}")
            old_models_removed += 1
    
    if old_models_removed == 0:
        print(f"  No old models to remove")
    
    # Copy new model to API directory
    print(f"\n📤 Deploying model to API service...")
    destination = api_models_path / latest_model_dir.name
    shutil.copytree(latest_model_dir, destination)
    print(f"  ✓ Copied to: {destination}")
    
    # List files in the deployed model
    print(f"\n📁 Deployed Model Contents:")
    for file in sorted(destination.iterdir()):
        file_size = file.stat().st_size / 1024  # KB
        print(f"  • {file.name:<35} ({file_size:>8.1f} KB)")
    
    print(f"\n{'='*80}")
    print(f"✅ DEPLOYMENT SUCCESSFUL!")
    print(f"{'='*80}")
    print(f"\n🌐 Model is ready to use in the API service:")
    print(f"   API Directory: {api_service_path}")
    print(f"   Model Path: data/models/{latest_model_dir.name}")
    print(f"\n💡 Next Steps:")
    print(f"   1. Restart your API service to load the new model")
    print(f"   2. Test the API endpoint: /predict")
    print(f"   3. Check model info at: /model-info")
    print(f"\n{'='*80}\n")
    
    return destination

if __name__ == "__main__":
    try:
        deployed_path = deploy_model_to_api()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
