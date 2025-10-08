"""
Test Enhanced Contextual Analysis Features
Demonstrates trend indicators, tags, and trading suggestions
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from src.data_scraper import DataScraper
from src.feature_engineering import FeatureEngineer
import pandas as pd
import json

def test_contextual_features():
    """Test the new contextual analysis features"""
    
    print("="*80)
    print("🧪 TESTING ENHANCED CONTEXTUAL FEATURES")
    print("="*80)
    
    # Initialize components
    print("\n📡 Fetching recent BTC data...")
    scraper = DataScraper(symbol="BTCUSDT", interval="1m")
    df = scraper.fetch_historical_("1 hour ago UTC")
    print(f"✓ Fetched {len(df)} candles")
    
    # Generate features
    print("\n🔧 Generating features...")
    fe = FeatureEngineer()
    df_features = fe.generate_all_features(df, create_targets=False)
    print(f"✓ Generated {df_features.shape[1]} features")
    
    # Test 1: Trend Indicators
    print("\n" + "="*80)
    print("📊 TEST 1: TREND INDICATORS")
    print("="*80)
    
    trend_info = fe.calculate_trend_indicators(df_features)
    
    print("\n🎯 Trend Analysis:")
    print(f"  Short-term Trend:  {trend_info['short_term_trend'].upper()}")
    print(f"  Long-term Trend:   {trend_info['long_term_trend'].upper()}")
    print(f"  Trend Strength:    {trend_info['trend_strength'].upper()}")
    
    print("\n📈 Trend Metrics:")
    for key, value in trend_info['trend_metrics'].items():
        print(f"  {key:25s}: {value}")
    
    # Test 2: Contextual Tags
    print("\n" + "="*80)
    print("🏷️  TEST 2: CONTEXTUAL TAGS")
    print("="*80)
    
    tags = fe.generate_contextual_tags(df_features)
    
    print(f"\n✓ Generated {len(tags)} tags:")
    for tag in tags:
        print(f"  • {tag}")
    
    # Test 3: Trading Suggestions
    print("\n" + "="*80)
    print("💡 TEST 3: TRADING SUGGESTIONS")
    print("="*80)
    
    # Simulate different prediction scenarios
    scenarios = [
        {"name": "Strong Buy Signal", "prediction": 1, "confidence": 0.75},
        {"name": "Weak Buy Signal", "prediction": 1, "confidence": 0.45},
        {"name": "Strong Sell Signal", "prediction": 2, "confidence": 0.80},
        {"name": "No Movement", "prediction": 0, "confidence": 0.60},
    ]
    
    for scenario in scenarios:
        print(f"\n📍 Scenario: {scenario['name']}")
        print(f"   Prediction: {scenario['prediction']} | Confidence: {scenario['confidence']:.1%}")
        
        suggestion = fe.generate_trading_suggestion(
            prediction=scenario['prediction'],
            confidence=scenario['confidence'],
            trend_info=trend_info,
            tags=tags
        )
        
        print(f"\n   ➡️  Action: {suggestion['action']} ({suggestion['conviction']} conviction)")
        print(f"   ⚠️  Risk Level: {suggestion['risk_level']}")
        print(f"   📊 Score Breakdown:")
        print(f"      Confidence Boost: {suggestion['score_breakdown']['confidence_boost']}")
        print(f"      Trend Score:      {suggestion['score_breakdown']['trend_score']}")
        print(f"      Total Score:      {suggestion['score_breakdown']['total_score']}")
        
        print(f"   💬 Reasoning:")
        for reason in suggestion['reasoning'][:3]:  # Show first 3 reasons
            print(f"      • {reason}")
        if len(suggestion['reasoning']) > 3:
            print(f"      ... and {len(suggestion['reasoning']) - 3} more reasons")
    
    # Test 4: Full JSON Response
    print("\n" + "="*80)
    print("📦 TEST 4: FULL JSON RESPONSE EXAMPLE")
    print("="*80)
    
    full_response = {
        "prediction": 1,
        "prediction_label": "Large Upward Movement Expected",
        "confidence": 0.68,
        "trend": {
            "short_term": trend_info['short_term_trend'],
            "long_term": trend_info['long_term_trend'],
            "strength": trend_info['trend_strength']
        },
        "tags": tags,
        "suggestion": fe.generate_trading_suggestion(1, 0.68, trend_info, tags),
        "current_price": float(df_features['close'].iloc[-1])
    }
    
    print("\n" + json.dumps(full_response, indent=2, default=str))
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n💡 Next Steps:")
    print("   1. Update prediction_api.py to include these enrichments")
    print("   2. Test with live API endpoint")
    print("   3. Update frontend to display trend/tags/suggestions")
    print("\n" + "="*80)

if __name__ == "__main__":
    test_contextual_features()
