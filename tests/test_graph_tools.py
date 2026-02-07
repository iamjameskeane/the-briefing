"""
Integration test for Constellation graph tools.
Run with: PYTHONPATH=. ./venv/bin/python3 tests/test_graph_tools.py
"""

import sys
import os
from pathlib import Path

# Ensure we can import from the root
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import execute_tool_call, get_supabase_client

# Known test data from live query
TEST_EVENT_ID = "b23bb3a3-1c44-40ba-8c14-d83099672d53"
TEST_ENTITY_NAME = "Russia"

def test_tool(name, args):
    print(f"\n🚀 Testing tool: {name}")
    print(f"   Args: {args}")
    try:
        result = execute_tool_call(name, args)
        print(f"   ✅ Result (first 200 chars):\n{result[:200]}...")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def run_all_tests():
    print("🧪 Starting Graph Tools Integration Tests...")
    
    # 1. Test get_event_graph
    test_tool("get_event_graph", {"event_id": TEST_EVENT_ID})
    
    # 2. Test get_entity_relationships
    test_tool("get_entity_relationships", {"entity_name": TEST_ENTITY_NAME})
    
    # 3. Test get_causal_chain
    test_tool("get_causal_chain", {"event_id": TEST_EVENT_ID, "max_depth": 3})
    
    # 4. Test get_impact_chain
    test_tool("get_impact_chain", {"event_id": TEST_EVENT_ID, "max_depth": 3})
    
    # 5. Test get_entity_events
    test_tool("get_entity_events", {"entity_name": TEST_ENTITY_NAME, "limit": 3})

if __name__ == "__main__":
    run_all_tests()
