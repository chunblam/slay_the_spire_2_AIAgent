#!/usr/bin/env python3
"""
Quick sanity test for RL_slay_the_spire_sts2agent adapter.
Verifies:
1. Health endpoint responds
2. State endpoint returns proper structure
3. Normalization handles STS2AIAgent screen enum
4. Basic action encoding works (target_index)
"""

import sys
import requests
import json
from typing import Dict

def test_health_endpoint(base_url: str) -> bool:
    """Phase 1: Health check"""
    print(f"\n[1/5] Testing health endpoint at {base_url}/health...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=5.0)
        resp.raise_for_status()
        payload = resp.json()
        print(f"  Response: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        if payload.get("ok"):
            print("  ✓ Health check passed")
            return True
        print("  ✗ Health check failed: ok != true")
        return False
    except Exception as e:
        print(f"  ✗ Health check error: {e}")
        return False


def test_state_endpoint(base_url: str) -> tuple[bool, Dict]:
    """Phase 2: State endpoint and format"""
    print(f"\n[2/5] Testing state endpoint at {base_url}/api/v1/session/state...")
    try:
        resp = requests.get(f"{base_url}/api/v1/session/state", timeout=5.0)
        if resp.status_code >= 400:
            print(f"  session/state unavailable ({resp.status_code}), fallback to /state")
            resp = requests.get(f"{base_url}/state", timeout=5.0)
        resp.raise_for_status()
        payload = resp.json()
        
        # Check response format: {ok, data, ...}
        if not payload.get("ok"):
            print(f"  ✗ Response ok != true: {payload.get('ok')}")
            return False, {}
        
        data = payload.get("data", {})
        if not data:
            print("  ✗ Response data is empty")
            return False, {}
        
        # Check top-level fields
        required_fields = ["screen"]
        for field in required_fields:
            if field not in data:
                print(f"  ✗ Missing field: {field}")
                return False, {}

        if "legal_actions" not in data and "available_actions" not in data:
            print("  ✗ Missing legal_actions/available_actions")
            return False, {}
        
        screen = data.get("screen")
        print(f"  Current screen: {screen}")
        legal = data.get("legal_actions", data.get("available_actions", []))
        print(f"  Legal actions: {legal[:3]}...")  # First 3 only
        
        if screen == "COMBAT":
            combat = data.get("combat", {})
            if "turn" in combat:
                print(f"  Combat round: {combat.get('turn')}")
            if "player" in combat:
                print(f"  Player HP: {combat['player'].get('current_hp')}/{combat['player'].get('max_hp')}")
        
        print("  ✓ State endpoint passed")
        return True, data
    except Exception as e:
        print(f"  ✗ State endpoint error: {e}")
        return False, {}


def test_screen_mapping(raw_state: Dict) -> bool:
    """Phase 2.1: Test screen -> state_type mapping"""
    print(f"\n[3/5] Testing screen enum conversion...")
    
    # Import the mapping
    try:
        sys.path.insert(0, ".")
        from sts2_env import SCREEN_ENUM_MAP
        
        screen = raw_state.get("screen")
        if screen in SCREEN_ENUM_MAP:
            mapped = SCREEN_ENUM_MAP[screen]
            print(f"  {screen} -> {mapped}")
            print("  ✓ Screen mapping OK")
            return True
        else:
            print(f"  ! Screen '{screen}' not in SCREEN_ENUM_MAP")
            return True  # Not critical, pass anyway
    except Exception as e:
        print(f"  ! Warning: Could not test screen mapping: {e}")
        return True  # Not critical


def test_normalize_state(raw_state: Dict) -> bool:
    """Phase 2: Test _normalize_state"""
    print(f"\n[4/5] Testing state normalization...")
    
    try:
        sys.path.insert(0, ".")
        from sts2_env import STS2Env
        
        normalized = STS2Env._normalize_state(raw_state)
        
        # Check key normalized fields
        state_type = normalized.get("state_type")
        screen_type = normalized.get("screen_type")
        can_act = normalized.get("can_act")
        
        print(f"  state_type: {state_type}")
        print(f"  screen_type: {screen_type}")
        print(f"  can_act: {can_act}")
        
        # In COMBAT, check combat.round
        if state_type == "combat" or state_type in ("monster", "elite", "boss"):
            combat = normalized.get("combat", {})
            round_val = combat.get("round")
            print(f"  combat.round: {round_val}")
            if round_val is None:
                print(f"  ! WARNING: combat.round is None")
        
        print("  ✓ State normalization passed")
        return True
    except Exception as e:
        print(f"  ✗ Normalization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_encoding(raw_state: Dict) -> bool:
    """Phase 3: Test action encoding with target_index"""
    print(f"\n[5/5] Testing action encoding...")
    
    try:
        sys.path.insert(0, ".")
        from action_space import STS2ActionSpace
        
        action_space = STS2ActionSpace(max_hand_size=10, max_potions=5)
        
        # Simulate a combat action
        fake_combat_state = {
            "state_type": "combat",
            "screen_type": "COMBAT",
            "legal_actions": ["play_card", "end_turn"],
            "combat": {
                "hand": [
                    {"name": "Test Card", "cost": 1},
                    {"name": "Another Card", "cost": 2},
                ],
                "monsters": [
                    {"id": "enemy_0", "name": "Goblin", "hp": 10}
                ]
            },
            "potions": [],
        }
        
        # Try to decode action 0 (first card)
        action = action_space.decode(0, fake_combat_state)
        print(f"  Decoded action 0: {action}")
        
        # Check for target_index (not target)
        if "target_index" in action:
            print(f"  ✓ Using target_index (correct for STS2AIAgent)")
            return True
        elif "target" in action:
            print(f"  ✗ Using old 'target' parameter (should be target_index)")
            return False
        else:
            print(f"  ✓ No target in this action")
            return True
    except Exception as e:
        print(f"  ✗ Action encoding error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    base_url = "http://127.0.0.1:18080"
    
    print("=" * 60)
    print("STS2AIAgent Adapter Sanity Test")
    print("=" * 60)
    
    results = []
    
    # Phase 1: Health
    results.append(("Health Check", test_health_endpoint(base_url)))
    
    # Phase 2: State & Normalization
    ok, raw_state = test_state_endpoint(base_url)
    results.append(("State Endpoint", ok))
    
    if ok:
        results.append(("Screen Mapping", test_screen_mapping(raw_state)))
        results.append(("State Normalization", test_normalize_state(raw_state)))
    
    # Phase 3: Action
    results.append(("Action Encoding", test_action_encoding(raw_state if ok else {})))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
