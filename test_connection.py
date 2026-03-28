import argparse
import json
import requests


def test_connection(host: str = "127.0.0.1", port: int = 18080) -> bool:
    base_url = f"http://{host}:{port}"
    health_endpoint = f"{base_url}/health"
    session_endpoint = f"{base_url}/api/v1/session/state"
    fallback_endpoint = f"{base_url}/state"
    print(f"Testing: {health_endpoint}")

    try:
        health_resp = requests.get(health_endpoint, timeout=5)
        health_resp.raise_for_status()
        health_payload = health_resp.json()
        print(f"  health.ok: {health_payload.get('ok')}")

        print(f"Testing: {session_endpoint}")
        resp = requests.get(session_endpoint, timeout=5)
        if resp.status_code >= 400:
            print(f"  session/state unavailable ({resp.status_code}), fallback to /state")
            resp = requests.get(fallback_endpoint, timeout=5)
        resp.raise_for_status()
        payload = resp.json()
        state = payload.get("data", payload)

        phase = state.get("phase", "?")
        screen = state.get("screen", state.get("state_type", "?"))
        can_act = state.get("can_act", "?")
        legal = state.get("legal_actions", state.get("available_actions", []))

        print("Connection OK")
        print(f"  phase: {phase}")
        print(f"  screen/state_type: {screen}")
        print(f"  can_act: {can_act}")
        print(f"  legal_actions: {legal}")
        print("  state preview:")
        print(json.dumps(state, ensure_ascii=False)[:600])
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test STS2AIAgent connection")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    ok = test_connection(host=args.host, port=args.port)
    raise SystemExit(0 if ok else 1)
