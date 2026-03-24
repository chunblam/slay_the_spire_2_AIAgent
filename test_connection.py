"""
scripts/test_connection.py

测试与 STS2AIAgent Mod 的连接
运行前请确保游戏已启动并加载 Mod
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json


def test_connection(host="127.0.0.1", port=18080):
    """
    STS2AIAgent API: GET /health
    若 Mod 使用非默认端口，修改 port（当前项目默认 18080）。
    """
    base_url = f"http://{host}:{port}"
    path = "/health"
    print(f"🔌 测试连接: {base_url}{path}")

    try:
        resp = requests.get(f"{base_url}{path}", timeout=5)
        resp.raise_for_status()
        state = resp.json()
        if isinstance(state, dict) and state.get("status") == "error":
            print(f"❌ API 错误: {state.get('error', state)}")
            return False

        data = state.get("data", state) if isinstance(state, dict) else {}
        st = data.get("status", "?")
        print("✅ 连接成功!")
        print(f"   service_status: {st}")
        state_resp = requests.get(f"{base_url}/api/v1/session/state", timeout=5)
        state_resp.raise_for_status()
        state_data = state_resp.json().get("data", {})
        floor = state_data.get("floor", "?")
        gold = state_data.get("gold", "?")
        phase = state_data.get("phase", "?")
        print(f"   楼层: {floor}")
        print(f"   金币: {gold}")
        print(f"   phase: {phase}")

        print("\n📋 完整状态 (前500字符):")
        print(json.dumps(state_data, ensure_ascii=False)[:500])
        return True

    except requests.ConnectionError:
        print("❌ 连接失败! 请确认:")
        print("   1. 杀戮尖塔2 已运行")
        print("   2. STS2AIAgent Mod 已在游戏中启用")
        print("   3. 游戏设置中开启了 Mod")
        print(f"   4. 端口与路径正确（API: GET {path}，当前默认端口 18080）")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试 STS2AIAgent API 连接")
    parser.add_argument("--host", default="127.0.0.1", help="主机名，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=18080, help="Mod HTTP 端口，默认 18080")
    args = parser.parse_args()
    success = test_connection(host=args.host, port=args.port)
    sys.exit(0 if success else 1)