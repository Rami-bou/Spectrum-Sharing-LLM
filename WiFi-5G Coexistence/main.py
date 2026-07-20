import json
import os
from hardware import RFEnvironmentSimulator
from normalization import telemetry_to_semantic
from graph import app

def run_simulation():

    sim = RFEnvironmentSimulator()
    
    print("=" * 60)
    print("STARTING 5G/WI-FI COEXISTENCE LLM NEGOTIATION SIMULATION")
    print("=" * 60)

    # Simulation scenarios to test
    scenarios = [
        {"name": "Step 1: Normal Operation", "attack": False},
        {"name": "Step 2: Under Starvation Attack (Jamming Idle Gaps)", "attack": True}
    ]

    for scenario in scenarios:
        print(f"\n--- SCENARIO: {scenario['name']} ---")
        
        # 1. Capture raw hardware telemetry from Python simulator
        # Default starting allocation: 5ms / 5ms
        raw_telemetry = sim.step(
            allocation_nru_ms=5.0, 
            allocation_wifi_ms=5.0, 
            icc_agreed=False, 
            attack_active=scenario["attack"]
        )
        print(f"\n[Raw Hardware Telemetry]: {raw_telemetry}")
        
        # 2. Normalize raw telemetry into semantic context
        semantic_telemetry = telemetry_to_semantic(raw_telemetry)
        print(f"[Semantic Telemetry Context]: {json.dumps(semantic_telemetry, indent=2)}")
        
        # 3. Initialize LangGraph State
        initial_state = {
            "telemetry": semantic_telemetry,
            "iteration_count": 0,
            "negotiation_history": [],
            "wifi_latest_proposal": None,
            "nr_u_latest_proposal": None,
            "final_configuration": None,
            "status": "INIT"
        }
        
        # 4. Invoke LangGraph multi-agent loop
        print("\n--> Launching LLM Agent Negotiation Loop...")
        final_graph_output = app.invoke(initial_state)
        
        agreed_config = final_graph_output.get("final_configuration")
        print(f"\n[Final Graph Status]: {final_graph_output.get('status')}")
        print(f"[Agreed Hardware Configuration]: {json.dumps(agreed_config, indent=2)}")
        
        # 5. Apply agreed settings back to physical simulator to verify recovery
        if agreed_config:
            post_negotiation_telemetry = sim.step(
                allocation_nru_ms=agreed_config["allocation_wifi_ms"],
                allocation_wifi_ms=agreed_config["allocation_nru_ms"],
                icc_agreed=agreed_config["icc_agreed"],
                attack_active=scenario["attack"]
            )
            print(f"[Hardware Performance Post-Negotiation]:")
            print(f"  - Wi-Fi Airtime Granted: {post_negotiation_telemetry['wifi_airtime_ms']} ms")
            print(f"  - 5G NR-U Airtime Granted: {post_negotiation_telemetry['nru_airtime_ms']} ms")
            print(f"  - Wi-Fi Backlog Remaining: {post_negotiation_telemetry['wifi_queue_backlog_mb']} MB")
            print(f"  - Interference Silenced: {not post_negotiation_telemetry['interference_detected']}")

if __name__ == "__main__":
    run_simulation()