from state import GraphState

def safety_validator_node(state: GraphState) -> dict:
    """
    Acts as the safety gatekeeper. Checks agent proposals against 
    physical RF constraints before approving configuration changes.
    """
    wifi_prop = state.get("wifi_latest_proposal")
    nru_prop = state.get("nr_u_latest_proposal")
    
    # 1. Structure Verification
    if not wifi_prop or not nru_prop:
        return {"status": "failed_validation"}
        
    try:
        wifi_airtime = float(wifi_prop.get("requested_airtime_ms", 0.0))
        nru_airtime = float(nru_prop.get("requested_airtime_ms", 0.0))
    except (ValueError, TypeError):
        return {"status": "failed_validation"}
        
    wifi_icc = (wifi_prop.get("coordination_strategy") == "ICC_ACTIVE")
    nru_icc = (nru_prop.get("coordination_strategy") == "ICC_ACTIVE")
    
    # 2. Case A: Both agreed on ICC (Spatial Multiplexing)
    # Both radios can operate up to 10.0 ms concurrently without sum-airtime bounds
    if wifi_icc and nru_icc:
        if 0.0 <= wifi_airtime <= 10.0 and 0.0 <= nru_airtime <= 10.0:
            return {
                "final_configuration": {
                    "allocation_wifi_ms": wifi_airtime,
                    "allocation_nru_ms": nru_airtime,
                    "icc_agreed": True
                },
                "status": "accepted"
            }

    # 3. Case B: Standard Time-Division Multiplexing (STANDARD_LBT)
    # Total combined frame airtime cannot exceed 10.0 ms
    if (wifi_airtime + nru_airtime) <= 10.0 and wifi_airtime >= 0.0 and nru_airtime >= 0.0:
        return {
            "final_configuration": {
                "allocation_wifi_ms": wifi_airtime,
                "allocation_nru_ms": nru_airtime,
                "icc_agreed": False
            },
            "status": "accepted"
        }
        
    # 4. Out-of-bounds or invalid airtime combination
    return {"status": "failed_validation"}