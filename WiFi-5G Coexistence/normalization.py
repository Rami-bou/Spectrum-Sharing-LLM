from typing import Dict, Any

def telemetry_to_semantic(raw_telemetry: Dict[str, Any]) -> Dict[str, str]:
    """
    Translates raw hardware metrics into human-readable, semantic descriptions
    that the LLMs can easily parse and reason about.
    """
    semantic_state = {}

    # 1. Energy to Congestion Mapping
    energy = raw_telemetry.get("sensed_channel_energy_dbm", -100)
    if energy >= -59:
        semantic_state["channel_status"] = "CRITICAL_CONGESTION (Over -59 dBm threshold)"
    elif energy >= -79:
        semantic_state["channel_status"] = "MODERATE_CONGESTION (Over -79 dBm threshold)"
    else:
        semantic_state["channel_status"] = "CLEAR"

    # 2. Queue Backlog Normalization
    wifi_queue = raw_telemetry.get("wifi_queue_backlog_mb", 0)
    nru_queue = raw_telemetry.get("nru_queue_backlog_mb", 0)
    
    def evaluate_queue(queue_size: float) -> str:
        if queue_size > 20: return "HIGH_BACKLOG"
        if queue_size > 5: return "MODERATE_BACKLOG"
        return "LOW_OR_EMPTY"

    semantic_state["wifi_queue_status"] = evaluate_queue(wifi_queue)
    semantic_state["nr_u_queue_status"] = evaluate_queue(nru_queue)
    
    # 3. Add explicit contextual flags
    attack_status = raw_telemetry.get("attack_active", False)
    semantic_state["environment_warning"] = (
        "POSSIBLE_STARVATION_ATTACK_DETECTED" if attack_status else "NORMAL_OPERATION"
    )

    return semantic_state

def proposal_to_hardware(wifi_proposal: Dict[str, Any], nru_proposal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes the structured JSON output from the LLMs (after they agree) 
    and converts it back into safe, clamped hardware parameters.
    """
    
    # We extract the agreed upon airtime, defaulting to a safe split if missing
    wifi_ms = float(wifi_proposal.get("requested_airtime_ms", 5.0))
    nru_ms = float(nru_proposal.get("requested_airtime_ms", 5.0))
    
    # Enforce strict physical constraints (e.g., total frame cannot exceed 10ms)
    total_requested = wifi_ms + nru_ms
    if total_requested > 10.0:
        # Scale them down proportionally if the LLMs hallucinated > 10ms
        scaling_factor = 10.0 / total_requested
        wifi_ms *= scaling_factor
        nru_ms *= scaling_factor

    # Determine if ICC (Implicit Channel Coordination) was successfully agreed upon
    # Both agents must explicitly agree to "ICC_ACTIVE" in their proposals
    wifi_icc = wifi_proposal.get("coordination_strategy") == "ICC_ACTIVE"
    nru_icc = nru_proposal.get("coordination_strategy") == "ICC_ACTIVE"
    icc_agreed = wifi_icc and nru_icc

    return {
        "allocation_wifi_ms": round(wifi_ms, 2),
        "allocation_nru_ms": round(nru_ms, 2),
        "icc_agreed": icc_agreed
    }