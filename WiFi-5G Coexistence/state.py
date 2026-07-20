from typing import List, Dict, Any, Optional, TypedDict, Literal, Tuple, Annotated
import operator

class SystemState(TypedDict):
    telemetry: Dict[str, Any]
    
    iteration_count: int
    
    negotiation_history: Annotated[List[str], operator.add]
    
    wifi_latest_proposal: Optional[Dict[str, Any]]
    nr_u_latest_proposal: Optional[Dict[str, Any]]
    
    final_configuration: Optional[Dict[str, Any]]
    
    status: str