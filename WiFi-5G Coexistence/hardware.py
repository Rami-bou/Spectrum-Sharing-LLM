import random

class RFEnvironmentSimulator:
    def __init__(self):
        # Energy Detection (ED) Thresholds in dBm
        self.WIFI_ED_THRESHOLD = -79.0 
        self.NRU_ED_THRESHOLD = -59.0  
        
        # Perceived signal strength (power) at the opposing receiver in dBm
        self.nru_power_at_wifi = -50.0  # 5G is loud, easily trips Wi-Fi's -79 dBm threshold
        self.wifi_power_at_nru = -70.0  # Wi-Fi is quieter, does not trip 5G's -59 dBm threshold
        self.attacker_power_at_wifi = -65.0 

        # Initial Traffic Queues (Megabytes)
        self.wifi_queue_mb = 15.0
        self.nru_queue_mb = 25.0
        
        # Time window per simulation step (10 ms standard radio frame)
        self.frame_duration_ms = 10.0 
        self.data_rate_mbps = 150.0 # Standardized max throughput for this channel

    def calculate_energy(self, nru_transmitting, attacker_active, icc_active):
        """Calculates the physical energy sensed by the Wi-Fi AP."""
        sensed_energy = -100.0 # Baseline noise floor
        
        if nru_transmitting:
            # If the LLMs successfully negotiated Implicit Channel Coordination (ICC), 
            # 5G nullifies its signal at the Wi-Fi AP's location.
            if icc_active:
                sensed_energy = max(sensed_energy, -85.0) 
            else:
                sensed_energy = max(sensed_energy, self.nru_power_at_wifi)
                
        if attacker_active:
            sensed_energy = max(sensed_energy, self.attacker_power_at_wifi)
            
        return sensed_energy

    def step(self, allocation_nru_ms, allocation_wifi_ms, icc_agreed=False, attack_active=False):
        """
        Executes one 10ms timeframe based on the LLM agents' agreed configuration.
        """
        wifi_actual_tx_ms = 0.0
        nru_actual_tx_ms = 0.0
        
        # 1. Evaluate 5G NR-U Transmission
        # 5G checks the channel. Since Wi-Fi's power (-70) is below 5G's threshold (-59), 
        # 5G transmits aggressively regardless of Wi-Fi.
        if self.nru_queue_mb > 0:
            nru_actual_tx_ms = min(allocation_nru_ms, self.frame_duration_ms)
            
        # 2. Evaluate Wi-Fi Transmission (Listen-Before-Talk)
        # Wi-Fi checks the channel energy before transmitting.
        sensed_dbm = self.calculate_energy(nru_transmitting=(nru_actual_tx_ms > 0), 
                                           attacker_active=attack_active, 
                                           icc_active=icc_agreed)
        
        if sensed_dbm > self.WIFI_ED_THRESHOLD:
            # Channel is deemed busy. Wi-Fi backs off. 
            wifi_actual_tx_ms = 0.0
            interference_prob = 1.0 # High probability of interference
        else:
            # Channel is clear (or ICC created a safe blind spot). Wi-Fi transmits.
            wifi_actual_tx_ms = min(allocation_wifi_ms, self.frame_duration_ms)
            if icc_agreed:
                # Simultaneous transmission occurs without harmful interference.
                nru_actual_tx_ms = self.frame_duration_ms 
            interference_prob = 0.0
            
        # 3. Update Queues based on airtime
        wifi_cleared_mb = (wifi_actual_tx_ms / 1000.0) * self.data_rate_mbps
        nru_cleared_mb = (nru_actual_tx_ms / 1000.0) * self.data_rate_mbps
        
        self.wifi_queue_mb = max(0.0, self.wifi_queue_mb - wifi_cleared_mb)
        self.nru_queue_mb = max(0.0, self.nru_queue_mb - nru_cleared_mb)
        
        # 4. Generate Telemetry for the LangGraph State
        return {
            "sensed_channel_energy_dbm": sensed_dbm,
            "wifi_airtime_ms": wifi_actual_tx_ms,
            "nru_airtime_ms": nru_actual_tx_ms,
            "wifi_queue_backlog_mb": round(self.wifi_queue_mb, 2),
            "nru_queue_backlog_mb": round(self.nru_queue_mb, 2),
            "interference_detected": bool(interference_prob > 0.5),
            "attack_active": attack_active
        }