"""
Network Simulation Environment with Nonlinear Queue Dynamics,
Chaotic Traffic Generators, and Self-Similar Fractal Packet Flows.

Authors: Pawel Kuras (Kozminski), Dominik Strzalka (PRz), Faustyna Misiura (UP Sanok)
For submission to Chaos: An Interdisciplinary Journal of Nonlinear Science.
"""

import numpy as np

class TrafficGenerator:
    """Traffic generation supporting Poisson, Deterministic Chaotic Maps, and Self-Similar (Pareto ON/OFF)."""
    def __init__(self, mode="poisson", mean_rate=50.0, hurst=0.8, chaos_r=3.89, seed=None):
        self.mode = mode
        self.mean_rate = mean_rate
        self.hurst = hurst
        self.chaos_r = chaos_r
        self.rng = np.random.default_rng(seed)
        
        # Chaotic state (Logistic / Henon coupled)
        self.chaos_x = self.rng.uniform(0.1, 0.9)
        self.chaos_y = self.rng.uniform(0.1, 0.9)
        
        # Pareto ON/OFF parameters for self-similarity
        # H = (3 - alpha) / 2 => alpha = 3 - 2*H
        self.alpha = max(1.05, min(1.95, 3.0 - 2.0 * hurst))
        self.num_sources = 32
        self.source_states = self.rng.choice([0, 1], size=self.num_sources)
        self.source_remaining = np.ones(self.num_sources, dtype=int)
        self.source_rate = self.mean_rate / (self.num_sources * 0.5)

    def _sample_pareto(self, size):
        # Sample Pareto duration with shape alpha and scale xm=1
        u = self.rng.uniform(1e-4, 1.0, size=size)
        return np.maximum(1, np.floor((1.0 / u) ** (1.0 / self.alpha))).astype(int)

    def step(self):
        """Generate packet arrival count for the current discrete time step."""
        if self.mode == "poisson":
            return self.rng.poisson(self.mean_rate)
            
        elif self.mode == "chaos":
            # Logistic map with periodic perturbation to prevent lockup
            self.chaos_x = self.chaos_r * self.chaos_x * (1.0 - self.chaos_x)
            # Occasional infinitesimal noise injection keeps it on chaotic attractor
            if self.rng.random() < 0.01:
                self.chaos_x = (self.chaos_x + 0.01) % 1.0
            rate = self.mean_rate * (0.3 + 1.4 * self.chaos_x)
            # Instantaneous burst
            return int(np.round(rate))
            
        elif self.mode == "self_similar":
            # Update Pareto ON/OFF sources
            self.source_remaining -= 1
            expired = self.source_remaining <= 0
            if np.any(expired):
                count = np.sum(expired)
                # Flip state: 0 -> 1, 1 -> 0
                self.source_states[expired] = 1 - self.source_states[expired]
                self.source_remaining[expired] = self._sample_pareto(count)
            active = np.sum(self.source_states)
            rate = active * self.source_rate
            return self.rng.poisson(max(0.1, rate))
        else:
            return self.rng.poisson(self.mean_rate)


class NetworkNode:
    """Network Switch with finite buffer, nonlinear AQM drop policy, and service capacity."""
    def __init__(self, node_id, buffer_capacity=100.0, service_rate=60.0, aqm_threshold=60.0, aqm_slope=0.15):
        self.node_id = node_id
        self.capacity = buffer_capacity
        self.service_rate = service_rate
        self.aqm_threshold = aqm_threshold
        self.aqm_slope = aqm_slope
        self.queue = 0.0
        
        # Telemetry history for Lyapunov exponent calculation
        self.history = []

    def drop_probability(self):
        """Nonlinear sigmoidal drop probability modeling RED/CoDel AQM."""
        diff = self.queue - self.aqm_threshold
        exponent = np.clip(self.aqm_slope * diff, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-exponent))

    def update(self, arrivals):
        """Update queue state for one time step and return (departures, dropped)."""
        p_drop = self.drop_probability()
        accepted_arrivals = arrivals * (1.0 - p_drop)
        dropped_arrivals = arrivals * p_drop
        
        # Buffer overflow check
        available = self.capacity - self.queue
        if accepted_arrivals > available:
            dropped_arrivals += (accepted_arrivals - available)
            accepted_arrivals = max(0.0, available)
            
        self.queue += accepted_arrivals
        
        # Service / departures
        departures = min(self.queue, self.service_rate)
        self.queue -= departures
        
        self.history.append(self.queue)
        return departures, dropped_arrivals


class NetworkSystem:
    """Coupled 2-Switch Network (Edge Switch S1 -> Core Switch S2) with cross-traffic and TCP feedback."""
    def __init__(self, config_vector, traffic_mode="poisson", mean_rate=52.0, hurst=0.8, seed=42):
        self.traffic_mode = traffic_mode
        self.mean_rate = mean_rate
        self.hurst = hurst
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Decode 14-bit binary configuration vector p in {0, 1}^14
        # Features:
        # f0-f2: Switch 1 Service Rate modifier (50 to 75)
        # f3-f5: Switch 1 AQM threshold (40 to 80)
        # f6-f7: Switch 1 AQM slope (0.05 to 0.35)
        # f8-f10: Switch 2 Service Rate modifier (55 to 80)
        # f11-f13: Switch 2 AQM threshold (45 to 85)
        p = np.array(config_vector, dtype=int)
        
        s1_mu = 50.0 + 3.5 * (p[0]*4 + p[1]*2 + p[2])
        s1_th = 40.0 + 5.5 * (p[3]*4 + p[4]*2 + p[5])
        s1_sl = 0.05 + 0.10 * (p[6]*2 + p[7])
        
        s2_mu = 55.0 + 3.5 * (p[8]*4 + p[9]*2 + p[10])
        s2_th = 45.0 + 5.5 * (p[11]*4 + p[12]*2 + p[13])
        s2_sl = 0.12
        
        self.s1 = NetworkNode("S1_Edge", buffer_capacity=120.0, service_rate=s1_mu, aqm_threshold=s1_th, aqm_slope=s1_sl)
        self.s2 = NetworkNode("S2_Core", buffer_capacity=140.0, service_rate=s2_mu, aqm_threshold=s2_th, aqm_slope=s2_sl)
        
        self.traffic_gen = TrafficGenerator(mode=traffic_mode, mean_rate=mean_rate, hurst=hurst, seed=seed)
        self.cross_traffic_gen = TrafficGenerator(mode=traffic_mode, mean_rate=15.0, hurst=hurst, seed=seed+100)
        
        # TCP feedback delay buffer
        self.feedback_delay = 5
        self.recent_loss = [0.0] * self.feedback_delay

    def simulate(self, steps=300):
        """Simulate the network for `steps` time units and record telemetry."""
        total_arrivals = 0.0
        total_departures = 0.0
        total_drops = 0.0
        total_delay = 0.0
        
        q1_series = []
        q2_series = []
        
        for t in range(steps):
            # Delayed TCP congestion response: drop rate suppresses incoming rate
            avg_past_loss = np.mean(self.recent_loss)
            pacing_factor = max(0.4, 1.0 - 1.8 * avg_past_loss)
            
            raw_arrivals = self.traffic_gen.step()
            arrivals_1 = raw_arrivals * pacing_factor
            
            dep_1, drop_1 = self.s1.update(arrivals_1)
            
            # Switch 2 gets departures from S1 plus cross-traffic
            cross = self.cross_traffic_gen.step()
            arrivals_2 = dep_1 + cross
            dep_2, drop_2 = self.s2.update(arrivals_2)
            
            total_arrivals += (arrivals_1 + cross)
            total_departures += dep_2
            total_drops += (drop_1 + drop_2)
            total_delay += (self.s1.queue + self.s2.queue)
            
            q1_series.append(self.s1.queue)
            q2_series.append(self.s2.queue)
            
            instant_loss = (drop_1 + drop_2) / max(1.0, arrivals_1 + cross)
            self.recent_loss.pop(0)
            self.recent_loss.append(instant_loss)
            
        throughput = total_departures / steps
        avg_delay = total_delay / (steps * 2.0)
        drop_rate = total_drops / max(1.0, total_arrivals)
        
        # Compute Largest Lyapunov Exponent (LLE) from queue trajectory
        q1_arr = np.array(q1_series[50:]) # burn-in
        lle = self._estimate_lyapunov(q1_arr)
        
        # Nonlinear Utility calculation
        # Utility in [0, 1]
        # High throughput is good, high delay and loss are bad, positive Lyapunov exponent penalized
        # (stability preference)
        norm_t = np.clip(throughput / 65.0, 0.0, 1.0)
        norm_d = np.clip(avg_delay / 80.0, 0.0, 1.0)
        norm_l = np.clip(drop_rate / 0.15, 0.0, 1.0)
        chaos_penalty = 0.15 if lle > 0.05 else 0.0
        
        utility = 0.50 * norm_t + 0.25 * (1.0 - norm_d) + 0.25 * (1.0 - norm_l) - chaos_penalty
        utility = float(np.clip(utility, 0.0, 1.0))
        
        metrics = {
            "throughput": float(throughput),
            "delay": float(avg_delay),
            "drop_rate": float(drop_rate),
            "lyapunov": float(lle),
            "utility": float(utility),
            "q1_trajectory": q1_series,
            "q2_trajectory": q2_series
        }
        return metrics

    def _estimate_lyapunov(self, series, delay=2, embed_dim=3):
        """Estimate largest Lyapunov exponent via Rosenstein-style phase space divergence."""
        N = len(series)
        if N < 60:
            return 0.0
        # Phase space embedding
        M = N - (embed_dim - 1) * delay
        if M <= 10:
            return 0.0
        vectors = np.empty((M, embed_dim))
        for i in range(embed_dim):
            vectors[:, i] = series[i * delay : i * delay + M]
            
        # Find nearest neighbours excluding temporal autocorrelation
        divergences = []
        for i in range(min(M - 10, 50)):
            dists = np.linalg.norm(vectors - vectors[i], axis=1)
            # exclude temporal window
            dists[max(0, i-5):min(M, i+6)] = np.inf
            nn = np.argmin(dists)
            d0 = dists[nn]
            if d0 > 1e-5:
                # check distance at step + 4
                if i + 4 < M and nn + 4 < M:
                    d_future = np.linalg.norm(vectors[i+4] - vectors[nn+4])
                    divergences.append(np.log(max(1e-5, d_future / d0)) / 4.0)
                    
        return float(np.mean(divergences)) if divergences else 0.0
