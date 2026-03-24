import numpy as np
import os

class TraceGenerator:
    def __init__(self, num_traces=10000, trace_length=1000, seed=42):
        self.num_traces = num_traces
        self.trace_length = trace_length
        self.seed = seed
        np.random.seed(seed)
        
        self.modulus = 3329
        # The specific clock cycle where the multiplication happens
        self.operation_index = 500 
        
    def hamming_weight(self, value):
        return bin(value).count('1')
    
    def generate_traces(self, use_defense=False, blend_weight=1.0):
        print(f"Generating {self.num_traces} traces. Defense Active: {use_defense}")
        
        # s = secret key coefficients, u = random ciphertexts
        s = np.random.randint(0, self.modulus, self.num_traces)
        u = np.random.randint(0, self.modulus, self.num_traces)
        
        # Real mathematical operation
        result = (s * u) % self.modulus
        real_hw = np.array([self.hamming_weight(r) for r in result])
        
        if use_defense:
            # IN-BAND NOISE: Generate fake but mathematically valid HW
            fake_s = np.random.randint(0, self.modulus, self.num_traces)
            fake_u = np.random.randint(0, self.modulus, self.num_traces)
            fake_result = (fake_s * fake_u) % self.modulus
            fake_hw = np.array([self.hamming_weight(fr) for fr in fake_result])
            
            # Blend the true leakage with the fake leakage
            target_hw = (1.0 - blend_weight) * real_hw + (blend_weight * fake_hw)
        else:
            target_hw = real_hw

        traces = np.zeros((self.num_traces, self.trace_length), dtype=np.float32)
        
        for i in range(self.num_traces):
            # 1. Base electronic noise (Standard Deviation = 0.8)
            trace = np.random.normal(0, 0.8, self.trace_length).astype(np.float32)
            
            # 2. The Realistic Power Spike (with hardware capacitor decay)
            hw = target_hw[i]
            trace[self.operation_index] += hw * 2.0
            trace[self.operation_index + 1] += hw * 1.0
            trace[self.operation_index + 2] += hw * 0.5
            
            # 3. Temporal Jitter (Clock desynchronization: shift left/right by up to 5 cycles)
            jitter = np.random.randint(-5, 6)
            trace = np.roll(trace, jitter)
            
            traces[i] = trace
            
        # Group the secret coefficients into 5 distinct classes for the CNN to predict
        labels = np.digitize(s, bins=np.linspace(0, self.modulus, 6)) - 1
        labels = np.clip(labels, 0, 4)
        
        return traces, labels

    def save_data(self):
        # Generate and save Baseline
        b_traces, b_labels = self.generate_traces(use_defense=False)
        np.save("baseline_traces.npy", b_traces)
        np.save("baseline_labels.npy", b_labels)
        
        # Generate and save Protected (1.0 blend completely replaces the signal)
        p_traces, p_labels = self.generate_traces(use_defense=True, blend_weight=1.0)
        np.save("protected_traces.npy", p_traces)
        np.save("protected_labels.npy", p_labels)
        print("Data generation complete and saved to disk.")

if __name__ == "__main__":
    generator = TraceGenerator()
    generator.save_data()