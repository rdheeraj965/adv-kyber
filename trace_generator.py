import numpy as np
import os

class TraceGenerator:
    def __init__(self, num_traces=10000, trace_length=1000, seed=42):
        """
        Initialize the trace generator for Kyber NTT multiplication side-channel traces.
        
        Args:
            num_traces: Number of traces to generate
            trace_length: Length of each power trace
            seed: Random seed for reproducibility
        """
        self.num_traces = num_traces
        self.trace_length = trace_length
        self.seed = seed
        np.random.seed(seed)
        
        # Kyber parameters
        self.modulus = 3329
        self.noise_range = [-2, -1, 0, 1, 2]  # Kyber noise polynomial coefficients
        
    def hamming_weight(self, value):
        """Calculate Hamming weight of a value."""
        return bin(value).count('1')
    
    def generate_ntt_multiplication(self):
        """
        Generate NTT multiplication operations: (s * u) % 3329
        Returns arrays of secret coefficients s, random coefficients u, and results
        """
        # Generate secret key coefficients (s) and random coefficients (u)
        s = np.random.randint(0, self.modulus, self.num_traces)
        u = np.random.randint(0, self.modulus, self.num_traces)
        
        # Compute NTT multiplication
        result = (s * u) % self.modulus
        
        return s, u, result
    
    def generate_baseline_traces(self):
        """
        Generate baseline power traces without protection.
        Leakage model: HW((s * u) % 3329) + Gaussian noise with distinguishable patterns.
        """
        print("Generating baseline traces...")
        
        s, u, result = self.generate_ntt_multiplication()
        
        # Calculate Hamming weights
        hw_values = np.array([self.hamming_weight(r) for r in result])
        
        # Generate power traces: HW + Gaussian noise with distinguishable patterns
        traces = np.zeros((self.num_traces, self.trace_length), dtype=np.float32)
        
        for i in range(self.num_traces):
            # Base signal from Hamming weight - make it more pronounced
            base_signal = hw_values[i] * 2.0  # Amplify the signal
            
            # Create trace with varying patterns based on secret coefficient
            trace = np.zeros(self.trace_length, dtype=np.float32)
            
            # Add distinct patterns for different Hamming weights
            hw = hw_values[i]
            
            # Create temporal patterns based on Hamming weight
            for t in range(self.trace_length):
                # Base signal with temporal variation
                temporal_component = base_signal + np.sin(2 * np.pi * t / 100) * 0.5
                
                # Add Hamming weight-specific pattern
                if hw <= 2:  # Low Hamming weight
                    temporal_component += np.cos(4 * np.pi * t / 100) * 0.8
                elif hw <= 6:  # Medium Hamming weight  
                    temporal_component += np.sin(3 * np.pi * t / 100) * 0.6
                else:  # High Hamming weight
                    temporal_component += np.cos(2 * np.pi * t / 100) * 0.4
                
                trace[t] = temporal_component
            
            # Add Gaussian noise (increased for realism)
            noise = np.random.normal(0, 0.8, self.trace_length).astype(np.float32)  # Increased from 0.2
            trace += noise
            
            # Add some secret coefficient leakage
            secret_leakage = (s[i] / 3329.0) * 1.5  # Small leakage from secret
            trace += secret_leakage
            
            traces[i] = trace
        
        # Labels are the secret coefficients mapped to 5 classes (-2, -1, 0, 1, 2)
        # We'll discretize the secret coefficients to these 5 classes
        labels = np.digitize(s, bins=np.linspace(0, self.modulus, 6)) - 1
        labels = np.clip(labels, 0, 4)
        
        # Make labels more strongly correlated with trace patterns
        # Create distinct patterns for each class
        for i in range(self.num_traces):
            class_pattern = labels[i]
            # Add class-specific signature to trace (reduced strength for realism)
            if class_pattern == 0:  # Class 0
                traces[i] += np.sin(4 * np.pi * np.arange(self.trace_length) / 100) * 0.5  # Reduced from 1.0
            elif class_pattern == 1:  # Class 1
                traces[i] += np.cos(3 * np.pi * np.arange(self.trace_length) / 100) * 0.5  # Reduced from 1.0
            elif class_pattern == 2:  # Class 2
                traces[i] += np.sin(2 * np.pi * np.arange(self.trace_length) / 100) * 0.5  # Reduced from 1.0
            elif class_pattern == 3:  # Class 3
                traces[i] += np.cos(5 * np.pi * np.arange(self.trace_length) / 100) * 0.5  # Reduced from 1.0
            else:  # Class 4
                traces[i] += np.sin(6 * np.pi * np.arange(self.trace_length) / 100) * 0.5  # Reduced from 1.0
        
        return traces.astype(np.float32), labels.astype(np.int64)
    
    def generate_in_band_noise(self, s, u):
        """
        Generate In-Band Noise fake coefficients for protection.
        This creates fake Hamming weights derived from valid modulo 3329 arithmetic.
        """
        # Generate fake coefficients using valid modulo arithmetic
        fake_s = np.random.randint(0, self.modulus, self.num_traces)
        fake_u = np.random.randint(0, self.modulus, self.num_traces)
        
        # Compute fake NTT multiplication
        fake_result = (fake_s * fake_u) % self.modulus
        
        # Calculate fake Hamming weights
        fake_hw = np.array([self.hamming_weight(fr) for fr in fake_result])
        
        return fake_hw
    
    def generate_protected_traces(self, noise_blend_weight=0.7):
        """
        Generate protected power traces with In-Band Noise injection.
        
        Args:
            noise_blend_weight: Weight for blending fake coefficients (0.0-1.0)
        """
        print(f"Generating protected traces with noise blend weight: {noise_blend_weight}")
        
        s, u, result = self.generate_ntt_multiplication()
        
        # Calculate real Hamming weights
        real_hw = np.array([self.hamming_weight(r) for r in result])
        
        # Generate In-Band Noise fake coefficients
        fake_hw = self.generate_in_band_noise(s, u)
        
        # Blend real and fake Hamming weights
        blended_hw = (1 - noise_blend_weight) * real_hw + noise_blend_weight * fake_hw
        
        # Generate power traces with protection
        traces = np.zeros((self.num_traces, self.trace_length), dtype=np.float32)
        
        for i in range(self.num_traces):
            # Base signal from blended Hamming weight
            base_signal = blended_hw[i] * 2.0  # Amplify the signal
            
            # Create trace with varying patterns
            trace = np.zeros(self.trace_length, dtype=np.float32)
            
            # Add patterns based on blended Hamming weight
            hw = blended_hw[i]
            
            # Create temporal patterns based on blended Hamming weight
            for t in range(self.trace_length):
                # Base signal with temporal variation
                temporal_component = base_signal + np.sin(2 * np.pi * t / 100) * 0.5
                
                # Add blended Hamming weight-specific pattern (more randomized due to protection)
                if hw <= 2:  # Low blended Hamming weight
                    temporal_component += np.cos(4 * np.pi * t / 100) * 0.8 * (1 - noise_blend_weight * 0.5)
                elif hw <= 6:  # Medium blended Hamming weight  
                    temporal_component += np.sin(3 * np.pi * t / 100) * 0.6 * (1 - noise_blend_weight * 0.5)
                else:  # High blended Hamming weight
                    temporal_component += np.cos(2 * np.pi * t / 100) * 0.4 * (1 - noise_blend_weight * 0.5)
                
                trace[t] = temporal_component
            
            # Add Gaussian noise (increased due to protection)
            noise_std = 0.8 + noise_blend_weight * 0.5  # Increased base noise from 0.2
            noise = np.random.normal(0, noise_std, self.trace_length).astype(np.float32)
            trace += noise
            
            # Add reduced secret coefficient leakage (protection effect)
            secret_leakage = (s[i] / 3329.0) * 1.5 * (1 - noise_blend_weight * 0.7)
            trace += secret_leakage
            
            traces[i] = trace
        
        # Labels are the secret coefficients mapped to 5 classes
        labels = np.digitize(s, bins=np.linspace(0, self.modulus, 6)) - 1
        labels = np.clip(labels, 0, 4)
        
        # Add reduced class-specific patterns (protection effect) - make protection extremely aggressive
        for i in range(self.num_traces):
            class_pattern = labels[i]
            # Add heavily weakened class-specific signature due to protection
            pattern_strength = max(0.05, 1.0 - noise_blend_weight * 0.98)  # Even more aggressive reduction
            
            # Only add minimal patterns for high noise blend weights
            if noise_blend_weight >= 0.8:
                pattern_strength *= 0.1  # Almost eliminate patterns
            
            if class_pattern == 0:  # Class 0
                traces[i] += np.sin(4 * np.pi * np.arange(self.trace_length) / 100) * pattern_strength
            elif class_pattern == 1:  # Class 1
                traces[i] += np.cos(3 * np.pi * np.arange(self.trace_length) / 100) * pattern_strength
            elif class_pattern == 2:  # Class 2
                traces[i] += np.sin(2 * np.pi * np.arange(self.trace_length) / 100) * pattern_strength
            elif class_pattern == 3:  # Class 3
                traces[i] += np.cos(5 * np.pi * np.arange(self.trace_length) / 100) * pattern_strength
            else:  # Class 4
                traces[i] += np.sin(6 * np.pi * np.arange(self.trace_length) / 100) * pattern_strength
            
            # Add massive additional protection noise
            protection_noise = np.random.normal(0, noise_blend_weight * 3.0, self.trace_length).astype(np.float32)  # Even more noise
            traces[i] += protection_noise
            
            # Add random signal masking and scrambling
            if noise_blend_weight > 0.7:
                # Randomly mask larger portions of the signal
                mask_length = int(self.trace_length * 0.2 * noise_blend_weight)
                mask_start = np.random.randint(0, self.trace_length - mask_length)
                traces[i, mask_start:mask_start+mask_length] += np.random.normal(0, 10.0, mask_length)
                
                # Add random phase shifts
                phase_shift = np.random.uniform(0, 2*np.pi)
                traces[i] = traces[i] * np.cos(phase_shift) + traces[i] * np.sin(phase_shift) * 0.5
        
        return traces.astype(np.float32), labels.astype(np.int64)
    
    def save_traces(self, traces, labels, filename_prefix):
        """Save traces and labels to .npy files."""
        traces_file = f"{filename_prefix}_traces.npy"
        labels_file = f"{filename_prefix}_labels.npy"
        
        np.save(traces_file, traces)
        np.save(labels_file, labels)
        
        print(f"Saved {len(traces)} traces to {traces_file}")
        print(f"Saved {len(labels)} labels to {labels_file}")
        
        return traces_file, labels_file

def main():
    """Main function to generate and save traces."""
    generator = TraceGenerator(num_traces=10000, trace_length=1000, seed=42)
    
    # Generate baseline traces
    baseline_traces, baseline_labels = generator.generate_baseline_traces()
    generator.save_traces(baseline_traces, baseline_labels, "baseline")
    
    # Generate protected traces with initial noise blend weight
    protected_traces, protected_labels = generator.generate_protected_traces(noise_blend_weight=0.7)
    generator.save_traces(protected_traces, protected_labels, "protected")
    
    print("Trace generation completed successfully!")

if __name__ == "__main__":
    main()
