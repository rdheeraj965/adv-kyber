import trace_generator
import cnn_attacker

def run_pipeline():
    print("=== Adv-Kyber Pipeline ===")
    
    # 1. Generate Data
    generator = trace_generator.TraceGenerator(100000)
    generator.save_data()
    
    # 2. Test Vulnerability (Baseline)
    print("\n[TEST 1] Testing CNN against Baseline Traces...")
    baseline_acc = cnn_attacker.train_and_evaluate("baseline_traces.npy", "baseline_labels.npy", 20)
    print(f"Single-Trace Accuracy: {baseline_acc:.2f}%")
    
    # 3. Test Defense (Protected)
    print("\n[TEST 2] Testing CNN against Protected Traces (In-Band Noise)...")
    protected_acc = cnn_attacker.train_and_evaluate("protected_traces.npy", "protected_labels.npy", 20)
    print(f"Single-Trace Accuracy: {protected_acc:.2f}%")
    
    print("\n=== FINAL RESULTS ===")
    print(f"Vulnerability Target (>95%): {'PASS' if baseline_acc > 95 else 'FAIL'}")
    print(f"Defense Target (~20% Random Guessing): {'PASS' if protected_acc <= 25 else 'FAIL'}")

if __name__ == '__main__':
    run_pipeline()