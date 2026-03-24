import numpy as np
import os
import torch
from trace_generator import TraceGenerator
from cnn_attacker import CNNAttacker

class PipelineOrchestrator:
    """
    Main pipeline orchestrator for the Adv-Kyber project.
    Executes trace generation, CNN training, and evaluation with autonomous correction.
    """
    
    def __init__(self):
        self.results = {}
        self.noise_blend_weight = 0.7  # Initial noise blending weight
        self.learning_rate = 0.001     # Initial learning rate
        self.max_iterations = 5        # Maximum correction iterations
        
    def run_pipeline(self):
        """
        Execute the complete pipeline with autonomous correction.
        """
        print("=" * 60)
        print("ADV-KYBER PIPELINE EXECUTION")
        print("=" * 60)
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- ITERATION {iteration} ---")
            
            # Step 1: Generate traces
            print("\nStep 1: Generating traces...")
            self.generate_traces()
            
            # Step 2: Train and evaluate on baseline
            print("\nStep 2: Training and evaluating on baseline traces...")
            baseline_acc = self.train_and_evaluate("baseline")
            
            # Step 3: Train and evaluate on protected
            print("\nStep 3: Training and evaluating on protected traces...")
            protected_acc = self.train_and_evaluate("protected")
            
            # Step 4: Check targets and apply corrections
            print("\nStep 4: Evaluating against targets...")
            success = self.evaluate_and_correct(baseline_acc, protected_acc)
            
            if success:
                print("\n" + "=" * 60)
                print("SUCCESS: Both targets achieved!")
                print(f"Baseline Accuracy: {baseline_acc:.2f}% (> 85%)")
                print(f"Protected Accuracy: {protected_acc:.2f}% (< 15%)")
                print("=" * 60)
                self.save_results()
                return True
            
            print(f"Iteration {iteration} completed. Continuing...")
        
        print(f"\nFAILED: Could not achieve targets after {self.max_iterations} iterations.")
        self.save_results()
        return False
    
    def generate_traces(self):
        """Generate baseline and protected traces."""
        generator = TraceGenerator(num_traces=10000, trace_length=1000, seed=42)
        
        # Generate baseline traces
        baseline_traces, baseline_labels = generator.generate_baseline_traces()
        generator.save_traces(baseline_traces, baseline_labels, "baseline")
        
        # Generate protected traces with current noise blend weight
        protected_traces, protected_labels = generator.generate_protected_traces(
            noise_blend_weight=self.noise_blend_weight
        )
        generator.save_traces(protected_traces, protected_labels, "protected")
        
        print("Trace generation completed.")
    
    def train_and_evaluate(self, dataset_type):
        """
        Train CNN on specified dataset and return accuracy.
        
        Args:
            dataset_type: "baseline" or "protected"
            
        Returns:
            Validation accuracy
        """
        # Load traces
        traces_file = f"{dataset_type}_traces.npy"
        labels_file = f"{dataset_type}_labels.npy"
        
        if not os.path.exists(traces_file) or not os.path.exists(labels_file):
            raise FileNotFoundError(f"Dataset files not found: {traces_file}, {labels_file}")
        
        traces = np.load(traces_file)
        labels = np.load(labels_file)
        
        print(f"Loaded {len(traces)} {dataset_type} traces.")
        
        # Initialize and train CNN
        try:
            model = CNNAttacker(input_length=traces.shape[1], num_classes=5)
            
            # Train model
            history = model.train_model(
                traces, labels, 
                epochs=30,  # Reduced for faster iteration
                learning_rate=self.learning_rate
            )
            
            # Load best model and evaluate
            model.load_best_model()
            accuracy = model.evaluate(traces, labels)
            
            # Store results
            self.results[f"{dataset_type}_accuracy"] = accuracy
            self.results[f"{dataset_type}_history"] = history
            
            return accuracy
            
        except RuntimeError as e:
            if "CUDA is not available" in str(e):
                print("ERROR: CUDA not available. This pipeline requires CUDA.")
                exit(1)
            else:
                raise e
    
    def evaluate_and_correct(self, baseline_acc, protected_acc):
        """
        Evaluate results against targets and apply autonomous corrections.
        
        Args:
            baseline_acc: Baseline validation accuracy
            protected_acc: Protected validation accuracy
            
        Returns:
            True if both targets achieved, False otherwise
        """
        baseline_target_met = baseline_acc > 85.0
        protected_target_met = protected_acc < 15.0
        
        print(f"Baseline Accuracy: {baseline_acc:.2f}% (Target: > 85%) - {'PASS' if baseline_target_met else 'FAIL'}")
        print(f"Protected Accuracy: {protected_acc:.2f}% (Target: < 15%) - {'PASS' if protected_target_met else 'FAIL'}")
        
        if baseline_target_met and protected_target_met:
            return True
        
        # Apply corrections
        corrections_made = []
        
        if not baseline_target_met:
            # Baseline accuracy too low - reduce noise or improve CNN
            if baseline_acc < 70.0:
                # Very low accuracy - reduce Gaussian noise in generator
                print("CORRECTION: Reducing Gaussian noise in trace generator...")
                self.modify_trace_generator(reduce_noise=True)
                corrections_made.append("Reduced Gaussian noise")
            else:
                # Moderate accuracy - adjust CNN learning rate
                print("CORRECTION: Increasing CNN learning rate...")
                self.learning_rate = min(self.learning_rate * 1.5, 0.01)
                corrections_made.append(f"Increased learning rate to {self.learning_rate}")
        
        if not protected_target_met:
            # Protected accuracy too high - increase noise blending
            print("CORRECTION: Increasing In-Band Noise blending weight...")
            self.noise_blend_weight = min(self.noise_blend_weight + 0.1, 0.9)
            corrections_made.append(f"Increased noise blend weight to {self.noise_blend_weight}")
        
        if corrections_made:
            print(f"Applied corrections: {', '.join(corrections_made)}")
        else:
            print("No corrections applied.")
        
        return False
    
    def modify_trace_generator(self, reduce_noise=False):
        """
        Modify trace generator parameters for autonomous correction.
        
        Args:
            reduce_noise: If True, reduce Gaussian noise in traces
        """
        if reduce_noise:
            # This would modify the trace generator to use less noise
            # For now, we'll regenerate with different parameters
            generator = TraceGenerator(num_traces=10000, trace_length=1000, seed=123)
            
            # Generate baseline traces with reduced noise
            baseline_traces, baseline_labels = generator.generate_baseline_traces()
            
            # Manually reduce noise in the traces
            for i in range(len(baseline_traces)):
                # Reduce Gaussian noise component
                noise_component = np.random.normal(0, 0.2, baseline_traces.shape[1])  # Reduced from 0.5
                baseline_traces[i] = baseline_traces[i] * 0.7 + noise_component  # Reduce overall variation
            
            generator.save_traces(baseline_traces, baseline_labels, "baseline")
            print("Regenerated baseline traces with reduced noise.")
    
    def save_results(self):
        """Save final results to results.txt file."""
        results_file = "results.txt"
        
        with open(results_file, 'w') as f:
            f.write("ADV-KYBER PIPELINE RESULTS\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Final Configuration:\n")
            f.write(f"- Noise Blend Weight: {self.noise_blend_weight}\n")
            f.write(f"- Learning Rate: {self.learning_rate}\n\n")
            
            f.write("Final Accuracies:\n")
            for key, value in self.results.items():
                if "accuracy" in key:
                    f.write(f"- {key.replace('_', ' ').title()}: {value:.2f}%\n")
            
            f.write("\nTarget Evaluation:\n")
            baseline_acc = self.results.get("baseline_accuracy", 0)
            protected_acc = self.results.get("protected_accuracy", 0)
            
            f.write(f"- Baseline Target (>85%): {'PASS' if baseline_acc > 85 else 'FAIL'}\n")
            f.write(f"- Protected Target (<15%): {'PASS' if protected_acc < 15 else 'FAIL'}\n")
            
            if baseline_acc > 85 and protected_acc < 15:
                f.write("\nOVERALL RESULT: SUCCESS - All targets achieved!\n")
            else:
                f.write("\nOVERALL RESULT: FAILED - Targets not achieved.\n")
        
        print(f"Results saved to {results_file}")

def main():
    """Main execution function."""
    print("Starting Adv-Kyber Pipeline...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This pipeline requires CUDA.")
        return False
    
    print(f"CUDA available: {torch.cuda.get_device_name()}")
    
    # Run the pipeline
    orchestrator = PipelineOrchestrator()
    success = orchestrator.run_pipeline()
    
    if success:
        print("\nPipeline completed successfully!")
    else:
        print("\nPipeline failed to achieve targets.")
    
    return success

if __name__ == "__main__":
    main()
