from evaluate_advanced import AdvancedEvaluator
from models.advanced_ensemble import AdvancedEnsemble
from models.model_factory import ModelFactory
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os

def save_results_as_md_and_csv(results, output_dir="d:\\Pasca Sarjana (S2)\\AI\\TugasAkhir"):
    # Define file paths
    md_filepath = os.path.join(output_dir, "results_final.md")
    csv_filepath = os.path.join(output_dir, "results_final.csv")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save as Markdown
    with open(md_filepath, "w") as md_file:
        md_file.write("<!-- filepath: {} -->\n".format(md_filepath))
        md_file.write(df.to_markdown(index=False))
    print(f"✅ Results saved to: {md_filepath}")
    
    # Save as CSV
    df.to_csv(csv_filepath, index=False)
    print(f"✅ Results saved to: {csv_filepath}")

def format_results_for_final_output(model_results):
    """Format results for final output"""
    formatted_results = []
    for model_name, metrics in model_results.items():
        formatted_results.append({
            "Model": model_name,
            "Accuracy": metrics.get("accuracy", 0.0),
            "F1": metrics.get("f1_score", 0.0),
            "Size_MB": metrics.get("size_mb", 0.0),  # Example placeholder
            "Params_M": metrics.get("params_m", 0.0),  # Example placeholder
            "Inference_s": metrics.get("inference_s", 0.0),  # Example placeholder
            "Throughput": metrics.get("throughput", 0.0),  # Example placeholder
            "Notes": metrics.get("notes", ""),  # Example placeholder
        })
    return formatted_results

def main():
    print("🔍 Starting ADVANCED Model Evaluation...")
    
    # Initialize evaluator
    evaluator = AdvancedEvaluator()
    
    if evaluator.test_data is None:
        print("❌ Failed to load test data")
        return
    
    # Load models using factory
    print("\n🔄 Loading models...")
    factory = ModelFactory()
    models, tokenizers = factory.load_all_models()
    
    if not models:
        print("❌ No models loaded")
        return
    
    # Initialize advanced ensemble
    model_names = list(models.keys())
    ensemble = AdvancedEnsemble(models, tokenizers, model_names)
    
    # Get test data
    texts = evaluator.test_data[evaluator.text_column].fillna('').astype(str).tolist()
    labels = evaluator.test_data[evaluator.label_column].tolist()
    
    print(f"\n🧠 Testing with {len(texts)} samples...")
    
    # Evaluate advanced ensemble methods
    print("\n🧠 Evaluating ADVANCED Ensemble Methods...")
    advanced_results = ensemble.evaluate_all_methods(texts, labels)
    
    # Show results
    print("\n=== ADVANCED ENSEMBLE RESULTS ===")
    for method, metrics in advanced_results.items():
        accuracy = metrics['accuracy']
        f1 = metrics['f1_score']
        print(f"✅ {method}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        # Calculate improvement over baseline
        baseline_accuracy = 0.975  # Current best from previous results
        improvement = accuracy - baseline_accuracy
        if improvement > 0:
            print(f"   🎯 Improvement: +{improvement:.4f} ({improvement*100:.2f}%)")
        else:
            print(f"   📉 Change: {improvement:.4f} ({improvement*100:.2f}%)")
    
    # Find best method
    if advanced_results:
        best_method = max(advanced_results.keys(), key=lambda x: advanced_results[x]['accuracy'])
        best_accuracy = advanced_results[best_method]['accuracy']
        print(f"\n🏆 Best Advanced Method: {best_method} (Accuracy: {best_accuracy:.4f})")
    
    # Compare with baseline models
    print("\n📊 Comparing with Individual Models...")
    baseline_results = evaluator.compare_models()
    
    print("\n✅ Advanced evaluation completed!")
    
        
    formatted_results = format_results_for_final_output(advanced_results)
    save_results_as_md_and_csv(formatted_results)
    
if __name__ == "__main__":
    main()