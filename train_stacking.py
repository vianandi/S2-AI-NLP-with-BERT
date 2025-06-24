from models.model_factory import ModelFactory
from models.advanced_ensemble import AdvancedEnsemble
import pandas as pd

def main():
    # Load training data
    train_data = pd.read_csv("data/train.csv")
    train_texts = train_data['text'].tolist()
    train_labels = train_data['label'].tolist()
    
    # Load all models
    print("Loading all models...")
    factory = ModelFactory()
    factory.load_all_models()
    models_list, tokenizers_list, model_names = factory.get_model_lists()
    
    # Initialize ensemble
    ensemble = AdvancedEnsemble(models_list, tokenizers_list, model_names)
    
    # Train stacking ensemble
    print("Training stacking ensemble...")
    ensemble.train_stacking_ensemble(train_texts, train_labels)
    
    # Save meta-learner
    ensemble.save_meta_learner("models/meta_learner.pkl")
    print("Meta-learner saved successfully!")

if __name__ == "__main__":
    main()