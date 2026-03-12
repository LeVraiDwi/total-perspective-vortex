import argparse
import os
from train import Train
from predict import Predict

def main():
    parser = argparse.ArgumentParser(description="EEG Tool")
    # Global argument: can be used with or without a command
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose mode")

    # Create subparsers. required=True means they MUST pick 'train' or 'predict'
    subparsers = parser.add_subparsers(dest="command", required=False)

    # --- Setup for 'train' ---
    train_parser = subparsers.add_parser('train', help="Train mode")
    train_parser.add_argument('subject', type=int, help="Subject number")
    train_parser.add_argument('run', type=int, help="Run number")

    # --- Setup for 'predict' ---
    predict_parser = subparsers.add_parser('predict', help="Predict mode")
    predict_parser.add_argument('subject', type=int, help="Subject number")
    predict_parser.add_argument('run', type=int, help="Run number")
    
    args = parser.parse_args()
    
    # Logic to route to your functions
    if args.command == 'train':
        assert args.subject <= 109 and args.subject > 0, ("subject should be between 1 and 109")
        assert args.run <= 14  and args.run > 0, ("run should be between 1 and 14")
        
        print(f"Training on Subject {args.subject} run: {args.run}")
        Train(args.subject, args.run, args.verbose)
    
    elif args.command == 'predict':
        assert args.subject <= 109 and args.subject > 0, ("subject should be between 1 and 109")
        assert args.run <= 14  and args.run > 0, ("run should be between 1 and 14")
        print(f"Predicting with {args.model_path}...")
        Predict(args.subject, args.run)
    else:
        experiment = [[3,7,11], [4,8,12], [5,9,13], [6,7,14]]
        expAccuracy = []
        for nbExp in range(len(experiment)):
            exp = experiment[nbExp]
            totalAccuracy = 0
            for subject  in range(1, 109):
                for run  in exp:
                    if not os.path.exists(f"./model/Export_{subject}_{run}.pkl"):
                        Train(subject, run, visualize=False)
                    accuracy = Predict(subject, run)
                    print(f"experiment {nbExp}: subject {subject:}: accuracy = {accuracy}")
                    totalAccuracy += accuracy
            expAccuracy.append(totalAccuracy / 108)
        
        for nbexp in range(len(expAccuracy)):
            exp = experiment[nbexp]
            print(f"exp {nbexp}: accuracy = {exp}")

if __name__ == "__main__":
    # Exemple : Prédire sur le sujet 4, run 12 avec le modèle entraîné précédemment
    main()