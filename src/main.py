import argparse
import os
import mne
from train import Train
from predict import Predict

def main():
    parser = argparse.ArgumentParser(description="EEG Tool")
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
    mne.set_log_level('WARNING')
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
        experiment = [3,4,5,6,7,8]
        expAccuracy = []
        nb_subject = range(1, 110)
        for nbExp in range(len(experiment)):
            run = experiment[nbExp]
            totalAccuracy = 0
            for subject  in nb_subject:
                if not os.path.exists(f"./model/Export_{subject}_{run}.pkl"):
                    Train(subject, run, visualize=False, cross_val=False)
                accuracy = Predict(subject, run, verbose=False)
                print(f"experiment {nbExp}: subject {subject:}: accuracy = {accuracy}")
                totalAccuracy += accuracy
            expAccuracy.append(totalAccuracy / len(nb_subject))
        
        for nbexp in range(len(expAccuracy)):
            exp = expAccuracy[nbexp]
            print(f"exp {nbexp}: accuracy = {exp}")

if __name__ == "__main__":
    main()