from training.Trainer import Trainer
import argparse as ap
import sys

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Train SSD Network")
    parser.add_argument("-b", '--batch_size', default=8, type=int, help="Number of samples per batch. Defaults to 8")
    parser.add_argument("-ip", '--images_path', required=True, type=str, help="Path where training images are. Required")
    parser.add_argument("-ap", '--annotations_path', required=True, type=str, help="Path to annotations file. Required")
    parser.add_argument("-mp", '--model_path', required=True, type=str,
                        help="Path where trained model will be saved")
    parser.add_argument("-a", "--alpha", default=1e-4, type=float, help="Training alpha rate")
    parser.add_argument("-e", "--epochs", default=10, type=int, help="Number of training epochs")
    parser.add_argument("-f", "--frequent_save", help="Save model on every epoch", action="store_true")
    parser.add_argument("-ts", "--time_save", default=0, type=float, help="Save model every N minutes")
    parser.add_argument("-tk", "--time_kill", default=0, type=float, help="Kill training after N minutes")

    args = parser.parse_args()

    trainer = Trainer(batch_size=args.batch_size, alpha=args.alpha, images_path=args.images_path,
                      annotations_path=args.annotations_path, model_path=args.model_path, max_epoch=args.epochs,
                      save_every_epoch=args.frequent_save, save_after_minutes=args.time_save, 
                      kill_after_minutes=args.time_kill)

