from postprocessing.visualization import visualize_prediction
from dataset_generation.data_feeder import TrainImageGenerator
from dataset_generation.augmenter import NoAgumenter

if __name__ == "__main__":
    generator = TrainImageGenerator("../../datasets/micro/annotations.csv", "../../datasets/micro",
                                    batch_size=1, augumenter=NoAgumenter())

    img, label = generator.generate_sample(0)
    visualize_prediction(img, label)

