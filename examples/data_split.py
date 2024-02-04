"""Example of splitting the dataset into training and testing sets."""

from pathlib import Path

from sklearn.model_selection import train_test_split

from potdata.io.adaptor import YAMLCollectionAdaptor
from potdata.schema.datapoint import DataCollection


def data_split(filename, test_ratio=0.1):
    # Read the data from the YAML file using the provided YAMLCollectionAdaptor
    yaml_adaptor = YAMLCollectionAdaptor()
    data_collection = yaml_adaptor.read(filename)

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(
        data_collection.data_points, test_size=test_ratio, random_state=42
    )

    # Create DataCollections for the training and testing sets
    train_collection = DataCollection(data_points=train_data, label="Training Set")
    test_collection = DataCollection(data_points=test_data, label="Testing Set")

    # Convert filename to a Path object
    filename_path = Path(filename)

    # Write the training and testing sets to separate YAML files using YAMLCollectionAdaptor
    output_train_filename = filename_path.with_stem(filename_path.stem + "_train")
    output_test_filename = filename_path.with_stem(filename_path.stem + "_test")

    yaml_adaptor.write(train_collection, output_train_filename)
    yaml_adaptor.write(test_collection, output_test_filename)


if __name__ == "__main__":
    # Replace 'dataset.yaml' with the actual path to your dataset file
    data_split("dataset.yaml", test_ratio=0.1)
