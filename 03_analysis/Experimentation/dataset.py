"""Load the dataset."""
from aif360.datasets import AdultDataset, BinaryLabelDataset


def load_dataset():
    """Load the dataset."""
    adult = AdultDataset()
    # change the dataset according to your need.
    data = adult.convert_to_dataframe()[0] 
    binary_data = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=data,
        label_names=["income-per-year"],
        protected_attribute_names=["race"],
    )
    return binary_data.split([0.7], shuffle=True)
