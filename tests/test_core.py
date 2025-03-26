import pytest
from pathlib import Path
import pandas as pd
from data_processor.core import DataProcessor, ProcessingConfig

@pytest.fixture
def sample_config():
    return ProcessingConfig(
        input_columns=["col1", "col2"],
        output_columns=["col1_processed", "col2_processed"],
        operations={"operation1": "sum"}
    )

@pytest.fixture
def sample_data(tmp_path):
    # Create a temporary CSV file for testing
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4, 5, 6]
    })
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)
    return file_path

def test_data_processor_initialization(sample_config):
    processor = DataProcessor(sample_config)
    assert processor.config == sample_config

def test_process_file(sample_config, sample_data):
    processor = DataProcessor(sample_config)
    result = processor.process_file(sample_data)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == sample_config.input_columns