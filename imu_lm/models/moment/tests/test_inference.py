''' 
Ensure MOMENT can run an inference forward pass on each task it can perform.
Model structures are based on those given in the tutorials folder. 
A test is considered successful if a TimeSeriesOutput is generated for each model inference pass, without an error.
'''

from momentfm import MOMENTPipeline # This imports the MOMENT model
import torch
from pprint import pprint

# Create models of each type
def create_classification_model(num_channels, n_classes):
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-small", 
        model_kwargs={
            "task_name": "classification",
            "n_channels": num_channels,
            "num_class": n_classes
        },
    )   
    model.init()
    model.eval()
    return model

def create_forecasting_model():
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': 192,
            'head_dropout': 0.1,
            'weight_decay': 0,
            'freeze_encoder': True, # Freeze the patch embedding layer
            'freeze_embedder': True, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
        },
        # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
    )
    model.init()
    model.eval()
    return model

def create_anomaly_detection_model():
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={"task_name": "reconstruction"},  # For anomaly detection, we will load MOMENT in `reconstruction` mode
        # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
    )
    model.init()
    model.eval()
    return model

def create_imputation_model():
    # this is the same as anomaly detection 
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={'task_name': 'reconstruction'} # For imputation, we will load MOMENT in `reconstruction` mode
        # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
    )
    model.init()
    model.eval()
    return model

# Classification
def test_classification():
    model = create_classification_model(num_channels=4, n_classes=2)
    x = torch.randn(1, 4, 512)  
    output = model(x_enc=x)
    # print("Classification output:")
    # pprint(output)
    assert output is not None, "Model returned None value"

# Forecasting
def test_forecasting():
    model = create_forecasting_model()
    x = torch.randn(16, 1, 512)
    output = model(x_enc=x)
    # print("Forecasting output:")
    # pprint(output)
    assert output is not None, "Model returned None value"

# Anomaly Detection
def test_anomaly_detection():
    model = create_anomaly_detection_model()
    x = torch.randn(16, 1, 512)
    output = model(x_enc=x)
    # print("Anomaly detection output:")
    # pprint(output)
    assert output is not None, "Model returned None value"

# Imputation
def test_imputation():
    model = create_imputation_model()
    x = torch.randn(16, 1, 512)
    output = model(x_enc=x)
    # print("Imputation output:")
    # pprint(output)
    assert output is not None, "Model returned None value"
