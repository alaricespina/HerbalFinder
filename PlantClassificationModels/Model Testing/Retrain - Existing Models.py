from TensorflowPreTrainedModels import (
    EfficientNetV2B3Model,
    EfficientNetV2LModel,
    EfficientNetV2MModel,
    EfficientNetV2SModel,
    ResNet50V2Model,
    ResNet101V2Model,
    ResNet152V2Model,
    InceptionV3Model,
    VGG19Model
)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings 
warnings.filterwarnings("ignore")

Models = [
    #EfficientNetV2B3Model(),
    EfficientNetV2LModel(),
    EfficientNetV2MModel(),
    EfficientNetV2SModel(),
    #ResNet50V2Model(),
    ResNet101V2Model(),
    ResNet152V2Model(),
    #InceptionV3Model(), # -> Requires 75 x 75
    #VGG19Model() # -> Requires 75 x 75
]

if __name__ == "__main__":
    for model in Models:
        print(model.model_name)
        model.prepare_data(image_size = (75, 75), dataset_mode="scikit")
        model.prepare_model()
        model.fit_data(desired_size = 32, epochs = 10, ckpt_metric = "val_accuracy", ckpt_mode = "max")
        print("Done Training")




