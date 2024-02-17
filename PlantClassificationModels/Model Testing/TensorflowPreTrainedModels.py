from tensorflow.keras.applications.efficientnet_v2 import (
        EfficientNetV2B3,
        EfficientNetV2L,
        EfficientNetV2M,
        EfficientNetV2S
    )


from tensorflow.keras.applications.resnet_v2 import (
        ResNet50V2,
        ResNet101V2,
        ResNet152V2
    )

from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.applications.vgg19 import VGG19

from BaseClassifier import BaseClassifier


class EfficientNetV2B3Model(BaseClassifier):
    def __init__(self, model_name = "EfficientNetV2B3MODEL"):
        super().__init__(model_name = model_name)
    
    def prepare_model(self):
        self.model = EfficientNetV2B3(include_top = True,
                                      weights = None,
                                      input_shape = self.image_size,
                                      classes = self.num_unique_classes)
       
        self.model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])

class EfficientNetV2LModel(BaseClassifier):
    def __init__(self, model_name = "EfficientNetV2LMODEL"):
        super().__init__(model_name = model_name)
    
    def prepare_model(self):
        self.model = EfficientNetV2L(include_top = True,
                                      weights = None,
                                      input_shape = self.image_size,
                                      classes = self.num_unique_classes)      
        
        self.model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])

class EfficientNetV2MModel(BaseClassifier):
    def __init__(self, model_name = "EfficientNetV2MMODEL"):
        super().__init__(model_name = model_name)
    
    def prepare_model(self):
        self.model = EfficientNetV2M(include_top = True,
                                      weights = None,
                                      input_shape = self.image_size,
                                      classes = self.num_unique_classes)
       
        self.model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])

class EfficientNetV2SModel(BaseClassifier):
    def __init__(self, model_name = "EfficientNetV2SMODEL"):
        super().__init__(model_name = model_name)
    
    def prepare_model(self):
        self.model = EfficientNetV2S(include_top = True,
                                      weights = None,
                                      input_shape = self.image_size,
                                      classes = self.num_unique_classes)
        
        self.model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])

class ResNet50V2Model(BaseClassifier):
    def __init__(self, model_name = "ResNet50V2MODEL"):
        super().__init__(model_name = model_name)
    
    def prepare_model(self):
        self.model = ResNet50V2(include_top = True,
                                      weights = None,
                                      input_shape = self.image_size,
                                      classes = self.num_unique_classes)
       
        self.model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])

class ResNet101V2Model(BaseClassifier):
    def __init__(self, model_name = "ResNet101V2MODEL"):
        super().__init__(model_name = model_name)
    
    def prepare_model(self):
        self.model = ResNet101V2(include_top = True,
                                      weights = None,
                                      input_shape = self.image_size,
                                      classes = self.num_unique_classes)

        self.model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])

class ResNet152V2Model(BaseClassifier):
    def __init__(self, model_name = "ResNet152V2MODEL"):
        super().__init__(model_name = model_name)
    
    def prepare_model(self):
        self.model = ResNet152V2(include_top = True,
                                      weights = None,
                                      input_shape = self.image_size,
                                      classes = self.num_unique_classes)
        
        self.model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])

class InceptionV3Model(BaseClassifier):
    def __init__(self, model_name = "InceptionV3MODEL"):
        super().__init__(model_name = model_name)
    
    def prepare_model(self):
        self.model = InceptionV3(include_top = True,
                                      weights = None,
                                      input_shape = self.image_size,
                                      classes = self.num_unique_classes)
    
        self.model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])

class VGG19Model(BaseClassifier):
    def __init__(self, model_name = "VGG19MODEL"):
        super().__init__(model_name = model_name)
    
    def prepare_model(self):
        self.model = VGG19(include_top = True,
                                      weights = None,
                                      input_shape = self.image_size,
                                      classes = self.num_unique_classes)
     
        self.model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])
