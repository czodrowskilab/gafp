import pickle as pkl
from glob import iglob
from os import path
from typing import Union

from django.conf import settings

from ctrainlib.models import Classifier, Regressor


class ModelManager:
    """
    A instance of this class contains all loaded and available models and FPCDBs during
    runtime of the django server. All models and DBs can be (re-)loaded with the instance method
    *load_all_models()*.
    """
    __slots__ = ('models',)

    def __init__(self):
        self.models = {}

    def add_model(self, model: Union[Classifier, Regressor]) -> None:
        """
        Adds a model to the instance dictionary.

        Prints a warning, if a model with the same name is already loaded.

        Parameters
        ----------
        model : Union[Classifier, Regressor]
            A Classifier or Regressor instance
        """
        if model.name not in self.models:
            self.models[model.name] = model
        else:
            print(f'WARNING: Model "{model.name}" already exists, skipped')

    def load_all_models(self) -> None:
        """
        (Re-)loads all models using the path specified in the settings.
        """
        self.models.clear()
        for model_file in iglob(path.join(settings.MODEL_FOLDER, '*.model')):
            with open(model_file, 'rb') as m:
                self.add_model(pkl.load(m))


model_manager = ModelManager()
