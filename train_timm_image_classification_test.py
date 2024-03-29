import logging
from ikomia.utils.tests import run_for_test

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::train timm image classification =====")
    input_path = t.get_input(0)
    input_path.set_path(data_dict["datasets"]["classification"]["dataset_classification"])
    for model_name in ["resnet18", "resnext50_32x4d"]:
        params = t.get_parameters()
        params["model_name"] = model_name
        params["epochs"] = "1"
        params["batch_size"] = "1"
        t.set_parameters(params)
        yield run_for_test(t)
