import warnings
warnings.simplefilter("ignore")
import logging
from functools import partial
import torch
from torchvision.utils import save_image
from subprocess import Popen
from lightning.storage import Path
from lightning.components.python import TracerPythonScript
from lightning.components.serve import ServeGradio
import gradio as gr

logger = logging.getLogger(__name__)


class PyTorchLightningScript(TracerPythonScript):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, raise_exception=True, **kwargs)
        self.last_model_path = None
        self._process = None

    def configure_tracer(self):
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import Callback

        tracer = super().configure_tracer()

        class CollectURL(Callback):

            def __init__(self, work):
                self._work = work

            def on_train_start(self, trainer, *_):
                self._work._process = Popen(
                    f"tensorboard --logdir='{trainer.logger.log_dir}' --host {self._work.host} --port {self._work.port}",
                    shell=True,
                )

        def trainer_pre_fn(self, *args, work=None, **kwargs):
            kwargs['callbacks'].append(CollectURL(work))
            return {}, args, kwargs

        tracer = super().configure_tracer()
        tracer.add_traced(Trainer, "__init__", pre_fn=partial(trainer_pre_fn, work=self))
        return tracer

    def run(self, *args, **kwargs):
        self.script_args += [
            "--trainer.callbacks=ModelCheckpoint",
            "--trainer.callbacks.save_last=true",
        ]
        warnings.simplefilter("ignore")
        logger.info(f"Running train_script: {self.script_path}")
        super().run(*args, **kwargs)

    def on_after_run(self, res):
        lightning_module = res["cli"].trainer.lightning_module
        checkpoint = torch.load(res["cli"].trainer.checkpoint_callback.last_model_path)
        lightning_module.load_state_dict(checkpoint["state_dict"])
        lightning_module.to_torchscript("model_weight.pt")
        self.last_model_path = Path("model_weight.pt")

class ImageServeGradio(ServeGradio):

    inputs = [
        gr.inputs.Slider(0, 1000, label="Seed", default=42),
        gr.inputs.Slider(4, 64, label="Number of Digits", step=1, default=10),
    ]
    outputs = "image"
    examples = [[27, 5], [18, 4], [256, 8], [1337, 35]]

    def __init__(self, cloud_compute, *args, **kwargs):
        super().__init__(*args, cloud_compute=cloud_compute, **kwargs)
        self.model_path = None

    def run(self, model_path):
        self.model_path = model_path
        super().run()

    def predict(self, seed, num_digits):
        torch.manual_seed(seed)
        z = torch.randn(num_digits, 100)
        digits = self.model(z)
        save_image(digits, "digits.png", normalize=True)
        return "digits.png"

    def build_model(self):
        model = torch.load(self.model_path)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model
