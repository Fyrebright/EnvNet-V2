import os
import tensorflow as tf
import opts
from training import Trainer

# from keras.utils import plot_model


def memory_settings():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    # (Unsuccessfully) try to prevent NUMA warning 
    # https://github.com/tensorflow/tensorflow/issues/42738#issuecomment-688827946
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


def Main():
    memory_settings()
    opt = opts.parse()
    for split in opt.splits:
        print("+-- Split {} --+".format(split))
        trainer = Trainer(opt, split)
        trainer.Train(opt.nClasses, input_length=opt.inputLength)
        # break;


if __name__ == "__main__":
    Main()
