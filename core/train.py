from tools import MultiItemAverageMeter
from core.video_train import train_video_steps

def train(base, loaders, config,scaler):

    base.set_train()
    meter = MultiItemAverageMeter()
    if config.dataset=='vcm':
        loader = loaders.get_train_loader()
    elif config.dataset=='bupt':
        loader = loaders.train_iter

    meter = train_video_steps(loader,base,meter,scaler)


    return meter.get_val(), meter.get_str()
