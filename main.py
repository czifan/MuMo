import torch 
from config.opts import get_config
from train_and_eval import Trainer
from utils.utils import *
import os 
from utils.print_utils import *
import json 

if __name__ == '__main__':
    opts, parser = get_config()
    torch.set_default_dtype(torch.float32)

    logger = build_logging(os.path.join(opts.save_dir, "log.log"))
    printer = logger.info

    print_log_message('Arguments', printer)
    printer(json.dumps(vars(opts), indent=4, sort_keys=True))

    trainer = Trainer(opts=opts, printer=printer)
    trainer.run()