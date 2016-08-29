# Usage

## Forward
### prepare configuration file
check the example of [forward.cfg](https://github.com/torms3/kaffe/blob/master/python/forward.cfg.example)

output patch size for TitanX pascal 
J-Net: 32 x 158 x 158
MSF: 12 x 150 x 150

### run script

    python forward.py 0 path/of/forward.cfg

0 is GPU ID, but this is different from the ID you can know from `nvidia-smi`. `caffe device_query` will give you the IDs. Basically faster GPU gets assigned lower number.