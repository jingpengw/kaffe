#!/usr/bin/env python
__doc__ = """

Inference.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import shutil
import configparser
import h5py
import sys
import time
import numpy as np

import config
from DataProvider.python.forward import ForwardScanner

# Forward config.
cfg = config.ForwardConfig(sys.argv[1])

# Create an inference net.
#znet_path = "/import/basil-mip1/out_280000_c8"
znet_path = cfg.get('forward', 'model')
net = cfg.net(znet_path)

# Create net spec.
net_spec = dict()
'''for i in net.inputs:
    net_spec[i] = net.blobs[i].data.shape[-3:]'''
net_spec = {'input': (20, 256, 256)}
# Create data provider.
dp = cfg.get_data_provider(net_spec)

# Scan params.
scan_list   = eval(cfg.get('forward','scan_list'))
scan_params = eval(cfg.get('forward','scan_params'))
save_prefix = cfg.get('forward','save_prefix')

# Create scan spec.
scan_spec = dict()
#THIS NEEDS TO BE CHANGED FOR EVERY NET UGLY HACK BOHAHA
scan_spec = {'output': (3, 20, 256, 256)}

# Forward scan.
for dataset in dp.datasets:

    idx = dataset.params['dataset_id']
    print(('Forward scan dataset {}'.format(idx)))

    # Create ForwardScanner for the current dataset.
    fs = ForwardScanner(dataset, scan_spec, params=scan_params)

    # Scan loop.
    ins = fs.pull()  # Fetch initial inputs.
    while ins is not None:
        start = time.time()
        # Run forward pass.
        outs = dict()
        outs["output"] = net.forward(ins["input"])[0][...]
        if np.isnan(np.sum(outs["output"])):
            print("Nan busted!")
            import pdb; pdb.set_trace()
        # Extract output data.
        fs.push(outs)    # Push current outputs.
        # Elapsed time.
        print(('Elapsed: {}'.format(time.time() - start)))
        ins = fs.pull()  # Fetch next inputs.

def forward(config_file, device_id = 0):
    # Initialize.
    caffe.set_device( device_id )
    caffe.set_mode_gpu()

    # Forward config.
    cfg = config.ForwardConfig( config_file )

    # Create an inference net.
    net = cfg.net()

    # Create net spec.
    net_spec = dict()
    for i in net.inputs:
        net_spec[i] = net.blobs[i].data.shape[-3:]

    # Create data provider.
    dp = cfg.get_data_provider(net_spec)

    # Scan params.
    scan_list   = eval(cfg.get('forward', 'scan_list'))
    scan_params = eval(cfg.get('forward', 'scan_params'))
    save_prefix = cfg.get('forward','save_prefix')

    # Create scan spec.
    scan_spec = dict()
    for i in scan_list:
        scan_spec[i] = net.blobs[i].data.shape[-4:]

    # Forward scan.
    for dataset in dp.datasets:
        idx = dataset.params['dataset_id']
        print('Forward scan dataset {}'.format(idx))

        # Create ForwardScanner for the current dataset.
        fs = ForwardScanner(dataset, scan_spec, params=scan_params)

        # Scan loop.
        ins = fs.pull()  # Fetch initial inputs.
        while ins is not None:
            start = time.time()
            # Set inputs.
            for k, v in ins.iteritems():
                shape = (1,) + v.shape
                net.blobs[k].reshape(*shape)
                net.blobs[k].data[0,...] = v
            # Run forward pass.
            net.forward()
            # Extract output data.
            outs = dict()
            for k in scan_spec.iterkeys():
                outs[k] = net.blobs[k].data[0,...]
            fs.push(outs)    # Push current outputs.
            # Elapsed time.
            print('Elapsed: {}'.format(time.time() - start))
            ins = fs.pull()  # Fetch next inputs.
    return fs.outputs, save_prefix

if __name__ == '__main__':
    outputs, save_prefix = forward( sys.argv[2], device_id=int(sys.argv[1]) )
    # Save as file.
    for key in list(fs.outputs.data.keys()):
        start = time.time()
        fname = '{}_dataset{}_{}.h5'.format(save_prefix, idx+1, key)
        print(('Save {}...'.format(fname)))
        f = h5py.File(fname)
        output = outputs.get_data(key)
        f.create_dataset('/main', data=output)
        f.close()
        print(('File Saving: {}'.format(time.time() - start)))
        try:
            print ("Deleting temp folder...")
            shutil.rmtree('/tmp/{}'.format(os.getpid()));
        except:
            print ("Couldn't remove the temp folder.")
        else:
            print ("Temp folder deleted.")
