MVE_SL_DA_v1:
    DA_weight: 1.4 -> 1.0
    beta: 1.0 -> 0.4 for 75 epochs, then fixed at 0.4s
    lr: 1e-5
    batch_size: 32

MVE_SL_DA_v2:
    DA_weight: 1.4 -> 1.0
    beta: 1.0 -> 0.4 
    lr: 1e-5
    batch_size: 32
    
MVE_SL_DA_v3:
    DA_weight: 1.4 -> 1.0
    beta: 1.0 -> 0.4 
    lr: 3e-5
    batch_size: 32
    comment: suffered some unexpected problem, model froze

MVE_SL_DA_v3_attempt2:
    DA_weight: 1.4 -> 1.0
    beta: 1.0 -> 0.4 
    lr: 3e-5
    batch_size: 32
    comment: same as v3, trying different seed. Didn't help.

MVE_SL_DA_v4:
    DA_weight: 1.4 -> 1.0
    beta: 1.0 -> 0.4 
    lr: 3e-5
    batch_size: 64
    comment: higher batch size with LR expected to make gradient stabler. Great improvement.

MVE_SL_DA_v5:
    DA_weight: 1.4 -> 1.0
    beta: 1.0 -> 0.4 
    lr: 1e-5
    batch_size: 32
    comment: added regression layer, check performance improvement or worsens. Uncertainties are much smaller.

MVE_SL_DA_v6:
    DA_weight: 1.4 -> 1.0
    beta: 1.0 -> 0.4 
    lr: 3e-5
    batch_size: 256
    comment: go crazy w/ batch size with LR expected to make gradient stabler. Didn't learn variance so well.


MVE_SL_DA_v7:
    DA_weight: 1.4 -> 1.0
    beta: 1.0 -> 0.0
    lr: 3e-5
    batch_size: 64
    epochs: 250
    comment: Combine knowledge for v1 NN model.

MVE_SL_DA_v8:
    DA_weight: 1.4 -> 1.0
    beta: 1.0 -> 0.0
    lr: 3e-5
    batch_size: 64
    epochs: 250
    regression layer: true
    comment: Combine knowledge for v2 NN model.
