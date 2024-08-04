MVE_SL_DA_v1:
    DA_weight: 1.4 -> 1.0
    beta: 1.0 -> 0.4 for 75 epochs, then fixed at 0.4s
    lr: 1e-5

MVE_SL_DA_v2:
    DA_weight: 1.4 -> 1.0
    beta: 1.0 -> 0.4 
    lr: 1e-5

MVE_SL_DA_v3:
    DA_weight: 1.4 -> 1.0
    beta: 1.0 -> 0.4 
    lr: 3e-5
    comment: suffered some unexpected problem, model froze

MVE_SL_DA_v3_attempt2:
    DA_weight: 1.4 -> 1.0
    beta: 1.0 -> 0.4 
    lr: 3e-5
    comment: same as v3, trying different seed.


