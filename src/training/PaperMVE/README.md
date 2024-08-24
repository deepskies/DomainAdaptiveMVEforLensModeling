v1:
    DA Weight = 1.4
    beta = 1 -> 0.5 (150 ep), 0.5 (100 ep)
    batch size = 128
    lr = 3e-5
    seed = 13

v2:
    DA Weight = 1.4 -> 1.0 (150 ep) 1.0 (100 ep)
    beta = 1 -> 0.5 (150 ep), 0.5 (100 ep)
    batch size = 128
    lr = 3e-5
    seed = 13

v3:
    DA Weight = 1.4 -> 1.0 (250 ep)
    beta = 1 -> 0.5 (150 ep), 0.5 (100 ep)
    batch size = 128
    lr = 3e-5
    seed = 13

v4:
    DA Weight = 1.4 
    beta = 1 -> 0.5 (150 ep), 0.5 (50 ep), 0.0 (50 ep)
    batch size = 128
    lr = 3e-5 (200 ep), step 3e-5 / 25 (100 ep)
    seed = 13