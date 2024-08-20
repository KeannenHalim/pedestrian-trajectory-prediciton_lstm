conf_train = {
    'USE_GPU' : 1,
    'NUM_SAMPLE_CHECK' : 5000,
    'PRINT_EVERY' : 5,
    'CHECKPOINT_EVERY' : 1,
    'PATH_CHECKPOINT' : 'checkpoint/exp_15_with_model.pt',
    'RESTORE_FROM_CHECKPOINT' : True,
    'OUTPUT_DIR' : 'checkpoint/',
    'CHECKPOINT_NAME' : 'exp_15',
    #1 model gupta, 2 my model
    'MODEL':2,

    # Dataset option
    'TRAIN_DSET_PATH' : 'datasets_gveii/',
    'VAL_DSET_PATH' : 'datasets_gveii/',
    'DELIM' : '\t',
    'LOADER_NUM_WORKERS' : 4,
    'OBS_LEN' : 8,
    'PRED_LEN' : 8,
    'SKIP' : 6,

    # Optimization
    'BATCH_SIZE' : 64,
    'NUM_EPOCH' : 200,

    # Model Options
    'EMBEDDING_DIM' : 128,
    'MLP_DIM' : 64,
    'ACTIVATION' : 'tanh',

    # Generator Options
    'ENCODER_H_DIM_G' : 128,
    'DECODER_H_DIM_G' : 168,
    'NOISE_DIM' : (8,),
    'NOISE_TYPE' : 'gaussian',
    'CLIPPING_THRESHOLD_G' : 2.0,
    'G_LEARNING_RATE' : 1e-4,
    'G_STEPS' : 1,

    # Pool Net Option
    'BOTTLENECK_DIM' : 32,
    'DEGREE_OF_VISION' : 180,
    # weight [position, velocity, hidden_state]
    'WEIGHT_POOLING_FEATURES' : [1,1,1],

    # Discriminator Options
    'ENCODER_H_DIM_D' : 128,
    'D_LEARNING_RATE' : 1e-3,
    'D_STEPS' : 1,
    'CLIPPING_THRESHOLD_D' : 0,

    # Loss Options
    'L2_LOSS_WEIGHT' : 1,
    'BESK_K' : 20,
}

conf_test={
    'MODEL_PATH' : 'checkpoint/exp_15_with_model.pt',
    'MODEL_NAME' :'exp_15',
    'TEST_DSET_PATH':'datasets_gveii/',
    'EVAL_TYPE':'min',
    'NUM_SAMPLES':20,
    'DSET_NAME':'gveii'
}