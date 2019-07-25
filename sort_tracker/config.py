params = dict()

params['num_classes'] = 3

params['dataset'] = '/local/b/cam2/data/HumanBehavior/UCF101-Split'

params['epoch_num'] = 1
params['batch_size'] = 10
params['step'] = 10
params['num_workers'] = 4
params['learning_rate'] = 1e-2
params['momentum'] = 0.9
params['weight_decay'] = 1e-5
params['display'] = 10
params['pretrained'] = './SlowFast/weights/clip_len_64frame_sample_rate_1_checkpoint_49.pth.tar'
params['gpu'] = [0]
params['log'] = 'log'
params['save_path'] = 'UCF101'
params['clip_len'] = 64
params['frame_sample_rate'] = 1
