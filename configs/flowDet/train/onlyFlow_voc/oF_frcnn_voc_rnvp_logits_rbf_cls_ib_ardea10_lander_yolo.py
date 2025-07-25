_base_ = ['../../feat_ext/faster_rcnn_r50_fpn_voc0712OS_clsLogits_ardea10_lander.py']
flow_params=dict(
    random_seed = 3,
	num_cls = 2, ### <<<<<<<<<<---------- NUM CS CLASSES +1 ---------->>>>>>>>>>
	flow_type = "rnvp", # "rnvp", "residual", "nsf_ar"
	input_dim = 2, ### <<<<<<<<<<---------- NUM CS CLASSES +1 ---------->>>>>>>>>>
	blocks = 16,
	hidden_dim = 64,
	num_layers_st_net = 4,
	permutation = "affine", # "affine", "permute", "lu_permute"
	coupling = "affine", # String, "affine", "additive" 
	scale_map = "exp", # "sigmoid", "exp", "sigmoid_inv"
	init_zeros = True, # Bool, flag whether to initialize last layer of NNs with zeros
	actnorm = True,
	dropout = False,
	prior_type = "resampled_cls_ib", # "gauss", "gaussian_mixture", "resampled"
	######## base distribution specific ########
	base_learn_mean_var = False,
	############# for IB loss #############
	ib_beta=30., # 1., 30., 50
    ib_sigma=0.5, # 0.1, 0.5, 1.0
	############# resampled base distribution specific #############
	base_T = 100, # 50, 100, 500
	base_eps = 0.05, # 0.01, 0.05, 0.1
	base_a_hidden_layers = 3,
	base_a_hidden_units = 128,
	base_dropout = 0.1, 
	base_init_zeros = True,
	######## training specific ########
	num_class = 2, ### <<<<<<<<<<---------- NUM CS CLASSES +1 ---------->>>>>>>>>>
	batch_size = 1024,
	max_epochs = 400,
	val_freq = 5,
	early_stop_tolerance=0, # number of iter to stop when the model doesn't improve
	lr = 1e-4, # 5e-4
	weight_decay = 1e-5, # 1e-5, 1e-8
	max_norm = None, # 1e1, 5e1, 1e2, None
	optm = "Adam", # SGD, Adam, "Adamax"
)