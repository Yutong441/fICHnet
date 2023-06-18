config = {}
# directory
config["root"] = "data/"
config["save_prefix"] = "results/DWI/DWI"
config["archive"] = "logs/"
config["mask_root"] = "/data/rosanderson/ich_ct/processed2/"
# directory and prefix for all the files to be saved, including the model
# weight, loss history and metric history

# learning rate
config["num_epochs"] = 150
config["lr"] = 0.0001
config["L2"] = 0
config["L1"] = 0
config["regularised_layers"] = []
config["step_size"] = 10
config["gamma"] = 0.5
config["resume_training"] = False
config["pretrained"] = False
config["train_weight"] = True
config["old_save_path"] = "None"

# data
config["data_folder"] = [""]
# must be a list, if multiple paths are supplied, the first one is assumed to
# be the training/internal validation ones, all others are external validation

# config["label_dir"] = ["labels"]
# similar to config ["data_folder"], each label must correspond to the data

config["select_channels"] = None
config["select_depths"] = 18
config["select_num"] = None
config["num_workers"] = 2
config["pin_memory"] = True
config["batch_size"] = 4
config["device"] = "cuda:0"  # "cpu" or "cuda:0"
config["common_shape"] = [128, 128]
config["downsize"] = None
config["transform"] = "default"
config["depth_as_channel"] = False
config["random_sampler"] = False

# model
config["input_channels"] = 1
config["predict_class"] = 1
config["add_sigmoid"] = None
config["times_max"] = 1
config["model_type"] = "resnet18"
config["loss_type"] = "logistic_hazard"
config["sigma"] = 0.5  # optional
config["prob_fun"] = "sigmoid"  # optional; can be "sigmoid" or "gauss"
config["decoder"] = "0conv_2lin_0pool"
config["dropout"] = 0
config["step_linear"] = 2
config["add_STN"] = False
config["init_chan"] = 16
config["output_features"] = None

# evaluation
config["outcome_col"] = "mRS"
config["index_col"] = "anoID"
config["all_metrics"] = None
config["eval_every"] = 2
config["initial_metric"] = 0
config["metric"] = "concordance"
config["tolerance"] = 20
config["better"] = "pos"  # either "pos" or "neg"
config["pos_label"] = 2
config["interval"] = 182.25   # only useful for discrete time survival analysis

# visualise
config["gcam"] = "encoder.blocks.2"  # "features.32" for vgg13_bn
config["save_gradcam"] = True
config["level"] = 10
config["show_num"] = 10
config["x_features"] = ["Age", "Sex"]
config["y_features"] = ["mRS_dc"]
