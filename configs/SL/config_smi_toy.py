# Learning setting
config = dict(setting="SL",

              dataset=dict(name="toy_smi",
                           datadir="../data/conv_shift_clf_2",
                           feature="dss",
                           type="image"),

              dataloader=dict(shuffle=True,
                              batch_size=20,
                              pin_memory=True),

              model=dict(architecture='logreg_net',
                         type='pre-defined',
                         numclasses=2,
                         input_dim=2),

              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=20),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=True),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.01,
                             weight_decay=5e-4),

              scheduler=dict(type="cosine_annealing",
                             T_max=300),

              dss_args=dict(type="SMI",
                            fraction=0.1,
                            select_every=2,
                            kappa=0,
                            linear_layer=False,
                            selection_type='Supervised',
                            smi_func_type='fl2mi',
                            valid=True,
                            optimizer='NaiveGreedy',
                            metric='cosine',
                            eta=1,
                            stopIfZeroGain=False,
                            stopIfNegativeGain=False,
                            verbose=True),

              train_args=dict(num_epochs=300,
                              device="cuda",
                              print_every=10,
                              results_dir='results/',
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )
