# Learning setting

config = dict(setting="SL",

              dataset=dict(name="toy_da",
                           datadir="../data/toy_da",
                           feature="dss",
                           type="image",
                           daParams=dict(source_domains=["d0"],
                                                      target_domains=["d1"])),

              dataloader=dict(shuffle=True,
                              batch_size=16,
                              pin_memory=True),


              model=dict(architecture='TwoLayerNet',
                         type='pre-defined',
                         numclasses=2,
                         input_dim=2,
                         hidden_units=3,
                         pretrained=False),

              ckpt=dict(is_load=False,
                        is_save=True,
                        dir="results/",
                        save_every=20),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.001,
                             weight_decay=5e-4),

              scheduler=dict(type="cosine_annealing",
                             T_max=300),

              dss_args=dict(type="SMI",
                            query_size=100,
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

              train_args=dict(num_epochs=20,
                              device="cuda",
                              print_every=1,
                              results_dir="results/",
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              visualize=True,
                              return_args=[]
                              )
              )