# Learning setting

config = dict(setting="SL",

              dataset=dict(name="toy_da5",
                           datadir="../data/toy_da5",
                           feature="dss",
                           type="image",
                           daParams=dict(source_domains=["d0"],
                                                      target_domains=["d1"])),

              dataloader=dict(shuffle=True,
                              batch_size=16,
                              pin_memory=True),


              model=dict(architecture='ThreeLayerNet',
                         type='pre-defined',
                         numclasses=2,
                         input_dim=2,
                         # hidden_units=16,
                         h1=16,
                         h2=16,
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

              dss_args=dict(type="GradMatch",
                            fraction=0.1,
                            select_every=20,
                            lam=0.5,
                            selection_type='PerClassPerGradient',
                            v1=True,
                            valid=True,
                            kappa=0,
                            eps=1e-100,
                            linear_layer=True,
                            verbose=True),

              train_args=dict(num_epochs=40,
                              device="cuda",
                              print_every=1,
                              results_dir="results/",
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "trn_acc", "time"],
                              visualize=True,
                              return_args=[]
                              )
              )