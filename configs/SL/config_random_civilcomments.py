config = dict(setting="SL",

              dataset=dict(name="civilcomments",
                           datadir="../data",
                           feature="dss",
                           type="image"),

              dataloader=dict(shuffle=True,
                              batch_size=16,
                              pin_memory=True),

              model=dict(architecture='distilbert',
                         type='pre-defined',
                         numclasses=2),

              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=5),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=True),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.01,
                             weight_decay=5e-4),

              scheduler=dict(type="cosine_annealing",
                             T_max=300),

              dss_args=dict(type="Random",
                                fraction=0.1,
                                select_every=2,
                                kappa=0),

              train_args=dict(num_epochs=10,
                              device="cuda",
                              print_every=1,
                              results_dir='results/',
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "worst_acc", "time"],
                              return_args=[]
                              )
              )