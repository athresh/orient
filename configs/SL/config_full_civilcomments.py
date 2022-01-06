# Learning setting
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
                         target_resolution=(224, 224),
                         numclasses=2),

              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=20),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="adamw",
                             lr=1e-5,
                             weight_decay=0.01),

              scheduler=dict(type="cosine_annealing",
                             T_max=300),

              dss_args=dict(type="Full",
                            verbose=True),

              train_args=dict(num_epochs=10,
                              device="cuda",
                              print_every=1,
                              results_dir='results/',
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "worst_acc", "time"],
                              return_args=[],
                              visualize=False
                              )
              )
