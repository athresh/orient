# Learning setting

config = dict(setting="SL",

              dataset=dict(name="officehome",
                           datadir="../data/Officehome",
                           feature="dss",
                           type="image",
                           customImageListParams=dict(image_list_folder="../data/Officehome_list",
                                                      source_domains=["Real_World"],
                                                      target_domains=["Clipart"],
                                                      num_classes=65),
                           preprocess=dict(crop=224,
                                           resize=256,
                                           normalizer_mean=[0.485, 0.456, 0.406],
                                           normalizer_std=[0.229, 0.224, 0.225]),),

              dataloader=dict(shuffle=True,
                              batch_size=20,
                              pin_memory=True),


              model=dict(architecture='SiameseResNet50',
                         type='pre-defined',
                         numclasses=65,
                         pretrained=True),

              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/Officehome/',
                        save_every=20),

              loss=dict(type='ccsa',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.001,
                             weight_decay=1e-4),

              scheduler=dict(type="none",
                             T_max=305),

              dss_args=dict(type="SMI",
                            fraction=0.1,
                            select_every=20,
                            query_size=100,
                            kappa=0,
                            linear_layer=False,
                            selection_type='Supervised',
                            smi_func_type='fl2mi',
                            valid=True,
                            optimizer='NaiveGreedy',
                            similarity_criterion='gradient',
                            metric='cosine',
                            eta=1,
                            stopIfZeroGain=False,
                            stopIfNegativeGain=False,
                            verbose=True),

              train_args=dict(num_epochs=300,
                              ft_epochs=5,
                              train_type='ft',
                              device="cuda",
                              alpha=0.25,
                              print_every=1,
                              results_dir='results/',
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )
