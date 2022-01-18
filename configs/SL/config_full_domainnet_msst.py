# Learning setting

config = dict(setting="SL",

              dataset=dict(name="domainnet",
                           datadir="../data/domainnet",
                           feature="dss",
                           type="image",
                           customImageListParams=dict(image_list_folder="../data/domainnet_list",
                                                      source_domains=["real"],
                                                      target_domains=["clipart"],
                                                      num_classes=345),
                           preprocess=dict(crop=224,
                                                  resize=256,
                                                  normalizer_mean=[0.485, 0.456, 0.406],
                                                  normalizer_std=[0.229, 0.224, 0.225]),),

              dataloader=dict(shuffle=True,
                              batch_size=16,
                              pin_memory=True),


              model=dict(architecture='ResNet50',
                         type='pre-defined',
                         numclasses=345,
                         pretrained=True),

              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/domainnet/',
                        save_every=20),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.001,
                             weight_decay=5e-4),

              scheduler=dict(type="cosine_annealing",
                             T_max=300),


              dss_args=dict(type="Full",
                            verbose=True),

              train_args=dict(num_epochs=300,
                              device="cuda",
                              print_every=10,
                              results_dir='results/',
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )
