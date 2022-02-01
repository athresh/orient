import matplotlib.pyplot as plt
import numpy.random
from sklearn import datasets
import numpy as np
import torch
from sklearn.datasets import make_moons, make_blobs, make_classification
from scipy.spatial import ConvexHull
import random
import subprocess
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#
# def diff_class_prior(X,y,num_centers):
#
#     train_indices = []
#     val_indices = []
#     tst_indices = []
#
#     choice = [x / 100.0 for x in range(2,10)]
#
#     frac = list(np.random.choice(choice, size=int(num_centers/2), replace=True))
#
#     for i in range(int(num_centers/2)):
#         frac.append(0.2-frac[i])
#
#     if num_centers %2 ==1:
#         frac.append(0.1)
#
#     random.shuffle(frac)
#
#     for i in range(num_centers):
#         temp_indices = (set(np.where(y == i)[0]))
#         val_indices.extend(np.random.choice(list(temp_indices), size=int((frac[i]/num_centers) * len(y)), replace=False))
#         temp_indices = temp_indices.difference(val_indices)
#         tst_indices.extend(np.random.choice(list(temp_indices), size=int((frac[i]*2/num_centers) * len(y)), replace=False))
#         temp_indices = temp_indices.difference(tst_indices)
#         train_indices.extend(list(temp_indices))
#
#     return train_indices, val_indices, tst_indices
#
# def generate_covariate_shift(num_samples, num_features, file_name, centers_train, centers_test):
#     X, y = datasets.make_blobs(n_samples=int(num_samples*0.8), centers=centers_train,
#                                         n_features=num_features)
#     x_trn, x_val, y_trn, y_val = train_test_split(X, y, test_size=int(num_samples*0.1), random_state=42)
#     # sc = StandardScaler()
#     # x_trn = sc.fit_transform(x_trn)
#     # x_val = sc.fit_transform(x_val)
#     # center_box=(-8.75, 8.75))
#     plt.scatter(x_trn[:, 0], x_trn[:, 1], marker='o', c=y_trn,
#                 s=15, edgecolor='k')
#     plt.scatter(x_val[:, 0], x_val[:, 1], marker='o', c=y_val,
#                 s=15, edgecolor='k')
#     plt.show()
#     plt.scatter(x_trn[:, 0], x_trn[:, 1], marker='o', c=y_trn,
#                 s=15, edgecolor='k')
#     plt.scatter(x_val[:, 0], x_val[:, 1], marker='o', c=y_val,
#                 s=15, edgecolor='k')
#
#     X_test, y_test = datasets.make_blobs(n_samples=int(num_samples*0.2), centers=centers_test,
#                                         n_features=num_features)
#     # x_test = sc.fit_transform(X_test)
#     x_test = X_test
#     plt.scatter(x_test[:, 0], x_test[:, 1], marker='o', c=y_test,
#                 s=15, edgecolor='k', cmap=plt.get_cmap('cool'))
#     plt.show()
#     plt.scatter(x_test[:, 0], x_test[:, 1], marker='o', c=y_test,
#                 s=15, edgecolor='k', cmap=plt.get_cmap('cool'))
#     plt.show()
#
#     X_tot_train = np.concatenate((x_trn, y_trn.reshape((-1, 1))), axis=1)
#     X_tot_val = np.concatenate((x_val, y_val.reshape((-1, 1))), axis=1)
#     X_tot_tst = np.concatenate((x_test, y_test.reshape((-1, 1))), axis=1)
#     data_dir = "data/" + file_name
#     with open(data_dir+"/"+file_name + ".trn", 'w') as f:
#         np.savetxt(f, X_tot_train, delimiter=",")
#     with open(data_dir+"/"+file_name + ".val", 'w') as f:
#         np.savetxt(f, X_tot_val, delimiter=",")
#     with open(data_dir+"/"+file_name + ".tst", 'w') as f:
#         np.savetxt(f, X_tot_tst, delimiter=",")
#
#
# def generate_linear_seperable_data(num_samples, num_centers, num_features, file_name,same=None, noise_ratio = 0, probability = 1):
#     X, y, centers= datasets.make_blobs(n_samples=num_samples, centers=num_centers,
#                                        n_features=num_features, return_centers=True,center_box=(-5.75, 5.75))
#     #center_box=(-8.75, 8.75))
#     plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
#                 s=25, edgecolor='k')
#     plt.show()
#     train_indices = []
#     val_indices = []
#     tst_indices = []
#
#     shift = np.array([0,0])
#     min_cnt = np.min(num_samples)
#     if same == "class_imb":
#         for i in range(len(num_samples)):
#             temp_indices = (set(np.where(y == i)[0]))
#             val_indices.extend(
#                 np.random.choice(list(temp_indices), size=int(0.4 * min_cnt), replace=False))
#             temp_indices = temp_indices.difference(val_indices)
#             tst_indices.extend(
#                 np.random.choice(list(temp_indices), size=int(0.4 * min_cnt), replace=False))
#             temp_indices = temp_indices.difference(tst_indices)
#             train_indices.extend(list(temp_indices))
#     else:
#         for i in range(num_centers):
#             temp_indices = (set(np.where(y == i)[0]))
#             val_indices.extend(np.random.choice(list(temp_indices), size=int((0.1/num_centers) * len(y)), replace=False))
#             temp_indices = temp_indices.difference(val_indices)
#             tst_indices.extend(np.random.choice(list(temp_indices), size=int((0.2/num_centers) * len(y)), replace=False))
#             temp_indices = temp_indices.difference(tst_indices)
#             train_indices.extend(list(temp_indices))
#
#         if same == "covariate":
#             choice = [x / 10.0 for x in range(-30,30)]
#             remove = [x / 10.0 for x in range(-14,15)]
#             choice = list(set(choice).difference(remove))
#
#             shift = np.random.choice(choice, size=2, replace=True)
#
#         elif same == "expand":
#             factor = np.random.choice([i for i in range(2,5)], size=1, replace=True)
#             #factor = 2
#             print(factor)
#             for i in val_indices+tst_indices:
#                 curr_center = centers[y[i]]
#                 X[i] = factor*X[i] - (factor-1)*curr_center
#
#         elif same == "shrink":
#             factor = np.random.choice([i for i in range(2,5)], size=1, replace=True)
#             print(factor)
#             for i in val_indices+tst_indices:
#                 curr_center = centers[y[i]]
#                 X[i] = ((factor-1)*curr_center + X[i])/factor
#
#
#     data_dir = "data/"+file_name
#     # subprocess.run(["mkdir", data_dir])
#
#
#     if same == "noise":
#         noise_size = int(len(train_indices) * noise_ratio)
#         noise_indices = np.random.choice(list(train_indices), size=noise_size, replace=False)
#         y[noise_indices] = np.random.choice(np.arange(num_centers), size=noise_size, replace=True)
#
#
#
#     X_train = X[train_indices]
#     y_train = y[train_indices]
#
#     X_tot_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
#     plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train,
#                 s=25, edgecolor='k')
#     plt.savefig(data_dir + "/training_data.png")
#     plt.show()
#
#     X_val = X[val_indices] + shift
#     y_val = y[val_indices]
#     print(shift)
#     X_tot_val = np.concatenate((X_val, y_val.reshape((-1, 1))), axis=1)
#     plt.figure()
#     plt.scatter(X_val[:, 0], X_val[:, 1], marker='o', c=y_val,
#                 s=25, edgecolor='k')
#     plt.savefig(data_dir + "/validation_data.png")
#     plt.show()
#
#     X_tst = X[tst_indices] + shift
#     y_tst = y[tst_indices]
#     X_tot_tst = np.concatenate((X_tst, y_tst.reshape((-1, 1))), axis=1)
#     plt.figure()
#     plt.scatter(X_tst[:, 0], X_tst[:, 1], marker='o', c=y_tst,
#                 s=25, edgecolor='k')
#     plt.savefig(data_dir + "/test_data.png")
#     plt.show()
#
#     with open(data_dir+"/"+file_name + ".trn", 'w') as f:
#         np.savetxt(f, X_tot_train, delimiter=",")
#     with open(data_dir+"/"+file_name + ".val", 'w') as f:
#         np.savetxt(f, X_tot_val, delimiter=",")
#     with open(data_dir+"/"+file_name + ".tst", 'w') as f:
#         np.savetxt(f, X_tot_tst, delimiter=",")
#
# #generate_linear_seperable_data(5000, 4, 2, 'red_large_linsep_4')
# #generate_linear_seperable_data(10000, 4, 2, 'prior_shift_large_linsep_4',"prior")
# #generate_linear_seperable_data(10000, 4, 2, 'conv_shift_large_linsep_4',"covariate")
# #generate_linear_seperable_data(5000, 4, 2, 'expand_large_linsep_4',"expand")
# #generate_linear_seperable_data(num_samples = np.array([50, 500, 50, 500]), num_centers=None, num_features=2, file_name='class_imb_linsep_4', same='class_imb')
#
# def generate_gaussian_quantiles_data(num_samples, num_centers, num_features, file_name,same=None):
#     #num_labels = int(num_samples)/num_features
#     X, y, centers = datasets.make_gaussian_quantiles(mean=None, cov=1.0, n_samples=num_samples, n_features=2, n_classes=num_centers, shuffle=True, random_state=None)
#     plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
#                 s=25, edgecolor='k')
#     plt.show()
#     train_indices = []
#     val_indices = []
#     tst_indices = []
#
#     shift = np.array([0,0])
#
#     if same == "prior":
#         train_indices, val_indices, tst_indices = diff_class_prior(X,y,num_centers)
#     else:
#         for i in range(num_centers):
#             temp_indices = (set(np.where(y == i)[0]))
#             val_indices.extend(np.random.choice(list(temp_indices), size=int((0.1/num_centers) * len(y)), replace=False))
#             temp_indices = temp_indices.difference(val_indices)
#             tst_indices.extend(np.random.choice(list(temp_indices), size=int((0.2/num_centers) * len(y)), replace=False))
#             temp_indices = temp_indices.difference(tst_indices)
#             train_indices.extend(list(temp_indices))
#
#         if same == "covariate":
#             choice = [x / 10.0 for x in range(-30,30)]
#             remove = [x / 10.0 for x in range(-9,10)]
#             choice = list(set(choice).difference(remove))
#
#             shift = np.random.choice(choice, size=2, replace=True)
#
#         elif same == "expand":
#             factor = np.random.choice([i for i in range(2,5)], size=1, replace=True)
#             print(factor)
#             for i in val_indices+tst_indices:
#                 curr_center = centers[y[i]]
#                 X[i] = factor*X[i] - (factor-1)*curr_center
#
#         elif same == "shrink":
#             factor = np.random.choice([i for i in range(2,5)], size=1, replace=True)
#             print(factor)
#             for i in val_indices+tst_indices:
#                 curr_center = centers[y[i]]
#                 X[i] = ((factor-1)*curr_center + X[i])/factor
#
#     X_train = X[train_indices]
#     y_train = y[train_indices]
#     X_tot_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
#     plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train,
#                 s=25, edgecolor='k')
#     plt.show()
#     X_val = X[val_indices]
#     y_val = y[val_indices]
#     X_tot_val = np.concatenate((X_val, y_val.reshape((-1, 1))), axis=1)
#     plt.scatter(X_val[:, 0], X_val[:, 1], marker='o', c=y_val,
#                 s=25, edgecolor='k')
#     plt.show()
#     X_tst = X[tst_indices]
#     y_tst = y[tst_indices]
#     X_tot_tst = np.concatenate((X_tst, y_tst.reshape((-1, 1))), axis=1)
#     plt.scatter(X_tst[:, 0], X_tst[:, 1], marker='o', c=y_tst,
#                 s=25, edgecolor='k')
#     plt.show()
#
#     data_dir = "./data/"+file_name
#     subprocess.run(["mkdir", data_dir])
#
#     with open(data_dir+"/"+file_name + ".trn", 'w') as f:
#         np.savetxt(f, X_tot_train, delimiter=",")
#     with open(data_dir+"/"+file_name + ".val", 'w') as f:
#         np.savetxt(f, X_tot_val, delimiter=",")
#     with open(data_dir+"/"+file_name + ".tst", 'w') as f:
#         np.savetxt(f, X_tot_tst, delimiter=",")
#
# #generate_gaussian_quantiles_data(10000, 2, 2, 'gauss_2')
# #generate_gaussian_quantiles_data(10000, 2, 2, 'prior_shift_gauss_2',"prior")
# #generate_gaussian_quantiles_data(10000, 2, 2, 'conv_shift_gauss_2',"convariate")
#
# def generate_classification_data(num_samples, num_centers, file_name,same=None):
#     #num_labels = int(num_samples)/num_features
#     X, y= datasets.make_classification(n_samples=num_samples, n_features=2, n_redundant=0, n_informative=2, n_classes=num_centers,
#                                         n_clusters_per_class=1, class_sep=2) #, centers
#     plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
#                 s=25, edgecolor='k')
#     plt.show()
#     train_indices = []
#     val_indices = []
#     tst_indices = []
#
#     shift = np.array([0,0])
#
#     if same == "prior":
#         train_indices, val_indices, tst_indices = diff_class_prior(X,y,num_centers)
#     else:
#         for i in range(num_centers):
#             temp_indices = (set(np.where(y == i)[0]))
#             val_indices.extend(np.random.choice(list(temp_indices), size=int((0.1/num_centers) * len(y)), replace=False))
#             temp_indices = temp_indices.difference(val_indices)
#             tst_indices.extend(np.random.choice(list(temp_indices), size=int((0.2/num_centers) * len(y)), replace=False))
#             temp_indices = temp_indices.difference(tst_indices)
#             train_indices.extend(list(temp_indices))
#
#         if same == "covariate":
#             choice = [x / 10.0 for x in range(-30,30)]
#             remove = [x / 10.0 for x in range(-9,10)]
#             choice = list(set(choice).difference(remove))
#
#             shift = np.random.choice(choice, size=2, replace=True)
#
#         elif same == "expand":
#             factor = np.random.choice([i for i in range(2,5)], size=1, replace=True)
#             print(factor)
#             for i in val_indices+tst_indices:
#                 curr_center = centers[y[i]]
#                 X[i] = factor*X[i] - (factor-1)*curr_center
#
#         elif same == "shrink":
#             factor = np.random.choice([i for i in range(2,5)], size=1, replace=True)
#             print(factor)
#             for i in val_indices+tst_indices:
#                 curr_center = centers[y[i]]
#                 X[i] = ((factor-1)*curr_center + X[i])/factor
#
#     X_train = X[train_indices]
#     y_train = y[train_indices]
#     X_tot_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
#     plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train,
#                 s=25, edgecolor='k')
#     plt.show()
#     X_val = X[val_indices] +shift
#     y_val = y[val_indices]
#     X_tot_val = np.concatenate((X_val, y_val.reshape((-1, 1))), axis=1)
#     plt.scatter(X_val[:, 0], X_val[:, 1], marker='o', c=y_val,
#                 s=25, edgecolor='k')
#     plt.show()
#     X_tst = X[tst_indices] +shift
#     y_tst = y[tst_indices]
#     X_tot_tst = np.concatenate((X_tst, y_tst.reshape((-1, 1))), axis=1)
#     plt.scatter(X_tst[:, 0], X_tst[:, 1], marker='o', c=y_tst,
#                 s=25, edgecolor='k')
#     plt.show()
#
#     data_dir = "data/"+file_name
#     # subprocess.run(["mkdir", data_dir])
#
#     with open(data_dir+"/"+file_name + ".trn", 'w+') as f:
#         np.savetxt(f, X_tot_train, delimiter=",")
#     with open(data_dir+"/"+file_name + ".val", 'w+') as f:
#         np.savetxt(f, X_tot_val, delimiter=",")
#     with open(data_dir+"/"+file_name + ".tst", 'w+') as f:
#         np.savetxt(f, X_tot_tst, delimiter=",")
#
#
# #generate_classification_data(10000, 2, 'clf_2')
# #generate_classification_data(10000, 2, 'prior_shift_clf_2','prior')
# numpy.random.seed(42)
# # generate_classification_data(10000, 2, 'conv_shift_clf_2','covariate')
# # generate_linear_seperable_data(10000, 2, 2, 'conv_shift_clf_2', 'covariate')
# generate_covariate_shift(10000, 2, 'synthetic_covariate', centers_train = [(-2, 2), (2, -2)],
#                          centers_test=[(-0, 0), (4, -4)])

def noise_augment(x):
    scale = x.new_tensor(torch.from_numpy(np.eye(2) + np.random.normal(0, .01, size=(2, 2))))
    bias = x.new_tensor(torch.from_numpy(np.random.normal(0, .1, size=(2))))

    return x.mm(scale) + bias


def make_data(n_samples=50000, n_domains=2, plot=False, noisemodels=None,
              train_fraction=0.7, save_data=False, seed=None):
    if noisemodels is None:
        noisemodels = []

        angles = np.linspace(0, np.pi/5, n_domains)
        for _ in range(n_domains):
            # scale = np.eye(2) + np.random.normal(0,.1,size=(2,2))
            a = angles[_]
            # scale = np.array([[np.cos(a), np.sin(a)],
            #                   [-np.sin(a), np.cos(a)]])
            scale = np.array([[1, 0],
                              [0, 1]])
            bias = 0  # np.random.normal(0,.5,size=(1,2))
            bias = np.array([0.4, 0])
        noisemodels.append(lambda x: x.dot(scale) + bias)

    if seed is not None:
        np.random.seed(seed)

    n_total = n_samples * n_domains * 3

    # # X_train, y_train = make_blobs(n_samples=int(n_samples*0.7), centers=2)
    # cov = np.array([[1,1],[0,1]])
    # X_train = np.random.multivariate_normal(mean=[0.0,0.0],cov=cov,size=n_samples)
    # print(X_train.shape)
    # # y_train = np.random.randint(0,2,n_samples)
    # y_train = np.zeros(n_samples)
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # # X_train, y_train = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
    # #                                        n_redundant=0, n_clusters_per_class=1, class_sep=1)

    X, y = make_moons(n_samples=n_total, shuffle=True, noise=.1)
    # X, y = make_blobs(n_samples=n_total, shuffle=True, n_features=2)
    X = X.reshape(n_domains, 3, n_samples, 2)
    y = y.reshape(n_domains, 3, n_samples)
    mask_val = np.random.randint(0,n_samples+1, int(n_samples*0.01)+1)
    # mask_val = list(map(bool, np.random.choice(2, int(n_samples*0.01))))
    mask_val = [True if i in mask_val else False for i in range(n_samples)]
    # plt.scatter(X[0, 0, :, 0], X[0, 0, :, 1], c=y[0, 0, :])
    # plt.show()
    # plt.scatter(X[1, 0, :, 0], X[1,0, :, 1], c=y[1, 0, :])
    # plt.show()

    for domain, noise in enumerate(noisemodels):
        X[domain] = noise(X[domain])

    # Xs = torch.from_numpy(X).float()
    # ys = torch.from_numpy(y).float()
    # print(Xs)

    # d = [(X, y) for (X, y) in zip(Xs, ys)]
    mask_0 = [idx for idx, val in enumerate(y[1, 2, :]) if val == 0]
    mask_1 = [idx for idx, val in enumerate(y[1, 2, :]) if val == 1]
    # hull_0 = ConvexHull(X[1, 2, mask_0, :])
    # hull_1 = ConvexHull(X[1, 2, mask_1, :])

    dt = 0
    p0 = plt.scatter(X[0, dt, :, 0], X[0, dt, :, 1], c=y[0, 0, :], cmap=plt.get_cmap('spring'), label="source domain")
    # plt.title("Toy data domain 0")
    p1 = plt.scatter(X[1, dt, mask_val, 0], X[1, dt, mask_val, 1], c=y[1, 0, mask_val], cmap=plt.get_cmap('RdYlGn'), label="target domain")
    # p1 = plt.scatter(X[1, dt, :, 0], X[1, dt, :, 1], c=y[1, 0, :], cmap=plt.get_cmap('RdYlGn'), label="target domain")
    plt.legend(handles=p0.legend_elements()[0]+p1.legend_elements()[0], labels=["source domain class 0", "source domain class 1", "target domain class 0", "target domain class 1"])
    # plt.scatter(X[1, 1, mask_val, 0], X[1, 1, mask_val, 1], c=y[1, 1, mask_val], cmap=plt.get_cmap('RdYlGn'))
    # for simplex in hull_0.simplices:
    #     plt.plot(X[1, 2, hull_0.vertices, 0], X[1, 2, hull_0.vertices, 1], 'r--', lw=2)
    #     plt.plot(X[1, 2, hull_0.vertices[0], 0], X[1, 2, hull_0.vertices[0], 1], 'ro')
    # plt.legend(['first', 'last'])
    plt.title("Toy data train set and validation set")
    plt.show()



    if save_data:
        data_dir = '../data/toy_da2'
        for i in range(n_domains):
            np.savetxt(data_dir+'/d{}/d{}.trn'.format(i,i),np.insert(X[i,0,:,:],2,y[i,0,:],axis=1), delimiter=",")
            np.savetxt(data_dir + '/d{}/d{}.val'.format(i,i), np.insert(X[i,1,mask_val,:],2,y[i,1,mask_val],axis=1), delimiter=",")
            np.savetxt(data_dir + '/d{}/d{}.tst'.format(i,i), np.insert(X[i,2,:,:],2,y[i,2,:],axis=1), delimiter=",")
    return X, y

data = make_data(n_samples=10000, n_domains=2, plot=True, save_data=True, seed=0)