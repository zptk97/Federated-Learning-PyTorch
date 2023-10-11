import pickle

with open("../save/objects/cifar_cnn_5_C[0.1]_iid[0]_E[5]_B[100].pkl","rb") as fr:
    flex_200 = pickle.load(fr)

print(flex_200)