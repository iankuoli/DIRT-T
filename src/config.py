
models = ['UDA', 'VADA', 'DIRT-T']
datasets = ['MNIST', 'SVHN', 'STL', 'CIFAR10']

model_used = models[1]

data_dir_path = "/home/ian/Dataset/SVHN"
data_source = datasets[1]
data_target = datasets[0]

display_step = 10

batch_size = 256
dim_input = (32, 32)
dim_embed = 64
num_classes = 10

# Hyper-parameter for Domain Adaptation
lambda_d = 1
lambda_s = 0.1
lambda_t = 0.1
beta_t = 10
learning_rate = 1e-4

# Hyper-parameters for VAT
src_xi = 8.
src_epsilon = 1e-6
tar_xi = 8.
tar_epsilon = 1


