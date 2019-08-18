
models = ['VAT', 'VADA', 'DIRT-T']
datasets = ['MNIST', 'SVHN', 'STL', 'CIFAR10']

model_used = models[0]

data_source = datasets[0]
data_target = datasets[1]

display_step = 10

batch_size = 256
dim_embed = (64, 64)
dim_embed = 64
num_classes = 10

# Hyper-parameter for Domain Adaptation
lambda_d = 0.1
lambda_s = 0.1
lambda_t = 0.1
learning_rate = 1e-6

# Hyper-parameters for VAT
xi = 1e-8
epsilon = 8.


