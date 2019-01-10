import torch
import torch.nn as nn
import torch.optim as optim
import random
import generate_data as gd

random.seed(999)
torch.manual_seed(999)

# generator class
class Generator(nn.Module):

    def __init__(self, ngpu, D_in, H1, H2, p):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input layer - > first hidden layer
            nn.Linear(D_in, H1),
            nn.BatchNorm1d(H1),
            nn.ReLU(True),
            nn.Dropout(p),
            # first hidden layer -> second hidden layer
            nn.Linear(H1, H2),
            nn.BatchNorm1d(H2),
            nn.ReLU(True),
            nn.Dropout(p),
            # second hidden layer -> output layer
            nn.Linear(H2, 2),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# discriminator class
class Discriminator(nn.Module):

    def __init__(self, ngpu, H1, H2, p):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input layer - > first hidden layer
            nn.Linear(2, H1),
            nn.BatchNorm1d(H1),
            nn.ReLU(True),
            nn.Dropout(p),
            # first hidden layer -> second hidden layer
            nn.Linear(H1, H2),
            nn.BatchNorm1d(H2),
            nn.ReLU(True),
            nn.Dropout(p),
            # second hidden layer -> output layer
            nn.Linear(H2, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# hyper parameters
num_samples = 1000  # number of sample
D_in = 2  # input to generator
H1_G = 10  # 1st hidden layer size in the generator
H2_G = 10  # 2nd hidden layer size in the generator
H1_D = 10  # 1st hidden layer size in the discriminator
H2_D = 10  # 2nd hidden layer size in the discriminator
p = 0.2  # dropout probability
num_epochs = 3000
to_print = 500
reg = 0.0001  # regularization
lr_g = 0.0002  # generator learning rate
beta_g = 0.5  # Beta1 hyperparam for Adam optimizers
lr_d = 0.0002  # discriminator learning rate
beta_d = 0.5  # Beta1 hyperparam for Adam optimizers
batch_size = 32  # batch size of generator
k = 5  # generator to discriminator update ratio
target_dist = "line"  # the target distribution

# cpu/gpu configurations
ngpu = 0  # 0 for cpu and 1 for gpu
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# generate data
noise = torch.rand(num_samples, D_in, device=device)  # random data uniformly distributed
real_data = torch.from_numpy(gd.get_data(num_samples, target_dist)).float().to(device)  # read data

# create occurrence classs
netG = Generator(ngpu, D_in, H1_G, H2_G, p).to(device)
netD = Discriminator(ngpu, H1_D, H2_D, p).to(device)

# Binary cross entropy loss
criterion = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta_g, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta_d, 0.999))

# Lists to keep track of progress
points_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):

    # get random permutation of the noise and the true data
    noise_permutation_dis = torch.randperm(noise.size()[0])
    noise_permutation_gen = torch.randperm(noise.size()[0])
    real_permutation = torch.randperm(real_data.size()[0])
    dis_loss = gen_loss = D_x = D_G_z1 = D_G_z2 = gen_num_examples = 0

    # for each batch
    for i in range(0, real_data.size()[0], batch_size):

        # (1) Update D network: maximizing log(D(x)) + log(1 - D(G(z)))
        # An equivalent method is to minimize -(y_t*log(D(x) + (1 - y_t)*log(1 - D(G(z))))) with gradient descent
        # accumulate errors from genuine and fake examples in the batch and make the update step

        netD.train()
        netD.zero_grad()

        # get batch examples
        indices_noise = noise_permutation_dis[i:i + batch_size]
        indices_real = real_permutation[i:i + batch_size]
        batch_fake = noise[indices_noise]
        batch_real = real_data[indices_real]
        b_size = batch_real.size(0)

        # run true examples through the network
        label = torch.full((b_size,), real_label, device=device)  # set labels
        output = netD(batch_real).view(-1)  # get network output
        errD_real = criterion(output, label)  # calc. average BCE error
        errD_real.backward()  # calc gradients for discriminator
        D_x += output.sum().item()  # average value given to true distribution

        # run fake examples through the network
        fake = netG(batch_fake)
        label.fill_(fake_label)  # set labels
        output = netD(fake.detach()).view(-1)  # get network output without backprop
        errD_fake = criterion(output, label)  # calc. average BCE error
        errD_fake.backward()  # calc gradients for discriminator
        D_G_z1 += output.sum().item()  # average value given to fake distribution

        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        optimizerD.step()

        dis_loss += errD*2*b_size  # get the sum of errors

        # (2) Update G network: maximize log(D(G(z)))
        # Alternative is to minimize -y_t*log(D(G(z))) with labels of "1" and with gradient descent
        # update generator weights every k steps
        if i % k == k-1:

            netG.train()
            netG.zero_grad()

            # get batch examples
            indices_noise = noise_permutation_gen[i:i + batch_size]
            batch_fake = noise[indices_noise]
            b_size = batch_fake.size(0)

            label = torch.full((b_size,), real_label, device=device)  # fake labels are real for generator cost
            fake = netG(batch_fake)
            output = netD(fake).view(-1)  # get network output
            errG = criterion(output, label)  # calc. BCE error
            errG.backward()  # calc gradients for discriminator
            D_G_z2 += output.sum().item()  # average value given to fake distribution
            gen_num_examples += b_size

            optimizerG.step()
            gen_loss += errG*b_size  # get the sum of errors

    # save losses
    D_losses.append(dis_loss)
    G_losses.append(gen_loss)

    # Output training stats
    if epoch % to_print == to_print-1:
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch, num_epochs,
                 D_losses[-1].item(), G_losses[-1].item(), D_x/num_samples, D_G_z1/num_samples, D_G_z2/gen_num_examples))








