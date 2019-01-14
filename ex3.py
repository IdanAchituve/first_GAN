import torch
import torch.nn as nn
import torch.optim as optim
import generate_data as gd
import matplotlib.pyplot as plt
import os

#torch.manual_seed(2)
#torch.cuda.manual_seed(2)

os.makedirs("./plots/", exist_ok=True)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0)


def print_graphs(real_data, gen_points, target_dist):
    # print real data
    real_data = real_data.data.numpy()
    plt.subplot(1, 2, 1)
    plt.title("Real Data - " + target_dist + " model")
    plt.plot(real_data[:, 0], real_data[:, 1], '.', label="Real", color='blue')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    # print generator data
    gen_points = gen_points.data.numpy()
    plt.subplot(1, 2, 2)
    plt.title("Generator Data - " + target_dist + " model")
    plt.plot(gen_points[:, 0], gen_points[:, 1], '.', label="Gen", color='orange')
    plt.xlabel("x")
    plt.legend()

    plt.savefig("./plots/" + target_dist + ".png", bbox_inches='tight')
    plt.close()


# generator class
class Generator(nn.Module):

    def __init__(self, ngpu, D_in, H1, H2, H3, p):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input layer - > first hidden layer
            nn.Linear(D_in, H1),
            nn.BatchNorm1d(H1),
            nn.ReLU(True),
            #nn.Tanh(),
            nn.Dropout(p),
            # first hidden layer -> second hidden layer
            nn.Linear(H1, H2),
            nn.BatchNorm1d(H2),
            nn.ReLU(True),
            #nn.Tanh(),
            nn.Dropout(p),
            # second hidden layer -> third hidden layer
            nn.Linear(H2, H3),
            nn.BatchNorm1d(H3),
            nn.ReLU(True),
            #nn.Tanh(),
            nn.Dropout(p),
            # third hidden layer -> output layer
            nn.Linear(H3, 2),
            #nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# discriminator class
class Discriminator(nn.Module):

    def __init__(self, ngpu, H1, H2, H3, p):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input layer - > first hidden layer
            nn.Linear(2, H1),
            nn.BatchNorm1d(H1),
            #nn.Tanh(),
            nn.ReLU(True),
            nn.Dropout(p),
            # first hidden layer -> second hidden layer
            nn.Linear(H1, H2),
            nn.BatchNorm1d(H2),
            #nn.Tanh(),
            nn.ReLU(True),
            nn.Dropout(p),
            # second hidden layer -> third hidden layer
            nn.Linear(H2, H3),
            nn.BatchNorm1d(H3),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Dropout(p),
            # third hidden layer -> output layer
            nn.Linear(H3, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# hyper parameters
num_samples = 1000  # number of sample
D_in = 2  # input to generator
H1_G = 20  # 1st hidden layer size in the generator
H2_G = 30  # 2nd hidden layer size in the generator
H3_G = 20  # 2nd hidden layer size in the generator
H1_D = 20  # 1st hidden layer size in the discriminator
H2_D = 30  # 2nd hidden layer size in the discriminator
H3_D = 20  # 2nd hidden layer size in the discriminator
p_gen = 0  # dropout probability
p_dis = 0
num_epochs = 100
to_print = 100
reg = 0.0001  # regularization
lr_g = 0.0003  # generator learning rate
beta_g = 0.5  # Beta1 hyperparam for Adam optimizers
lr_d = 0.0001  # discriminator learning rate
beta_d = 0.5  # Beta1 hyperparam for Adam optimizers
momentum_d = 0.9
batch_size = 512  # batch size of generator
k = 1  # generator to discriminator update ratio
target_dist = "spiral"  # the target distribution

# cpu/gpu configurations
ngpu = 0  # 0 for cpu and 1 for gpu
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# create instances of the classes
netG = Generator(ngpu, D_in, H1_G, H2_G, H3_G, p_gen).to(device)
netG.apply(weights_init)  # init weights according to normal dist with mean 0 and std 0.02
netD = Discriminator(ngpu, H1_D, H2_D, H3_D, p_dis).to(device)
netD.apply(weights_init)  # init weights according to normal dist with mean 0 and std 0.02

# Binary cross entropy loss
criterion = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta_g, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta_d, 0.999))

# Lists to keep track of progress
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):

    # get random permutation of the noise and the true data
    dis_loss = gen_loss = D_x = D_G_z1 = D_G_z2 = g_grads_norm = d_grads_norm = 0

    # generate data
    real_data = torch.from_numpy(gd.get_data(batch_size, target_dist)).float().to(device)  # read data

    # (1) Update D network: maximizing log(D(x)) + log(1 - D(G(z)))
    # An equivalent method is to minimize -(y_t*log(D(x) + (1 - y_t)*log(1 - D(G(z))))) with gradient descent
    # accumulate errors from genuine and fake examples in the batch and make the update step

    netD.train()
    netD.zero_grad()

    # run true examples through the network
    label = torch.full((batch_size,), real_label, device=device)  # set labels
    output = netD(real_data).view(-1)  # get network output
    errD_real = criterion(output, label)  # calc. average BCE error
    errD_real.backward()  # calc gradients for discriminator
    D_x += output.sum().item()  # average value given to true distribution

    # run fake examples through the network
    noise = 2*torch.rand(batch_size, D_in, device=device) - 1
    fake = netG(noise)
    label.fill_(fake_label)  # set labels
    output = netD(fake.detach()).view(-1)  # get network output without backprop in the generator network
    errD_fake = criterion(output, label)  # calc. average BCE error
    errD_fake.backward()  # calc gradients for discriminator
    D_G_z1 += output.sum().item()  # average value given to fake distribution

    # Add the gradients from the all-real and all-fake batches
    errD = errD_real + errD_fake
    optimizerD.step()

    dis_loss += (errD*2*batch_size).item()  # get the sum of errors

    grads = list(netD.parameters())[0].grad.view(1,-1).squeeze(0)  # get params gradients of the first layer in the discriminator network
    d_grads_norm += torch.sqrt(torch.dot(grads, grads))

    # (2) Update G network: maximize log(D(G(z)))
    # Alternative is to minimize -y_t*log(D(G(z))) with labels of "1" and with gradient descent
    # update generator weights every k steps
    if epoch % k == k-1:

        netG.train()
        netG.zero_grad()

        noise = 2*torch.rand(batch_size, D_in, device=device) - 1
        label = torch.full((batch_size,), real_label, device=device)  # fake labels are real for generator cost
        fake = netG(noise)
        output = netD(fake).view(-1)  # get network output
        errG = criterion(output, label)  # calc. BCE error
        errG.backward()  # calc gradients for discriminator
        D_G_z2 += output.sum().item()  # average value given to fake distribution

        optimizerG.step()
        gen_loss += (errG*batch_size).item()  # get the sum of errors

        grads = list(netG.parameters())[0].grad.view(1, -1).squeeze(0)  # get params gradients of the first layer in the generator network
        g_grads_norm += torch.sqrt(torch.dot(grads, grads))

    # Output training stats
    if epoch % to_print == to_print -1:
        print_graphs(real_data, fake, target_dist)
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tGrads_D: %.4f\t Grads_G: %.4f'
              % (epoch, num_epochs,
                 dis_loss, gen_loss, D_x/batch_size,
                 D_G_z1/batch_size, D_G_z2/batch_size, d_grads_norm/(2*batch_size), g_grads_norm/batch_size))


torch.save({
            'modelG_state_dict': netG.state_dict(),
            'modelD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            }, "./plots/model")


real_data = torch.from_numpy(gd.get_data(num_samples, target_dist)).float().to(device)  # read data
noise = 2 * torch.rand(num_samples, D_in, device=device) - 1  # random data uniformly distributed between -1 and 1
# print points
for i in range(0, noise.size()[0], batch_size):

    netD.eval()
    netG.eval()

    batch_fake = noise[i:i+batch_size]
    b_size = batch_fake.size(0)

    fake = netG(batch_fake)
    gen_points = fake if i == 0 else torch.cat((gen_points, fake))


print_graphs(real_data, gen_points, target_dist)


