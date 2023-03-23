import pandas, string, torch, time, numpy as np
from sklearn.model_selection import train_test_split
from mlp import MLP

# 0.6574, 0.343, 0.0178, 0.0006         2M, 5e-4, 420 epochs, batch 32768, lambda=3, 216/108/54

N = 1000000                         # Number of training data points to use
MAX_WORD_LENGTH = 8
ALPHABET = string.ascii_lowercase + '_'
LEARNING_RATE = 1e-3
NUM_EPOCHS = 160
BATCH_SIZE = 131072                  # Number of data points worked on by GPU in parallel
LAMBDA = 3                          # L1 regularization factor

def one_hot_encode(strings, dists, device):
    strings = strings.tolist()
    dists = dists.tolist()
    indices_np = np.array([[ord(c) - ord('a') for c in s] for s in strings], dtype=np.int64)
    x_np = np.eye(len(ALPHABET), dtype=np.float32)[indices_np]
    x = torch.from_numpy(x_np).to(device)
    y = torch.as_tensor(dists, device=device)
    return x, y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")
fileName = 'mixed_data' + str(MAX_WORD_LENGTH) + '.csv'
data = pandas.read_csv(fileName, nrows=N, header=None, usecols=[0, 1], names=['strings', 'dists'])
data['dists'] = data['dists'].astype(int)

train_data, test_data = train_test_split(data, test_size=0.2)
train_strings, train_dists = train_data['strings'], train_data['dists']
test_strings, test_dists = test_data['strings'], test_data['dists']
train_data_x, train_data_y = one_hot_encode(train_strings, train_dists, device)
test_data_x, test_data_y = one_hot_encode(test_strings, test_dists, device)

def try_model(h1,h2,h3,learning_rate, l1_factor):
    model = MLP(MAX_WORD_LENGTH * len(ALPHABET),h1,h2,h3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.1)
    criterion_mse, criterion_l1 = torch.nn.MSELoss(), torch.nn.L1Loss()

    lowest_validation_loss = 1000.0
    for _ in range(NUM_EPOCHS):
        model.train()
        running_train_loss = 0.0
        num_batches = len(train_data_x) // BATCH_SIZE

        for i in range(num_batches):
            inputs,labels = train_data_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE],train_data_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            optimizer.zero_grad()
            outputs = model(inputs.float())
            mse_loss = criterion_mse(outputs, labels.float().unsqueeze(1))
            l1_loss = criterion_l1(torch.cat([p.view(-1) for p in model.parameters()], dim=0), torch.zeros_like(torch.cat([p.view(-1) for p in model.parameters()], dim=0)))
<<<<<<< HEAD
            # First grid search options -- yielded 256,128,64,0.01,1 -- lowest loss: 0.354
# h1s = [128, 256]
# h2s = [64, 128]
# h3s = [32, 64]
# lrs = [0.01, 0.0075, 0.005]
# lambdas = [0.01,0.1,1,5]
=======
            total_loss = mse_loss + l1_factor * l1_loss
            total_loss.backward()
            optimizer.step()
            running_train_loss += mse_loss.item()
>>>>>>> 8bb9fd5ae10e398d28ac5647c08970c049636f6c

<<<<<<< HEAD
# Second grid search options -- yielded 256,128,64,0.01,1 again -- lowest loss: 0.353
# h1s = [256]
# h2s = [128]
# h3s = [64]
# lrs = [0.015,0.01,0.0085]
# lambdas = [1]

# Third grid search options -- yielded 256,180,64,0.01,1  -- lowest loss: 0.351
# h1s = [256]
# h2s = [128,180]
# h3s = [64]
# lrs = [0.01]
# lambdas = [1]
=======
        model.eval()
        running_val_loss = 0.0
        test_batch_size = min(len(test_data_x), BATCH_SIZE)
        num_test_batches = len(test_data_x) // test_batch_size
        error_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        total_samples = 0
        elapsed_time = 0.0
        start_time = time.perf_counter()
        with torch.no_grad():
            for i in range(num_test_batches):
                inputs,labels = test_data_x[i*test_batch_size:(i+1)*test_batch_size],test_data_y[i*test_batch_size:(i+1)*test_batch_size]
                
                outputs = model(inputs.float())
                running_val_loss += criterion_mse(outputs, labels.float().unsqueeze(1)).item()
>>>>>>> 8bb9fd5ae10e398d28ac5647c08970c049636f6c

<<<<<<< HEAD
# Fourth grid search options -- yielded 320,230,64,0.01,2  -- lowest loss: 0.348
h1s = [256,280,320,400]
h2s = [200,230,260]
h3s = [64]
lrs = [0.01]
lambdas = [1,2]
=======
                # When finding the distribution of predictions for a real-world scenario (integer values)
                # Round the outputs to the nearest integer
                rounded_outputs = torch.round(outputs)
                errors = torch.abs(rounded_outputs - labels.float().unsqueeze(1))
                for error in error_counts:
                    error_counts[error] += torch.sum(errors == error).item()
>>>>>>> 8bb9fd5ae10e398d28ac5647c08970c049636f6c

<<<<<<< HEAD
def do_grid_search():
    lowest_validation_loss = 1000.0
    best_h1, best_h2, best_h3, best_lr, best_l1 = 0, 0, 0, 0, 0
    for h1 in h1s:
        for h2 in h2s:
            for h3 in h3s:
                for lr in lrs:
                    for l1_fac in lambdas:
                        val_loss = try_model(h1,h2,h3,lr,l1_fac)
=======
                total_samples += errors.shape[0]
        end_time = time.perf_counter()
        elapsed_time += (end_time - start_time)
        # For benchmarking how long it takes to feedforward
        # print("Time: ", elapsed_time)
        # print("Total # points: ", len(test_data_x))
        # print("words/sec: ", len(test_data_x) / elapsed_time)
        # print("batches/sec: ", num_test_batches / elapsed_time)
        avg_val_loss = running_val_loss / num_test_batches
        if avg_val_loss < lowest_validation_loss:
            lowest_validation_loss = avg_val_loss
        scheduler.step(avg_val_loss)
    return lowest_validation_loss

# First grid search options -- yielded 256,128,64,0.01,1 -- lowest loss: 0.354
# h1s = [128, 256]
# h2s = [64, 128]
# h3s = [32, 64]
# lrs = [0.01, 0.0075, 0.005]
# lambdas = [0.01,0.1,1,5]

# Second grid search options -- yielded 256,128,64,0.01,1 again -- lowest loss: 0.353
# h1s = [256]
# h2s = [128]
# h3s = [64]
# lrs = [0.015,0.01,0.0085]
# lambdas = [1]

# Third grid search options -- yielded 256,180,64,0.01,1  -- lowest loss: 0.351
# h1s = [256]
# h2s = [128,180]
# h3s = [64]
# lrs = [0.01]
# lambdas = [1]

# Fourth grid search options -- yielded 320,230,64,0.01,2  -- lowest loss: 0.348
h1s = [256,280,320,400]
h2s = [200,230,260]
h3s = [64]
lrs = [0.01]
lambdas = [1,2]

def do_grid_search():
    lowest_validation_loss = 1000.0
    best_h1, best_h2, best_h3, best_lr, best_l1 = 0, 0, 0, 0, 0
    for h1 in h1s:
        for h2 in h2s:
            for h3 in h3s:
                for lr in lrs:
                    for l1_fac in lambdas:
                        val_loss = try_model(h1,h2,h3,lr,l1_fac)
                        if val_loss < lowest_validation_loss:
                            lowest_validation_loss = val_loss
                            best_h1 = h1
                            best_h2 = h2
                            best_h3 = h3
                            best_lr = lr
                            best_l1 = l1_fac
                        print(h1,h2,h3,lr,l1_fac, "   ", val_loss)

    print(best_h1, best_h2, best_h3, best_lr, best_l1)
    print(lowest_validation_loss)
>>>>>>> 8bb9fd5ae10e398d28ac5647c08970c049636f6c