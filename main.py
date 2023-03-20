import pandas, string, torch, time, numpy as np
from sklearn.model_selection import train_test_split
from mlp import MLP

# 0.6574, 0.343, 0.0178, 0.0006         2M, 5e-4, 420 epochs, batch 32768, lambda=3, 216/108/54

N = 2000000                         # Number of training data points to use
MAX_WORD_LENGTH = 8
ALPHABET = string.ascii_lowercase + '_'
LEARNING_RATE = 5e-4
NUM_EPOCHS = 10000
BATCH_SIZE = 32768                  # Number of data points worked on by GPU in parallel
LAMBDA = 3                          # L1 regularization factor

def one_hot_encode(strings, dists, device):
    strings = strings.tolist()
    dists = dists.tolist()
    indices_np = np.array([[ord(c) - ord('a') for c in s] for s in strings], dtype=np.int64)
    x_np = np.eye(len(ALPHABET), dtype=np.float32)[indices_np]
    x = torch.from_numpy(x_np).to(device)
    y = torch.as_tensor(dists, device=device)
    return x, y

def main():
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

    model = MLP(MAX_WORD_LENGTH * len(ALPHABET),216,108,54).to(device)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.1)

    criterion_mse, criterion_l1 = torch.nn.MSELoss(), torch.nn.L1Loss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_train_loss = 0.0
        num_batches = len(train_data_x) // BATCH_SIZE

        for i in range(num_batches):
            inputs,labels = train_data_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE],train_data_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            optimizer.zero_grad()
            outputs = model(inputs.float())
            mse_loss = criterion_mse(outputs, labels.float().unsqueeze(1))
            l1_loss = criterion_l1(torch.cat([p.view(-1) for p in model.parameters()], dim=0), torch.zeros_like(torch.cat([p.view(-1) for p in model.parameters()], dim=0)))
            total_loss = mse_loss + LAMBDA * l1_loss
            total_loss.backward()
            optimizer.step()
            running_train_loss += mse_loss.item()

        model.eval()
        running_val_loss = 0.0
        running_rounded_val_loss = 0.0
        num_test_batches = len(test_data_x) // BATCH_SIZE
        error_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        total_samples = 0
        elapsed_time = 0.0
        with torch.no_grad():
            for i in range(num_test_batches):
                inputs,labels = test_data_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE],test_data_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

                start_time = time.time()
                outputs = model(inputs.float())
                end_time = time.time()
                elapsed_time += (end_time - start_time)
                running_val_loss += criterion_mse(outputs, labels.float().unsqueeze(1)).item()

                # Round the outputs to the nearest integer
                rounded_outputs = torch.round(outputs)
                errors = torch.abs(rounded_outputs - labels.float().unsqueeze(1))
                for error in error_counts:
                    error_counts[error] += torch.sum(errors == error).item()

                total_samples += errors.shape[0]

        # Print the portion of each error value
        print("Buckets: ", end='')
        for error, count in error_counts.items():
            portion = count / total_samples
            print(f"{portion:.2%}", end='     ')
        print('')

        #test_x, test_y = one_hot_encode_list(['abcdefg'], [3.0], 'cuda')
        #output = model(test_x.float())
        #print("abcdefg: ", output)

        avg_train_loss = running_train_loss / num_batches
        avg_val_loss = running_val_loss / num_test_batches
        avg_rounded_val_loss = running_rounded_val_loss / num_test_batches
        print(f'Epoch {epoch+1}: train loss={avg_train_loss:.4f}, val loss={avg_val_loss:.4f}, time={elapsed_time:.4f}')
        scheduler.step(avg_val_loss)

main()