'''
THIS SCRIPT IS FOR NEW FUNCITONS TO PREPOROCESS THE SIGNALS
'''


import pandas as pd
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt


''' This is for inspecting the signals and getting the max and min values of the signals'''

def inspect_signals(ids, prefix, directory='.\\SIGNALS_FINAL'):

    values = {}
    for id in ids:
        id = int(id)
        filename = f'{prefix}{id}.h5'
        filepath = os.path.join(directory, filename)

        if os.path.exists(filepath):
            with h5py.File(filepath, 'r') as f:
                dataset_name = f'{prefix}{id}_dataset'
                data = f[dataset_name][:7, :]
                print(f"shape of data: {data.shape}")
                print(f"total length of data: {data.size}")
                # Flatten the data to calculate the 95th percentile
                flattened_data = data.flatten()
                print(f"shape of flattened data: {flattened_data.shape}")
                max_val = np.max(flattened_data)
                min_val = np.min(flattened_data)
                median_val = np.median(flattened_data)
                values[id] = {'max': max_val, 'min': min_val, 'median': median_val}

                print("done")
    return values


''' This is for segment-based normalization the signals'''

def cut_signals(ids, samples, save_directory, directory='.\\SIGNALS_FINAL'):
    # Create save directory if it does not exist
    os.makedirs(save_directory, exist_ok=True)

    for id in ids:
        id = int(id)
        filename = f'EMG_{id}.h5'
        filepath = os.path.join(directory, filename)

        if os.path.exists(filepath):
            with h5py.File(filepath, 'r') as f:
                dataset_name = f'EMG_{id}_dataset'
                data = f[dataset_name][:7, :]
                print(f"shape of data: {data.shape}")
                
                _, N = data.shape

                if N > samples:
                    # Shorten the signal to keep only the last `num_samples` samples
                    adjusted_emg_data = data[:, -samples:]
                    print(f"shape of adjusted data: {adjusted_emg_data.shape}")
                else:
                    # If the signal is shorter, prepend zeros
                    zeros_to_add = samples - N
                    adjusted_emg_data = np.pad(data, ((0, 0), (zeros_to_add, 0)), 'constant', constant_values=0)
                    print('zeros to add:', zeros_to_add)
                    print(f"shape of adjusted data: {adjusted_emg_data.shape}")

                #plot the signals
                #plot_emg_data(data, adjusted_emg_data, id)
                
                
            # Save the normalized data
            save_path = os.path.join(save_directory, f'EMG_{id}.h5')
            with h5py.File(save_path, 'w') as f:
                dataset_name = f'EMG_{id}_dataset'
                f.create_dataset(dataset_name, data= adjusted_emg_data)
            print(f'File saved: {save_path}')

        else:
            print(f'File does not exist: {filepath}')

        


''' This is for visualizing the signals, just comprovation'''

def plot_emg_data(original_emg_data, adjusted_emg_data, id):
    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot original EMG data
    axs[0].plot(original_emg_data[0, :], label='Channel 1')
    axs[0].plot(original_emg_data[1, :], label='Channel 2')
    axs[0].plot(original_emg_data[2, :], label='Channel 3')
    axs[0].plot(original_emg_data[3, :], label='Channel 4')
    axs[0].plot(original_emg_data[4, :], label='Channel 5')
    axs[0].plot(original_emg_data[5, :], label='Channel 6')
    axs[0].plot(original_emg_data[6, :], label='Channel 7')
    axs[0].legend()
    axs[0].set_title(f'Original EMG_{id}')

    # Plot adjusted EMG data
    axs[1].plot(adjusted_emg_data[0, :], label='Channel 1')
    axs[1].plot(adjusted_emg_data[1, :], label='Channel 2')
    axs[1].plot(adjusted_emg_data[2, :], label='Channel 3')
    axs[1].plot(adjusted_emg_data[3, :], label='Channel 4')
    axs[1].plot(adjusted_emg_data[4, :], label='Channel 5')
    axs[1].plot(adjusted_emg_data[5, :], label='Channel 6')
    axs[1].plot(adjusted_emg_data[6, :], label='Channel 7')
    axs[1].legend()
    axs[1].set_title(f'Adjusted EMG_{id}')

    # Adjust layout
    plt.tight_layout()
    plt.show()

'''This is for normalization of the signals by the 95th percentile of the SUBJECT'''
def process_and_normalize(ids, directory, save_directory):
    data_points = []
    total_length_check = 0  # check total number of data points

    # First pass: Collect data points to compute the 95th percentile
    for id in ids:
        id = int(id)
        filename = f'EMG_{id}.h5'
        filepath = os.path.join(directory, filename)

        if os.path.exists(filepath):
            # print(f'Loading file: {filepath}')
            with h5py.File(filepath, 'r') as f:
                dataset_name = f'EMG_{id}_dataset'
                data = f[dataset_name][:7, :]
                flattened_data = data.flatten()
                data_points.extend(flattened_data)
                total_length_check += flattened_data.size  # Accumulate the total number of data points
        else:
            print(f'File does not exist: {filepath}')

    if not data_points:
        print("No files were loaded, exiting function.")
        return

    # Verify that the length of data_points matches the accumulated total_length_check
    print(f"Total data points collected: {len(data_points)}")
    print(f"Total length check: {total_length_check}")
    print('-----------------------------------')

    assert len(data_points) == total_length_check, "Mismatch in total data points calculated."

    # Calculate the 95th percentile from all collected data points
    p95 = np.percentile(data_points, 95)
    print(f"95th Percentile of all data: {p95}")
    print('-----------------------------------')

    # Second pass: Normalize and save each file
    for id in ids:
        id = int(id)
        filename = f'EMG_{id}.h5'
        filepath = os.path.join(directory, filename)

        if os.path.exists(filepath):
            with h5py.File(filepath, 'r') as f:
                dataset_name = f'EMG_{id}_dataset'
                data = f[dataset_name][:7, :]
                normalized_data = data / p95  # Normalize the data

            # Save the normalized data
            save_path = os.path.join(save_directory, f'SN_EMG_{id}.h5')
            with h5py.File(save_path, 'w') as f:
                dataset_name = f'SN_EMG_{id}_dataset'
                
                """ 
                print(f'Saving file: {filepath}')
                print()
                print(f'Shape of normalized data: {normalized_data.shape}')
                print(f'Original data shape: {data.shape}')
                print()
                """
                f.create_dataset(dataset_name, data=normalized_data)
            """  
            print(f'File saved: {save_path}')
            print('-----------------------------------')
            """
        else:
            print(f'File does not exist: {filepath}')


# This function normalizes the EMG signals with 95th percentile and saves them to a new directory
''' This is for the normalization by RECORDING (OR INDIVIDUAL ID). First we get all of the EMG signals
calculate the 95th percentile then normalize all of the signals of this RECORDING and save them to a new directory
'''

def process_and_normalize_individual(ids, directory, save_directory):
    # Create save directory if it does not exist
    os.makedirs(save_directory, exist_ok=True)

    for id in ids:
        id = int(id)
        filename = f'EMG_{id}.h5'
        filepath = os.path.join(directory, filename)

        if os.path.exists(filepath):
            with h5py.File(filepath, 'r') as f:
                dataset_name = f'EMG_{id}_dataset'
                data = f[dataset_name][:7, :]
                print(f"shape of data: {data.shape}")
                print(f"total length of data: {data.size}")
                # Flatten the data to calculate the 95th percentile
                flattened_data = data.flatten()
                print(f"shape of flattened data: {flattened_data.shape}")
                p95 = np.percentile(flattened_data, 95)
                print(f"95th Percentile for file {filename}: {p95}")
                
                # Normalize the data
                normalized_data = data / p95
                
            # Save the normalized data
            save_path = os.path.join(save_directory, f'NT_EMG_{id}.h5')
            with h5py.File(save_path, 'w') as f:
                dataset_name = f'NT_EMG_{id}_dataset'
                f.create_dataset(dataset_name, data=normalized_data)
            print(f'File saved: {save_path}')

        else:
            print(f'File does not exist: {filepath}')
        



'''This is for stacking data for training the model'''

def stack_signals(ids, directory, prefix):
    stacked_signals = []

    for id in ids:
        id = int(id)
        filename = f'{prefix}{id}.h5'  # Use the provided prefix
        filepath = os.path.join(directory, filename)
        
        print(f"Checking file: {filepath}")
        
        if os.path.exists(filepath):
            print('File exists')

            with h5py.File(filepath, 'r') as f:
                # Read the dataset directly
                dataset_name = f'{prefix}{id}_dataset'
                data = f[dataset_name][:7, :]  # Only take the first 7 channels
                
                print(f'Opening {filename}...')
                
                # Stack the data
                stacked_signals.append(data)

        else:
            print(f"File does not exist: {filepath}")

    # Stack the signals along the last axis
    stacked_signals = np.stack(stacked_signals)
    print(f"Shape of stacked signals: {stacked_signals.shape}")
    return stacked_signals


  