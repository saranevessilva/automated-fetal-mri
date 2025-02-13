import h5py


def print_hdf5_structure(filename):
    # Open the HDF5 file in read mode
    with h5py.File(filename, 'r') as file:
        print(f"File structure for {filename}:")
        
        # Recursively print the structure of the file
        def print_group(name, obj):
            print(f"{name}: {type(obj)}")
            if isinstance(obj, h5py.Dataset):
                print(f" - Shape: {obj.shape}, Data Type: {obj.dtype}")
                if obj.attrs:
                    print(" - Attributes:")
                    for key, value in obj.attrs.items():
                        print(f"   * {key}: {value}")
            elif isinstance(obj, h5py.Group):
                if obj.attrs:
                    print(" - Attributes:")
                    for key, value in obj.attrs.items():
                        print(f"   * {key}: {value}")
        
        # Call the function for every item in the file
        file.visititems(print_group)


# Specify the HDF5 file name here
file_name = '/home/sn21/data/t2-stacks/test/haste_cor_wholeuterus_2025-01-07-122351_69.h5'

# Run the function to print the header information
print_hdf5_structure(file_name)

# Open the HDF5 file
with h5py.File(file_name, 'r') as hdf:
    # Access the header dataset
    header_data = hdf['dataset/images_0/header']

    # Access specific fields: read_dir, phase_dir, and slice_dir
    read_dir = header_data['read_dir']
    phase_dir = header_data['phase_dir']
    slice_dir = header_data['slice_dir']

    # Print the values of the read_dir, phase_dir, and slice_dir for each entry
    for i in range(len(read_dir)):
        print(f"Entry {i+1}:")
        print(f"  Read Direction: {read_dir[i]}")
        print(f"  Phase Direction: {phase_dir[i]}")
        print(f"  Slice Direction: {slice_dir[i]}")
