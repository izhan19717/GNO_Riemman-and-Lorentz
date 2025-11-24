import h5py
import os

path = "data/pdebench/2D_DarcyFlow_beta1.0_Train.hdf5"
print(f"Checking {path}...")

if not os.path.exists(path):
    print("File does not exist!")
else:
    try:
        with h5py.File(path, 'r') as f:
            print("Keys:", list(f.keys()))
            for k in f.keys():
                print(f"{k}: {f[k].shape}, {f[k].dtype}")
                
                # Try to read first element
                try:
                    print(f"  First element shape: {f[k][0].shape}")
                    if k == 't-coordinate':
                        # print(f"  Values: {f[k][:10]}...")
                        pass
                    if k == 'x-coordinate':
                        # print(f"  Values: {f[k][:10]}...")
                        pass
                        
                    if k == 'tensor':
                        # Check variation along axis 1
                        data = f[k][:5] # Read first 5 samples
                        # data shape: (5, 201, 1024)
                        diff = data[:, 0, :] - data[:, -1, :]
                        print(f"  Diff between t=0 and t=end: {diff.min()}, {diff.max()}")
                        if (diff == 0).all():
                            print("  Data is CONSTANT along axis 1.")
                        else:
                            print("  Data VARIES along axis 1 (Time dependent).")
                            
                except Exception as e:
                    print(f"  Error reading first element: {e}")
                    
    except Exception as e:
        print(f"Error opening file: {e}")
