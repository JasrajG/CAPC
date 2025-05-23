import scipy.io
import numpy as np

def convert_mat_to_npy(mat_file_path, output_npy_path):
    #converts matlab dataset to numpy array and saves it as .npy
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        print(f"Successfully loaded MATLAB file: {mat_file_path}")

        print("keys in .mat file:", mat_data.keys())

        if 'label_lab' in mat_data: #replace the specific keyname in the ''
            numpy_array = mat_data['label_lab']#replace here again within the ''
            np.save(output_npy_path, numpy_array)
            print(f"Numpy array saved to: {output_npy_path}")
        
        else:
             print(f"Error: The specified data key {'your_data_key'} was not found in the MATLAB file.")
             print("Please inspect the keys above and update the script.")
    
    except FileNotFoundError:
        print(f"Error: MATLAB file not found at {mat_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    matlab_file = input("Enter the path to your .mat file:")
    npy_file = input("Enter the path to save the output numpy .npy file")
    convert_mat_to_npy(matlab_file, npy_file)
