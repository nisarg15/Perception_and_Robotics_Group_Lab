import os
import sys
import numpy as np
sys.path.insert(0, '../utils')
sys.path.insert(0, '../models')
sys.path.insert(0, '../trained_models')
import argparse
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import math
import time
import pickle
import pprint   
import time


newwwww_xyz = []

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='scene15_dyn_test_06_000000', help='full path to input point cloud')
parser.add_argument ('--output_path', type=str, default='../log/outputs/results', help='full path to input point cloud')
parser.add_argument('--gpu_idx', type=int, default=0, help='index of gpu to use, -1 for cpu')
parser.add_argument('--trained_model_path', type=str, default='../trained_models/DeepFit',
                    help='path to trained model')
parser.add_argument('--mode', type=str, default='DeepFit', help='how to compute normals. use: DeepFit | classic')
parser.add_argument('--k_neighbors', type=int, default=256, help='number of neighboring points for each query point')
parser.add_argument('--jet_order', type=int, default=3,
                    help='order of jet to fit: 1-4. if in DeepFit mode, make sure to match training order')
parser.add_argument('--compute_curvatures', type=bool, default=True, help='true | false indicator to compute curvatures')
args = parser.parse_args()

all_values = []
all_dot_values = []
final_cuboids = []
xyz_lst = []
n_lst = []
n_flow = []
xyz_lst_new = []

# def vector_magnitude(x, y):
#     return math.sqrt(x**2 + y**2)
def normal_flow(x,y,z):

    column_vector = np.array([[x], [y]])
    #coloum_vector = np.transpose(coloum_vector)
    squared_norm = np.linalg.norm((column_vector))**2
    result = (-z * (((column_vector)) / squared_norm))
    result = result*166.666666667
    
    return  result
def avg(x):
    squared_sum = np.sum(x**2)
    avg_squared_sum = squared_sum / np.size(x)
    avg = np.sqrt(avg_squared_sum)
    return avg
    

def divide_cuboid(cuboid_dimensions,x,y):
    divisions = []
    
    # Get the dimensions of the cuboid
    length, width = cuboid_dimensions
    
    # Calculate the dimensions of each divided cuboid
    divided_length = length / 2
    divided_width = width / 2
   
    
    # Generate the divisions
    for i in range(2):
        for j in range(2):
         
                x_start = (i * divided_length)+x

                x_end = x_start + divided_length

                y_start = j * divided_width+y
                y_end = y_start + divided_width
                
                division = {
                    'x': (x_start, x_end),
                    'y': (y_start, y_end)
                }
                divisions.append(division)
    return divisions




folder = "scene15_dyn_test_06_000000"   
pc = "enigma"  
points_xy = np.load(args.input + '/dataset_events_xy.npy').astype('float32')
points_t = np.load(args.input + '/dataset_events_t.npy').astype('float32')

t_file = f"/home/{pc}/prg/DeepFit/tutorial/{folder}/dataset_normal_flow/t.npy"
t_end_file = f"/home/{pc}/prg/DeepFit/tutorial/{folder}/dataset_normal_flow/t_end.npy"
t1 = np.load(t_file)
t2 = np.load(t_end_file)





all_values = []
all_dot_values = []
final_cuboids = []
xyz_lst = []
n_lst = []

idx_frame = 1
ts = points_t[0]


counts = 0


start_time = time.time()
while ts <= points_t[-1]:
    t1 = list(t1)
    t2 = list(t2)
    samples = [0,4,20,84,340,1364,5460,21844,87380]
    
    


    left_indices = np.searchsorted(points_t, t1[0])
  
    
 

   
    right_indices = np.searchsorted(points_t, t2[0], side='right')

    first_element = t1.pop(0)
    second_element = t2.pop(0)
   



    xyz = np.array([points_xy[left_indices:right_indices, 0], points_xy[left_indices:right_indices, 1], 10000*points_t[left_indices:right_indices]-np.min(10000*points_t[left_indices:right_indices])]).T
#     k = 20 # You can adjust this value based on your needs

# # Create a NearestNeighbors object
#     nn = NearestNeighbors(n_neighbors=k)

#     # Fit the NearestNeighbors model to your point cloud data
#     nn.fit(xyz)

#     # Find the k-nearest neighbors for each point
#     distances, indices = nn.kneighbors(xyz)


#     # Calculate normals based on the neighbors' positions (example calculation)
#     normals = np.zeros_like(xyz)  # Initialize normals array

#     for i in range(len(xyz)):
#         neighbors = xyz[indices[i]]  # Get neighbors' positions
#         centroid = np.mean(neighbors, axis=0)  # Calculate centroid
#         normal = xyz[i]- centroid  # Calculate normal vector
#         normals[i] = normal / np.linalg.norm(normal)
#         newwwww_xyz.append(xyz[i][0:2])
        


#     for i in normals:
#             normal__flow = normal_flow(i[0],i[1],i[2])
#             normal__flow = [item for sublist in normal__flow for item in sublist]  
#             n_lst.append(normal__flow) 
    xyz[:,2] = np.array (xyz[:,2], dtype=int)
  

 

    
    points = np.array(xyz)
    X, Y, Z = xyz[:,0],xyz[:,1],xyz[:,2]
    cuboid = [640,480]



    all_cuboids = []
    final_cuboids = []
    first_cuboid = { 'cuboid' : cuboid,
            'x': [0, 640],
            'y': [0, 480]

    }
    all_cuboids.append(first_cuboid)




    
    for i in range (100000000):
      
        
        
        if all_cuboids == []:
           
            break
        first_dict = all_cuboids[0]

        divisions = divide_cuboid(first_dict['cuboid'],first_dict['x'][0],first_dict['y'][0])
        # print("Print Divisions")
        # pprint.pprint(divisions)
        res_x = list(map(itemgetter('x'), divisions))
        res_y = list(map(itemgetter('y'), divisions))
     




    

        if i == samples[0]:
           
            cuboid[0] = cuboid[0]/2
            cuboid[1] = cuboid[1]/2


            samples.pop(0)

        for j in range (4):
           
            
            x_start = res_x[j][0]
            x_end = res_x[j][1]
            y_start = res_y[j][0]
            y_end = res_y[j][1]


            x_values = points[:, 0]  
            y_values = points[:, 1] 
            z_values = points[:, 2]  

            x_mask = np.logical_and(x_start <= x_values, x_values <= x_end)
            y_mask = np.logical_and(y_start<= y_values, y_values <= y_end)
            z_mask = np.logical_and(int(Z[0]) <= z_values, z_values <= int(Z[0]+(t2[0]*10000)))
        
        

        

            combined_mask = np.logical_and(np.logical_and(x_mask, y_mask), z_mask)
            selected_points = points[combined_mask]
            num_of_points = np.shape(selected_points)
            f_points = selected_points
           
           
            
            
            if num_of_points[0]<3:
               
                continue
            selected_points = np.array(selected_points)
            

            centroid = np.mean(selected_points, axis=0)
            selected_points -= centroid

            
            

            selected_points = np.array(selected_points)
            # Compute the covariance matrix
            covariance_matrix = np.cov(selected_points.T)
            # Perform eigenvalue decomposition on the covariance matrix
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
         

            # Find the eigenvector corresponding to the smallest eigenvalue
            min_eigenvalue_index = np.argmin(eigenvalues) 
            normal_vector = eigenvectors[:, min_eigenvalue_index]
            
            
            dot_product = np.dot(selected_points, normal_vector)
            av_pro = avg(dot_product)
            normal__flow = normal_flow(normal_vector[0],normal_vector[1],normal_vector[2])
            normal__flow = [item for sublist in normal__flow for item in sublist]    
      

           
            

            if av_pro<3:
    
                    sss = np.shape(f_points)
                    for n in range (sss[0]):
                        xyz_lst.append([f_points[n][0],f_points[n][1]])
                    
                        n_flow.append(normal__flow)
        
            else:
               
                small_cuboids= { 'cuboid' : cuboid,
                        'x': [x_start,x_end],
                        'y': [y_start,y_end]
                    }
                all_cuboids.append(small_cuboids)
                
        all_cuboids.pop(0)
     

    # print(np.shape(xyz_lst))
    # print

    # try:
    #     index = xyz_lst.index([554,13])
    #     print(f"The value {i} is at index: {index}")
    # except ValueError:
    #     print("not found")
    # exit()


    
    
    # plt.plot(all_values,all_dot_values)
    # plt.show()
    # normals_output_file_name = os.path.join(args.input, 'normals', os.path.splitext(os.path.basename(args.input))[0] + '_'+str(idx_frame)+'.normals')
    # xyz_output_file_name = os.path.join(args.input, 'xyz', os.path.splitext(os.path.basename(args.input))[0] + '_'+str(idx_frame)+'.xyz')
    normal_flow_file_name = os.path.join(args.input, 'nf', os.path.splitext(os.path.basename(args.input))[0] + '_'+str(idx_frame)+'.nf')
    new_xyz_file_name = os.path.join(args.input, 'nf', os.path.splitext(os.path.basename(args.input))[0] + '_'+str(idx_frame)+'.xyz')

    # np.savetxt(xyz_output_file_name, xyz_lst_new, delimiter=' ')
    # np.savetxt(normals_output_file_name, n_lst, delimiter=' ')



    with open(new_xyz_file_name, 'wb') as file:
        pickle.dump(xyz_lst, file)
    with open(normal_flow_file_name, 'wb') as file:
        pickle.dump(n_flow, file)

    print("files created")
    print(idx_frame)

    xyz_lst = []
    n_lst = []
    all_dot_values = []
    all_values = []
    n_flow = []

    idx_frame+=1
    counts = counts+1
   
end_time = time.time()
time_taken = end_time - start_time
print(f"Time taken: {time_taken} seconds")

