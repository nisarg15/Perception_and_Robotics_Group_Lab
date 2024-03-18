Clone the GitHub repository and place it in the home 
Download the data with the link given below and place 2 folders in the folder called prg/DeepFit/tutorial\

https://drive.google.com/drive/folders/1zqbM87yC7e--9tMDCIDwKHfmsReMYwWI?usp=sharing


````
old_compute_normals.py (Buggy Method that works)

````
* Change the variable named default in the line to the name of the sequence to load
"parser.add_argument('--input', type=str, default='scene10_dyn_train_00_000000', help='full path to input point cloud')"
* Output
    pickle files of x,y coordinates, and normal flow associated to in the folder called nf



````
compute_normals.py (Fast Method that doesn't work)

````
* Change the variable named default in the line to the name of the sequence to load
"parser.add_argument('--input', type=str, default='scene10_dyn_train_00_000000', help='full path to input point cloud')"
* Output
    pickle files of x,y coordinates, and normal flow associated to in the folder called nf



````
normal_flow_supervised.py

````
* This code will take the normal flow and x,y coordinates from the nf 
 folder and compare it with the ground truth

 Change the variable named pc according to your pc name and the variable name folder according to the name of the sequence 
