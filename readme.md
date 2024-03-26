Clone the GitHub repository and place it in the home 

````
compute_normals.py 

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

 ## Output
 ### Output Video


https://github.com/nisarg15/Perception_and_Robotics_Group_Lab/assets/89348092/0a1a8c2f-ca1d-4fe5-83f5-fc7c3b8d95a1

 

Nisarg_Upadhyay
