## Creating Dataset
### Point Cloud
The point clouds are obtained using the EVIMOv2 Dataset <br>
The link to the Dataset is provided below <br>
https://better-flow.github.io/evimo/download_evimo_2.html <br>
Download the .npz of samsung_mono of the Motion Segmentation / Object Recognition dataset<br>

Extract all the files and combine all the dataset sequences from /npz_samsung_mono_imo/samsung_mono/imo/evel and /home/enigma/Downloads/npz_samsung_mono_imo/samsung_mono/imo/train in one folder named dataset.<br>
In every subfolder create an empty folder by the name of "nf"

### Ground Truth
To obtain the ground truth download the from the below given GoogleDrive link<br>
https://drive.google.com/drive/folders/1tC0S10VbLei8HYtGcf8pPDxz997U76sX?usp=sharing<br>
Extract the zip files of each sequence and place them in the sub folders of same name which are present in the dataset folder created for the pointcloud data.

## Code
### Computing the Normal Flow
Clone the GitHub repository and place it in the dataset folder

````
compute_normals.py 

````
* Change the default variable to the subfolder's name for which normal flow is to be generated in the line given below in the code<br>
### "parser.add_argument('--input', type=str, default='scene10_dyn_train_00_000000', help='full path to input point cloud')"<br>
* Update data_set_folder to the dataset folder's location.<br>
* Change the name of the variable folder whos normalflow is to genrate.<br>
* Ensure you're in the dataset directory when executing the code<br>

* Output
    pickle files of x,y coordinates, and normal flow will be genrated in the empty folder called nf

### Generating ouput videos and excel files

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
