
from model import SiameseModel
from lib import *

def_working_directory = "/home/Automated-Signature-Verification/Terminal_scripts"
def_training_set_path = "./sign_data/train"
def_test_set_path = "./sign_data/test"

def_real_extention = ""
def_fake_extention = "_forg"

def_global_image_height = 400
def_global_image_width = 700


train_master_index = []
test_master_index = []


############################### FUNCTIONS ###############################

def get_single_perm (is_train, location):
    
    """
    Retuns a list containing all [anchor, positive, negative] path permutations 
    in a given folder.
    
    Note : `location` must be an integer denoting the folder number.
    """
   
    index = []
        
    if(is_train):
        root_path = training_set_path
    else:
        root_path = test_set_path
    
    originals = os.listdir(os.path.join(root_path, location + real_extention))
    forgeries = os.listdir(os.path.join(root_path, location + fake_extention))

    originals.sort()
    forgeries.sort()
    
    for i in range(len(originals)):
        for j in range(i+1,len(originals)):
            for k in range(len(forgeries)):
                
                path_anchor = os.path.join(root_path, location + real_extention, originals[i])
                path_real = os.path.join(root_path, location + real_extention, originals[j])
                path_fake = os.path.join(root_path, location + fake_extention, forgeries[k])
                
                index.append([path_anchor, path_real, path_fake])
  
    return index


def build_file_index (is_train = True):
    
    """
    Returns a master index file that contains all the 
    [anchor, real, fake] path permuataions across the whole dataset.    
    """
        
    if(is_train):
        directory = training_set_path
    else:
        directory = test_set_path
    
    folder_list = os.listdir(directory)
    folder_list.sort()
                           
    master_index = []
                           
    for i in range(len(folder_list)):
        if(i%2==0):
            master_index = master_index + get_single_perm(is_train, folder_list[i])    
    
    return master_index



def get_image_from_path (image_path):
    
    """
    Returns a numpy array containing the image at the provided `path`.
    """
    
    image = cv2.imread(str(image_path))
    image = np.array(image)
    return image



def print_img (image):
    
    """
    Plots a simgle image. 
    Accepts raw data, NOT path.
    """    
    
    imgplot = plt.imshow(image)
    plt.show()


def resize_images (
	new_dim1,
	new_dim2, 
	is_train = True
	):

    '''
    Transforms all images in the dataset into correct sizes.
    Then rewrites them at their original path.
    '''
    
    if(is_train):
        directory = training_set_path
    else:
        directory = test_set_path

    # Get all Folders    
    folder_list = os.listdir(directory)
    folder_list.sort()            
        
    for folder in folder_list:
        
        # Get all Images in a folder
        image_list = os.listdir(os.path.join(directory,folder))
        image_list.sort()    
                                  
        for image_name in image_list:
            
            # Get image path
            image_path = str(os.path.join(directory,folder,image_name))
            
            # Get image
            image = get_image_from_path(image_path)
            
            # Resize image
            resized_image = cv2.resize(image, (new_dim2, new_dim1))
            
            # Overwrite the Image
            cv2.imwrite(image_path, resized_image)        


############################### MAIN ###############################

if __name__ == "__main__":

	# Parsing arguments.

	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--working-directory", 
		"-w", 
		default = def_working_directory,
		type = str,
		help = "Path to directry contain notebook and dataset."
	)
	
	parser.add_argument(
		"--training-set-path", 
		"-tr", 
		default = def_training_set_path,
		type = str,
		help = "Path to training set."
	)

	parser.add_argument(
		"--test-set-path", 
		"-ts",
		default = def_test_set_path,
		type = str, 
		help = "Path to test set."
	)

	parser.add_argument(
		"--real-extention", 
		"-re", 
		default = def_real_extention,
		type = str,
		help = "Extention applied to folders containg real files."
	)

	parser.add_argument(
		"--fake-extention", 
		"-fe", 
		default = def_fake_extention,
		type = str,
		help = "Extention applied to folders containg fake files."
	)	

	parser.add_argument(
		"--global-image-height", 
		"-d1", 
		default=def_global_image_height,
		type=int,
		help="Height (in pixel) to which all images would be reshaped."
	)

	parser.add_argument(
		"--global-image-width", 
		"-d2", 
		default=def_global_image_width,
		type=int,
		help="Width (in pixel) to which all images would be reshaped."
	)

	args = parser.parse_args()

	working_directory = args.working_directory
	training_set_path = args.training_set_path
	test_set_path = args.test_set_path
	real_extention = args.real_extention
	fake_extention = args.fake_extention
	global_image_height = args.global_image_height
	global_image_width = args.global_image_width

    # __________________________________________________________________________________


	os.chdir(working_directory)

	print("Started preprocessing...")

	resize_images(global_image_height, global_image_width, is_train = True)
	resize_images(global_image_height, global_image_width, is_train = False)

	print("Resizing images Done.")

	# List of all the path permutations in the train file
	train_master_index = build_file_index(is_train = True)

	# List of all the path permutations in the test file
	test_master_index = build_file_index(is_train = False)

	fileObject = open('master_indices.pkl', 'wb')
	pickle.dump([train_master_index, test_master_index],fileObject)  
	fileObject.close()

	print("Building file indexes Done.")	

	print("Training set size : ", len(train_master_index))
	print("Test set size : ", len(test_master_index))	
	print("Sample Image File : ")

	print_img(get_image_from_path('./sign_data/train/001/001_01.PNG'))

