

from model import SiameseModel
from lib import *

from model_train import def_triplet_loss_margin
from model_preprocess import def_global_image_height, def_global_image_width


def_threshold = def_triplet_loss_margin
global_image_height = def_global_image_height
global_image_width = def_global_image_width


############################### FUNCTIONS ###############################

def get_image_from_path(image_path):
	
	"""
	Returns a numpy array containing the image at the provided `path`.
	"""

	image = cv2.imread(str(image_path))
	image = np.array(image)
	return image



def get_encodings(matrix):

	'''
	Accepts a Tensor and returns its encoding.
	'''

	if torch.cuda.is_available():
		matrix = Variable(matrix.cuda())    
	else:
		matrix = Variable(matrix)

	matrix = matrix.float()
	matrix_enc = model(matrix)

	return matrix_enc



def return_diff(anchors_enc, positives_enc, negatives_enc):
 
	'''
	Accepts the encodings of three tensors.
	
	Returns d(E1,E2) and d(E1,E3) where d(A,B) is the 
	Frobenius norm of the vector A-B.
	
	Returns the result as a pair of numpy arrays.
	'''

	assert(anchors_enc.shape == positives_enc.shape)
	assert(anchors_enc.shape == negatives_enc.shape)
	
	num = anchors_enc.shape[0]
	pos_diff_vec = []
	neg_diff_vec = []
	
	anchors_enc = anchors_enc.cpu().detach().numpy()
	positives_enc = positives_enc.cpu().detach().numpy()
	negatives_enc = negatives_enc.cpu().detach().numpy()
	
	for i in range(num):
		pos_diff = np.linalg.norm(anchors_enc[i] - positives_enc[i])
		neg_diff = np.linalg.norm(anchors_enc[i] - negatives_enc[i])
		
		pos_diff_vec.append(pos_diff)
		neg_diff_vec.append(neg_diff)
   
	return np.array(pos_diff_vec), np.array(neg_diff_vec)    


def process(matrix_path):

    """
    Processes an array of images.
    Takes their path as input and returns thier encoding.
    Helper function for `assess`.
    """
    
    matrix = get_image_from_path(matrix_path)
    matrix = cv2.resize(matrix, (global_image_width, global_image_height))
    matrix = np.rollaxis(matrix, 2, 0)
    matrix = matrix.reshape(1, 3, global_image_height, global_image_width)        
    matrix = torch.from_numpy(matrix)
    matrix_enc = get_encodings(matrix)

    return matrix_enc


def assess(anchor_path, image1_path, image2_path, threshold):
    
    """
    Takes in three image paths  : anchor - Image corresponding to data in database.
                                : image1 - First input image
                                : image2 - Second input image
    
    and prints which among image1 and image2 is close to anchor image.
    
    If difference is less than `threshold` then notifies accordingly.
    
    """
    
    anchor_enc = process(anchor_path)               
    image1_enc = process(image1_path)               
    image2_enc = process(image2_path)               

    image1_dist_list, image2_dist_list = return_diff(anchor_enc, image1_enc, image2_enc)
    image1_dist = image1_dist_list[0]
    image2_dist = image2_dist_list[0]
    
    diff = image1_dist - image2_dist
        
    # Return output
    
    print("=======================================================")   
    
    # Found forgery
    if(abs(diff) >= threshold):
        if(image1_dist < image2_dist):
            print("VERDICT : Second signature (s2) appears to forged.")
        else:
            print("VERDICT : First signature (s1) appears to forged.")
    
    # Cannot finnd forgery
    else:
        print("VERDICT : Unable to determine authenticity.")
        print("\tEither provide different samples or lower threshold.")
        print("\tAlso, please cross-check the Achor provided.")

    print("---> Current Threshold \t\t\t : {:.4f}".format(threshold))
    print("---> Difference between Encodings \t : {:.4f}".format(abs(diff)))        
    print("---> Distance between Anchor and Image_1 : {:.4f}".format(image1_dist))
    print("---> Distance between Anchor and Image_2 : {:.4f}".format(image2_dist))

    print("=======================================================")


############################### MAIN ###############################

if __name__ == "__main__":

	# Parsing arguments.

	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--anchor-path", 
		"-a"
	)

	parser.add_argument(
		"--image1-path", 
		"-s1"
	)
	
	parser.add_argument(
		"--image2-path", 
		"-s2"
	)
	
	parser.add_argument(
		"--threshold", 
		"-t",
		default = def_threshold,
		type = float,
		help = "Threshold for distance between images."
	)

	args = parser.parse_args()

	if (args.anchor_path == None or 
		args.anchor_path == None or 
		args.anchor_path == None):

		print("Incorrect arguments. Exiting.")
		exit(0)

    # __________________________________________________________________________________

	model = SiameseModel()

	if torch.cuda.is_available():
	    model.cuda()	

	model.load_state_dict(torch.load('state_dicts/state_dict_margin=1.pt'))
	model.eval()

	assess(args.anchor_path, args.image1_path, args.image2_path, args.threshold)
