
from lib import *
from model_preprocess import get_image_from_path


def transform_image(path):
    
    """
    Rotate the image so that the model can process it.
    """
    
    image = get_image_from_path(path)
    image = np.rollaxis(image, 2, 0)
    return image


##################### DATASET DEFINITION #####################

class SignatureDataset(Dataset):
        
    def __init__(self, train_master_index, test_master_index, is_train=True, transform=None):  
        
        self.is_train = is_train
        self.transform = transform
        self.train_master_index = train_master_index
        self.test_master_index = test_master_index
        
    def __getitem__(self, index):
        
        if(self.is_train):
            master_index = self.train_master_index
        else:
            master_index = self.test_master_index
        
        # Get paths
        anchor_path = master_index[index][0]
        real_path = master_index[index][1]
        fake_path = master_index[index][2]
            
        # Transform images
        anchor = transform_image(anchor_path)
        real = transform_image(real_path)
        fake = transform_image(fake_path)
        
        if self.transform is not None:
            anchor = self.transform(anchor)
            real = self.transform(real)
            fake = self.transform(fake)
        
        return anchor, real, fake
    
    
    def __len__(self):

        if(self.is_train):
            return len(self.train_master_index)
        else:
            return len(self.test_master_index)
        
        