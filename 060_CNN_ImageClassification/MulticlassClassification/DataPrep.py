#%% packages
import requests
import urllib.request
from pathlib import Path

#%%
def get_dog_images(breed, count=100, type='train'):
    available_breeds = ['affenpinscher', 'african', 'airedale', 'akita', 'appenzeller', 'australian', 'basenji', 'beagle', 'bluetick', 'borzoi', 'bouvier', 'boxer', 'brabancon', 'briard', 'buhund', 'bulldog', 'bullterrier', 'cattledog', 'chihuahua', 'chow', 'clumber', 'cockapoo', 'collie', 'coonhound', 'corgi', 'cotondetulear', 'dachshund', 'dalmatian', 'dane', 'deerhound', 'dhole', 'dingo', 'doberman', 'elkhound', 'entlebucher', 'eskimo', 'finnish', 'frise', 'germanshepherd', 'greyhound', 'groenendael', 'havanese', 'hound', 'husky', 'keeshond', 'kelpie', 'komondor', 'kuvasz', 'labradoodle', 'labrador', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese', 'mastiff', 'mexicanhairless', 'mix', 'mountain', 'newfoundland', 'otterhound', 'ovcharka', 'papillon', 'pekinese', 'pembroke', 'pinscher', 'pitbull', 'pointer', 'pomeranian', 'poodle', 'pug', 'puggle', 'pyrenees', 'redbone', 'retriever', 'ridgeback', 'rottweiler', 'saluki', 'samoyed', 'schipperke', 'schnauzer', 'setter', 'sheepdog', 'shiba', 'shihtzu', 'spaniel', 'springer', 'stbernard', 'terrier', 'tervuren', 'vizsla', 'waterdog', 'weimaraner', 'whippet', 'wolfhound']
    for i in range(count):
        if breed not in available_breeds:
            raise ValueError('Breed not available')
        else:
            API = f'https://dog.ceo/api/breed/{breed}/images/random'
        
        img_url = requests.get(API).json()['message']
        # create folder if necessary
        breed_path = f"{type}/{breed}"
        Path(breed_path).mkdir(parents=True, exist_ok=True)
        
        # download an image and save it in folder
        img = urllib.request.urlretrieve(img_url, f"{breed_path}/{breed}_{i}.jpg")
    return None

def get_breeds():
    API = 'https://dog.ceo/api/breeds/list/all'
    breeds = requests.get(API).json()['message']
    return breeds.keys()

# %% get train images
get_dog_images('affenpinscher')
get_dog_images('corgi')
get_dog_images('akita')
#%% get test images
get_dog_images('affenpinscher', type='test', count=20)
get_dog_images('corgi', type='test', count=20)
get_dog_images('akita', type='test', count=20)

# %%
