import cv2

from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from deepproblog.examples.Autonomous_driving.version_2.data.AD_generate_datasets_baseline_0 import get_dataset


def show_transformed_img(img_path, transform=None):
    if transform is None:
        transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
    transformed_img = transform(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)).permute(1, 2, 0)
    plt.imshow(transformed_img)
    plt.show()


transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((32, 32), antialias=False, interpolation=InterpolationMode.NEAREST_EXACT),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
show_transformed_img('/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/balanced/version_3_env_3/medium/2/0_iter7frame23.png', transform)