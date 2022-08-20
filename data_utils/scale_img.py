from PIL import Image
import os
from tqdm import tqdm

if __name__ == "__main__":
    down_scale = 4
    result_root_folder = "../data/WaymoDataset/images/"
    save_folder = "../data/scaled_images"
    os.makedirs(save_folder, exist_ok=True)
    for file in os.walk(result_root_folder):
        for file in tqdm(file[2]):
            if ".png" in file:
                img = Image.open(os.path.join(result_root_folder, file)).convert('RGB')
                width = img.width // down_scale
                height = img.height // down_scale
                img = img.resize((width, height),Image.Resampling.LANCZOS)
                img.save(os.path.join(save_folder,file),"PNG")
                print()

'''
width = img_info['width'] // self.img_downscale
            height = img_info['height'] // self.img_downscale

            if self.split == 'val':
                img = Image.open(os.path.join(
                    self.root_dir, 'images', img_info['image_name'])).convert('RGB')
                if self.img_downscale != 1:
                    img = img.resize((width, height),
                                     Image.Resampling.LANCZOS)  # cv2.imshow("123.png",cv2.cvtColor(np.array(img),cv2.COLOR_BGR2RGB)),cv2.waitKey()
                img = self.transform(img)  # (3,h,w)
                img = img.view(3, -1).permute(1, 0)
'''
