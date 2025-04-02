import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

def get_aerial_image(x, y, offset=20, save_path=None):
    """
    Fetches an aerial image from PDOK WMS for given x, y coordinates (EPSG:28992) and an offset.
    
    Parameters:
        x (float): X-coordinate (RD)
        y (float): Y-coordinate (RD)
        offset (float): Buffer distance to create the bounding box
        save_path (str): If provided, the image will be saved to this path.
        
    Returns:
        PIL.Image object if successful, otherwise None.
    """
    # Create bounding box around the point (min_x, min_y, max_x, max_y)
    bbox = f"{x - offset},{y - offset},{x + offset},{y + offset}"
    wms_url = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": "Actueel_orthoHR",  # or "Actueel_ortho25" if desired
        "crs": "EPSG:28992",
        "bbox": bbox,
        "width": 500,
        "height": 500,
        "format": "image/jpeg"
    }
    
    response = requests.get(wms_url, params=params)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        if save_path:
            image.save(save_path)
            print(f"Image saved to {save_path}")
        return image
    else:
        print("Failed to fetch image. HTTP status code:", response.status_code)
        return None

# For testing when run standalone:
if __name__ == "__main__":
    # Example test coordinates
    x = 259294.98
    y = 470400.44
    get_aerial_image(x, y, offset=20)
