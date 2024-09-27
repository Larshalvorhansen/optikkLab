import numpy as np
import matplotlib.pyplot as plt

def get_striped_spectrogram(spectrogram, spectral_lines=None):
    width = spectrogram.shape[1]
    if spectral_lines is not None:
        assert len(spectral_lines) < width
    else:
        spectral_lines = []
        for _ in range(2):
            index = np.random.randint(0, width, dtype=int)
            amplitude = np.random.uniform(0.1, 1)
            fwhm = np.random.uniform(0.1, 2)
            spectral_lines.append((index, amplitude, fwhm))

    for index, amplitude, fwhm in spectral_lines:
        # add a gaussian line to the spectrogram at the given index
        spectrogram += amplitude * np.exp(-((np.arange(width) - index) ** 2) / (2 * fwhm ** 2))

    return spectrogram

def get_uniform_spectrogram(spectrogram, value=0.5):
    spectrogram = np.ones((spectrogram.shape[0], spectrogram.shape[1])) * value
    return spectrogram

def apply_vignette(image, strength=0.5, center=(0.5, 0.5)):
    """
    Applies a vignette effect to an image using NumPy.

    Args:
        image: The input image as a NumPy array.
        strength (float, optional): Controls the intensity of the vignette effect. 
            Higher values create a stronger darkening towards the edges. Defaults to 0.5.
        center (tuple, optional): A tuple representing the center of the vignette effect 
            as a percentage of the image size. Defaults to (0.5, 0.5) which is the center.

    Returns:
        A new NumPy array representing the image with the vignette effect applied.
    """
    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate center coordinates as pixels
    center_x, center_y = int(width * center[0]), int(height * center[1])

    # Create a grayscale mask with a radial gradient
    x, y = np.ogrid[0:height, 0:width]
    radius = max(center_x, center_y, width - center_x, height - center_y)
    mask = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (radius**2) * strength))

    # Expand mask to same channel format as the image
    mask = np.dstack([mask] * 3) if len(image.shape) == 3 else mask

    # Apply the mask to the image using weighted multiplication
    return image * mask


def create_fake_spectrogram(width, height, profile = "stripes", smile=False, vignetting=False, noise=False, spectral_lines=None):
    """
    Create a fake spectrogram image using NumPy.

    Args:

        width (int): The width of the spectrogram image.
        height (int): The height of the spectrogram image.
        profile (str, optional): The type of spectrogram profile to generate. 
            Options are "stripes" (default), "uniform", or "blank".
        smile (bool, optional): Whether to add a smile effect to the spectrogram.
        vignetting (bool, optional): Whether to add a vignetting effect to the spectrogram.
        noise (bool, optional): Whether to add random noise to the spectrogram.
        spectral_lines (list, optional): A list of tuples representing spectral lines to add to the spectrogram.
            Each tuple should contain the index, amplitude, and full width at half maximum (fwhm) of the line.

    Returns:
        A NumPy array representing the fake spectrogram image.
    """
        
    # Generate a blank spectrogram
    spectrogram = np.ones((height, width)) * 0.05
    if profile == "stripes":
        spectrogram = get_striped_spectrogram(spectrogram,spectral_lines=spectral_lines)
    elif profile == "uniform":
        def get_uniform_spectrogram(spectrogram):
            spectrogram = np.ones((height, width)) * 0.5
            return spectrogram
        spectrogram = get_uniform_spectrogram(spectrogram)
    else:
        spectrogram = np.zeros((height, width))

    # Add smile effect if enabled
    if smile:
        # shift all pixels away from the center to the right to create a smile effect

        # curve as a gaussian function
        curve = np.linspace(-1, 1, width)
        curve = np.exp(-curve ** 2 / 4)
        curve = curve * height // 2

        # shift each row by the corresponding amount
        for i in range(height):
            spectrogram[i] = np.roll(spectrogram[i], -int(curve[i]))
        
        
    # Add vignetting effect if enabled
    if vignetting:
        spectrogram = apply_vignette(spectrogram, strength=1, center=(0.5, 0.5))

    # Add noise if enabled
    if noise:
        spectrogram += np.random.normal(0, 0.01, (height, width))

    return spectrogram

from PIL import Image

def save_spectrogram_as_png(data, filename):
  """
  Saves a NumPy array as a PNG image file.

  Args:
      data: The NumPy array to be saved. 
          - For grayscale images, data should have shape (height, width).
          - For RGB images, data should have shape (height, width, 3).
      filename (str): The filename to save the image as (including .png extension).
  """
  # Check data type and dimensions
  if not isinstance(data, np.ndarray):
    raise TypeError("Input data must be a NumPy array.")
  
  if len(data.shape) == 2:
    # Grayscale image, convert to mode 'L' (luminance)
    mode = 'L'
  elif len(data.shape) == 3 and data.shape[2] == 3:
    # RGB image, convert to mode 'RGB'
    mode = 'RGB'
  else:
    raise ValueError("Data must be a 2D grayscale image or a 3D RGB image.")

  # Ensure data is within valid range (0-255) for uint8 image type
  data = np.clip(data*255, 0, 255).astype(np.uint8)  

  # Create a PIL image from the NumPy array
  img = Image.fromarray(data, mode=mode)

  # Save the image as a PNG file
  img.save(filename)

# Example usage
#   spectrogram = create_fake_spectrogram(512, 512, profile="uniform", smile=True, vignetting=True, noise=True)
#   spectrogram = create_fake_spectrogram(512, 512, profile="stripes", smile=True, vignetting=True, noise=True)
#   plt.imshow(spectrogram, cmap='hot', origin='lower')
#   plt.colorbar()
#   plt.clim(0, 1)
#   plt.title('Mock Spectrogram')
#   plt.xlabel('Wavelength Axis')
#   plt.ylabel('Spatial Axis')
#   plt.show()

#   # plot line 400 and line 256
#   plt.plot(spectrogram[400])
#   plt.plot(spectrogram[256])
#   plt.legend(['Line 400', 'Line 256'])
#   plt.ylim(0, 1)
#   plt.title('Spectrogram Lines')
#   plt.xlabel('Wavelength Axis')
#   plt.ylabel('Intensity')
#   plt.show()