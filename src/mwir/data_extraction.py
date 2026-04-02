"""
Extract data from MWIR ascii files
"""

import numpy as np
import matplotlib.pyplot as plt


def extract_thermal_data(filepath):
    """
    Extract thermal data from MWIR ASCII file.
    
    Reads an ASCII file starting from line 17 (after [Data] header).
    Each line contains 640 temperature values separated by ';'.
    Commas are decimal separators and are converted to dots.
    
    Args:
        filepath (str): Path to the ASCII file
        
    Returns:
        np.ndarray: A 512x640 array of float values (thermal data)
    """
    thermal_data = []
    
    with open(filepath, 'r', encoding='iso-8859-1') as file:
        lines = file.readlines()
    
    # Start from line 17 (index 16) where the [Data] section begins
    for i in range(16, len(lines)):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Replace commas with dots to convert European decimal format to standard
        line = line.replace(',', '.')
        
        # Split by semicolon to get individual values
        values = line.split(';')
        
        # Convert to float and filter out empty strings
        row_data = [float(val) for val in values if val.strip()]
        
        # Only add rows with 640 elements
        if len(row_data) == 640:
            thermal_data.append(row_data)
    
    # Convert to numpy array and ensure it's 512x640
    thermal_array = np.array(thermal_data, dtype=np.float32)
    
    # Verify dimensions
    if thermal_array.shape != (512, 640):
        raise ValueError(
            f"Expected shape (512, 640), but got {thermal_array.shape}. "
            f"Check if the file contains exactly 512 lines with 640 values each."
        )
    
    return thermal_array


def display_thermal_image(thermal_data, title="Thermal Image", cmap='hot', show_colorbar=True):
    """
    Display thermal data as an image.
    
    Args:
        thermal_data (np.ndarray): 2D array of temperature values (512x640)
        title (str): Title for the image. Default: "Thermal Image"
        cmap (str): Colormap name. Default: 'hot' (good for thermal images)
                   Other options: 'viridis', 'plasma', 'inferno', 'magma', 'twilight', etc.
        show_colorbar (bool): Whether to display a colorbar. Default: True
        
    Returns:
        None (displays the image in a matplotlib window)
    """
    plt.figure(figsize=(12, 8))
    
    # Display the thermal data as an image
    im = plt.imshow(thermal_data, cmap=cmap, aspect='auto', origin='upper')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X Pixel', fontsize=12)
    plt.ylabel('Y Pixel', fontsize=12)
    
    if show_colorbar:
        cbar = plt.colorbar(im, label='Temperature (°C)', pad=0.02)
    
    plt.tight_layout()
    plt.show()


def save_thermal_image(thermal_data, output_path, title="Thermal Image", cmap='hot', show_colorbar=True, dpi=100):
    """
    Save thermal data as an image file.
    
    Args:
        thermal_data (np.ndarray): 2D array of temperature values (512x640)
        output_path (str): Path where to save the image (e.g., 'thermal_image.png')
        title (str): Title for the image. Default: "Thermal Image"
        cmap (str): Colormap name. Default: 'hot'
        show_colorbar (bool): Whether to include a colorbar. Default: True
        dpi (int): Resolution in dots per inch. Default: 100
        
    Returns:
        None (saves image to file)
    """
    plt.figure(figsize=(12, 8), dpi=dpi)
    
    # Display the thermal data as an image
    im = plt.imshow(thermal_data, cmap=cmap, aspect='auto', origin='upper')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X Pixel', fontsize=12)
    plt.ylabel('Y Pixel', fontsize=12)
    
    if show_colorbar:
        cbar = plt.colorbar(im, label='Temperature (°C)', pad=0.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Image saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Example usage
    filepath = "data/infratec_mwir/Julian_31_03_1_Ascii/ISM_Infratec_31162925_0019_1801.asc"
    thermal_data = extract_thermal_data(filepath)
    print(thermal_data.shape)  # Should print (512, 640)
    display_thermal_image(thermal_data, title="Thermal Image from MWIR Data")