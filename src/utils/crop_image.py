import cv2
import os

def crop_top_and_bottom(image_path: str):
    """
    Corta a imagem em duas partes (metade superior e metade inferior).

    Args:
        image_path (str): Caminho da imagem.

    Raises:
        ValueError: Se a imagem não puder ser lida.

    Returns:
        tuple: (metade_superior, metade_inferior)
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível ler a imagem: {image_path}")

        height, width = image.shape[:2]

        # Metade da altura
        mid = height // 2

        # Metade superior e metade inferior
        top_half = image[0:mid, 0:width]
        bottom_half = image[mid:height, 0:width]

        return top_half, bottom_half

    except Exception as e:
        raise ValueError(f"Erro ao processar imagem: {e}")
