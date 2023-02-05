class ImageQualityMatrix:
    # Class variables
    image_name = ''
    brisque_score = ''

    def __init__(self, image_name, brisque_score):
        self.image_name = image_name
        self.brisque_score = brisque_score