from matplotlib import pyplot as plt
import matplotlib.patches as patches

size_x = 224
size_y = 224

OBJ_PRESENT_CONFIDENCE = 0.5
# predictions = N * S * S * [class:<0..19>, [P, x, y, w, h], [P, x, y, w, h]......[P, x, y, w, h]]
# images = N * C * w * h
class disp():
    def __init__(self, S, B, C):
        self.S = S
        self.B = B
        self.C = C

    def show(self, images, predictions):
        S = self.S
        B = self.B
        C = self.C
        ThisBox = predictions[0]
        # image = images[0].permute(1, 2, 0)
        image = images
        assert ThisBox.shape == (S, S, (C+5*B))

        img_width, img_height = image.size
        # Create figure and axes
        fig, ax = plt.subplots()
        for y in range(S):
            for x in range(S):
                cell = ThisBox[x,y]
                if cell[20] < OBJ_PRESENT_CONFIDENCE and cell[25] < OBJ_PRESENT_CONFIDENCE:
                    continue

                if cell[20] > cell[25]:
                    box = cell[21:25]
                else:
                    box = cell[26:30]

                width = box[2]/S
                height = box[3]/S
                cell_width = img_width / S
                cell_height = img_height / S
                bb_x = cell_width * (y + box[0])
                bb_y = cell_height * (x + box[1])
                bb_width = width * img_width
                bb_height = height * img_height

                bb_top_left_x = bb_x - bb_width/2
                bb_top_left_y = bb_y - bb_height/2

                print(bb_top_left_x, bb_top_left_y, bb_width, bb_height)

                # Display the image
                ax.imshow(image)

                # Create a Rectangle patch
                rect = patches.Rectangle((bb_top_left_x, bb_top_left_y), bb_width, bb_height, linewidth=1, edgecolor='r', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)
        plt.show()