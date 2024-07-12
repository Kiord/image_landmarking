import pygame
import click
import cv2
import numpy as np
import os
import pickle

SCALE_FACTOR = 1.1
LANDMARK_RADIUS = 8

COMMANDS = \
    '+-------------------------------------------------------------------+\n'\
    '|                         COMMANDS                                  |\n'\
    '+-------------------------------------------------------------------+\n'\
    '|- right click\t\t: place a landmark\t                              \n'\
    '|- left click\t\t: move a landmark                                  \n'\
    '|- mouse wheel\t\t: zoom                                            \n'\
    '|- wheel click\t\t: translation                                     \n'\
    '|- lctrl + left click\t: mark landmark as occluded                  \n'\
    '|- backspace\t\t: remove last placed landmark                       \n'\
    '|- "c" key\t\t: clear the landmarks on the current image            \n'\
    '|- right arrow\t\t: next image                                      \n'\
    '|- left arrow\t\t: previous image                                   \n'\
    '+--------------------------------------------------------------------\n'

IMAGE_EXTENSIONS=('jpg','jpeg', 'jpe', 'jp2', 'png', 'tiff', 'tif', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jfif', 'tiff', 'tif', 'bmp', 'dib')

class Dimensions:
    def __init__(self, display_width, imshape):
        self.display_width = display_width
        self.update_shape(imshape)
    def update_shape(self, imshape):
        self.imshape = imshape
        self.display_dim = (self.display_width, int(self.display_width * (self.imshape[0] / float(self.imshape[1]))))
        self.ratio_dim = min(imshape) / min(self.display_dim)


def is_image(filename):
    return filename.split('.')[-1].lower() in IMAGE_EXTENSIONS

def cvimage_to_pygame(image):
    return pygame.surfarray.make_surface(image.swapaxes(0,1))

def get_image_paths_dict(images_dir_path):
    return {name.split('.')[0]: images_dir_path + os.sep + name for name in os.listdir(images_dir_path) if is_image(name)}

def imread(f):
    print('reading', f)
    img = cv2.imread(f)
    code = cv2.COLOR_GRAY2RGB if len(img.shape)==2 else cv2.COLOR_BGR2RGB
    img = cv2.cvtColor(img, code)
    return img

def local_to_global_coordinates(xy, scale, tx, ty, dimensions):
    return xy[0] * scale * dimensions.ratio_dim + tx,  xy[1] * dimensions.ratio_dim * scale + ty

def local_to_global_delta(xy, scale, dimensions):
    return xy[0] * scale * dimensions.ratio_dim,  xy[1] * dimensions.ratio_dim * scale

def global_to_local_coordinates(xy, scale, tx, ty, dimensions):
    return (xy[0] - tx) / (scale * dimensions.ratio_dim),  (xy[1] - ty) / (scale * dimensions.ratio_dim)

def blit_at(window, image, scale, tx, ty, dimensions):
    h, w = image.shape[:2]
    real_h, real_w =  int(scale * h), int(scale * w)
    cropped = image[ty:ty+real_h, tx:tx+real_w, :]
    img = cv2.resize(cropped, (dimensions.display_dim[0], dimensions.display_dim[1]), interpolation = cv2.INTER_NEAREST)
    window.blit(cvimage_to_pygame(img), (0,0))


class Landmark():
    def __init__(self, XY, activated=True):
        self.XY = tuple(XY)
        self.activated = activated

    def display(self, window, scale, tx, ty, font, text, is_last, dimensions):
        if self.activated:
            color = [0, 0, 255]
        else:
            color = [0, 255, 0]
        if is_last:
            color = [255, color[1]//2, color[2]//2]
        xy = global_to_local_coordinates(self.XY, scale, tx, ty, dimensions)
        pygame.draw.circle(window, color, xy, LANDMARK_RADIUS)
        text_image = font.render(text, True, color)
        window.blit(text_image, dest=(xy[0] + LANDMARK_RADIUS, xy[1] + LANDMARK_RADIUS))
        

class LandmarkList(list):

    def display(self, window, scale, tx, ty, font, dimensions):
        for i, l in enumerate(self):
            l.display(window, scale, tx, ty, font, str(i), i == len(self) - 1, dimensions)
            
    def to_arrays(self):
        coords = np.array([l.XY for l in self])
        activations = np.array([l.activated for l in self])
        return coords, activations
    
    def from_arrays(coords, activations):
        res = LandmarkList()
        for i in range(coords.shape[0]):
            x, y = coords[i, :]
            activation = activations[i]
            res.append(Landmark((x, y), activation))
        return res

class ViewManager():
    def __init__(self, window, dimensions, nb_landmarks=None):
        self.reset_transform()
        self.window = window
        self.translating = False
        self.need_update = True
        self.font = pygame.font.SysFont('arial.ttf', 30)
        self.control = False
        self.dragged_landmark = None
        self.nb_landmarks = nb_landmarks
        self.dimensions = dimensions

    
    def reset_transform(self):
        self.scale = 1
        self.tx, self.ty = 0, 0
    
    def request_update(self):
        self.need_update = True

    def _correct_t(self):
        self.tx = max(min(int(self.tx), int(self.dimensions.imshape[1] * (1 - self.scale))), 0)
        self.ty = max(min(int(self.ty), int(self.dimensions.imshape[0] * (1 - self.scale))), 0)

    def _update(self, image, landmarks):
        blit_at(self.window, image, self.scale, self.tx, self.ty, self.dimensions)
        landmarks.display(self.window, self.scale, self.tx, self.ty, self.font, self.dimensions)
        pygame.display.update()

    def pick_landmark(self, pos, landmarks):
        for landmark in reversed(landmarks): # reverse just to pick the one on top of a possible stack
            xy = global_to_local_coordinates(landmark.XY, self.scale, self.tx, self.ty, self.dimensions)
            dx = xy[0] - pos[0]
            dy = xy[1] - pos[1]
            distance = (dx ** 2 + dy ** 2) ** 0.5
            if distance <= LANDMARK_RADIUS * 2:
                return landmark

    def process_events(self, events, image, landmarks):
        changed = False
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == 8 and len(landmarks) > 0 and self.dragged_landmark is None:
                    landmarks.pop()
                    changed = True
                if event.key == pygame.K_c and self.dragged_landmark is None:
                    landmarks.clear()
                    changed = True
                if event.key == pygame.K_LCTRL:
                    self.control = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LCTRL:
                    self.control = False
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                if event.button == 3 and not self.control:
                    if self.nb_landmarks is None or self.nb_landmarks > len(landmarks):
                        XY = local_to_global_coordinates(pos, self.scale, self.tx, self.ty, self.dimensions)
                        ldm = Landmark(XY, activated=True)
                        landmarks.append(ldm)
                        changed = True
                if event.button == 1:
                    self.dragged_landmark = None
                if event.button == 2:
                    self.translating = False
                
            
            if event.type == pygame.MOUSEMOTION:
                if self.dragged_landmark is not None:
                    pos = pygame.mouse.get_pos()
                    XY = local_to_global_coordinates(pos, self.scale, self.tx, self.ty, self.dimensions)
                    self.dragged_landmark.XY = XY
                    changed = True
                
                if self.translating :
                    dXY = local_to_global_delta(event.rel, self.scale, self.dimensions)
                    self.tx -= int(dXY[0])
                    self.ty -= int(dXY[1])
                    self._correct_t()
                    changed = True

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos() 
                XY = local_to_global_coordinates(pos, self.scale, self.tx, self.ty, self.dimensions)
                if event.button == 4 or event.button == 5:
                    prev_scale = self.scale
                    if event.button == 4:
                        self.scale = min(1, self.scale * SCALE_FACTOR)
                    else:
                        self.scale /= SCALE_FACTOR
                    scale_ratio = self.scale / prev_scale
                    tx_display = pos[0] * (1 - scale_ratio)
                    ty_display = pos[1] * (1 - scale_ratio)
                    self.tx, self.ty = local_to_global_coordinates((tx_display, ty_display), prev_scale, self.tx, self.ty, self.dimensions)
                    self._correct_t()
                    changed = True
                if event.button == 1:
                    if not self.control:
                        self.dragged_landmark = self.pick_landmark(pos, landmarks)
                if event.button == 2:
                    self.translating = True
                if event.button == 1:
                    if self.control:
                        landmark = self.pick_landmark(pos, landmarks)
                        if landmark is not None:
                            landmark.activated = not landmark.activated
                            changed = True
        
        if changed or self.need_update:
            self.need_update = False
            self._update(image, landmarks)

class ImageLoader():
    def __init__(self, image_paths_dict):
        self.image_paths_dict = image_paths_dict
        self.image_caches = dict()
        self.labels = list(self.image_paths_dict.keys())

    def __getitem__(self, i):
        label = self.labels[i]
        if label in self.image_caches:
            image = self.image_caches[label]
        else:
            image = imread(self.image_paths_dict[label])
            self.image_caches[label] = image
        return label, image

    def __len__(self):
        return len(self.labels)
    
    def get_caption(self, i):
        return f'{i+1}/{len(self)} : {self.labels[i]}'

def manual_annotation(image_paths_dict, display_width=1000, input_landmarks=None, nb_landmarks=None, specific=None):
    pygame.init()
    loader = ImageLoader(image_paths_dict)
    cpt = 0
    if specific is not None:
        cpt = loader.labels.index(specific)


    imshape = loader[cpt][1].shape[:2]
    dimensions = Dimensions(display_width, imshape)
    window = pygame.display.set_mode(dimensions.display_dim)

    pygame.display.set_caption(loader.get_caption(cpt))
    if os.path.exists('icon.png'):
        pygame.display.set_icon(pygame.image.load('icon.png'))
    pygame.display.update()
    if input_landmarks is None:
        all_landmarks = {label:LandmarkList() for label in image_paths_dict}
    else:
        all_landmarks = {label:LandmarkList.from_arrays(arrays[0], arrays[1]) for label, arrays in input_landmarks.items()}
    viewer = ViewManager(window, dimensions, nb_landmarks=nb_landmarks)
    done = False
    
    complete = False
    while not done:
        events = pygame.event.get()

        label, image = loader[cpt]
        new_imshape = image.shape[:2]
        if dimensions.imshape != new_imshape:
            dimensions.update_shape(new_imshape)
            window = pygame.display.set_mode(dimensions.display_dim)
        landmarks = all_landmarks[label]

        viewer.process_events(events, image, landmarks)

        for event in events:
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                if nb_landmarks is not None:
                    complete = all([len(x) == nb_landmarks for x in all_landmarks.values()])
                    if complete:
                        print('Done ! Every images are fully annotated. Exiting.')
                    else:
                        print('[WARNING] Not all images are fully annotated ! Exiting.')
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT and cpt < len(loader) - 1:
                    cpt += 1
                    pygame.display.set_caption(loader.get_caption(cpt))
                    viewer.request_update()
                if event.key == pygame.K_LEFT and cpt > 0:
                    cpt -= 1
                    pygame.display.set_caption(loader.get_caption(cpt))
                    viewer.request_update()
 
    pygame.quit()
    all_landmarks_arrays = {label:landmarks.to_arrays() for label, landmarks in all_landmarks.items()}
    return all_landmarks_arrays


@click.command()
@click.argument('images_dir_path', type=click.Path(exists=True), required=True)
@click.option('-nl', '--nb_landmarks', type=int, default=None, help='Number of landmarks to be expecting.')
@click.option('-dw', '--display_width', type=int, default=1000, help='Width of the viewport')
@click.option('-s', 'specific', type=str, default=None, help='Starting image')
def cli(images_dir_path, nb_landmarks, display_width, specific):
    landmarks_path = images_dir_path + os.sep + 'landmarks.pickle'
    input_landmarks = None
    if os.path.exists(landmarks_path):
        with open(landmarks_path, 'rb') as handle:
            input_landmarks = pickle.load(handle)
    image_paths_dict = get_image_paths_dict(images_dir_path)
    print(COMMANDS)
    all_landmarks_arrays = manual_annotation(image_paths_dict, display_width, input_landmarks, nb_landmarks, specific)

    with open(landmarks_path, 'wb') as handle:
        pickle.dump(all_landmarks_arrays, handle)

if __name__ == '__main__':
    cli()