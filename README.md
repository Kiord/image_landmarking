# image_landmarking
A tool for image landmark annotation written in Python



https://github.com/user-attachments/assets/08aafeda-f245-436b-a28d-4a1c6902bc59



## Usage

`python image_landmarking.py <path to image folder> -nl <number of landmarks per image> -dw <display width>`

example:

`python image_landmarking.py images -dw 720`

To save the result, quit the application by closing the window or pressing `esc`. The result will appear in the image folder as `landmarks.pickle`.

## Notes

This tool uses Pygame and OpenCV to interactively place, move, remove and tag landmarks on a serie of images. Image keypoints are very useful in many computer vision application including face alignment.

## Controls

The controls are displayed in the console when launching the application.

| Control    | Action |
| -------- | ------- |
| right click  | Place a landmark under the cursor  |
| left click | Grab a landmark under the cursor |
| mouse wheel    | zoom   |
| wheel click and drag   | translation  |
| lctrl + left click    | Falg landmark as occluded  |
| backspace    | remove last placed landmark |
| c    | clear the landmarks on the current image |
| right arrow    | next image |
| left arrow    | previous image |


## Color coding

Landmarks are color coded to ease the process.

| Color    | Meaning |
| -------- | ------- |
| blue  | Default |
| magenta | Last placed |
| green or orange | Flagged as occluded |


## Requirements
- python 3.7+
- pygame
- opencv-python
- numpy
- click
