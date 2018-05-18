import pdb
import Tkinter as tk
from PIL import ImageTk, Image
from common import *


class DisplayPhotos:
  def __init__(self, image_path=None):
    self.image_path = image_path
    self._root = tk.Tk()
    self._photo = None
    self._cv = None
    self._button = [None, None]
    self._button_width = 5
    self._button_height = 1
    self._pad = 5
    self._canvas_bg = "white"
    self.result = None

  def display_window(self):
    self.display_photo()
    self.display_botton()
    self._root.mainloop()

  def display_photo(self):
    logging.debug(self.image_path)
    self._photo = ImageTk.PhotoImage(Image.open(self.image_path))
    self._cv = tk.Canvas(width=self._photo.width() + 2*self._pad, 
                    height= self._photo.height() + 2*self._pad, 
                    bg=self._canvas_bg)

    self._cv.pack(side='top')
    # Padding of 10 and 10 with a NW anchor, default unit is pixel
    # Adding 1 to place it _really_ center because self._cv.winfo_width()
    # is 2 more than what's set above, perhaps because of its own size?
    self._cv.create_image(self._pad + 1, self._pad + 1, image=self._photo,
                          anchor='nw')

  def display_botton(self):
    # Frame to contain the two buttons
    frame = tk.Frame(self._root, padx=self._pad, pady=self._pad)

    self._button[0] = tk.Button(frame, width=self._button_width,
        height=self._button_height, text="Yes", command=self.return_yes)
    self._button[0].pack(side="left", expand=True)

    self._button[1] = tk.Button(frame, width=self._button_width,
        height=self._button_height, text="No", command=self.return_no)
    self._button[1].pack(side="left", expand=True)
    frame.pack(side='bottom', anchor="sw")

  def return_yes(self):
    self.result = (self.image_path, True)
    self._root.destroy()

  def return_no(self):
    self.result = (self.image_path, False)
    self._root.destroy()


if __name__ == "__main__":
  for i in range(1, 10):
    dp = DisplayPhotos("/home/chaturvedi/workspace/misc/try_keras/tiny-imagenet-200/val/images/val_1023.JPEG")
    dp.display_window()
    print dp.result


