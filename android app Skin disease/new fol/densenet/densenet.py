from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.lang import Builder
# from kivy.uix.checkbox import CheckBox
# from kivy.uix.accordion import Accordion,AccordionItem
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.image import Image
# from PIL import Image as img
# from kivy.clock import Clock
# import cam

import tensorflow as tf
from PIL import Image as PIL_Image
import numpy as np
import cv2 as cv

label_mapping = {
    0: 'nv',
    1: 'mel',
    2: 'bkl',
    3: 'bcc',
    4: 'akiec',
    5: 'vasc',
    6: 'df'
}

# Builder.load_string('''
# <MyApps>:
#     orientation:'vertical'
#     Button3:
#         text:'Play'
            
                    
                    
# ''')
def_button_size = ('85dp','56dp')
class MyApp(App):
    def build(self):
        # Create a BoxLayout
        # self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        self.floatlayout = FloatLayout()#size=(100,100)
        self.layout = RelativeLayout()#size=(100,100)
        self.gridlayout= GridLayout(cols=2)
        self.gridlayout.spacing = (110,0)
        self.boxlayout=BoxLayout(size=('512dp','384dp'),
                                 size_hint=(None,None),
                                 
                                )
        self.loaded_model = tf.keras.models.load_model('./DenseNetV2M.h5')
        # Create a Label
        self.camera = Camera(resolution=(-1,-1),
                            #  size=('512dp','384dp'),
                            #  size_hint=(1,1)
                            allow_stretch= True
                             
                             )
        self.camera.play = False
        
        self.label = Label(text="Unpredict",
                           size=def_button_size,
                           size_hint_y=None,
                           size_hint_x=None)
        # self.camera.size = (480,480)
        # self.camera.size_hint=(None,None)
        # self.camera.pos = (150,150)
        self.image = Image()
        self.image.allow_stretch = True
        # self.image.size = (480,480)
        # self.image.size_hint=(None,None)
        # self.image.pos = (150,150)
        
        # Create a Button
        button = Button(text='Click Me!!')
                        # ,
                        # height='48dp',
                        # size_hint_y=None,
                        # size= '68dp'
                        # )
    
        button.size = def_button_size
        button.size_hint=(None,None)
        # button.pos= (350,654)
        


        button.bind(on_press=self.on_button_click)
        button2 = Button(text="take",
                         size = def_button_size,
                         size_hint_y=None,
                         
                         size_hint_x=None,
                        #  pos=(400,20)
                         )
        button2.bind(on_press=self.on_press__)
        # Add widgets to the layout
        self.cam = False
        self.layout.add_widget(self.boxlayout)
        self.gridlayout.add_widget(button)
        self.gridlayout.add_widget(button2)
        self.gridlayout.add_widget(self.label)
       
        self.layout.add_widget(self.gridlayout)
        
        
        return self.layout
    
    def on_press__(self,instance):
        picture = self.camera.texture
        # self.camera.export_to_png("test.jpg")
        texture = self.camera.texture
        image_data = texture.pixels  # Görüntü verisini al
        image_size = (texture.width, texture.height)


        image = PIL_Image.frombytes(mode='RGBA', size=image_size, data=image_data)
        # image = image.transpose(PIL_Image.FLIP_TOP_BOTTOM)  # Kivy'deki terslik düzeltmesi
        image = image.convert("RGB")
        image.save("./test.jpg")
        self.image.source = "./test.jpg"
        self.image.reload();
        
        if(self.cam == True):
            self.layout.remove_widget(self.camera)
            self.layout.add_widget(self.image,10)
            self.label.text = self.__predict()
            self.cam = False
        else:
            self.layout.remove_widget(self.image)
            self.layout.add_widget(self.camera,10)
            self.cam = True
        
        return
        {#region
        # self.image.texture = picture
        # size = picture.size
        # pixels = picture.pixels
        # pil_image = img.frombytes(mode='RGBA', size=size,data=pixels)
        # numpypicture=numpy.array(pil_image)
        # self.image.texture = numpypicture
        # # if self.label1.text == "blabla":
            # self.label1.text = "falan felan"
        # else:
            # self.label1.text = "blabla"
        }
    def on_button_click(self, instance):
        
        if self.cam == False:
            self.layout.add_widget(self.camera,10)
            self.cam = True
        self.camera.play = not self.camera.play
    
    #eveluate predict of ai
    def __predict(self):
        
        
    
        # Img =  np.asarray(PIL_Image.open('./test.jpg').resize((56,56)))
        img = cv.resize(cv.imread('./test.jpg'),(224, 224),interpolation=cv.INTER_AREA)
        test = img.reshape((1,224, 224,3))
        test = test/255.0
        # print(test)

        pred = self.loaded_model.predict(test)
        pred = np.array(list(map(lambda x: np.argmax(x), pred)))
        print(label_mapping.get(int(pred[0])))
        return label_mapping.get(int(pred[0]))
        
        # self.gridlayout.remove_widget(self.label)
        # self.gridlayout.add_widget(self.label)

        
        

if __name__ == '__main__':
    MyApp().run()



