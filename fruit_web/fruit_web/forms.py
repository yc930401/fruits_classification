from django import forms

class ImageForm(forms.Form):
    imagefile = forms.ImageField(label='Uploaded Image', required=False)