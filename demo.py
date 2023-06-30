
from ChestXrayDataset import ChestXrayDataset
from MyModel import MyModel
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torch
import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import NoiseTunnel
from matplotlib import cm
import torchvision.transforms.functional as TF
import io
from io import BytesIO

print("Caricamento dataset...")
dataset = ChestXrayDataset(img_dir='/Users/ale_strange/Desktop/dvntn9yhd2-1/aunione', transform=None)
print("Dataset caricato")
print("Proseguire nella GUI")

# %%
data_transform = transforms.Compose([
                            #transforms.RandomHorizontalFlip(1), 
                            #transforms.RandomVerticalFlip(1),
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(0.5, 0.5)
                            ])

# %%
test_set = dataset.get_testset()
normals = [image for image, label in test_set if label == 0]
pneumonias = [image for image, label in test_set if label == 1]
covids = [image for image, label in test_set if label == 2]
idx_to_labels = {0: 'normal', 1: 'pneumonia', 2: 'covid'}

# %%
model = MyModel(num_classes=3, Trained=True)


integrated_gradients = IntegratedGradients(model)
noise_tunnel = NoiseTunnel(integrated_gradients)


# %%
def show_map(input_img, pred_label_idx):
    #tolgo l'handler del bottone
    map_button.pack_forget()

    attributions_ig_nt = noise_tunnel.attribute(input_img, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx, internal_batch_size=10, stdevs=0.5)
    plottable_attributions = np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0))
    activation_image = viz.visualize_image_attr(
        plottable_attributions,
        method="heat_map",
        cmap=cm.seismic,
        show_colorbar=True
    )
    # Salva la figura come immagine in memoria
    buffer = BytesIO()
    activation_image[0].savefig(buffer, format='png')
    buffer.seek(0)
    # Carica l'immagine da BytesIO come oggetto PIL Image
    pil_image = Image.open(buffer)

    left_padding = 65
    top_padding = 50
    right_padding = 50
    bottom_padding = 30
    cropped_image = pil_image.crop((left_padding, top_padding, pil_image.width - right_padding, pil_image.height - bottom_padding))
    resized = cropped_image.resize((300,300))

    # Crea un oggetto PhotoImage
    photo_image = ImageTk.PhotoImage(resized)
    canvas2.create_image(0, 0, anchor="nw", image=photo_image)
    canvas2.image = photo_image
    canvas2.pack()
    


def plot_random_image(class_name):
    random_index = np.random.randint(0, len(class_name))
    img = class_name[random_index]

    img = np.asarray(img)
    img_tk = ImageTk.PhotoImage(Image.fromarray(img))

    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk  

    img = Image.fromarray(img) 
    input_img = data_transform(img)
    input_img = input_img.unsqueeze(0)
    output = model(input_img)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1) 
    pred_label_idx.squeeze_() 
    predicted_label = idx_to_labels[pred_label_idx.item()]

    result_label.config(text="Predicted: {} ({})".format(predicted_label, prediction_score.squeeze().item()))


    map_button.pack()
    map_button.config(command=lambda: show_map(input_img, pred_label_idx.item()))


    
def conferma():
    global img_tk  

    selected_option = var.get()
    if selected_option == "normal":
        plot_random_image(normals)
        
    elif selected_option == "pneumonia":
        plot_random_image(pneumonias)

    elif selected_option == "covid":
        plot_random_image(covids)

    canvas2.pack_forget()
    


# Crea la finestra principale
window = tk.Tk()
window.title("Menu")

# Centra la finestra principale
window_width = 500
window_height = 800
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Testo introduttivo
intro_label = tk.Label(window, text="Benvenuti nella Demo di 'Chest XRay Illness Detector' \n Avrete anche la possibilità di capire sulla base di cosa "+
                       "è stata fatta la predizione (Attenzione: generare l'Activation Map potrebbe richiedere fino ad un minuto). \n" +
                       "Scegli una classe di immagine da testare:", wraplength=500, pady=5)
intro_label.pack()

var = tk.StringVar()
option1 = tk.Radiobutton(window, text="NORMAL", variable=var, value="normal")
option1.pack()
option2 = tk.Radiobutton(window, text="PNEUMONIA", variable=var, value="pneumonia")
option2.pack()
option3 = tk.Radiobutton(window, text="COVID", variable=var, value="covid")
option3.pack()

# Bottone di conferma
confirm_button = tk.Button(window, text="Elabora", command=conferma)
confirm_button.pack()

# Canvas per disegnare l'immagine
canvas = tk.Canvas(window, width=224, height=224)
canvas.pack()

# Per il risultato
result_label = tk.Label(window, text="")
result_label.pack()

# Crea il secondo bottone "Clicca qui"
map_button = tk.Button(window, text="Mostra Activation Map")

# Canvas per disegnare l'attivazione
canvas2 = tk.Canvas(window, width=300, height=300)

window.mainloop()



