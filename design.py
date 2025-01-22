import io
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import scrolledtext
import translators as ts
from threading import Thread
from PIL import Image, ImageGrab
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import easyocr

class CRNN (nn.Module):
    def __init__(self, input_height, num_classes):
        super(CRNN, self).__init__()
        self.filter_size = 64
        self.image_size = input_height // 4
        self.hidden_size = 256

        self.lstm_hidden = 256
        self.lstm_layer = 2

        self.attention_head = 8

        self.conv1 = nn.Conv2d(1, self.filter_size, 3, 1, 1)
        self.conv12 = nn.Conv2d(self.filter_size, self.filter_size, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.filter_size)
        self.bn12 = nn.BatchNorm2d(self.filter_size)
        self.conv2 = nn.Conv2d(self.filter_size, self.filter_size * 2, 3, 1, 1)
        self.conv22 = nn.Conv2d(self.filter_size *2, self.filter_size*2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(self.filter_size * 2)
        self.bn22 = nn.BatchNorm2d(self.filter_size*2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.filter_size * 2 * self.image_size, self.hidden_size)

        self.lstm1 = nn.LSTM(self.hidden_size, self.lstm_hidden, num_layers=self.lstm_layer, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(self.lstm_hidden*2, self.lstm_hidden, num_layers=self.lstm_layer, bidirectional= True, batch_first=True)

        self.attention = nn.MultiheadAttention(self.lstm_hidden*2, self.attention_head, batch_first=True)

        self.layer_norm1 = nn.LayerNorm(self.lstm_hidden*2)
        self.layer_norm2 = nn.LayerNorm(self.lstm_hidden*2)
        self.fc2 = nn.Linear(self.lstm_hidden*2, num_classes+1)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv12(output)
        output = self.bn12(output)
        output = self.relu(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv22(output)
        output = self.bn22(output)
        output = self.relu(output)
        output = self.pool(output)

        bs, c, h, w = output.size()
        output = output.permute(0, 3, 1, 2)
        output = output.view(bs, w, -1)

        output = self.fc1(output)

        output1, _ = self.lstm1(output)
        output1 = self.layer_norm1(output1)

        attention_out, _ = self.attention(output1, output1, output1)

        output2, _ = self.lstm2(attention_out)
        output2 = self.layer_norm2(output2)

        output = output1 + output2
        output = self.fc2(output)
        return output

class capture_window(tk.Toplevel):
    def __init__(self, *args, **kwargs):
        super(capture_window, self).__init__()
        self.label = tk.Label(self, bg="#262729")
        self.label.pack(fill="both", expand=True)
        self.label.bind("<ButtonPress-1>", self.start_move)
        self.label.bind("<ButtonRelease-1>", self.stop_move)
        self.label.bind("<B1-Motion>", self.do_move)

    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def stop_move(self, event):
        self.x = None
        self.y = None

    def do_move(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        self.xnow = self.winfo_x() + deltax
        self.ynow = self.winfo_y() + deltay
        self.geometry(f"+{self.xnow}+{self.ynow}")

class query_window(tk.Toplevel):
    def __init__(self, *args, **kwargs):
        super(query_window, self).__init__()
        self.scrolledText = scrolledtext.ScrolledText(self, wrap="word", bg="#262729", font="Arial 20", fg="white", state="disabled")

        self.scrolledText.pack()
        self.scrolledText.bind("<ButtonPress-1>", self.start_move)
        self.scrolledText.bind("<ButtonRelease-1>", self.stop_move)
        self.scrolledText.bind("<B1-Motion>", self.do_move)

    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def stop_move(self, event):
        self.x = None
        self.y = None

    def do_move(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        self.xnow = self.winfo_x() + deltax
        self.ynow = self.winfo_y() + deltay
        self.geometry(f"+{self.xnow}+{self.ynow}")

class onScreenTranslator():
    def __init__(self):
        self.language_dict = {
            'afrikaans': 'af',
            'albanian': 'sq',
            'amharic': 'am',
            'arabic': 'ar',
            'armenian': 'hy',
            'azerbaijani': 'az',
            'basque': 'eu',
            'belarusian': 'be',
            'bengali': 'bn',
            'bosnian': 'bs',
            'bulgarian': 'bg',
            'catalan': 'ca',
            'cebuano': 'ceb',
            'chichewa': 'ny',
            'chinese (simplified)': 'zh-cn',
            'chinese (traditional)': 'zh-tw',
            'corsican': 'co',
            'croatian': 'hr',
            'czech': 'cs',
            'danish': 'da',
            'dutch': 'nl',
            'english': 'en',
            'esperanto': 'eo',
            'estonian': 'et',
            'filipino': 'tl',
            'finnish': 'fi',
            'french': 'fr',
            'frisian': 'fy',
            'galician': 'gl',
            'georgian': 'ka',
            'german': 'de',
            'greek': 'el',
            'gujarati': 'gu',
            'haitian creole': 'ht',
            'hausa': 'ha',
            'hawaiian': 'haw',
            'hebrew': 'he',
            'hindi': 'hi',
            'hmong': 'hmn',
            'hungarian': 'hu',
            'icelandic': 'is',
            'igbo': 'ig',
            'indonesian': 'id',
            'irish': 'ga',
            'italian': 'it',
            'japanese': 'ja',
            'javanese': 'jw',
            'kannada': 'kn',
            'kazakh': 'kk',
            'khmer': 'km',
            'korean': 'ko',
            'kurdish (kurmanji)': 'ku',
            'kyrgyz': 'ky',
            'lao': 'lo',
            'latin': 'la',
            'latvian': 'lv',
            'lithuanian': 'lt',
            'luxembourgish': 'lb',
            'macedonian': 'mk',
            'malagasy': 'mg',
            'malay': 'ms',
            'malayalam': 'ml',
            'maltese': 'mt',
            'maori': 'mi',
            'marathi': 'mr',
            'mongolian': 'mn',
            'myanmar (burmese)': 'my',
            'nepali': 'ne',
            'norwegian': 'no',
            'odia': 'or',
            'pashto': 'ps',
            'persian': 'fa',
            'polish': 'pl',
            'portuguese': 'pt',
            'punjabi': 'pa',
            'romanian': 'ro',
            'russian': 'ru',
            'samoan': 'sm',
            'scots gaelic': 'gd',
            'serbian': 'sr',
            'sesotho': 'st',
            'shona': 'sn',
            'sindhi': 'sd',
            'sinhala': 'si',
            'slovak': 'sk',
            'slovenian': 'sl',
            'somali': 'so',
            'spanish': 'es',
            'sundanese': 'su',
            'swahili': 'sw',
            'swedish': 'sv',
            'tajik': 'tg',
            'tamil': 'ta',
            'telugu': 'te',
            'thai': 'th',
            'turkish': 'tr',
            'ukrainian': 'uk',
            'urdu': 'ur',
            'uyghur': 'ug',
            'uzbek': 'uz',
            'vietnamese': 'vi',
            'welsh': 'cy',
            'xhosa': 'xh',
            'yiddish': 'yi',
            'yoruba': 'yo',
            'zulu': 'zu',
        }

        self.language_list = list(self.language_dict.keys())
        self.language_code = list(self.language_dict.values())
        self.language_list = [i.capitalize() for i in self.language_list]
        self.language_dict = dict()
        for i in range(len(self.language_list)):
            self.language_dict[self.language_list[i]] = self.language_code[i]
        self.engine_list = ["Google Translate", "Bing", "Yandex", "DeepL", "Sogou", "Argos", "Baidu"]
        self.engine_dict = {
            "Google Translate": "google",
            "Bing": "bing",
            "Yandex": "yandex",
            "DeepL": "deepl",
            "Sogou": "sogou",
            "Argos": "argos",
            "Baidu": "baidu"
        }
        self.captureW = None
        self.queryW = None
        self.resultW = None
        self.resultText = None
        self.queryText = None
        self.dark = "#262729"
        self.running=False
        self.model_path = "model/CSAR-MBiLNet.pth"
        self.checkpoint = torch.load(self.model_path, weights_only=True)
        self.vocab = self.checkpoint["vocab"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CRNN(input_height=32, num_classes=len(self.vocab))
        self.model.load_state_dict(self.checkpoint["model"])
        self.model.to(self.device)

        self.main_window = tk.Tk()
        self.main_window.title("On Screen Translator By Ronny")

        self.main_window.configure(bg=self.dark)
        self.whitevar = tk.IntVar()
        self.whitevar.set(1)
        # Top Buttons
        self.main_frame = tk.Frame(self.main_window, bg=self.dark)
        self.main_frame.pack(fill="x", padx=5, pady=5)

        self.btn_capture = ttk.Button(self.main_frame, text="Capture Window", command=self.create_capture_window)
        self.btn_capture.pack(side="left", padx=5)

        self.btn_query = ttk.Button(self.main_frame, text="Query Window", command=self.create_query_window)
        self.btn_query.pack(side="left", padx=5)

        self.btn_result = ttk.Button(self.main_frame, text="Result Window", command=self.create_result_window)
        self.btn_result.pack(side="left", padx=5)

        self.button_white1 = tk.Checkbutton(self.main_frame, text="White Text", variable=self.whitevar, onvalue=1, offvalue=0, bg=self.dark, fg="white", selectcolor="black", activebackground=self.dark, activeforeground="white")
        self.button_white1.pack(side="left", padx=5)

        # Top ScrolledText for input
        self.input_text = scrolledtext.ScrolledText(self.main_window, wrap="word", height=5, bg=self.dark, fg="white", font="Arial 20")
        self.input_text.pack(fill="both", padx=10, pady=5, expand=True)

        # Translation Options Frame
        self.options_frame = tk.Frame(self.main_window, bg=self.dark)
        self.options_frame.pack(fill="x", padx=10, pady=5)

        # Translation Engine
        self.engine_label = tk.Label(self.options_frame, text="Engine:", fg="white", bg=self.dark)
        self.engine_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.engine_combo = ttk.Combobox(self.options_frame, values=self.engine_list, state="readonly")
        self.engine_combo.grid(row=0, column=1, padx=5, pady=5)
        self.engine_combo.current(0)

        # Source Language
        self.from_label = tk.Label(self.options_frame, text="From:", fg="white", bg=self.dark)
        self.from_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        self.from_combo = ttk.Combobox(self.options_frame, values=self.language_list, state="readonly")
        self.from_combo.grid(row=0, column=3, padx=5, pady=5)
        self.from_combo.current(45)

        # Target Language
        self.to_label = tk.Label(self.options_frame, text="To:", fg="white", bg=self.dark)
        self.to_label.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        self.to_combo = ttk.Combobox(self.options_frame, values=self.language_list, state="readonly")
        self.to_combo.grid(row=0, column=5, padx=5, pady=5)
        self.to_combo.current(21)

        # Swap and Clear Buttons
        self.swap_button = ttk.Button(self.options_frame, text="⇆ Swap", command=self.swap)
        self.swap_button.grid(row=0, column=6, padx=5, pady=5)

        # Bottom ScrolledText for output
        self.output_text = scrolledtext.ScrolledText(self.main_window, wrap="word", height=5, bg=self.dark, fg="white",
                                                font="Arial 20")
        self.output_text.pack(fill="both", padx=10, pady=5, expand=True)

        self.match_input()
        self.match_output()
        self.thread_translate()
        self.check_capture_window()
        self.main_process_thread()
        # Run the application
        self.main_window.mainloop()

    def decode_ctc_output(self, output, vocab):
        """
        Decode the output of a CTC model into text using best path decoding
        Args:
            output: Model output tensor of shape [seq_len, batch_size, num_classes]
            vocab: List of characters in the vocabulary
        Returns:
            decoded_text: String of decoded text
        """
        # Get probabilities and best path
        # probs = torch.nn.functional.softmax(output, dim=2)
        # predicted_indices = torch.argmax(probs, dim=2)
        predicted_indices = torch.argmax(output, dim=2)
        predicted_indices = predicted_indices.squeeze().cpu().numpy()

        # Merge repeated characters and remove blanks
        decoded_text = []
        previous_index = None  # Changed from -1 to None for clarity

        for idx in predicted_indices:
            # Convert to int to avoid any numpy type issues
            idx = int(idx)

            # Skip if it's a blank token (index 0) or repeated character
            if idx != 0 and idx != previous_index:  # blank = 0
                # Subtract 1 from index since 0 is reserved for blank
                char_idx = idx - 1
                if char_idx < len(vocab):  # Add boundary check
                    decoded_text.append(vocab[char_idx])
            previous_index = idx

        return ''.join(decoded_text)  # Join characters into a string

    def detect_text_and_convert_to_tensors(self, img):
        # Initialize EasyOCR reader
        transform = transforms.Compose([
            transforms.Grayscale(),  # Convert image to grayscale
            transforms.Resize((32,)),  # Resize height to 28, keep width dynamic
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize([0.5, ], [0.5,])  # Normalize if needed
        ])
        reader = easyocr.Reader(['ja'])  # Specify the languages you need

        # Read image
        image_bytes = io.BytesIO()
        img.save(image_bytes, format="PNG")
        image = Image.open(image_bytes)
        image = image.convert("L")

        # image = ImageOps.invert(image)
        threshold = 190

        if self.whitevar.get():
            image = image.point(lambda x: 255 if x < threshold else 0, '1')
        else:
            image = image.point(lambda x: 0 if x < threshold else 255, '1')


        # Perform text detection
        imagenp = np.array(img)

        results = reader.readtext(imagenp)

        # If no text is detected, return an empty list
        if not results:
            return [transform(image)]

        # List to hold lists of tensor images
        tensor_images_list = []

        # Define the transformation

        for (bbox, text, prob) in results:
            # Unpack bounding box coordinates
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))

            # Crop the image to the bounding box

            cropped_image = image.crop((*top_left, *bottom_right))

            # Apply the transformation
            tensor_image = transform(cropped_image)


            # Append each tensor to a separate list
            tensor_images_list.append(tensor_image)

        return tensor_images_list

    def predict(self, tensor_image_list):
        self.model.eval()  # Set the model to evaluation mode

        # evaluation mode

        prediction = []
        for i in tensor_image_list:
            img = i.unsqueeze(0).to(self.device)
            # print(img.shape)
            # Perform inference
            with torch.inference_mode():
                try:
                    outputs = self.model(img)
                    # print(f"Raw outputs shape: {outputs.shape}")

                    outputs = outputs.permute(1, 0, 2)  # Permute to (sequence_length, batch_size, num_classes)
                    outputs = torch.nn.functional.log_softmax(outputs, 2)
                    # print(f"Permuted outputs shape: {outputs.shape}")

                    predicted_text = self.decode_ctc_output(outputs, self.vocab)
                    prediction.append(predicted_text)
                except:
                    pass
        return prediction

    def main_process_thread(self):
        t1 = Thread(target=self.main_process)
        t1.start()


    def main_process(self):
        try:
            if self.running:
                #screenshot get PIL image
                x = self.captureW.winfo_rootx()
                y = self.captureW.winfo_rooty()
                width = x + self.captureW.winfo_width()
                height = y + self.captureW.winfo_height()

                screenshot = ImageGrab.grab(bbox=(x, y, width, height))

                detection_result = self.detect_text_and_convert_to_tensors(screenshot)

                prediction_list = self.predict(detection_result)
                prediction_result = ''.join(prediction_list)
                if prediction_result.endswith("墜"):
                    prediction_result = prediction_result[0:-2]
                self.set_input_normal(self.input_text, prediction_result)


        except Exception as e:

            pass
        self.main_window.after(100, self.main_process_thread)

    def check_capture_window(self):
        try:
            if self.captureW is not None and self.captureW.state() == "normal":
                self.running=True
            else:
                self.running-False
        except:
            pass

        self.main_window.after(100, self.check_capture_window)

    #me trying to improve performance with this
    def thread_translate(self):
        t1 = Thread(target=self.translate)
        t1.start()



    def translate(self):

        src = self.language_dict[self.from_combo.get()]
        dest = self.language_dict[self.to_combo.get()]
        txt = self.retrieve_input(self.input_text)
        if txt != "":
            engine = self.engine_dict[self.engine_combo.get()]
            try:
                a = ts.translate_text(txt, translator=engine, from_language=src, to_language=dest)
            except:
                a = txt

            self.set_input(self.output_text, a)

        self.main_window.after(2000, self.thread_translate)
        
    def match_input(self):
        try:
            if self.queryW.winfo_exists():
                a = self.retrieve_input(self.input_text)
                b = self.retrieve_input(self.queryW.scrolledText)
                if a != b:

                    self.set_input(self.queryW.scrolledText, a)
        except:
            pass
        self.main_window.after(100, self.match_input)

    def match_output(self):
        try:
            if self.resultW.winfo_exists():
                a = self.retrieve_input(self.output_text)
                b = self.retrieve_input(self.resultW.scrolledText)
                if a != b:

                    self.set_input(self.resultW.scrolledText, a)
        except:
            pass
        self.main_window.after(100, self.match_output)

    def retrieve_input(self, tb1: scrolledtext.ScrolledText):
        nice = tb1.get("1.0", "end")
        return nice

    def set_input(self, tb: scrolledtext.ScrolledText, value: str):
        tb.configure(state="normal")
        tb.delete("1.0", "end")
        tb.insert("end", value)
        tb.configure(state="disabled")

    def set_input_normal(self, tb: scrolledtext.ScrolledText, value: str):
        tb.configure(state="normal")
        tb.delete("1.0", "end")
        tb.insert("end", value)

    def change_capture_opacity(self, val):
        try:
            if self.captureW.winfo_exists():
                self.captureW.attributes("-alpha", float(val))
        except Exception as e:
            print(e)

    def create_capture_window(self):
        if self.captureW is None or not self.captureW.winfo_exists():
            self.captureW = capture_window(self.main_window)
            self.captureW.title("Capture Window")
            self.captureW.configure(bg=self.dark)
            self.captureW.geometry("1280x360+0+0")
            self.captureW.wm_attributes("-topmost", True)
            self.captureW.attributes("-alpha", 0.8)

            # Opacity Slider
            self.slider_frame = tk.Frame(self.captureW, bg=self.dark)
            self.slider_frame.pack(side="right", padx=5, anchor="e")
            self.button_white1 = tk.Checkbutton(self.slider_frame, text="White Text", variable=self.whitevar, onvalue=1,
                                                offvalue=0, bg=self.dark, fg="white", selectcolor="black",
                                                activebackground=self.dark, activeforeground="white")
            self.button_white1.pack(side="right", padx=5)
            #self.opacity_label_main = tk.Label(self.slider_frame, text="Capture Window Opacity:", fg="white", bg=self.dark)
            #self.opacity_label_main.pack(side="left", padx=5)

            style = ttk.Style()
            style.configure("TScale", background=self.dark)
            self.opacity_slider = ttk.Scale(self.slider_frame, from_=0.1, to=1, value=0.8, orient="horizontal", style="TScale",
                                       command=lambda val: self.change_capture_opacity(val))

            self.opacity_slider.pack(side="right", padx=5)
        else:
            self.captureW.destroy()
            self.captureW = None

    def create_result_window(self):
        if self.resultW is None or not self.resultW.winfo_exists():
            self.resultW = query_window(self.main_window)
            self.resultW.title("Result Window")
            self.resultW.configure(bg=self.dark)
            self.resultW.geometry("480x240+500+0")
            self.resultW.wm_attributes("-topmost", True)

        else:
            self.resultW.destroy()
            self.resultW = None

    def create_query_window(self):
        if self.queryW is None or not self.queryW.winfo_exists():
            self.queryW = query_window(self.main_window)
            self.queryW.title("Query Window")
            self.queryW.configure(bg="#262729")
            self.queryW.geometry("480x240+0+0")
            self.queryW.wm_attributes("-topmost", True)


        else:
            self.queryW.destroy()
            self.queryW = None

    def swap(self):
        x = self.to_combo.current()
        y = self.from_combo.current()

        self.to_combo.current(y)
        self.from_combo.current(x)

onScreenTranslator()
