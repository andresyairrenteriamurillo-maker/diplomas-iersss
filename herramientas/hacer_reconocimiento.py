"""Crea la plantilla 'Reconocimiento' a partir de la de Mención de Honor."""
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

src, out = sys.argv[1], sys.argv[2]
FONT_TITULO = sys.argv[3] if len(sys.argv) > 3 else "/tmp/ttf/gv.ttf"

img = cv2.imread(src)  # BGR
H, W = img.shape[:2]
b, g, r = [img[:, :, i].astype(int) for i in range(3)]

# ---------- 1. Borrar "Mención de Honor" de la cinta ----------
x0, x1, y0, y1 = 340, 1200, 378, 508
mask = np.zeros((H, W), np.uint8)
zona = np.zeros((H, W), bool); zona[y0:y1, x0:x1] = True
blanco = (r > 110) & (b > 150)                      # letras blancas / grises
oro = (r > 150) & (b < 140)                         # bordes dorados de la cinta
mask[zona & blanco] = 255
mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=4)   # incluye bordes y sombra
mask[~zona] = 0
mask[oro & (mask > 0)] = 0
# no tocar el oro: además protege 2px alrededor del oro
oro_m = cv2.dilate(oro.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=int(sys.argv[4]) if len(sys.argv)>4 else 4) > 0
mask[oro_m] = 0
limpio = cv2.inpaint(img, mask, 9, cv2.INPAINT_TELEA)
# suavizado solo dentro de la zona borrada para quitar manchas
blur = cv2.GaussianBlur(limpio, (0, 0), 3)
m3 = cv2.GaussianBlur(mask, (0, 0), 3).astype(float)[:, :, None] / 255.0
limpio = (limpio * (1 - m3) + blur * m3).astype(np.uint8)

# reparar el borde dorado donde lo cruzaba la tilde de la ó
parche = limpio[368:394, 690:728].copy()
limpio[368:394, 726:764] = parche
for k in range(4):  # suavizar las uniones
    a = (k + 1) / 5
    for xx in (726 + k, 763 - k):
        lado = 725 if xx < 745 else 764
        limpio[368:394, xx] = (limpio[368:394, xx] * a + limpio[368:394, lado] * (1 - a)).astype(np.uint8)

# ---------- 2. Borrar "Confiere la presente" ----------
cx0, cx1, cy0, cy1 = 568, 968, 300, 356
limpio[cy0:cy1, cx0:cx1] = (255, 255, 255)

pil = Image.fromarray(cv2.cvtColor(limpio, cv2.COLOR_BGR2RGB)).convert("RGBA")

# ---------- 3. Escribir "Confiere el presente" ----------
f = ImageFont.truetype("/tmp/ttf/cg6.ttf", 50)
t = "Confiere el presente"
capa = Image.new("RGBA", pil.size, (0, 0, 0, 0)); d = ImageDraw.Draw(capa)
w = d.textlength(t, font=f)
d.text(((cx0 + cx1) / 2 - w / 2, 343), t, font=f, fill=(20, 20, 20, 255), anchor="ls")
pil = Image.alpha_composite(pil, capa)

# ---------- 4. Escribir "Reconocimiento" en la cinta ----------
titulo = "Reconocimiento"
tam = 150
ft = ImageFont.truetype(FONT_TITULO, tam)
cx, base = (x0 + x1) / 2 - 8, 484
while True:
    bb = ImageDraw.Draw(Image.new("L", (10, 10))).textbbox((0, 0), titulo, font=ft, anchor="ls", stroke_width=3)
    if bb[2] - bb[0] <= 780 and -bb[1] <= 112: break
    tam -= 2; ft = ImageFont.truetype(FONT_TITULO, tam)
px = cx - (bb[2] + bb[0]) / 2

sombra = Image.new("RGBA", pil.size, (0, 0, 0, 0))
ImageDraw.Draw(sombra).text((px + 5, base + 6), titulo, font=ft, anchor="ls", fill=(0, 10, 40, 200), stroke_width=3, stroke_fill=(0, 10, 40, 200))
sombra = sombra.filter(ImageFilter.GaussianBlur(4))
pil = Image.alpha_composite(pil, sombra)

letra = Image.new("RGBA", pil.size, (0, 0, 0, 0))
ImageDraw.Draw(letra).text((px, base), titulo, font=ft, anchor="ls", fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(255, 255, 255, 255))
# leve degradado gris abajo para dar volumen como el original
alfa = np.asarray(letra.split()[3]).astype(float) / 255
grad = np.ones((H, W, 3)) * 255
ys = np.arange(H)[:, None]
sh = np.clip((ys - (base - 60)) / 70.0, 0, 1) * 28
grad = grad - sh[:, :, None] * np.array([1, 0.9, 0.6])
letra_rgb = Image.fromarray(np.clip(grad, 0, 255).astype(np.uint8)).convert("RGBA")
letra_rgb.putalpha(letra.split()[3])
pil = Image.alpha_composite(pil, letra_rgb)

pil.convert("RGB").save(out, quality=92)
print("tamaño título", tam)
