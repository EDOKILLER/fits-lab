import numpy as np, math
from PIL import Image, ImageDraw, ImageFilter, ImageFont
S=1024; rng=np.random.default_rng(7)
yy,xx=np.mgrid[0:S,0:S]/S
# fondo: viola-blu notte con alone
bg=np.zeros((S,S,3)); r=np.hypot(xx-0.5,yy-0.45)
bg[...,0]=18+30*np.exp(-r**2/0.08); bg[...,1]=14+18*np.exp(-r**2/0.08); bg[...,2]=46+60*np.exp(-r**2/0.1)
img=Image.fromarray(bg.clip(0,255).astype(np.uint8))
# maschera della beuta (Erlenmeyer)
def beuta_poly(s=1.0):
    cx=0.5*S; top=0.16*S; collo=0.085*S; base=0.33*S; spalla=0.40*S; fondo=0.86*S
    return [(cx-collo,top),(cx+collo,top),(cx+collo,spalla),(cx+base,fondo-0.04*S),(cx+base-0.05*S,fondo),
            (cx-base+0.05*S,fondo),(cx-base,fondo-0.04*S),(cx-collo,spalla)]
m=Image.new('L',(S,S),0); ImageDraw.Draw(m).polygon(beuta_poly(),fill=255)
liq=Image.new('L',(S,S),0); d=ImageDraw.Draw(liq); d.polygon(beuta_poly(),fill=255)
d.rectangle([0,0,S,0.50*S],fill=0)  # livello del liquido
# nebulosa nel liquido
neb=np.zeros((S,S,3))
for c,(cx,cy,sx,sy,a) in zip([(255,70,160),(255,150,60),(90,140,255),(255,220,180)],
        [(0.42,0.70,0.16,0.09,1.0),(0.6,0.66,0.12,0.12,0.9),(0.5,0.78,0.25,0.07,0.8),(0.5,0.68,0.05,0.05,1.2)]):
    g=a*np.exp(-(((xx-cx)/sx)**2+((yy-cy)/sy)**2)/2)
    for k in range(3): neb[...,k]+=c[k]*g
# filamenti
for _ in range(9):
    t=np.linspace(0,1,200); x0,y0=rng.uniform(0.3,0.7),rng.uniform(0.58,0.82); ang=rng.uniform(0,np.pi)
    xs=x0+0.15*np.cos(ang)*(t-0.5)+0.02*np.sin(8*t+rng.uniform(0,6)); ys=y0+0.08*np.sin(ang)*(t-0.5)
    for X,Y in zip(xs,ys):
        g=np.exp(-((xx-X)**2+(yy-Y)**2)/(2*0.005**2))*0.03
        neb[...,0]+=255*g; neb[...,1]+=120*g; neb[...,2]+=200*g
neb=(neb*0.55).clip(0,200)
base_liq=np.array([40,20,80])
nebimg=Image.fromarray((neb+base_liq).clip(0,255).astype(np.uint8))
img.paste(nebimg,(0,0),liq)
# stelle nel liquido
d=ImageDraw.Draw(img)
L=np.array(liq)>0
for _ in range(70):
    x,y=rng.integers(0,S,2)
    if L[y,x]:
        rr=rng.choice([2,3,4,6],p=[.5,.3,.15,.05]); d.ellipse([x-rr,y-rr,x+rr,y+rr],fill=(255,250,235))
# bolle a forma di pixel (quadratini) che salgono dal liquido
for i,(x,y,l,c) in enumerate([(0.47,0.44,0.035,(120,220,255)),(0.53,0.36,0.028,(255,210,90)),(0.49,0.28,0.022,(255,120,200)),(0.52,0.21,0.016,(160,255,200))]):
    d.rectangle([x*S-l*S/2,y*S-l*S/2,x*S+l*S/2,y*S+l*S/2],fill=c,outline=(255,255,255),width=3)
# vetro: contorno chiaro e riflesso
vetro=Image.new('RGBA',(S,S),(0,0,0,0)); dv=ImageDraw.Draw(vetro)
dv.line(beuta_poly()+[beuta_poly()[0]],fill=(200,235,255,255),width=18,joint='curve')
dv.rectangle([0.5*S-0.115*S,0.135*S,0.5*S+0.115*S,0.175*S],fill=(200,235,255,255))
dv.line([(0.36*S,0.80*S),(0.47*S,0.43*S)],fill=(255,255,255,110),width=14)
# linea del liquido
dv.line([(0.5*S-0.135*S,0.50*S),(0.5*S+0.135*S,0.50*S)],fill=(255,190,240,200),width=6)
vetro=vetro.filter(ImageFilter.GaussianBlur(1))
img=img.convert('RGBA'); img.alpha_composite(vetro)
# fascia «LAB» dorata e scritta FITS
fb=ImageDraw.Draw(img)
try: font=ImageFont.truetype('DejaVuSans-Bold.ttf',int(0.075*S))
except OSError: font=ImageFont.load_default()
fb.rounded_rectangle([0.25*S,0.885*S,0.75*S,0.975*S],radius=int(0.03*S),fill=(232,180,60,255))
tb=fb.textbbox((0,0),'FITS LAB',font=font); 
fb.text((S/2-(tb[2]-tb[0])/2-tb[0], 0.93*S-(tb[3]-tb[1])/2-tb[1]),'FITS LAB',font=font,fill=(30,18,60,255))
# angoli arrotondati
mask=Image.new('L',(S,S),0); ImageDraw.Draw(mask).rounded_rectangle([0,0,S-1,S-1],radius=int(0.18*S),fill=255)
out=Image.new('RGBA',(S,S),(0,0,0,0)); out.paste(img,(0,0),mask)
out=out.resize((512,512),Image.LANCZOS)
out.save('/home/claude/fitslab/fitslab.png')
out.save('/home/claude/fitslab/fitslab.ico',sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])
out.resize((128,128),Image.LANCZOS).save('/tmp/icona_128.png')
