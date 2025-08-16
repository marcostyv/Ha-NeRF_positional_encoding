import torch
from torch import nn

	## --------- Para definir las funciones torch triangular, cuadrática y derivados ----------

    # Seno triangular
def tri(z):
	z = torch.remainder(z, 2 * torch.pi)
	out = 2 * torch.abs(z / torch.pi - 1) - 1
	return out

    # Seno triangular + pi/2
def tri_shifted(z):
    return tri(z + torch.pi / 2)

    # Seno triangular + pi
def tri_shifted_pi(z):
    return tri(z + torch.pi)

    # Seno triangular + 3pi/2
def tri_shifted_3pi2(z):
    return tri(z + 3*torch.pi/2)

    # Seno cuadrado
def square(z):
    z = torch.remainder(z, 2 * torch.pi)  
    out = torch.where(z < torch.pi, -1.0, 1.0)
    return out

    # Seno cuadrado + pi/2
def square_shifted(z):
    return square(z + torch.pi / 2)

    # Seno cuadrado + pi
def square_shifted_pi(z):
    return square(z + torch.pi)

    # Seno + pi
def sin_shifted_pi(z):
    return torch.sin(z + torch.pi)

    # Seno  + 3pi/2
def sin_shifted_3pi2(z):
    return torch.sin(z + 3*torch.pi/2)

	## --------------------------------------------------------------------------------

class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, funcs=None, logscale=True): ## Se añade el argumento funcs.
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
            
            ## Se cogen las funcs seleccionadas
        self.funcs = funcs if funcs is not None else [torch.sin, torch.cos] ## Se definen las funciones para ser usadas.

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs) 

        ''' 
                    En el caso inicial tenemos el código de arriba, con este código nosotros estamos metiendo un valor
                    de N_emb_xyz = 15. Por lo tanto, max_logscale = 14 y N_freqs = 15. 
                    La parte importante viene dentro del "if logscale":
                        - Primero sacamos una lista de valores con torch.linspace(0,14,15) -> Esto es sacame 15 valores desde 0 a 14. 
                          En este caso serían los 15 numeros del 0 al 14
                        - Despues hacemos 2^lista de numeros. Por tanto generamos un [2^0,2^1,....2^14]. Estas van a ser todas las frecuencias
                          que usemos para cada valor de lo que usamos para represenatar una posicion con los senos.

        '''
    def forward(self, x):
        """
        Inputs:
            x: (B, 3) ---->>>> La B son los puntos en el espacio y el 3 indica que cada uno tiene 3 dimensiones.
                               Esto especifica la forma del dato del vector de posicion.
                               ejemplo: x[0][0] es el valor de posicion del punto 0 en el eje x 
        Outputs:
            out: (B, 3 + 3 * N_freqs * número_de_funciones) ---->>> La forma a la salida sera otra matriz pero tiene [3poscionesNormales + 3posiciones*Nfuncs*N_frecs] columnas.
        """


        out = [x] ## Posición a codificar.
        for freq in self.freqs: ## Para cada frecuencia de las anteriores ([2^0,2^1,....2^14]).
            for func in self.funcs: ## Para cada funcion dentro de las funciones definidas.
                out += [func(freq*x)] ## La salida es una suma de las funciones usadas y cada argumento es el valor de la posicion*frecuencias. Se da una lista de tensores.
                

	
	
        return torch.cat(out, -1) ## Devuelve el valor de la posición codificada.

        

        



class NeRF(nn.Module):
    def __init__(self, typ,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_random=False):

        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir

        # self.encode_appearance = encode_appearance
        self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_random = False if typ=='coarse' else encode_random

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                        nn.Linear(W+in_channels_dir+self.in_channels_a, W//2), nn.ReLU(True))
        self.static_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())


    def forward(self, x, sigma_only=False, output_random=True):

        if sigma_only:
            input_xyz = x
        elif output_random:
            input_xyz, input_dir, input_a, input_random_a = \
                  torch.split(x, [self.in_channels_xyz,
                                  self.in_channels_dir,
                                  self.in_channels_a,
                                  self.in_channels_a], dim=-1)
            input_dir_a = torch.cat((input_dir, input_a), dim=-1)
            input_dir_a_random = torch.cat((input_dir, input_random_a), dim=-1)
        else:
            input_xyz, input_dir_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a], dim=-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        static_sigma = self.static_sigma(xyz_) # (B, 1)
        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding) # (B, 3)
        static = torch.cat([static_rgb, static_sigma], 1) # (B, 4)

        if output_random:
            dir_encoding_input_random = torch.cat([xyz_encoding_final.detach(), input_dir_a_random.detach()], 1)
            dir_encoding_random = self.dir_encoding(dir_encoding_input_random)
            static_rgb_random = self.static_rgb(dir_encoding_random) # (B, 3)
            return torch.cat([static, static_rgb_random], 1) # (B, 7)

        return static
