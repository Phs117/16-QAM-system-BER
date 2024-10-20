# -------------------------------------------------------------------------
# Bibliotecas
# -------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from numpy import log2, sqrt
from scipy.special import erfc
from scipy import signal
from scipy.signal import upfirdn
from scipy.spatial.distance import cdist
from PIL import Image
from tqdm import tqdm
import abc
import galois

#%%
class Equalizer():
    # Base class: Equalizer (Abstract base class)
    # Attribute definitions:
    #    self.N: length of the equalizer
    #    self.w : equalizer weights
    #    self.delay : optimized equalizer delay
    def __init__(self,N): # constructor for N tap FIR equalizer
        self.N = N
        self.w = np.zeros(N)
        self.opt_delay = 0
        
    @abc.abstractmethod
    def design(self): #Abstract method
        "Design the equalizer for the given impulse response and SNR"
    
    def convMatrix(self,h,p):
        """
        Construct the convolution matrix of size (N+p-1)x p from the
        input matrix h of size N. (see chapter 1)
        Parameters:
            h : numpy vector of length L
            p : scalar value
        Returns:
            H : convolution matrix of size (L+p-1)xp
        """
        col=np.hstack((h,np.zeros(p-1)))
        row=np.hstack((h[0],np.zeros(p-1)))
        
        from scipy.linalg import toeplitz
        H=toeplitz(col,row)
        return H
        
    def equalize(self,inputSamples):
        """
        Equalize the given input samples and produces the output
        Parameters:
            inputSamples : signal to be equalized
        Returns:
            equalizedSamples: equalized output samples
        """
        #convolve input with equalizer tap weights
        equalizedSamples = np.convolve(inputSamples,self.w)
        return equalizedSamples    
        
class zeroForcing(Equalizer): #Class zero-forcing equalizer
    def design(self,h,delay=None): #override method in Equalizer abstract class
        """
        Design a zero forcing equalizer for given channel impulse response (CIR).
        If the tap delay is not given, a delay optimized equalizer is designed
        Parameters:
            h : channel impulse response
            delay: desired equalizer delay (optional)
        Returns: MSE: Mean Squared Error for the designed equalizer
        """
        L = len(h)
        H = self.convMatrix(h,self.N) #(L+N-1)xN matrix - see Chapter 1
        # compute optimum delay based on MSE
        Hp = np.linalg.pinv(H) #Moore-Penrose Pseudo inverse
        #get index of maximum value using argmax, @ for matrix multiply
        opt_delay = np.argmax(np.diag(H @ Hp))
        self.opt_delay = opt_delay #optimized delay
        
        if delay==None:
            delay=opt_delay
        elif delay >=(L+self.N-1):
            raise ValueError('Given delay is too large delay (should be < L+N-1')
        
        k0 = delay
        d=np.zeros(self.N+L-1);d[k0]=1 #optimized position of equalizer delay
        self.w=Hp @ d # Least Squares solution, @ for matrix multiply
        MSE=(1-d.T @ H @ Hp @ d) #MSE and err are equivalent,@ for matrix multiply
        return MSE
#%%
# -------------------------------------------------------------------------
# Funções
# -------------------------------------------------------------------------
def qamPe(M, EbN0dBs):
    gamma_s = log2(M) * (10**(EbN0dBs/10))
    SERs = 1 - (1 - (1 - 1/sqrt(M)) * erfc(sqrt(3/2 * gamma_s / (M-1))))**2
    return SERs

def qamConst(M, Ex=1):
    d = np.sqrt(6 * Ex / (M - 1))
    QamSym = np.arange(-(np.sqrt(M) - 1), np.sqrt(M), 2) * d / 2
    QamSym = (np.ones((int(np.sqrt(M)), 1)) @ QamSym.reshape(1, -1) + 
              1j * QamSym.reshape(-1, 1)[::-1] @ np.ones((1, int(np.sqrt(M)))))
    QamSym = QamSym.reshape(-1)
    
    return QamSym

def awgn(x, EbN0dB, oversampling):
    snr_dB = EbN0dB + 10 * np.log10(log2(M)) - 10 * np.log10(oversampling)
    Nx = len(x)
    snr = 10 ** (snr_dB / 10)
    Px = np.sum(abs(x) ** 2) / Nx
    sigma2 = Px / snr
    if np.isrealobj(x):
        w = np.sqrt(sigma2) * np.random.randn(Nx)
    else:
        w = np.sqrt(sigma2 / 2) * (np.random.randn(Nx) + 1j * np.random.randn(Nx))
    y = x + w
    
    return y

def detectorIQ(Sym, SymRE):
    XA = np.column_stack((np.real(SymRE), np.imag(SymRE)))
    XB = np.column_stack((np.real(Sym), np.imag(Sym)))
    d = cdist(XA, XB, metric='euclidean')
    ind = np.argmin(d, axis=1)
    x_hat = np.array([Sym[ind[i]] for i in range(len(ind))])
    
    return x_hat  

def srrc(beta, Tb, L, Nb):
    Fs = L / Tb
    Ts = 1 / Fs
    t = np.arange(-Nb * Tb / 2, Nb * Tb / 2 + Ts, Ts)
    p = (1 / Tb) * ((np.sin(np.pi * t * (1 - beta) / Tb) + 
                     (4 * beta * t / Tb) * np.cos(np.pi * t * (1 + beta) / Tb)) /
                    ((np.pi * t / Tb) * (1 - (4 * beta * t / Tb) ** 2)))
    
    p[t == 0] = (1 / Tb) * (1 - beta + 4 * beta / np.pi)
    p[t == Tb / (4 * beta)] = (beta / (np.sqrt(2) * Tb)) * (
        (1 + (2 / np.pi)) * np.sin(np.pi / (4 * beta)) + 
        (1 - (2 / np.pi)) * np.cos(np.pi / (4 * beta)))
    p[t == -Tb / (4 * beta)] = (beta / (np.sqrt(2) * Tb)) * (
        (1 + (2 / np.pi)) * np.sin(np.pi / (4 * beta)) + 
        (1 - (2 / np.pi)) * np.cos(np.pi / (4 * beta)))
    
    return (t, p)

def eyediagram(x, n, title):
    plt.figure(figsize = (10,8),dpi = 300
               )
    num_traces = len(x) // n
    for i in range(num_traces):
        trace = x[i * n:(i + 1) * n]
        plt.plot(trace, color='blue', alpha=0.5)
    
    plt.title(f'Diagrama de olho - {title}')
    plt.xlabel('Tempo')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
    
def encode(image_path, M):
    qam_alphabet = qamConst(M) # Gera um array com os simbolos M-QAM
    image = Image.open(image_path).convert('L') # Converte a imagem para a escala Grey, onde cada pixel é representado por um valor de 0-255
    data = np.array(image).flatten() # Transforma os pixeis da imagem 2-D em um Vetor 1-D
    binary_data = ''.join(format(byte, '08b') for byte in data) # Percorre cada valor do pixel em data e converte para uma string binaria de 8 bits e gera uma string continua
    bits_per_symbol = int(np.log2(len(qam_alphabet))) # Calcula o numero de bits por simbolo
    symbols = [qam_alphabet[int(binary_data[i:i+bits_per_symbol], 2)] for i in range(0, len(binary_data), bits_per_symbol)] #Mapea os bits para simbolos QAM, basicamente pega um bloco de bits (M = 16, ent 4 bits) e converte para simbolos Qam
    symbols_array = np.array(symbols) # Gera um array numpy
    
    # Plotar a imagem original
    plt.figure(dpi = 300)
    plt.imshow(image, cmap='gray')
    plt.title("Imagem Transmitida")
    plt.axis('off')
    plt.show()
    
    return symbols_array, image.size

def decode(symbols, M, image_shape,EbN0dB):
    qam_alphabet = qamConst(M)
    reverse_alphabet = {symbol: format(i, f'0{int(np.log2(len(qam_alphabet)))}b') for i, symbol in enumerate(qam_alphabet)}
    binary_data = ''.join(reverse_alphabet[symbol] for symbol in symbols)
    byte_data = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
    int_data = [int(byte, 2) for byte in byte_data]
    image_data = np.array(int_data, dtype=np.uint8).reshape(image_shape)
    image = Image.fromarray(image_data)
    
    # Plotar a imagem decodificada
    plt.figure(dpi = 300)
    plt.imshow(image, cmap='gray')
    plt.title(f"Imagem Recebida com Eb/N0 = {EbN0dB} dB")
    plt.axis('off')
    plt.show()
    
    return image

def plotarConst(M,SymRE,EbN0dB):
    QamSyms = qamConst(M)
    plt.figure(figsize = (10,8),dpi = 300)
    plt.title(f'Constelação {M}-QAM para Eb/N0 = {EbN0dB} dB')
    plt.plot(np.real(SymRE),np.imag(SymRE),'.',label = 'Simbolos Recebidos',color = 'red')
    plt.plot(np.real(QamSyms),np.imag(QamSyms),'.',label = 'Simbolos Transmitidos',color = 'blue',markersize = 10)
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.legend(loc='upper right',ncol=2)
    plt.show()
    
def plotarResposta(h,Fs,title):
    #CFR
    F,H = signal.freqz(h,1,2048,whole = True,fs = Fs)
    gain = 20*np.log10(np.fft.fftshift(abs(H)))
    #Plot
    plt.figure(figsize=(10, 6),dpi = 300)
    plt.plot((F - Fs/2)/1000, gain)
    plt.axis([-Fs/2000,Fs/2000,-15,15])
    plt.title(f'CFR para h = {title}')
    plt.xlabel('f (kHz)')
    plt.ylabel('Ganho (dB)')
    plt.grid(True)
    plt.show()

def de2bi(matriz, num_bits):
    matriz_bits = [[np.binary_repr(valor, width=num_bits) for valor in linha] for linha in matriz]
    matriz_expandida = []
    for linha in matriz_bits:
        for bit_pos in range(num_bits):
            linha_bits = [int(bits[bit_pos]) for bits in linha]
            matriz_expandida.append(linha_bits)
    return np.array(matriz_expandida)

def bi2de(matriz_bits):
    num_rows = bit_symbols.shape[0]
    num_cols = bit_symbols.shape[1]

    matriz_decimal = []
    for i in range(num_rows):
        bit_string = ''.join(str(bit_symbols[i, j]) for j in range(num_cols))
        valor_decimal = int(bit_string, 2)
        matriz_decimal.append(valor_decimal)
    
    return np.array(matriz_decimal)

def bi2de_matriz(matriz_expandida, num_bits):
    num_linhas_expandida = len(matriz_expandida)
    num_colunas = len(matriz_expandida[0])
    
    # Calculando o número original de linhas da matriz
    num_linhas_original = num_linhas_expandida // num_bits
    
    # Inicializando a matriz original
    matriz_original = []
    
    for i in range(num_linhas_original):
        linha_original = []
        for j in range(num_colunas):
            # Construindo o valor binário a partir das linhas expandida
            valor_binario = ''.join(str(matriz_expandida[i * num_bits + k][j]) for k in range(num_bits))
            # Convertendo o valor binário para decimal
            valor_decimal = int(valor_binario, 2)
            linha_original.append(valor_decimal)
        matriz_original.append(linha_original)
    
    return np.array(matriz_original)

#%%
# -------------------------------------------------------------------------
# Parametros
# -------------------------------------------------------------------------

Fs = 44100           # Frequencia de Amostragem do Sinal Continuo (Hz)
Ts = 1 / Fs          # Periodo de Amostragem (s)
oversampling = 10    # Fator Fs/R
R = Fs / oversampling  # Taxa de Transmissao em simbolos/s (baud rate)
T = 1 / R            # Periodo de Simbolo (s)
dell = 25            # Resposta do filtro formatador se estende por (2*dell) periodos de simbolo
rolloff = 0.5        # Fator de rolloff dos filtros Tx e Rx

# Filtro de Tx+Rx
filtro = srrc(rolloff, T, oversampling, 2*dell) 
pulso = filtro[1]
pulso = pulso / np.linalg.norm(pulso)

# Canal Passa Baixas Simulado usando um Filtro de Butterworth 
fc = R / 2 * (1 + rolloff)  # Largura de banda do sinal transmitido
bn, an = signal.butter(5, 2 * fc / Fs)  # Filtro passa-baixas de largura de banda fc

# Canal do exercicio de simulação 4
h_k = np.array([0.19 +1j*0.56, 0.45 - 1j*1.28, -0.14 - 1j*0.53, -0.19 + 1j*0.23, 0.33 +1j*0.51])
#plotarResposta(h_k,Fs,'Canal distorcido')

# Equalizador
#from DigiCommPy.equalizers import zeroForcing
# Equalizer Design Parameters
N = 18

zf = zeroForcing(N)
mse = zf.design(h = h_k)
w = zf.w # Coeficientes do filtro
#plotarResposta(w,Fs,'Canal ZF')

h_sys = zf.equalize(h_k) # Efeito do canal e do equalizador
#plotarResposta(h_sys,Fs,'Canal equalizado')

# Canal sem distorção
#h = np.array([1])
#plotarResposta(h,Fs)

# Plotar resposta dos 3 canais
plt.figure(figsize=(10,6),dpi = 300)

F,H = signal.freqz(h_k,1,2048,whole = True,fs = Fs)
gain = 20*np.log10(np.fft.fftshift(abs(H)))
#Plot do canal com distorção
plt.plot((F - Fs/2)/1000, gain,label = 'h_isi')

# Plot do canal ZF
F,H = signal.freqz(w,1,2048,whole = True,fs = Fs)
gain = 20*np.log10(np.fft.fftshift(abs(H)))
plt.plot((F - Fs/2)/1000, gain,label = 'h_zf')

# Plot do canal equalizado
F,H = signal.freqz(h_sys,1,2048,whole = True,fs = Fs)
gain = 20*np.log10(np.fft.fftshift(abs(H)))
plt.plot((F - Fs/2)/1000, gain,label = 'h_sys')

plt.axis([-Fs/2000,Fs/2000,-15,15])
plt.title(f'CFR para $N_1 + N_2 = {N}$')
plt.xlabel('f (kHz)')
plt.ylabel('Ganho (dB)')
plt.grid(True)
plt.legend()
plt.show()
#%%
# -------------------------------------------------------------------------
# Parametros de simulação
# -------------------------------------------------------------------------

# Caminho da imagem a ser transmitida
image_path = 'image.tif'

ArrayDeM = [16]
EbN0dBs = np.arange(0,20,1)
#nSym = 10**6 # Simulação monte carlo

BER_simulada = np.zeros(len(EbN0dBs))
BER_teorica = np.zeros(len(EbN0dBs)) 

plt.style.use('bmh')
markers = ['o-', 's', '^', 'd']
linestyles = ['-', '--', '-.', ':']
   
for i, M in enumerate(ArrayDeM):
    bit = np.log2(M)
    QamSym = qamConst(M=16,Ex=10)
    
    # Codificação de canal
    rs = galois.ReedSolomon(63,59)
    GF = rs.field
    num_blocos = 1000
    
    matriz_blocos = np.array([GF.Random(rs.k) for _ in range(num_blocos)])
    matriz_encode = rs.encode(matriz_blocos)
    matriz_encode_bits = de2bi(matriz_encode,6)
    bit_symbols = matriz_encode_bits.reshape(-1,2)

    # Modulação
    symbols = bi2de(bit_symbols)
    alfabeto = [-3, -1, 1, 3]
    pam = np.array([alfabeto[val] for val in symbols])
    pam_I = pam[0::2]
    pam_Q = pam[1::2]
    QamSyms_tx = pam_I + 1j*pam_Q
    
    
    sinal_tx = upfirdn([1], QamSyms_tx, oversampling) 
    sinal_tx_filtrado = np.convolve(sinal_tx, pulso)
    sinal_rx = signal.filtfilt(h_k, 1, sinal_tx_filtrado) 

    
    for j, EbN0dB in tqdm(enumerate(EbN0dBs), total=len(EbN0dBs), desc=f'{M} - QAM '): # Adição do ruido AWGN
        sinal_rx_ruido = awgn(sinal_rx, EbN0dB, oversampling)
        sinal_rx_casado = np.convolve(sinal_rx_ruido, pulso[::-1])
        
        sinal_rx_casado = signal.filtfilt(w,1,sinal_rx_casado)   # Equalização
        
        qam_rx = sinal_rx_casado[::oversampling]
        QamSyms_rx = qam_rx[dell*2:len(qam_rx)-dell*2] 
        QamSyms_quant = detectorIQ(QamSym, QamSyms_rx)
        
        pam_I_rx = np.real(QamSyms_quant)
        pam_Q_rx = np.imag(QamSyms_quant)

        pam_rx = np.zeros(len(pam))
        pam_rx[0::2] = pam_I_rx
        pam_rx[1::2] = pam_Q_rx
        
        # Demodulação
        symbols_rx = (pam_rx + 3) / 2
        symbols_rx = np.array(symbols_rx,dtype = int)

        # Decodificação de canal
        bit_symbols_rx = np.array([list(format(value, '02b')) for value in symbols_rx], dtype=int)
        matriz_encode_bits_rx = bit_symbols_rx.reshape(-1,63)
        matriz_encode_rx = bi2de_matriz(matriz_encode_bits_rx, 6)
        matriz_blocos_rx = rs.decode(matriz_encode_rx)
        
        BER_simulada[j] = np.sum(matriz_blocos_rx != matriz_blocos)/(num_blocos*63*6)
        BER_teorica[j] = qamPe(M, EbN0dB) / bit
        
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(EbN0dBs, BER_teorica, label=f'Teórica {M}-QAM', color='grey', linestyle=linestyles[i])
    plt.plot(EbN0dBs, BER_simulada, markers[i], label=f'Simulação {M}-QAM', color='black')
    plt.yscale('log')
    plt.ylabel('BER')
    plt.xlabel('Eb/N0(dB)')  
    plt.ylim(10**-7, 10**0)   
    plt.legend(ncol=4) 
   
plt.title("BER para M-QAM em um canal AWGN")     
plt.show()
