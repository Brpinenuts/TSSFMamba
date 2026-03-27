# TSSFMamba
Code of papaer: TSSFMamba: Temporal-Spatial-Spectral Fusion State Space Model   for  Dim Moving  Target Detection in Hyperspectral Image Sequences



<img width="2350" height="1502" alt="flos" src="https://github.com/user-attachments/assets/0297f2c4-0e35-4cfc-b069-8db2e075e289" />


## Abstract

<table>
<tr>
<td>

Dim moving target detection (DMTD) in hyperspectral image sequences (HSIS) aims to identify
potential small moving anomalous targets with low contrast relative to the background of HSIS and
has garnered substantial attention in various remote sensing photography and surveying applications.
Recently, owing to their prominent nonlocal representations and linear complexity, Mamba-based
approaches have drawn growing attention. Our study pioneers the integration of Mamba into
DMTD task in HSIS, presenting TSSFMamba, which introduces a novel temporal-spatial-spectral
fusion Mamba detection architecture that gives consideration to the advantages of lightweight and
high precision for dim moving target detection. The overall network adopts multiscale encoder-
decoder reconstruction learning architecture which consists of spatial-spectral-temporal Mamba block
for long-range nonlocal feature fusion and temporal-spatial-spectral decoupled convolution block
for lightweight local representation enhancement. Furthermore, a Haar discrete wavelet transform
convolution module is designed to explicitly capture the discriminative frequency characteristics
between targets and the background, thereby facilitating the effective suppression of anomalous
targets during background reconstruction. To suppress false alarms caused by background clutter
in various scenarios, a motion-consistency optical flow estimation module is introduced to perform
motion optical flow estimation and model the motion difference representation between moving
targets and background clutter. Extensive experiments on diverse DMTD datasets in both real
scenes and simulated HSIS demonstrate that TSSFMamba achieves state-of-the-art detection accuracy
while requiring the lowest parameter quantity and running time, confirming its superiority in both
effectiveness and efficiency. The code can be available at http://github.com/Brpinenuts/TSSFMamba.



</details>

</td>
</tr>
</table>
