# Counting-with-Focus-for-Free
Code for Counting with Focus for Free, ICCV, 2019

<h2> Code environments </h2>
<br> 1. Ubantu 16.0 or higher
<br> 2. CUDA 8.0 and Cudnn 7.5 or higher
<br> 3. GPU memory 10GB or higher
<br> 4. Python 2.7
<br> 5. Tensorflow 1.04 or or higher

<h2> Data preprocessing </h2>
<h3> Datasets </h3>
<br> 1. ShanghaiTech
<br> 2. TRANCOS
<br> 3. Dublin Cell Counting
<br> 4. WIDER FACE
<br> 5. UCF-QNRF
<h3> Density map generation </h3>
Bsed on equation (1) and (7), for datasets with dense objects, ie, ShanghaiTech Part_A, TRANCOS and UCF-QNRF, we use our proposed non-uniform kernel with beta=0.3 and k=5. For ShanghaiTech Part_B and DCC, we set the Gaussian kernel variance to sigma=5 and sigma=10 respectively. For WIDER FACE, we obtain the Gaussian kernel variance by leveraging the box annotations.
<h3> Segmentation map generation </h3>
Bsed on equation(2), we use the same sigma as density map generation.
<h3> Global density generation </h3>
Bsed on equation(4) and (5), we use M=8 density levels for ShanghaiTech Part_A and UCF-QNRF, and 4 for the other datasets.

<h3> Code </h3>
You can find these map generation codes in folder ¨data¨

<h2> Training </h2>
<br> 1. Prepare your data according to data preprocessing step.
<br> 2. Set the experiment settings in ¨code/tr_param.ini¨ in which phase = train, and set other parameters accordingly (refer to our paper).
<br> 3. run ¨python code/main.py¨

<h2> Testing </h2>
<br> 1. Prepare your data according to data preprocessing step.
<br> 2. Set the experiment settings in ¨code/tr_param.ini¨ in which phase = test, and set other parameters accordingly (refer to our paper).
<br> 3. run ¨python code/main.py¨
