# Counting-with-Focus-for-Free
[Zenglin Shi, Pascal Mettes, and Cees G. M. Snoek. Counting with Focus for Free, ICCV, 2019](https://arxiv.org/pdf/1903.12206.pdf)
![image](https://github.com/shizenglin/Counting-with-Focus-for-Free/blob/master/image/overview.png)
<p> &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 Overview of our approach </p>

<h2> Requirements </h2>
     1. CUDA 8.0 and Cudnn 7.5 or higher
<br> 2. GPU memory 10GB or higher
<br> 3. Python 2.7
<br> 4. Tensorflow 1.04 or or higher

<h2> Data preprocessing </h2>
<h3> Datasets </h3>
     1. ShanghaiTech partA and partB
<br> 2. TRANCOS
<br> 3. Dublin Cell Counting
<br> 4. WIDER FACE
<br> 5. UCF-QNRF
<h3> Density map generation </h3>
Bsed on equation (1) and (7), for datasets with dense objects, ie, ShanghaiTech Part_A, TRANCOS and UCF-QNRF, we use our proposed non-uniform kernel with beta=0.3 and k=5. For ShanghaiTech Part_B and DCC, we set the Gaussian kernel variance to sigma=5 and sigma=10 respectively. For WIDER FACE, we obtain the Gaussian kernel variance by leveraging the box annotations. You can find the code in folder ¨data/getDmap.m¨.
<h3> Segmentation map generation </h3>
Bsed on equation(2), we use the same sigma as density map generation. You can find the code in folder ¨data/getPmap.m¨.
<h3> Global density generation </h3>
Bsed on equation(4) and (5), we use M=8 density levels for ShanghaiTech Part_A and UCF-QNRF, and 4 for the other datasets.

<h2> Training </h2>
     1. Prepare your data following the step of ¨data preprocessing¨.
<br> 2. Set the experiment settings in ¨code/tr_param.ini¨ in which phase = train, and set other parameters accordingly (refer to our paper).
<br> 3. Run ¨python code/main.py¨

<h2> Testing </h2>
     1. Prepare your data following the step of ¨data preprocessing¨.
<br> 2. Set the experiment settings in ¨code/tr_param.ini¨ in which phase = test, and set other parameters accordingly (refer to our paper).
<br> 3. Run ¨python code/main.py¨

<h2> Tips </h2>
     1. The dataload function in this code is a little bit slow, you can impove it by using Dataset API in TF.
<br> 2. Please generate your groundtruth map following the step of ¨data preprocessing¨ if you want to reproduce our reported numbers in Table 6.
<br> 3. You may get different results for each run because some random functions are used even with a fixed random seeds.
<br> 4. Please report your questions in ¨Issues¨, we can deal with them together.
<h2> Citation </h2>
Please cite our paper when you use this code.

