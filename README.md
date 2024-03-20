This repository contains data and program used in the manuscript ["Deep learning with the potential to provide accurate and reliable seafloor topography for the public"] submitted to Nature Geoscience on 21th Mar, 2024, by Yang Liu, Sanzhong Li, Weichao Yan, Zhuoyan Zou, Yi Sun, Haohao Cheng, Yongbo Yu, Ziying Li, Lixin Wu. 

CONTENTS:

[Data]: This directory contains train data and test data for six ocean area, including the Hatteras Plain, the Gulf of Guinea, the Bellingshausen Sea, the Mariana Trench, the South China Sea, the Arabian Sea. For each ocean areas, 13 variables—including gravity anomalies, residual gravity anomalies, vertical gravity gradients, residual vertical gravity gradients, east component of vertical deflection, north component of vertical deflection, isostatic gravity anomalies, sediment thickness, the depth to Moho, magnetic anomalies and coordinates (longitude and latitude)—were provided in a csv format. The reference sources of all these data can be found in online Data Availability of this paper.

[Code]: This directory contains programs related to deep learning model, model testing, uncertainty evaluation and interpretability evaluation.

[Reconstruct]: This directory contains seafloor topography and uncertainty distributions for six ocean areas mapped by deep learning, as well as other traditional models (ETOPO and SIO) and GMM.