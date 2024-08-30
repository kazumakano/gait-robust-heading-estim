# Gait-Robust Heading Estimation Using Horizontal Acceleration for Smartphone-based PDR
<p align="center">
    <img alt="overview" src="https://github.com/kazumakano/gait-robust-heading-estim/assets/78717643/ba033488-c38e-43cf-ad0a-62b188788305" width="80%" />
</p>

<p align="center">
    <a href="https://ceur-ws.org/Vol-3581/191_WiP.pdf">
        <img alt="pdf badge" src="https://img.shields.io/badge/CEUR--WS.org-PDF-000080" />
    </a>
</p>

## Abstract
This study tackles heading estimation for Pedestrian Dead Reckoning (PDR) with smartphones.
In dealing with changes in the holding posture of smartphones, it works to consider the relationship between sensor orientation and heading.
However, the existing methods lack robustness to various gaits, such as sideways and backward walking.
Therefore, we propose a novel method considering various spatiotemporal features of horizontal acceleration with deep learning.
The proposed method calculates horizontal acceleration in the global coordinate system from measured acceleration, gravitational acceleration, and rotation vector.
Then, it inputs the horizontal acceleration over a certain period into a deep neural network model and predicts the unit vector directed to the mean heading during that period.
We created a dataset covering multiple gaits and evaluated the method using four models: Convolutional Neural Network (CNN), Bidirectional Long Short-Term Memory (BiLSTM), DualCNN-LSTM, and DualCNNTransformer.
Consequently, we found that the proposed method was more robust to gaits than the existing methods, with the DualCNN-LSTM and DualCNN-Transformer models achieving the highest accuracy.

## Citation
```bib
@inproceedings{gait-robust-heading-estim,
    author={Kano, Kazuma and Yoshida, Takuto and Katayama, Shin and Urano, Kenta and Yonezawa, Takuro and Kawaguchi, Nobuo},
    booktitle={WiP Proceedings of the Thirteenth International Conference on Indoor Positioning and Indoor Navigation - Work-in-Progress Papers (IPIN-WiP 2023)},
    editor={Kaiser, Susanna and Franke, Norbert and Mutschler, Christopher},
    month={12},
    title={Gait-Robust Heading Estimation Using Horizontal Acceleration for Smartphone-based PDR},
    volume={3581},
    year={2023}
}
```
