# Literature reviewed for the interim report

All downloaded papers live in this folder. The PDFs are kept locally and are not committed; the publisher or arXiv link
for each paper is given instead. "Protocol" says whether test patients were
also seen in training (intra-patient) or kept separate (inter-patient).

| # | Paper | Protocol | Local file | Row owner |
|---|---|---|---|---|
| 1 | P. de Chazal, M. O'Dwyer, R. B. Reilly, IEEE TBME 51(7):1196-1206, 2004. doi:10.1109/TBME.2004.827359 | inter-patient (defines DS1/DS2) | `dechazal_2004_morphology_rr_features.pdf` | Tanishq |
| 2 | C. Zou et al., IEEE JTEHM 10, Art. 1900508, 2022. doi:10.1109/JTEHM.2022.3202749 (open access) | inter-patient | `zou_2022_rf_segment_label.pdf` | Tanishq |
| 3 | C. Lakshminarayan, T. Basil, arXiv:1607.03822, 2016 | inter-patient | `lakshminarayan_2016_feature_extraction_ml.pdf` | Tanishq |
| 4 | M. Kachuee, S. Fazeli, M. Sarrafzadeh, IEEE ICHI 2018, pp. 443-444. arXiv:1805.00794 | intra-patient | `kachuee_2018_ecg_deep_transferable_representation.pdf` | Parth |
| 5 | T. Wang, C. Lu, Y. Sun, M. Yang, C. Liu, C. Ou, Entropy 23(1):119, 2021. doi:10.3390/e23010119 | inter-patient | `wang_2021_cwt_cnn_rr_entropy.pdf` | Parth |
| 6 | P. Rajpurkar et al., arXiv:1707.01836, 2017 | patient-disjoint (not MIT-BIH) | `rajpurkar_2017_cardiologist_level_arrhythmia_cnn.pdf` | Parth |
| 7 | R. Salvi, AIxSET 2024, pp. 316-319. doi:10.1109/AIxSET62544.2024.00054 | random 80:20 split on a different (12-lead rhythm) database | `salvi_2024_explainable_cnn_bigru.pdf` | Ishaan |
| 8 | H. Sigurthorsdottir et al., Computing in Cardiology 47, 2020. doi:10.22489/CinC.2020.198 | hidden test sets | `sigurthorsdottir_2020_crnn_ecg.pdf` | Ishaan |
| 9 | S. Mousavi, F. Afghah, IEEE ICASSP 2019, pp. 1308-1312. arXiv:1812.07421 | intra- and inter-patient | `mousavi_2019_seq2seq_ecg_arrhythmia.pdf` | Riddhi |
| 10 | S. Hu, W. Cai, T. Gao, J. Zhou, M. Wang, Physiol. Meas. 42(12):125001, 2021. doi:10.1088/1361-6579/ac3e88 | patient-adapted (DS1 + first 5 min of DS2 records for training) | `hu_2021_transformer_heartbeat.pdf` | Riddhi |
| - | Z. Ahmad, A. Tabassum, L. Guan, N. M. Khan, IEEE Access 9:100615-100626, 2021 (cited as an intra-patient example) | intra-patient | `ahmad_2021_multimodal_fusion_ecg.pdf` | - |

Corrections relative to the synopsis reference list: the random-forest
segment-label paper is by **Zou et al.** (not "Li et al."); Mousavi et al. 2019
has two authors (Mousavi, Afghah); Salvi (2024) reports 94.44% accuracy and does
not use MIT-BIH (it classifies 5 rhythm types in a PhysioNet 12-lead database);
Hu et al. (2021) train on the first 5 min of each DS2 record, so their result is
patient-adapted rather than purely inter-patient.
