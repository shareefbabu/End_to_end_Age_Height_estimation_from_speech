# Data Preparation to create npz files for training and validation
# Generate 1 sec chunks shifted by 0.1s from input speech data and 
# store in output file along with targets
# Size of the speech chunks could be varied in dataprep_ht_age.py  

# For MALE SPEAKERS   Creating the data.. 
# Comment this after creating the npz file 
python dataprep_ht_age.py ../list/male_tr_mfcc_vad_ht_age.list ../list/male_val_mfcc_vad_ht_age.list male_tr_val_data_targets_200fr_10frshift_60mfcc_height.npz  

# This is the neural network traiing 
python cnn_fstat_2output_HT_age.py  ../model/means ../model/tr_mfcc_zstat_ht_3layers_256_512_256_assign_mean_asWeights_batch128_epch100_lr_e-5_total_fit1.h5  ../model/ht_age_svr15360_male.csv ../model/ht_age_biasSVR_male.txt  male_tr_mfcc_vad_ht_age.list male_ts_mfcc_vad_ht_age.list male_val_mfcc_vad_ht_age.list HT_AGE_onelayer_150frames_with_10frms_Overlap_inputs_svrwts_with_valsplit_lr_2e-4_dropout_dot3_loop15_epch1_ saved_gmmmeans_SVRwts_multiple_inp_age_ht male_tr_val_data_targets_200fr_10frshift_60mfcc_height.npz

# For FEMALE SPEAKERS
python dataprep_ht_age.py ../list/female_tr_mfcc_vad_ht_age.list ../list/female_val_mfcc_vad_ht_age.list female_tr_val_data_targets_200fr_10frshift_60mfcc_height.npz  

python cnn_fstat_2output_HT_age.py  ../model/means ../model/tr_mfcc_zstat_ht_3layers_256_512_256_assign_mean_asWeights_batch128_epch100_lr_e-5_total_fit1.h5  ../model/ht_age_svr15360_female.csv ../model/ht_age_biasSVR_female.txt  ../list/female_tr_mfcc_vad_ht_age.list ../list/female_ts_mfcc_vad_ht_age.list ../list/female_val_mfcc_vad_ht_age.list HT_AGE_onelayer_200frames_with_10frms_Overlap_inputs_svrwts_with_valsplit_lr_2e-4_dropout_dot3_loop10_epch1_ saved_gmmmeans_SVRwts_multiple_inp_age_ht female_tr_val_data_targets_200fr_10frshift_60mfcc_height.npz 

