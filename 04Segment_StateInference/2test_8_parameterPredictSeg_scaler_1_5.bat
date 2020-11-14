
:: set modelPath=./saveModel/model-1550

::set modelPath=./saveModel_20200115_1ManualViaWeightForSeg1_9960_1250/model-1250

::set modelPath=./saveModel_20200201_1ManualViaEntropyForSeg1_9950_1550/model-1550

set modelPath=./saveModel/model-1599

if 1==1 (

python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_iw_1.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_iw_2.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_iw_3.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_iw_4.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_iw_5.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_ph_1.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_ph_2.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_ph_3.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_ph_4.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_ph_5.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_rp_1.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_rp_2.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_rp_3.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_rp_4.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_rp_5.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_sd_1.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_sd_2.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_sd_3.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_sd_4.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_sd_5.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_wd_1.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_wd_2.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_wd_3.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_wd_4.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user1_wd_5.mat --dataDir Data_DiscretizeCsi/user1_test_data --modelDir %modelPath%


)



if 1==1 (

python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_iw_1.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_iw_2.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_iw_3.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_iw_4.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_iw_5.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_ph_1.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_ph_2.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_ph_3.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_ph_4.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_ph_5.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_rp_1.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_rp_2.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_rp_3.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_rp_4.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_rp_5.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_sd_1.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_sd_2.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_sd_3.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_sd_4.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_sd_5.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_wd_1.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_wd_2.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_wd_3.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_wd_4.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user2_wd_5.mat --dataDir Data_DiscretizeCsi/user2_test_data --modelDir %modelPath%

)



if 1==1 (

python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_iw_1.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_iw_2.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_iw_3.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_iw_4.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_iw_5.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_ph_1.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_ph_2.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_ph_3.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_ph_4.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_ph_5.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_rp_1.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_rp_2.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_rp_3.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_rp_4.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_rp_5.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_sd_1.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_sd_2.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_sd_3.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_sd_4.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_sd_5.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_wd_1.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_wd_2.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_wd_3.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_wd_4.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user3_wd_5.mat --dataDir Data_DiscretizeCsi/user3_test_data --modelDir %modelPath%


)


if 1==1 (

python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_iw_1.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_iw_2.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_iw_3.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_iw_4.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_iw_5.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_ph_1.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_ph_2.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_ph_3.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_ph_4.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_ph_5.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_rp_1.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_rp_2.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_rp_3.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_rp_4.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_rp_5.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_sd_1.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_sd_2.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_sd_3.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_sd_4.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_sd_5.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_wd_1.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_wd_2.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_wd_3.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_wd_4.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user4_wd_5.mat --dataDir Data_DiscretizeCsi/user4_test_data --modelDir %modelPath%

)



if 1==1 (

python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_iw_1.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_iw_2.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_iw_3.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_iw_4.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_iw_5.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_ph_1.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_ph_2.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_ph_3.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_ph_4.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_ph_5.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_rp_1.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_rp_2.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_rp_3.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_rp_4.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_rp_5.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_sd_1.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_sd_2.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_sd_3.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_sd_4.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_sd_5.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%


python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_wd_1.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_wd_2.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_wd_3.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_wd_4.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%
python 2test_8_parameterPredictSeg_scaler.py --csiFile user5_wd_5.mat --dataDir Data_DiscretizeCsi/user5_test_data --modelDir %modelPath%

)

pause
