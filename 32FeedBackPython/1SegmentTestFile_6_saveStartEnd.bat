
::Train
if 1==0 (

python 1SegmentTestFile_6_saveStartEnd.py --SegTestPred PredictE_20200115_1ManualNotWeightForSeg1_9960_1550_1_5
cd..
cd 12CnnActionCode
python 4test_7_parameterEntropy.py --modelDir ./saveModel_20200115_1ManualSegForAction1_9500_750/model-750

)


::------------------------ only6------------------------------------------------------------------------------

:: NotWeightSegm-----ManualSegForAction
if 1==0 (

python 1SegmentTestFile_6_saveStartEnd.py --SegTestPred PredictE_20200115_1ManualNotWeightForSeg1_9960_1550_only6
cd..
cd 12CnnActionCode
python 4test_7_parameterEntropy.py --modelDir ./saveModel_20200115_1ManualSegForAction1_9500_750/model-750

)




:: NotWeightSegm-----ViaWeightSegForAction
if 1==0 (

python 1SegmentTestFile_6_saveStartEnd.py --SegTestPred PredictE_20200115_1ManualNotWeightForSeg1_9960_1550_only6
cd..
cd 12CnnActionCode
python 4test_7_parameterEntropy.py --modelDir ./saveModel_20200115_2ViaWeightSegForAction1_9542_1400/model-1400

)


:: ViaWeightSegm-----ManualSegForAction
if 1==0 (

python 1SegmentTestFile_6_saveStartEnd.py --SegTestPred PredictE_20200115_1ManualViaWeightForSeg1_9960_1250_only6
cd..
cd 12CnnActionCode
python 4test_7_parameterEntropy.py --modelDir ./saveModel_20200115_1ManualSegForAction1_9500_750/model-750

)

:: ViaWeightSegm-----ViaWeightSegForAction
if 1==0 (

python 1SegmentTestFile_6_saveStartEnd.py --SegTestPred PredictE_20200115_1ManualViaWeightForSeg1_9960_1250_only6
cd..
cd 12CnnActionCode
python 4test_7_parameterEntropy.py --modelDir ./saveModel_20200115_2ViaWeightSegForAction1_9542_1400/model-1400

)

::------------------------ test------------------------------------------------------------------------------
if 1==1 (

python 1SegmentTestFile_6_saveStartEnd.py --SegTestPred PredictE_20200115_1ManualViaWeightForSeg1_9960_1250_only6
cd..
cd 12CnnActionCode
python 4test_7_parameterEntropy.py --modelDir ./saveModel_20200115_2ViaWeightSegForAction1_9542_1400/model-1400

)

pause

